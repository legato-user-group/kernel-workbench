"""Evaluate an unchanged HF checkpoint, with MTP disabled and no quantization."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import traceback

from .evaluation import HFEngine, PlainEngine, benchmark, check, make_inputs, reference_capture, save, sha_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lengths', required=True)
    parser.add_argument('--new-tokens', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], required=True)
    parser.add_argument('--expected-gpu', required=True)
    parser.add_argument('--reference-device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--kernels', choices=['torch', 'fla', 'fla-delta'], default='torch')
    parser.add_argument('--skip-hf-timing', action='store_true')
    args = parser.parse_args()
    out = args.output
    started = time.time()

    def stage(name):
        save(out / 'progress.json', {'stage': name, 'elapsed_s': time.time() - started})
        print(name, flush=True)

    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .model import Qwen3_5ForCausalLM

        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        device = torch.cuda.get_device_properties(0)
        if args.expected_gpu not in device.name:
            raise ValueError(f'Expected {args.expected_gpu}; got {device.name}')
        if args.dtype == 'bf16' and device.major < 8:
            raise ValueError('Native BF16 GPU measurement requires compute capability >= 8.0')
        models = json.loads(args.models.read_text())
        if any(models['candidate'][k] != models['reference'][k] for k in ('repo', 'revision')):
            raise ValueError('Unchanged-checkpoint evaluation requires the same reference and candidate revision')
        lengths = [int(n) for n in args.lengths.split(',')]
        capacity = max(lengths) + args.new_tokens + 64
        engine_name = f'{args.dtype}_plain_cudagraph'
        gate = {'max_mean_kl': .005, 'margin_threshold': .5, 'strong_mismatches_allowed': 0,
                'allclose_atol': .02, 'allclose_rtol': .02, 'allclose_is_diagnostic': True}
        save(out / 'environment.json', {'gpu': device.name, 'memory_gib': device.total_memory / 2**30,
             'compute_capability': [device.major, device.minor], 'torch': torch.__version__, 'cuda': torch.version.cuda,
             'packages': {n: importlib.metadata.version(n) for n in ['transformers', 'triton', 'safetensors', 'huggingface_hub']},
             'nvidia_smi': subprocess.check_output(['nvidia-smi', '-q'], text=True)})
        config_path = Path(models['candidate']['path']) / 'config.json'
        checkpoint_config = json.loads(config_path.read_text())
        save(out / 'model_config.json', checkpoint_config)
        save(out / 'contract.json', {'reference': f'official HF BF16 cached eager; SDPA; StaticCache; {args.reference_device}',
             'candidate': f'unchanged HF checkpoint; {args.dtype}; MTP disabled; no quantization',
             'candidate_dtype': args.dtype, 'engines': [engine_name], 'check_engines': [engine_name],
             'runner_id': out.parent.parent.name, 'mtp_enabled': False, 'quantization': None,
             'kernels': args.kernels, 'model_config_file': 'results/model_config.json',
             'lengths': lengths, 'new_tokens': args.new_tokens, 'warmup': args.warmup, 'repeats': args.repeats,
             'hf_timing_requested': False, 'cache_capacity': capacity, 'gate': gate,
             'teacher_forcing_positions': args.new_tokens, 'state_comparison': 'not checked; logits and cache position checked',
             'timing': 'separate unprofiled wall/CUDA Event passes; model load, cache reset and graph capture excluded',
             'models': {role: {k: v for k, v in m.items() if k != 'files'} for role, m in models.items()}})
        tokenizer = AutoTokenizer.from_pretrained(models['reference']['path'])
        inputs, text = make_inputs(tokenizer, lengths)
        save(out / 'inputs.json', {'category': 'prose', 'repeated_source': True, 'chat_template': False,
             'add_special_tokens': False, 'source_text': text,
             'cases': {str(n): {'input_ids': ids[0].tolist(), 'sha256': sha_tensor(ids)} for n, ids in inputs.items()}})
        stage('load_hf_bf16_' + args.reference_device)
        reference_model = AutoModelForCausalLM.from_pretrained(models['reference']['path'], dtype=torch.bfloat16,
                          attn_implementation='sdpa').to(args.reference_device).eval().requires_grad_(False)
        reference_engine = HFEngine(reference_model, capacity)
        references = {}
        reference_dir = out.parents[2].parent / 'models/cache/references' / out.parent.parent.name / out.parent.name
        reference_dir.mkdir(parents=True, exist_ok=True)
        for length, ids in inputs.items():
            stage(f'hf_reference_{length}')
            reference = reference_capture(reference_engine, ids.to(args.reference_device), args.new_tokens)
            references[length] = reference
            save_file({k: reference[k] for k in ('tokens', 'logits')}, str(reference_dir / f'{length}.safetensors'))
            save(out / f'reference_{length}.json', {'generated_ids': reference['tokens'][0].tolist(),
                 'logits_sha256': sha_tensor(reference['logits']),
                 'reference_replay_max_abs_difference': reference['replay_max_abs_difference']})
        del reference_engine, reference_model
        gc.collect()
        torch.cuda.empty_cache()
        stage('load_checkpoint_without_mtp_or_quantization')
        dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
        model = Qwen3_5ForCausalLM.from_pretrained_dir(models['candidate']['path'], dtype=dtype,
                                                     device='cuda', kernels=args.kernels, mtp=False)
        fused = model.fuse_projections()
        save(out / 'candidate_load.json', {'mtp_loaded': model.mtp is not None, 'quantization': None,
             'fused_projection_groups': fused, 'layers': model.config.num_hidden_layers, 'dtype': str(dtype)})
        engine = PlainEngine(model, capacity, name=engine_name)
        ids = inputs[lengths[0]].cuda()
        engine.target(ids)
        stage('prepare_cuda_graph')
        engine.decoder.prepare()
        checks = []
        for length, ids_cpu in inputs.items():
            ids = ids_cpu.cuda()
            stage(f'correctness_{length}')
            first = check(engine, ids, references[length], gate)
            checks.append(first)
            if length == lengths[0]:
                repeated = check(engine, ids, references[length], gate)
                repeated.update(reset_repeat=True, bitwise_equal_to_first=repeated['logits_sha256'] == first['logits_sha256'])
                checks.append(repeated)
            save(out / 'correctness.json', {'gate': gate, 'checks': checks})
            stage(f'benchmark_{length}')
            timing = {'engine': engine_name, 'input_tokens': length}
            def emit(key, value):
                timing[key] = value
                save(out / f'timing_{engine_name}_{length}.json', timing)
            benchmark(engine, ids, args.new_tokens, args.warmup, args.repeats, emit)
        save(out / 'exit.json', {'success': True, 'all_quality_checks_passed': all(c['passed'] for c in checks),
                                'elapsed_s': time.time() - started})
        stage('complete')
    except Exception:
        save(out / 'exit.json', {'success': False, 'traceback': traceback.format_exc(), 'elapsed_s': time.time() - started})
        raise
    finally:
        save(out.parent / 'session_status.json', {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                                                 'state': 'evaluation_process_exited'})


if __name__ == '__main__':
    main()
