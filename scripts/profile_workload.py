"""Warmed model workload shared by timeline and counter collection."""
from contextlib import ExitStack, contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import traceback


def module_id(path):
    prefix = 'model.language_model.'
    if path == 'model.language_model':
        return 'execution.model'
    if path.startswith(prefix + 'layers.'):
        return 'layer.' + path[len(prefix + 'layers.'):]
    return {'model.language_model.embed_tokens': 'embedding',
            'model.language_model.norm': 'final_norm',
            'model.language_model.rotary_emb': 'position_embeddings'}.get(path, path)


@contextmanager
def module_ranges(model, trace):
    """Observe actual module/function calls; never infer ownership from kernel names."""
    import torch.nn.functional as functional
    from models.qwen3_5 import model as implementation, attention_kernels
    handles, patches, active, modules = [], [], [], {}

    @contextmanager
    def scope(label, info):
        modules.setdefault(label, {'id': label, **info})
        active.append(label)
        try:
            with trace.module(label):
                yield
        finally:
            active.pop()

    for path, module in model.named_modules():
        if not path or path == 'model':
            continue
        label = module_id(path)
        info = {'label': type(module).__name__, 'torch_path': path, 'kind': 'nn.Module'}
        if hasattr(module, 'in_features'):
            info['label'] += f' · {module.in_features} → {module.out_features}'
        weight = getattr(module, 'weight', None)
        if weight is not None:
            info['weight_shape'] = list(weight.shape)
        stack = []
        def before(module, args, label=label, info=info, stack=stack):
            context = scope(label, info)
            context.__enter__()
            stack.append(context)
        def after(module, args, output, stack=stack):
            if stack:
                stack.pop().__exit__(None, None, None)
        handles += [module.register_forward_pre_hook(before),
                    module.register_forward_hook(after, always_call=True)]

    def patch(owner, name, suffix, label, fixed=False):
        original = getattr(owner, name)
        @wraps(original)
        def wrapped(*args, **kwargs):
            if not active and not fixed:
                return original(*args, **kwargs)
            tag = suffix if fixed else active[-1] + '.' + suffix
            with scope(tag, {'label': label, 'kind': 'function', 'torch_path': original.__qualname__}):
                return original(*args, **kwargs)
        setattr(owner, name, wrapped)
        patches.append((owner, name, original))

    patch(model, 'decode_logits', 'execution.model', 'Decoder 실행·상태', fixed=True)
    for owner, name, suffix, label in [
        (implementation, 'torch_chunk_gated_delta_rule', 'delta', 'Chunk gated delta rule'),
        (implementation, 'torch_recurrent_gated_delta_rule', 'delta', 'Recurrent gated delta rule'),
        (implementation, 'apply_rotary_pos_emb', 'rope', 'RoPE 적용'),
        (attention_kernels, 'decode_attention', 'attention', 'Decode attention + KV'),
        (functional, 'scaled_dot_product_attention', 'attention', 'Scaled dot-product attention'),
        (functional, 'conv1d', 'conv1d', 'Causal depthwise convolution'),
        (functional, 'silu', 'silu', 'SiLU'),
    ]:
        patch(owner, name, suffix, label)
    try:
        yield modules
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def run(directory, trace, *, model_name, output, single_decode=False, counter_phase=None):
    import torch
    from safetensors.torch import load_file
    if model_name != 'qwen3_5':
        raise ValueError(f'No checkpoint profiling workload for {model_name}')
    from models.qwen3_5.evaluation import PlainEngine, check, reference_capture, save, sha_tensor
    from models.qwen3_5.model import Qwen3_5ForCausalLM

    directory = Path(directory).resolve()
    results = directory / 'results'
    contract = json.loads((results / 'contract.json').read_text())
    if contract.get('quantization') is not None or contract.get('mtp_enabled') is not False:
        raise ValueError('Profiling requires an unchanged checkpoint with MTP disabled')
    if len(contract['lengths']) != 1:
        raise ValueError('Profile one input length per experiment')
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    n, count = contract['lengths'][0], contract['new_tokens']
    inputs = json.loads((results / 'inputs.json').read_text())['cases'][str(n)]
    ids = torch.tensor([inputs['input_ids']], device='cuda')
    if sha_tensor(ids) != inputs['sha256']:
        raise ValueError('Input tensor identity mismatch')
    weights = os.environ.get('KWB_PROFILE_WEIGHTS', contract['models']['candidate']['path'])
    dtype = getattr(torch, {'fp16': 'float16', 'bf16': 'bfloat16', 'fp32': 'float32'}[contract['candidate_dtype']])
    model = Qwen3_5ForCausalLM.from_pretrained_dir(weights, dtype=dtype, device='cuda',
                                                 kernels=contract['kernels'], mtp=False)
    model.fuse_projections()
    engine = PlainEngine(model, contract['cache_capacity'], name=contract['engines'][0])
    engine.target(ids)
    engine.decoder.prepare()
    reference_dir = directory.parents[2] / 'models/cache/references' / directory.parent.name / directory.name
    identity = json.loads((results / f'reference_{n}.json').read_text())
    reference_path = reference_dir / f'{n}.safetensors'
    if not reference_path.is_file():
        ledger = json.loads((results / 'correctness.json').read_text())['checks']
        original = next(c for c in ledger if c['engine'] == engine.name and c['input_tokens'] == n and not c.get('reset_repeat'))
        captured = reference_capture(engine, ids, count)
        checked = {'passed': original['passed'] and sha_tensor(captured['logits']) == original['logits_sha256']
                   and captured['tokens'][0].tolist() == identity['generated_ids']
                   and captured['replay_max_abs_difference'] == 0,
                   'positions': count, 'logits_sha256': sha_tensor(captured['logits']),
                   'validation_scope': 'exact full-logit hash and token equality to saved HF-validated candidate; independent reset replay'}
    elif counter_phase:
        # The same evaluation already checked all positions and reset parity.
        # Repeating 255 graph replays under NCU needlessly instruments >500k
        # nodes before collection. The captured output is checked below as well.
        ledger = json.loads((results / 'correctness.json').read_text())['checks']
        checks = [c for c in ledger if c['engine'] == engine.name and c['input_tokens'] == n]
        checked = {'passed': bool(checks) and all(c['passed'] for c in checks)
                   and any(c.get('reset_repeat') and c.get('bitwise_equal_to_first') for c in checks),
                   'source': 'results/correctness.json', 'validation_scope': 'reuse full HF/reset gate from this evaluation; profile output checked independently',
                   'positions': contract['teacher_forcing_positions']}
    else:
        reference = load_file(str(reference_path))
        if sha_tensor(reference['logits']) != identity['logits_sha256'] or reference['tokens'][0].tolist() != identity['generated_ids']:
            raise ValueError('HF reference tensor identity mismatch')
        checked = check(engine, ids, reference, contract['gate'])
    save(output / 'correctness.json', checked)
    if not checked['passed']:
        raise ValueError('Profiling correctness gate failed')

    def generate(annotate=False):
        if not annotate:
            engine.reset()
        with trace.range('inference') if annotate else ExitStack():
            with trace.phase('prefill') if annotate else ExitStack():
                hidden, logits = engine.target(ids)
            with trace.module('execution.sampling') if annotate else ExitStack():
                token = logits[:, -1].argmax(-1, keepdim=True)
            tokens = [token]
            with trace.phase('decode') if annotate else ExitStack():
                for step in range(1 if single_decode else count - 1):
                    with trace.step(step) if annotate else ExitStack():
                        with trace.module('execution.decode') if annotate else ExitStack():
                            next_logits = engine.step(token)
                        with trace.module('execution.sampling') if annotate else ExitStack():
                            token = next_logits[:, -1].argmax(-1, keepdim=True)
                        tokens.append(token)
            with trace.module('execution.output') if annotate else ExitStack():
                return torch.cat(tokens, dim=1)

    for _ in range(contract['warmup']):
        generate()
    engine.reset()
    if counter_phase in ('decode', 'unknown'):
        _, logits = engine.target(ids)
        if counter_phase == 'decode':
            first = logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    modules = {}
    try:
        if counter_phase == 'prefill':
            with trace.range('inference'), trace.phase('prefill'):
                _, logits = engine.target(ids)
                generated = logits[:, -1].argmax(-1, keepdim=True)
        elif counter_phase == 'decode':
            with trace.range('inference'), trace.phase('decode'), trace.step(0):
                token = engine.step(first)[:, -1].argmax(-1, keepdim=True)
                generated = torch.cat([first, token], dim=1)
        elif counter_phase == 'unknown':
            with trace.range('inference'):
                token = logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.cudart().cudaProfilerStop()
            tokens = [token]
            for _ in range(count - 1):
                token = engine.step(token)[:, -1].argmax(-1, keepdim=True)
                tokens.append(token)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            with trace.range('inference'):
                generated = torch.cat(tokens, dim=1)
        else:
            with module_ranges(model, trace) as modules:
                with trace.range('graph_capture'):
                    engine.decoder.prepare()
                engine.reset()
                generated = generate(annotate=True)
        torch.cuda.synchronize()
    except Exception:
        save(output / 'workload_error.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    actual = generated[0].cpu().tolist()
    baseline = json.loads((results / f'timing_{engine.name}_{n}.json').read_text())
    expected = baseline['generation_ids'][:len(actual)]
    if actual != expected:
        raise ValueError('Profiled output differs from independent generation')
    return {'engine': engine.name, 'input_tokens': n, 'output_tokens': len(actual),
            'checkpoint': contract['models']['candidate']['repo'],
            'checkpoint_revision': contract['models']['candidate']['revision'],
            'input_ids_sha256': inputs['sha256'],
            'decode_steps': len(actual) - 1, 'output_ids_sha256': sha_tensor(generated),
            'matches_unprofiled_generation': True,
            'matches_hf_greedy': actual == identity['generated_ids'][:len(actual)], 'correctness': checked,
            'scope': 'one warmed generation; graph decode; profiler timings are diagnostic',
            'module_structure': (list(modules.values()) + [
                {'id': 'execution.decode', 'label': 'Decode 입력·실행', 'kind': 'control'},
                {'id': 'execution.sampling', 'label': 'Greedy token 선택', 'kind': 'control'},
                {'id': 'execution.output', 'label': '출력 token 결합', 'kind': 'control'},
            ] if modules else []),
            'module_attribution': ('forward hooks + function NVTX; graph construction node IDs' if modules else
                                   'unattributed; counter specimens have no module annotation')}
