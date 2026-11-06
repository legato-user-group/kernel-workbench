"""Reusable HF reference and INT8/MTP evaluation using the local Qwen3.5 package.

Generation and comparison behavior follows infer-opt; source provenance is recorded
in provenance.json. Model weights are explicit runtime inputs, never embedded here.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
from models.common.timing import summarize_samples as summary


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def sha_tensor(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def measure(fn):
    import torch
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    begin.record()
    value = fn()
    end.record()
    torch.cuda.synchronize()
    return {'wall_ms': (time.perf_counter()-start)*1000,
            'cuda_event_ms': begin.elapsed_time(end)}, value


def target_prefill(model, ids, cache):
    hidden = model.model.language_model(ids, cache)
    return hidden, model.lm_head(hidden[:, -1:])


class HFEngine:
    def __init__(self, model, capacity):
        from transformers import StaticCache
        self.model = model
        self.cache = StaticCache(config=model.config, max_cache_len=capacity)
        self.name = 'hf_bf16'

    def reset(self):
        self.cache.reset()

    def target(self, ids):
        output = self.model(input_ids=ids, past_key_values=self.cache,
                            use_cache=True, logits_to_keep=1)
        return None, output.logits

    def step(self, token):
        return self.target(token)[1]

    def decode(self, ids, hidden, first, count):
        import torch
        token = first
        tokens = [first]
        for _ in range(count - 1):
            token = self.step(token)[:, -1].argmax(-1, keepdim=True)
            tokens.append(token)
        return torch.cat(tokens, dim=1)


class PlainEngine:
    def __init__(self, model, capacity, mode='cudagraph', name=None):
        from .model import FixedShapeDecoder
        self.model = model
        self.cache = model.new_cache(1, capacity)
        self.decoder = FixedShapeDecoder(model, self.cache, mode)
        self.name = name or 'int8_plain_' + mode

    def reset(self):
        self.cache.reset()

    def target(self, ids):
        return target_prefill(self.model, ids, self.cache)

    def step(self, token):
        return self.decoder.step(token)

    decode = HFEngine.decode


class MTPEngine:
    """Split the existing prefill API at first-token readiness; keep all draft work timed."""
    def __init__(self, model, capacity, k=3):
        from .speculative import MTPSpeculativeDecoder
        self.model = model
        self.k = k
        # MTPSpeculativeDecoder adds 4*(k+1)+k internal slots.
        self.decoder = MTPSpeculativeDecoder(model, capacity-4*(k+1)-k,
                                             k=k, mode='cudagraph', verification='parallel')
        self.cache = self.decoder.cache
        self.name = 'int8_mtp3_cudagraph'

    def reset(self):
        from .speculative import SpecStats
        self.cache.reset()
        self.decoder.mtp_cache.reset()
        self.decoder.stats = SpecStats()

    def target(self, ids):
        return target_prefill(self.model, ids, self.cache)

    def initialize_drafts(self, ids, hidden, first):
        import torch
        from .speculative import draft
        cache = self.decoder.mtp_cache
        output = self.model.mtp_hidden(torch.cat([ids[:, 1:], first], dim=1), hidden, cache)[:, -1:]
        token = self.model.lm_head(output).argmax(-1)
        drafts = draft(self.model, output, token, cache, self.k-1)
        self.decoder.tokens.copy_(torch.cat([first, token, *drafts], dim=1))

    def decode(self, ids, hidden, first, count):
        import torch
        self.initialize_drafts(ids, hidden, first)
        tokens = first[0].tolist()
        while len(tokens) < count:
            tokens += self.decoder.round()
        self.last_emitted = len(tokens)-1
        return torch.tensor([tokens[:count]], device=ids.device)


class VerifyEngine:
    """Teacher-force the existing parallel verification kernel, four positions at a time."""
    def __init__(self, model, capacity):
        import torch
        self.model = model
        self.cache = model.new_cache(1, capacity)
        self.cache.allocate_spec_buffers(4)
        self.tokens = torch.zeros((1, 4), device='cuda', dtype=torch.long)
        self.graph = None
        self.name = 'int8_mtp_parallel_verify4'

    def reset(self):
        self.cache.reset()

    def target(self, ids):
        return target_prefill(self.model, ids, self.cache)

    def forward(self):
        return self.model.lm_head(self.model.model.language_model.forward_verify(self.tokens, self.cache))

    def prepare(self):
        import torch
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.forward()

    def block(self, tokens):
        import torch
        width = tokens.shape[1]
        self.tokens[:, :width].copy_(tokens)
        if width < 4: self.tokens[:, width:].copy_(tokens[:, -1:])
        self.graph.replay()
        output = self.output[:, :width].clone()
        self.cache.commit(torch.tensor([width], device=tokens.device))
        self.cache.seen_tokens += width
        return output


def logits_metrics(want, got, step, gate):
    import torch
    # Reuse the original infer-opt comparison, including KL direction.
    from .validation import kl_and_top1
    want, got = want.to(got.device).float().reshape(1, -1), got.float().reshape(1, -1)
    kl, agreement, maxdiff = kl_and_top1(want, got)
    finite = bool(torch.isfinite(want).all() and torch.isfinite(got).all())
    top = want.topk(2, dim=-1).values[0]
    margin = float(top[0]-top[1])
    diff = got-want
    return {'step':step, 'kl':kl, 'top1_equal':agreement == 1., 'hf_margin':margin,
            'strong_mismatch':agreement != 1. and margin > gate['margin_threshold'],
            'max_abs_logit_diff':maxdiff,
            'relative_l2':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(want).clamp_min(1e-12)),
            'allclose_failed_elements':int((~torch.isclose(got, want, atol=.02, rtol=.02)).sum()),
            'finite':finite}


def check(engine, ids, reference, gate):
    import torch
    engine.reset()
    _, logits = engine.target(ids)
    outputs = [logits[:, -1].cpu().clone()]
    forced = reference['tokens'].cuda()
    if isinstance(engine, VerifyEngine):
        for start in range(0, forced.shape[1]-1, 4):
            block = engine.block(forced[:, start:min(start+4, forced.shape[1]-1)])
            outputs.extend(block[:, i].cpu().clone() for i in range(block.shape[1]))
    else:
        for i in range(forced.shape[1]-1):
            outputs.append(engine.step(forced[:, i:i+1])[:, -1].cpu().clone())
    actual = torch.cat(outputs)
    rows = [logits_metrics(reference['logits'][i], actual[i], i, gate) for i in range(len(actual))]
    expected_position = ids.shape[1]+forced.shape[1]-1
    position = int(engine.cache.position.item())
    mean_kl = statistics.mean(r['kl'] for r in rows[1:])
    passed = (all(r['finite'] for r in rows) and position == expected_position
              and rows[0]['kl'] <= gate['max_mean_kl'] and mean_kl <= gate['max_mean_kl']
              and not any(r['strong_mismatch'] for r in rows))
    return {'engine':engine.name, 'input_tokens':ids.shape[1], 'positions':len(rows),
            'prefill_kl':rows[0]['kl'], 'decode_mean_kl':mean_kl,
            'decode_max_kl':max(r['kl'] for r in rows[1:]),
            'top1_agreement':statistics.mean(r['top1_equal'] for r in rows),
            'strong_mismatches':sum(r['strong_mismatch'] for r in rows),
            'cache_position':position, 'expected_cache_position':expected_position,
            'logits_sha256':sha_tensor(actual), 'passed':passed, 'steps':rows}


def benchmark(engine, ids, count, warmup, repeats, emit, trace=None, previous=None):
    results = {}
    source = ('unprofiled' if trace is None or getattr(trace, 'independent_timing', False)
              else 'profile_collection')
    def first():
        hidden, logits = engine.target(ids)
        return hidden, logits[:, -1].argmax(-1, keepdim=True)
    for metric in ('prefill_forward', 'engine_ttft', 'decode', 'generation'):
        samples = list((previous or {}).get(metric, {}).get('samples', []))
        if len(samples) > repeats:
            raise ValueError('Saved sample count exceeds requested repeats')
        if len(samples) == repeats:
            results[metric] = summary(samples, source)
            continue
        remaining = repeats-len(samples)
        for index in range(warmup+remaining):
            engine.reset()
            if metric == 'prefill_forward':
                fn = lambda:engine.target(ids)
            elif metric == 'engine_ttft':
                fn = first
            elif metric == 'decode':
                hidden, token = first()
                fn = lambda:engine.decode(ids, hidden, token, count)
            else:
                def fn():
                    with trace.phase('prefill') if trace else nullcontext():
                        h, logits = engine.target(ids)
                    t = logits[:, -1].argmax(-1, keepdim=True)
                    with trace.phase('decode') if trace else nullcontext():
                        return engine.decode(ids, h, t, count)
            if index < warmup:
                fn()
                import torch
                torch.cuda.synchronize()
            else:
                with trace.range('inference') if trace and metric == 'generation' else nullcontext():
                    sample, value = measure(fn)
                if metric in ('decode', 'generation'):
                    sample['returned_output_tokens'] = value.shape[1]
                    assert value.shape[1] == count
                    sample['emitted_decode_tokens'] = (engine.last_emitted if isinstance(engine, MTPEngine) else count-1)
                    if isinstance(engine, MTPEngine):
                        sample['mtp_acceptance'] = engine.decoder.stats.as_dict()
                    sample['output_ids_sha256'] = sha_tensor(value)
                    if index == warmup:
                        emit('generation_ids', value.cpu().tolist()[0])
                samples.append(sample)
                emit(metric, summary(samples, source))
        results[metric] = summary(samples, source)
    results['decode_ms_per_token'] = results['decode']['wall_ms']['median']/(count-1)
    results['decode_tokens_per_second'] = (count-1)*1000/results['decode']['wall_ms']['median']
    return results


def make_inputs(tokenizer, lengths):
    import torch
    from .validation import PROSE
    text = ''.join(PROSE.format(i=i) for i in range(1, 97))
    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids
    assert tokens.shape[1] >= max(lengths)
    return {length:tokens[:, :length].clone() for length in lengths}, text


def reference_capture(engine, ids, count):
    import torch
    engine.reset()
    _, logits = engine.target(ids)
    tokens, values = [], []
    for step in range(count):
        if step: logits = engine.step(token)
        values.append(logits[:, -1].cpu().clone())
        token = logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(token.cpu().clone())
    ref = {'tokens':torch.cat(tokens,dim=1), 'logits':torch.cat(values)}
    # A separate cached replay supplies the reference values used by candidates.
    engine.reset()
    _, logits = engine.target(ids)
    replay = [logits[:, -1].cpu().clone()]
    for step in range(count-1):
        replay.append(engine.step(ref['tokens'][:,step:step+1].to(ids.device))[:, -1].cpu().clone())
    replay = torch.cat(replay)
    ref['replay_max_abs_difference'] = float((replay.float()-ref['logits'].float()).abs().max())
    assert torch.equal(replay.argmax(-1), ref['tokens'][0])
    ref['logits'] = replay
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', default='128,512,1024,4096')
    parser.add_argument('--new-tokens', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--resume', action='store_true', help='Reuse this session\'s pinned references and raw HF samples')
    parser.add_argument('--skip-hf-timing', action='store_true', help='Use HF for correctness only; time the INT8 engines')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', '/tmp/infer-opt-models/huggingface')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/infer-opt-models/inductor')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    started = time.time()
    def stage(name):
        save(out/'progress.json', {'stage':name,'elapsed_s':time.time()-started})
        print(name, flush=True)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        p = torch.cuda.get_device_properties(0)
        assert 'A100' in p.name and p.total_memory/2**30 >= 75
        models = json.loads(Path(args.models).read_text())
        lengths = [int(n) for n in args.lengths.split(',')]
        capacity = max(lengths)+args.new_tokens+64
        gate = {'max_mean_kl':.005,'margin_threshold':.5,'strong_mismatches_allowed':0,
                'allclose_atol':.02,'allclose_rtol':.02,'allclose_is_diagnostic':True}
        environment = {'gpu':p.name,'memory_gib':p.total_memory/2**30,'torch':torch.__version__,
                       'cuda':torch.version.cuda,'packages':{n:importlib.metadata.version(n) for n in
                       ['transformers','flash-linear-attention','triton','huggingface_hub']},
                       'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)}
        if args.resume:
            old = json.loads((out/'contract.json').read_text())
            assert old['lengths']==lengths and old['new_tokens']==args.new_tokens and old['repeats']==args.repeats
            assert old['gate']==gate and old['cache_capacity']==capacity
            for role in ('reference','candidate'):
                assert old['models'][role]['revision']==models[role]['revision']
            save(out/'resume.json', {'started_at_unix':started,'reuse_reference_tensors':True,
                                    'fresh_warmup_before_remaining_samples':args.warmup,
                                    'skip_hf_timing':args.skip_hf_timing})
        save(out/'environment.json', environment)
        save(out/'contract.json', {'reference':'official HF BF16 cached eager; SDPA; StaticCache',
            'candidate':'published INT8 group-128 + INT8 MTP head; BF16 lm_head',
            'lengths':lengths,'new_tokens':args.new_tokens,'warmup':args.warmup,'repeats':args.repeats,
            'hf_timing_requested':not args.skip_hf_timing,
            'cache_capacity':capacity,'gate':gate,'teacher_forcing_positions':args.new_tokens,
            'timing':'separate unprofiled wall/CUDA Event passes; initial MTP drafts included in decode',
            'state_comparison':'not checked; logits and cache position checked',
            'models':{role:{k:v for k,v in m.items() if k != 'files'} for role,m in models.items()}})
        tokenizer = AutoTokenizer.from_pretrained(models['reference']['path'])
        inputs, text = make_inputs(tokenizer, lengths)
        save(out/'inputs.json', {'category':'prose','repeated_source':True,'chat_template':False,
            'add_special_tokens':False,'source_text':text,'cases':{
                str(n):{'input_ids':ids[0].tolist(),'sha256':sha_tensor(ids)} for n,ids in inputs.items()}})
        references = {}
        temporary = Path('/tmp/infer-opt-models/evaluation-reference')
        temporary.mkdir(exist_ok=True)
        from safetensors.torch import load_file, save_file
        if args.resume:
            references={n:load_file(str(temporary/f'{n}.safetensors')) for n in lengths}
            for n, reference in references.items():
                meta=json.loads((out/f'reference_{n}.json').read_text())
                assert sha_tensor(reference['logits'])==meta['logits_sha256']
                assert reference['tokens'][0].tolist()==meta['generated_ids']
        pending=[]
        for n in lengths:
            path=out/f'timing_hf_bf16_{n}.json'
            old=json.loads(path.read_text()) if args.resume and path.exists() else {}
            if not all(old.get(k,{}).get('sample_count')==args.repeats for k in
                       ('prefill_forward','engine_ttft','decode','generation')):
                pending.append(n)
        if args.skip_hf_timing:
            pending=[]
        if pending or not args.resume:
            stage('load_hf_bf16')
            model, loading = AutoModelForCausalLM.from_pretrained(models['reference']['path'],
                dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa',
                output_loading_info=True)
            assert not loading.get('missing_keys') and not loading.get('mismatched_keys'), loading
            model.eval().requires_grad_(False)
            engine = HFEngine(model, capacity)
            engine.target(inputs[lengths[0]].cuda())  # materialize lazy cache buffers
            if not args.resume:
                for n, ids_cpu in inputs.items():
                    stage(f'hf_reference_{n}')
                    references[n] = reference_capture(engine, ids_cpu.cuda(), args.new_tokens)
                    reference = references[n]
                    save_file({k:reference[k] for k in ('tokens','logits')},str(temporary/f'{n}.safetensors'))
                    save(out/f'reference_{n}.json', {'generated_ids':reference['tokens'][0].tolist(),
                        'logits_sha256':sha_tensor(reference['logits']),
                        'reference_replay_max_abs_difference':reference['replay_max_abs_difference'],
                        'text':tokenizer.decode(reference['tokens'][0])})
            for n in pending:
                stage(f'benchmark_hf_{n}')
                path = out/f'timing_hf_bf16_{n}.json'
                partial=json.loads(path.read_text()) if args.resume and path.exists() else {}
                def emit(k,v):
                    partial[k]=v; save(path, partial)
                benchmark(engine,inputs[n].cuda(),args.new_tokens,args.warmup,args.repeats,emit,previous=partial)
            del engine, model
            gc.collect(); torch.cuda.empty_cache()
        stage('load_int8_mtp_checkpoint')
        from .model import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_quantized_dir(models['candidate']['path'],device='cuda',kernels='fla-delta')
        assert model.mtp is not None and model.config.num_hidden_layers == 64
        fused = model.fuse_projections()
        save(out/'candidate_load.json', {'mtp_loaded':True,'fused_projection_groups':fused,
                                       'allocated_gib':torch.cuda.memory_allocated()/2**30})
        checks = []
        plain_outputs = {}
        for kind in ('plain','verify','mtp'):
            stage('prepare_'+kind)
            engine = (PlainEngine(model, capacity) if kind=='plain' else
                      VerifyEngine(model, capacity) if kind=='verify' else MTPEngine(model, capacity))
            ids = inputs[lengths[0]].cuda()
            engine.reset(); hidden, logits = engine.target(ids)
            if kind=='mtp': engine.initialize_drafts(ids,hidden,logits[:,-1].argmax(-1,keepdim=True))
            if kind=='plain': engine.decoder.prepare()
            elif kind=='verify': engine.prepare()
            else: engine.decoder.prepare()
            for n, ids_cpu in inputs.items():
                ids = ids_cpu.cuda()
                if kind != 'mtp':
                    stage(f'teacher_force_{kind}_{n}')
                    checked = check(engine,ids,references[n],gate)
                    checks.append(checked)
                    save(out/'correctness.json', {'gate':gate,'checks':checks})
                if kind=='verify': continue
                stage(f'benchmark_{kind}_{n}')
                path = out/f'timing_{engine.name}_{n}.json'
                partial = {}
                def emit(k,v):
                    partial[k]=v; save(path,partial)
                benchmark(engine,ids,args.new_tokens,args.warmup,args.repeats,emit)
                generated = torch.tensor(partial['generation_ids'])
                if kind=='plain': plain_outputs[n] = generated
                else:
                    agreement = generated == plain_outputs[n]
                    save(out/f'mtp_parity_{n}.json', {'matches_plain_int8':bool(agreement.all()),
                        'token_agreement':float(agreement.float().mean()),
                        'first_divergence':None if bool(agreement.all()) else int((~agreement).nonzero()[0]),
                        'reference':'same-checkpoint plain INT8; independent from HF distribution gate',
                        'emitted_decode_tokens':engine.last_emitted,
                        'returned_tokens':args.new_tokens})
            if kind != 'mtp':
                stage('reset_repeat_'+kind)
                repeated=check(engine,inputs[lengths[0]].cuda(),references[lengths[0]],gate)
                first=next(c for c in checks if c['engine']==engine.name)
                repeated['reset_repeat']=True
                repeated['bitwise_equal_to_first']=repeated['logits_sha256']==first['logits_sha256']
                checks.append(repeated)
                save(out/'correctness.json',{'gate':gate,'checks':checks})
            del engine, hidden, logits
            gc.collect(); torch.cuda.empty_cache()
        stage('complete')
        save(out/'exit.json', {'success':True,'elapsed_s':time.time()-started,
             'hf_distribution_gate_passed':all(c['passed'] for c in checks),
             'speedups_reported':False})
    except BaseException:
        save(out/'exit.json', {'success':False,'traceback':traceback.format_exc(),
                              'elapsed_s':time.time()-started})
        traceback.print_exc()
        raise


_PREPARED = None


def run_report(config, trace, model_description):
    """Standard Workbench runner for one case, using fresh suite reference artifacts."""
    global _PREPARED
    import torch
    from safetensors.torch import load_file
    options = config.get('runner_options', {})
    if not options.get('weights') or not options.get('reference_dir'):
        raise ValueError('Set weights and reference_dir; run the HF evaluation suite first to capture its pinned references.')
    key = json.dumps(config, sort_keys=True)
    n = config['input_tokens']
    if _PREPARED is None or _PREPARED['key'] != key:
        _PREPARED = None
        gc.collect(); torch.cuda.empty_cache()
        from .model import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_quantized_dir(options['weights'],device='cuda',kernels='fla-delta')
        model.fuse_projections()
        reference = load_file(str(Path(options['reference_dir'])/f'{n}.safetensors'))
        if reference['tokens'].shape[1] != config['output_tokens']:
            raise ValueError('Reference continuation length differs from output_tokens')
        inputs = json.loads(Path(options['input_manifest']).read_text())
        reference_root = Path(options['input_manifest']).parent
        contract = json.loads((reference_root/'contract.json').read_text())
        identity = json.loads((reference_root/f'reference_{n}.json').read_text())
        lineage = json.loads((Path(options['weights'])/'quantize_info.json').read_text())
        if (lineage['source'] != contract['models']['reference']['repo'] or
                lineage['source_revision'] != contract['models']['reference']['revision']):
            raise ValueError('HF reference revision does not match checkpoint lineage')
        if (sha_tensor(reference['logits']) != identity['logits_sha256'] or
                reference['tokens'][0].tolist() != identity['generated_ids']):
            raise ValueError('Reference tensors do not match their saved identity')
        ids = torch.tensor([inputs['cases'][str(n)]['input_ids']],device='cuda')
        if ids.shape[1] != n or sha_tensor(ids) != inputs['cases'][str(n)]['sha256']:
            raise ValueError('Input IDs do not match their manifest')
        capacity = options.get('cache_capacity', n+config['output_tokens']+64)
        engine = MTPEngine(model,capacity)
        engine.reset(); hidden, logits=engine.target(ids)
        engine.initialize_drafts(ids,hidden,logits[:,-1].argmax(-1,keepdim=True))
        engine.decoder.prepare()
        verifier = VerifyEngine(model, capacity)
        verifier.reset(); verifier.target(ids); verifier.prepare()
        gate = {'max_mean_kl':options.get('max_mean_kl', .005),'margin_threshold':.5}
        correctness = check(verifier, ids, reference, gate)
        del verifier
        gc.collect(); torch.cuda.empty_cache()
        _PREPARED={'key':key,'engine':engine,'ids':ids,'reference':reference,
                   'correctness':correctness,'warmed':False}
    state=_PREPARED
    from .analysis import supplement
    analysis = supplement(config)
    partial={}
    warmup = 0 if state['warmed'] else config.get('warmup',3)
    result=benchmark(state['engine'],state['ids'],config['output_tokens'],warmup,1,
                     lambda k,v:partial.update({k:v}),trace=trace)
    state['warmed'] = True
    return {'model':model_description,'config':config,'gpu':{'name':torch.cuda.get_device_name()},
            'total_ms':result['generation']['wall_ms']['median'],
            'phase_ms':{'prefill':result['prefill_forward']['wall_ms']['median'],
                        'decode':result['decode']['wall_ms']['median']},
            'evaluation':result,'correctness':state['correctness'],
            **{k: analysis[k] for k in ('runner_id', 'estimates', 'kernel_plan', 'estimate_scope', 'analysis_notes', 'phase_estimate_reasons')},
            'prepared_model_persistent':True,'warmup_iterations':warmup,
            'versions':{'torch':torch.__version__},
            'warnings':['Phases are measured in separate passes; their sum is not generation time.',
                        'HF teacher-forcing evidence comes from pinned reference tensors captured by the evaluation suite.',
                        'Correctness failure makes timings diagnostic; do not publish validated speedups.']}
