"""CPU-only architecture and logical projection costs for the pinned checkpoint.

These are source-derived costs, not kernel timings or observed DRAM traffic.
MTP decode costs require round counts that the recovered evaluation did not save.
"""
import json
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def model_config():
    path = Path(__file__).with_name('config.json')
    if not path.is_file():
        raise FileNotFoundError('Model configuration missing; run scripts/prepare_models.py first')
    return json.loads(path.read_text())['text_config']
RUNNER_ID = 'qwen35_int8_mtp'


def runner_identity(config):
    if config.get('runner_id'):
        return config['runner_id']
    return f'{RUNNER_ID}_prose_{config["input_tokens"]}'


def layer_ops(prefix, kind, c=None):
    c = c or model_config()
    h, f = c['hidden_size'], c['intermediate_size']
    if kind == 'linear_attention':
        k = c['linear_key_head_dim'] * c['linear_num_key_heads']
        v = c['linear_value_head_dim'] * c['linear_num_value_heads']
        matrices = [('linear_attn.in_proj_qkvz', h, 2*k+2*v),
                    ('linear_attn.in_proj_b', h, c['linear_num_value_heads']),
                    ('linear_attn.in_proj_a', h, c['linear_num_value_heads']),
                    ('linear_attn.out_proj', v, h)]
    else:
        q, kv = c['num_attention_heads']*c['head_dim'], c['num_key_value_heads']*c['head_dim']
        matrices = [('self_attn.qkv_proj', h, 2*q+2*kv), ('self_attn.o_proj', q, h)]
    matrices += [('mlp.gate_up_proj', h, 2*f), ('mlp.down_proj', f, h)]
    return [(prefix+'.'+name, a, b) for name, a, b in matrices]


def describe(config):
    c = config.get('model_config') or model_config()
    def operations(prefix, kind):
        matrices = [{'id':name, 'label':f'{name.rsplit(".",1)[-1]} · {a} → {b}'}
                    for name,a,b in layer_ops(prefix,kind,c)]
        mlp = [node for node in matrices if '.mlp.' in node['id']]
        attention = [node for node in matrices if '.mlp.' not in node['id']]
        linear = kind == 'linear_attention'
        attention[-1:-1] = ([{'id':prefix+'.linear_attn.conv1d', 'label':'Causal depthwise conv + SiLU'},
                            {'id':prefix+'.linear_attn.delta', 'label':'Delta rule + recurrent state'}]
                           if linear else [{'id':prefix+'.self_attn.attention', 'label':'SDPA / decode attention + KV'}])
        return [{'id':prefix+'.input_layernorm', 'label':'Input RMSNorm'},
                {'id':prefix+('.linear_attn' if linear else '.self_attn'),
                 'label':'GatedDeltaNet' if linear else 'Gated full attention', 'children':attention},
                {'id':prefix+'.post_attention_layernorm', 'label':'Post-attention RMSNorm'},
                {'id':prefix+'.mlp', 'label':'MLP · SwiGLU', 'children':mlp}]

    children = [{'id': 'embedding', 'label': f'Token embedding · {c["vocab_size"]} × {c["hidden_size"]}'}]
    for i, kind in enumerate(c['layer_types']):
        prefix = f'layer.{i}'
        children.append({'id': prefix, 'label': f'Layer {i} · {"GatedDeltaNet" if kind == "linear_attention" else "Full attention"}',
                         'children': operations(prefix,kind)})
    children += [{'id': 'final_norm', 'label': 'Final RMSNorm'},
                 {'id': 'lm_head', 'label': f'{(config.get("dtype") or "bf16").upper()} lm_head · {c["hidden_size"]} → {c["vocab_size"]}'}]
    if config.get('mtp_enabled', True):
        children.append({'id': 'mtp', 'label': 'MTP · full-attention decoder', 'children': [
            {'id': 'mtp.fc', 'label': f'Normalized embedding + hidden · {2*c["hidden_size"]} → {c["hidden_size"]}'},
            {'id': 'mtp.layer.0', 'label': 'MTP decoder layer', 'children': operations('mtp.layer.0', 'full_attention')}]})
    return {'name': config.get('model_name', 'Qwen/Qwen3.5-27B'), 'hidden': c['hidden_size'], 'layers': c['num_hidden_layers'],
            'linear_attention_layers': c['layer_types'].count('linear_attention'),
            'full_attention_layers': c['layer_types'].count('full_attention'), 'children': children,
            'source': 'pinned checkpoint config.json + models/qwen3_5/model.py'}


def supplement(config):
    """Count projection GEMMs only. Never infer MTP round counts from latency."""
    CONFIG = config.get('model_config') or model_config()
    quantized = config.get('quantization', 'int8') is not None
    precision = 'fp16_tensor' if config.get('dtype') == 'fp16' else 'bf16_tensor'
    n, steps, batch = config['input_tokens'], config['output_tokens']-1, config['batch']
    engine = config.get('runner', '')
    mtp = 'mtp' in engine and engine != 'int8_plain_cudagraph'
    rows, plan = [], []
    matrices = [op for i,kind in enumerate(CONFIG['layer_types']) for op in layer_ops(f'layer.{i}', kind, CONFIG)]
    matrices.append(('lm_head', CONFIG['hidden_size'], CONFIG['vocab_size']))
    for phase in ('prefill', 'decode'):
        if phase == 'decode' and (mtp or steps == 0):
            continue
        for module, a, b in matrices:
            is_head = module == 'lm_head'
            tokens = 1 if is_head or phase == 'decode' else n
            calls = 1 if phase == 'prefill' else steps
            m = batch*tokens
            # INT8 and BF16 scale reads, plus explicit dequantization for prefill.
            weights = 2*a*b if is_head or not quantized else a*b + 2*(a//128)*b
            temporary = 4*a*b if phase == 'prefill' and not is_head and quantized else 0
            traffic = calls*(weights + temporary + 2*m*(a+b))
            path = 'fp32' if quantized and phase == 'decode' and not is_head else precision
            implementation = ('torch.nn.functional.linear · ' + (config.get('dtype') or 'bf16') if is_head or not quantized else
                              'qwen3_5.quant.dequantize_int8 + F.linear' if phase == 'prefill' else
                              'qwen3_5.quant_kernels.w8a16_group_gemv')
            rows.append({'module': module, 'phase': phase, 'flops': {path: 2*m*a*b*calls},
                         'bytes': traffic, 'source': 'estimated', 'calls': calls,
                         'shape': f'[{m}, {a}] × [{b}, {a}]ᵀ',
                         'traffic_model': 'logical projection weights/activation and optional quantization temporaries; cache reuse omitted'})
            plan.append({'module': module, 'phase': phase, 'implementation': implementation,
                         'shape': f'[{m}, {a}] × [{b}, {a}]ᵀ', 'calls': calls,
                         'source': 'static_source', 'duration_ms': None})
    # These operations are present in the implementation, but not priced by the projection model.
    for i,kind in enumerate(CONFIG['layer_types']):
        ops = ([('linear_attn.conv1d', 'PyTorch causal conv / elementwise'),
                ('linear_attn.delta', config.get('kernels', 'fla-delta') + ' gated delta rule')]
               if kind == 'linear_attention' else
               [('self_attn.attention', 'PyTorch SDPA / qwen3_5.attention_kernels.decode_attention')])
        for name, implementation in ops:
            plan.append({'module': f'layer.{i}.{name}', 'phase': 'prefill', 'implementation': implementation,
                         'shape': 'see checkpoint head/state dimensions', 'calls': 1, 'source': 'static_source', 'duration_ms': None})
    if mtp:
        plan.append({'module': 'mtp', 'phase': 'decode', 'implementation': 'MTPSpeculativeDecoder · draft + parallel verify4 + commit',
                     'shape': 'k=3; actual rounds unavailable', 'calls': None, 'source': 'static_source', 'duration_ms': None})
    return {'runner_id': runner_identity(config), 'model': describe(config), 'estimates': rows, 'kernel_plan': plan,
            'estimate_scope': 'projection GEMMs only; logical traffic, not observed DRAM; attention/DeltaNet/norm/elementwise omitted',
            'phase_estimate_reasons': {'decode': 'MTP round·acceptance 기록이 없어 decode 연산량을 복원할 수 없습니다.'} if mtp else {},
            'analysis_notes': ['모델 구조와 구현 경로는 고정 checkpoint 설정 및 runner 소스에서 복원했습니다.',
                               '커널별 시간·호출·DRAM 카운터는 미수집입니다. 구현 경로의 호출 수는 소스 기반 추정입니다.',
                               'Roofline 점은 projection GEMM 연산량과 논리적 트래픽 모형을 실측 구간 시간으로 나눈 값입니다.'],
            'provenance': {'config_sha256': config.get('model_config_sha256', 'f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c'),
                           'checkpoint_revision': config.get('checkpoint_revision', 'e41d469d00ef99639f592c1cfc5ab6e0a9db75bf'),
                           'kind': 'source_derived_analysis; no new GPU measurement'}}
