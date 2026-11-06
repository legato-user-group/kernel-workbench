"""Self-contained synthetic decoder; custom runners implement describe + run."""
from contextlib import contextmanager
import time


def describe(config):
    nodes = [{'id': 'embedding', 'label': 'Token embedding', 'kind': 'Embedding'}]
    for i in range(4):
        nodes.append({'id': f'layer.{i}', 'label': f'Decoder layer {i}', 'kind': 'Transformer',
                      'children': [{'id': f'layer.{i}.{op}', 'label': label, 'kind': op}
                                   for op,label in [('norm','LayerNorm'),('qkv','QKV projection'),
                                      ('attention','Causal attention + KV cache'),('out','Output projection'),('mlp','MLP · GELU')]]})
    nodes.append({'id':'head','label':'LM head','kind':'Linear'})
    return {'name':'Tiny Decoder · synthetic weights', 'hidden':256,'layers':4,'heads':4,
            'vocab':4096,'synthetic':True,'children':nodes}


def run(config, trace):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[config['dtype']]
    b, length, outputs = config['batch'], config['input_tokens'], config['output_tokens']
    h, vocab, heads = 256,4096,4
    path = config['dtype'] + '_tensor' if config['dtype'] != 'fp32' else 'fp32'
    model = nn.ModuleDict({'embedding':nn.Embedding(vocab,h), 'head':nn.Linear(h,vocab,bias=False),
        'layers':nn.ModuleList([nn.ModuleDict({'norm':nn.LayerNorm(h),'qkv':nn.Linear(h,3*h,bias=False),
        'out':nn.Linear(h,h,bias=False),'up':nn.Linear(h,4*h,bias=False),'down':nn.Linear(4*h,h,bias=False)}) for _ in range(4)])}).cuda().to(dtype).eval()
    inputs = torch.randint(vocab,(b,length),device='cuda')
    size = torch.empty((),dtype=dtype).element_size()
    estimates, steps, phases = [], [], {}
    active_phase = ''
    @contextmanager
    def op(name, flops, traffic, enabled):
        with trace.module(name):
            yield
        if enabled:
            estimates.append({'phase':active_phase,'module':name,'name':name,'flops':{path:flops},'bytes':traffic*size})
    def forward(tokens, cache, enabled):
        n = tokens.shape[1]; count=b*n
        with op('embedding',0,count*h,enabled): x=model['embedding'](tokens)
        for i, layer in enumerate(model['layers']):
            p=f'layer.{i}.'
            with op(p+'norm', count*h*8,count*h*2,enabled): z=layer['norm'](x)
            with op(p+'qkv',2*count*h*3*h, h*3*h+count*4*h,enabled):
                q,k,v=layer['qkv'](z).view(b,n,3,heads,h//heads).permute(2,0,3,1,4).unbind(0)
            if cache[i] is not None:
                k=torch.cat((cache[i][0],k),dim=2); v=torch.cat((cache[i][1],v),dim=2)
            cache[i]=(k,v); context=k.shape[2]
            with op(p+'attention',4*b*n*context*h, b*(n+2*context+n)*h,enabled):
                attn=F.scaled_dot_product_attention(q,k,v,is_causal=(n>1)).transpose(1,2).reshape(b,n,h)
            with op(p+'out',2*count*h*h,h*h+count*2*h,enabled): x=x+layer['out'](attn)
            with op(p+'mlp',16*count*h*h,8*h*h+count*10*h,enabled): x=x+layer['down'](F.gelu(layer['up'](x)))
        with op('head',2*b*h*vocab,h*vocab+b*(h+vocab),enabled):
            return model['head'](x[:,-1:]).argmax(-1)
    with torch.inference_mode():
        for _ in range(config.get('warmup',1)):
            cache=[None]*4; token=forward(inputs,cache,False)
            for _ in range(max(0,outputs-1)): token=forward(token,cache,False)
        torch.cuda.synchronize()
        with trace.range('inference'):
            start=time.perf_counter(); cache=[None]*4
            for phase in ('prefill','decode'):
                active_phase=phase
                with trace.phase(phase):
                    before=time.perf_counter()
                    if phase=='prefill': token=forward(inputs,cache,True)
                    else:
                        for step in range(outputs-1):
                            t=time.perf_counter()
                            with trace.step(step): token=forward(token,cache,True)
                            # Do not synchronize each token: preserve natural scheduling.
                            steps.append({'step':step,'host_submit_ms':(time.perf_counter()-t)*1000})
                    torch.cuda.synchronize()
                    phases[phase]=(time.perf_counter()-before)*1000
            total=(time.perf_counter()-start)*1000
    return {'model':describe(config),'config':config,'total_ms':total,'phase_ms':phases,
            'steps':steps,'estimates':estimates,'gpu':{'name':torch.cuda.get_device_name()},
            'versions':{'torch':torch.__version__,'cuda':torch.version.cuda},
            'warnings':['Estimates cover modeled tensor operations; omit allocator/cache effects and some elementwise work.']}
