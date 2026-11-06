"""Qwen3.5 inference implementation imported from infer-opt.

Heavy GPU dependencies are imported only when model classes are requested.
"""

def __getattr__(name):
    if name in ('Qwen3_5ForCausalLM','Qwen3_5TextConfig','Qwen3_5Cache','FixedShapeDecoder'):
        from . import model
        return getattr(model,name)
    raise AttributeError(name)
