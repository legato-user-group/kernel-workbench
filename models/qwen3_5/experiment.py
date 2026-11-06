"""Collection interface for the shared Qwen3.5 implementation."""
from .analysis import describe


def run(config, trace):
    import torch
    from .evaluation import run_report
    with torch.inference_mode():
        return run_report(config, trace, describe(config))
