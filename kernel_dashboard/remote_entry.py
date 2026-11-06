"""GPU runtime entrypoint for profiling and separate, profiler-free timings."""
import importlib.util
import json
import math
import statistics
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path


class Trace:
    @contextmanager
    def range(self, label):
        import torch
        torch.cuda.nvtx.range_push(label)
        try: yield
        finally: torch.cuda.nvtx.range_pop()
    def phase(self, name): return self.range('phase:'+name)
    def module(self, name): return self.range('module:'+name)
    def step(self, number): return self.range('step:'+str(number))


class NoTrace:
    """Same runner interface with no profiler or NVTX calls."""
    def range(self, label): return nullcontext()
    def phase(self, name): return nullcontext()
    def module(self, name): return nullcontext()
    def step(self, number): return nullcontext()


def collect_baseline(runner, config):
    samples = []
    metadata = None
    repeats = config.get('baseline_repeats', 5)
    for _ in range(repeats):
        result = runner.run(dict(config), NoTrace())
        sample = {'total_ms': result['total_ms'], 'phase_ms': dict(result['phase_ms'])}
        values = [sample['total_ms'], *[sample['phase_ms'][p] for p in ('prefill', 'decode')]]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or values[0] <= 0:
            raise ValueError('Baseline runner must return finite synchronized inference timings')
        if metadata and result['gpu'] != metadata['gpu']:
            raise ValueError('GPU changed between baseline samples')
        metadata = result
        samples.append(sample)
    return {'source': 'unprofiled', 'statistic': 'median', 'sample_count': len(samples),
            'samples': samples, 'total_ms': statistics.median(s['total_ms'] for s in samples),
            'phase_ms': {p: statistics.median(s['phase_ms'][p] for s in samples) for p in ('prefill', 'decode')},
            'config': config, 'gpu': metadata['gpu'], 'versions': metadata.get('versions', {}),
            'warmup_per_sample': config.get('warmup', 1), 'runner_reinitialized_per_sample': True}


if __name__ == '__main__':
    config = json.loads(Path(sys.argv[1]).read_text())
    spec = importlib.util.spec_from_file_location('runner', sys.argv[2])
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    if '--benchmark' in sys.argv[4:]:
        result = collect_baseline(runner, config)
    else:
        result = runner.run(config, Trace())
        result['model'] = runner.describe(config)
    Path(sys.argv[3]).write_text(json.dumps(result))
