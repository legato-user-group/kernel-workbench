"""Request-level latency statistics shared by collection and dashboard analysis."""
import math
import statistics


def summarize_samples(samples, source='unprofiled'):
    if not samples:
        raise ValueError('At least one timing sample is required')
    result = {'samples': samples, 'sample_count': len(samples), 'source': source}
    for key in ('wall_ms', 'cuda_event_ms'):
        values = [sample[key] for sample in samples]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f'Invalid raw timing sample: {key}')
        values.sort()
        position = (len(values) - 1) * .9
        lower = int(position)
        p90 = values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
        result[key] = {'median': statistics.median(values), 'min': values[0],
                       'max': values[-1], 'p90_across_samples': p90}
    return result
