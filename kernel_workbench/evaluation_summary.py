"""Compute evaluation summaries from raw evidence without rendering or writing files."""
from .evaluation_source import EvaluationSource, ENGINES, PHASES, PLAIN, MTP


def summarize(directory):
    return summarize_source(EvaluationSource(directory))


def summarize_source(source):
    contract = source.contract
    rows = []
    for (engine, length), timing in source.timings.items():
        counts = {phase: timing[phase]['sample_count'] for phase in PHASES}
        eligible = {phase: source.quality_ok(engine, length) and timing[phase]['source'] == 'unprofiled'
                    and counts[phase] == contract['repeats'] for phase in PHASES}
        quality = source.check(engine, length)
        steps = contract['new_tokens'] - 1
        rows.append({
            'engine': engine, 'input_tokens': length, 'requested_output_tokens': contract['new_tokens'],
            'complete': all(count == contract['repeats'] for count in counts.values()),
            'stable_output_hashes': source.stable(engine, length), 'sample_counts': counts,
            'metric_eligible': eligible, 'prefill_ms': timing['prefill_forward']['wall_ms']['median'],
            'engine_ttft_ms': timing['engine_ttft']['wall_ms']['median'],
            'decode_ms': timing['decode']['wall_ms']['median'],
            'decode_ms_per_token': timing['decode']['wall_ms']['median'] / steps,
            'decode_cuda_event_ms_per_token': timing['decode']['cuda_event_ms']['median'] / steps,
            'generation_ms': timing['generation']['wall_ms']['median'],
            'quality_gate_passed': bool(quality and quality['passed']) if engine in source.engines else None,
            'mtp_plain_parity': source.parity.get(length) if engine == MTP else None,
            'validated_comparison_eligible': engine in source.engines and all(eligible.values()),
        })
    comparisons = []
    for length in contract['lengths']:
        pair = {row['engine']: row for row in rows if row['input_tokens'] == length}
        plain, mtp = pair.get(PLAIN), pair.get(MTP)
        if plain and mtp and plain['metric_eligible']['decode'] and mtp['metric_eligible']['decode']:
            comparisons.append({'input_tokens': length, 'metric': 'decode',
                                'samples_per_engine': contract['repeats'],
                                'plain_ms_per_token': plain['decode_ms_per_token'],
                                'mtp_ms_per_token': mtp['decode_ms_per_token'],
                                'speedup': plain['decode_ms'] / mtp['decode_ms']})
    exit_state = source.read('results/exit.json', {})
    expected_engines = len(source.engines) + int(bool(contract.get('hf_timing_requested')))
    complete = (bool(exit_state.get('success')) and len(rows) == expected_engines * len(contract['lengths'])
                and all(row['complete'] for row in rows) and source.reset_ok
                and len(source.checks) == len(source.check_engines) * (len(contract['lengths']) + 1)
                and (MTP not in source.engines or
                     all(source.read(f'results/mtp_parity_{n}.json') is not None for n in contract['lengths'])))
    return {'complete': complete, 'exit': exit_state, 'session_status': source.read('session_status.json'),
            'environment': source.read('results/environment.json', {}), 'contract': contract, 'rows': rows,
            'correctness': source.correctness, 'validated_decode_comparisons': comparisons,
            'profiling': 'not collected', 'limits': [contract.get('state_comparison', 'State comparison not recorded')]}
