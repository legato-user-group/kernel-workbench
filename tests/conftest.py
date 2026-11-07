"""Small synthetic evidence; tests do not depend on ignored GPU result files."""
import json

import pytest

from kernel_workbench.evaluation_source import PHASES, PLAIN, MTP, VERIFY


@pytest.fixture
def evidence(tmp_path):
    directory = tmp_path / 'experiments' / 'test_qwen' / '2026-01-01'

    def write(name, value):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    gate = {'max_mean_kl': .005, 'margin_threshold': .5, 'strong_mismatches_allowed': 0}
    contract = {'lengths': [128, 512, 1024, 4096], 'new_tokens': 256, 'warmup': 3,
                'repeats': 3, 'batch': 1, 'cache_capacity': 4416, 'hf_timing_requested': False, 'gate': gate,
                'models': {key: {'repo': f'test/{key}', 'revision': 'fixture'} for key in ('candidate', 'reference')}}
    write('results/contract.json', contract)
    write('registration.json', {'id_prefix': 'test-qwen', 'series': 'fixture'})
    write('invocation.json', {'model': 'qwen3_5'})
    write('results/hardware.json', {'gpu': 'NVIDIA A100-SXM4-80GB', 'bandwidth': 2039e9,
                                    'peaks': {'fp32': 19.5e12, 'bf16_tensor': 312e12, 'fp16_tensor': 312e12},
                                    'source': 'fixture'})
    write('session_status.json', {'observed_at_utc': '2026-01-01T00:00:00Z'})
    write('results/environment.json', {'gpu': 'NVIDIA A100-SXM4-80GB', 'memory_gib': 79.25,
                                       'torch': 'fixture', 'cuda': 'fixture', 'packages': {}})
    checks = []
    for length in contract['lengths']:
        for engine in (PLAIN, VERIFY):
            checks.append({'engine': engine, 'input_tokens': length, 'passed': length != 512,
                           'prefill_kl': .001, 'decode_mean_kl': .02 if length == 512 else .001,
                           'strong_mismatches': 0, 'top1_agreement': 1})
        if length != 4096:
            write(f'results/mtp_parity_{length}.json', {'input_tokens': length,
                  'matches_plain_int8': length != 128, 'token_agreement': .75 if length == 128 else 1})
        for engine in (PLAIN, MTP):
            if length == 4096 and engine == MTP:
                continue
            timing = {}
            for phase in PHASES:
                count = 2 if length == 1024 and engine == MTP and phase == 'generation' else 3
                value = (6 if engine == PLAIN else 3) if phase == 'decode' else 1
                samples = [{'wall_ms': value, 'cuda_event_ms': value - .1, 'output_ids_sha256': 'fixture'}
                           for _ in range(count)]
                timing[phase] = {'samples': samples, 'sample_count': count, 'source': 'unprofiled',
                                 'wall_ms': {'median': -999}, 'cuda_event_ms': {'median': -999}}
            write(f'results/timing_{engine}_{length}.json', timing)
    checks.extend({'engine': engine, 'input_tokens': 128, 'reset_repeat': True,
                   'passed': True, 'bitwise_equal_to_first': True} for engine in (PLAIN, VERIFY))
    write('results/correctness.json', {'gate': gate, 'checks': checks})
    write('results/inputs.json', {'fixture': True})
    return directory


@pytest.fixture
def experiment_definition(tmp_path):
    def create(length=512):
        path = tmp_path / f'experiment_{length}' / 'config.json'
        path.parent.mkdir()
        path.write_text(json.dumps({'name': f'test_prose_{length}', 'model': 'qwen3_5',
                                    'input_tokens': length, 'output_tokens': 256, 'warmup': 3,
                                    'baseline_repeats': 10}))
        return path
    return create
