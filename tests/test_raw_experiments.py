import json

from fastapi.testclient import TestClient

import kernel_workbench.app as service
from kernel_workbench.experiments import ExperimentStore, compare
from kernel_workbench.raw_experiments import RawExperiments
from kernel_workbench.reports import ReportStore


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


def test_raw_dashboard_needs_no_upload_and_never_writes_during_reads(evidence, monkeypatch):
    directory = evidence.parent.parent
    before = snapshot(directory)
    raw = RawExperiments(directory)
    reports = ReportStore(directory, raw_repository=raw)
    monkeypatch.setattr(service, 'store', reports)
    monkeypatch.setattr(service, 'experiments', ExperimentStore(directory / 'attempts', raw_repository=raw))
    with TestClient(service.app) as client:
        runs = client.get('/api/runs').json()
        assert len(runs) == 7 and all(r['source'] == 'experiment_raw' for r in runs)
        experiments = client.get('/api/experiments').json()
        assert len(experiments) == 4
        for experiment in experiments:
            comparison = experiment['comparison']
            if experiment['id'].endswith('-1024'):
                assert comparison['speedup']['decode'] == 2
                assert 'total_ms' not in comparison['speedup']
            else:
                assert comparison['speedup'] == {}
        ident = 'test-qwen-1024-mtp3'
        result = client.get(f'/api/runs/{ident}/result').json()
        assert result['model']['layers'] == 64 and result['kernel_plan']
        assert result['phase_points']['prefill']['source'] == 'estimated'
        assert result['timings']['sample_counts']['total_ms'] == 2
        path = evidence / 'results/timing_int8_mtp3_cudagraph_1024.json'
        assert client.get(f'/api/runs/{ident}/artifacts/baseline.json').content == path.read_bytes()
        assert client.put(f'/api/runs/{ident}/artifacts/baseline.json', json={}).status_code == 409
        assert client.get(f'/api/runs/{ident}/artifacts/../../other').status_code == 404
    assert snapshot(directory) == before
    assert raw.issues == []


def test_raw_file_changes_are_reflected_without_reimport(evidence):
    directory = evidence.parent.parent
    raw = RawExperiments(directory)
    ident = 'test-qwen-1024-mtp3'
    old = raw.result(ident)['timings']['prefill']
    path = evidence / 'results/timing_int8_mtp3_cudagraph_1024.json'
    data = json.loads(path.read_text())
    for sample in data['prefill_forward']['samples']:
        sample['wall_ms'] += 10
    path.write_text(json.dumps(data))
    assert raw.result(ident)['timings']['prefill'] == old + 10
    path.unlink()
    assert not raw.contains(ident)
    assert not compare(raw.attempt('test-qwen-prose-1024'), ReportStore(directory, raw_repository=raw))['speedup']


def test_empty_store_and_reader_do_not_create_directories(tmp_path):
    directory = tmp_path / 'does-not-exist'
    raw = RawExperiments(directory)
    assert ReportStore(directory, raw_repository=raw).records() == []
    assert ExperimentStore(directory / 'attempts', raw_repository=raw).records() == []
    assert not directory.exists()


def test_reset_failure_withholds_raw_comparison(evidence):
    path = evidence / 'results/correctness.json'
    correctness = json.loads(path.read_text())
    next(c for c in correctness['checks'] if c.get('reset_repeat'))['bitwise_equal_to_first'] = False
    path.write_text(json.dumps(correctness))
    directory = evidence.parent.parent
    raw = RawExperiments(directory)
    attempt = raw.attempt('test-qwen-prose-1024')
    assert attempt['correctness']['status'] == 'failed'
    assert not compare(attempt, ReportStore(directory, raw_repository=raw))['speedup']


def test_unchanged_checkpoint_uses_its_own_architecture_and_needs_no_mtp_parity(evidence):
    from pathlib import Path
    from kernel_workbench.evaluation_summary import summarize
    contract_path = evidence / 'results/contract.json'
    contract = json.loads(contract_path.read_text())
    engine = 'fp16_plain_cudagraph'
    contract.update(lengths=[1024], engines=[engine], check_engines=[engine],
                    candidate_dtype='fp16', mtp_enabled=False, quantization=None, kernels='torch',
                    model_config_file='results/model_config.json', runner_id='qwen35_08b_prose_1024',
                    candidate='unchanged checkpoint; fp16; MTP disabled; no quantization')
    contract['models']['candidate']['repo'] = 'Qwen/Qwen3.5-0.8B'
    contract_path.write_text(json.dumps(contract))
    config = json.loads(Path('models/qwen3_5/config.json').read_text())
    config['text_config'].update(num_hidden_layers=24, layer_types=['linear_attention']*18+['full_attention']*6,
                                 hidden_size=1024, intermediate_size=3584, num_attention_heads=8,
                                 num_key_value_heads=2, linear_num_value_heads=16, linear_num_key_heads=16)
    (evidence / 'results/model_config.json').write_text(json.dumps(config))
    correctness_path = evidence / 'results/correctness.json'
    correctness = json.loads(correctness_path.read_text())
    checks = [c for c in correctness['checks'] if c['engine'] == 'int8_plain_cudagraph'
              and (c['input_tokens'] == 1024 or c.get('reset_repeat'))]
    for check in checks:
        check['engine'] = engine
    correctness['checks'] = checks
    correctness_path.write_text(json.dumps(correctness))
    original = evidence / 'results/timing_int8_plain_cudagraph_1024.json'
    (evidence / f'results/timing_{engine}_1024.json').write_bytes(original.read_bytes())
    (evidence / 'results/mtp_parity_1024.json').unlink()
    (evidence / 'results/exit.json').write_text('{"success": true}')
    raw = RawExperiments(evidence.parent.parent)
    runs = raw.records()
    assert len(runs) == 1
    result = raw.result(runs[0]['id'])
    assert result['model']['layers'] == 24 and result['model']['name'] == 'Qwen/Qwen3.5-0.8B'
    assert result['config']['dtype'] == 'fp16' and result['config']['quantization'] is None
    assert not any(n['id'] == 'mtp' for n in result['model']['children'])
    assert not any('int8' in p['implementation'] for p in result['kernel_plan'])
    assert all('fp16_tensor' in row['flops'] for row in result['estimates'])
    attempt = raw.attempts()[0]
    assert attempt['correctness']['status'] == 'passed'
    assert attempt['baseline_run_id'] is None and attempt['candidate_run_id'] == runs[0]['id']
    assert 'MTP 생성 token parity 미검증' not in attempt['correctness']['details']
    assert summarize(evidence)['complete']
