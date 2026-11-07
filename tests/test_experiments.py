import asyncio
import copy
import json
import subprocess
import pytest
from fastapi.testclient import TestClient
import kernel_workbench.app as service
from kernel_workbench.experiments import ExperimentStore
from kernel_workbench.reports import ReportStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'store', ReportStore(tmp_path / 'runs'))
    monkeypatch.setattr(service, 'experiments', ExperimentStore(tmp_path / 'experiments'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Experiment storage must never execute a command')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbidden)
    with TestClient(service.app) as client:
        yield client


def upload_pair(client, candidate_change=None):
    baseline = json.load(open('tests/example_result.json'))
    baseline['timings'].update(source='unprofiled', total_ms=10, prefill=4, decode=6)
    candidate = copy.deepcopy(baseline)
    candidate['config']['runner'] = 'optimized_runner'
    candidate['timings'].update(total_ms=5, prefill=2, decode=3)
    if candidate_change:
        candidate_change(candidate)
    for ident, result in [('baseline', baseline), ('candidate', candidate)]:
        assert client.post('/api/runs', json={'id': ident, 'result': result}).status_code == 201


def attempt(**changes):
    return {'id': 'trial-1', 'series': 'decoder-gemv', 'title': 'Fuse weight loading',
            'changes': 'Fused dequantization and GEMV.', 'code_revision': 'abc123',
            'environment': 'SSH A100', 'baseline_run_id': 'baseline',
            'candidate_run_id': 'candidate',
            'correctness': {'status': 'passed', 'details': 'Compared full logits and KV cache.'},
            'conclusion': 'Reduced memory traffic.', 'next_steps': 'Tune split-K.', **changes}


def test_experiments_link_results_and_calculate_comparison(client):
    upload_pair(client)
    response = client.post('/api/experiments', json=attempt())
    assert response.status_code == 201
    assert response.headers['location'] == '/api/experiments/trial-1'
    saved = response.json()
    comparison = saved['comparison']
    assert comparison['eligible'] and comparison['reasons'] == []
    assert comparison['speedup'] == {'total_ms': 2, 'prefill': 2, 'decode': 2}
    assert comparison['latency_reduction_pct']['total_ms'] == 50
    assert saved['code_revision'] == 'abc123'
    assert client.get(response.headers['location']).json() == saved
    assert client.get('/api/experiments').json() == [saved]
    assert client.get('/api/experiments?series=decoder-gemv').json() == [saved]
    assert client.get('/api/experiments?series=other').json() == []
    assert client.post('/api/experiments', json=attempt()).status_code == 409
    # Immutable reports/attempts retain the evidence of previous trials.
    assert client.patch('/api/experiments/trial-1', json={'changes': 'overwrite'}).status_code == 405
    assert client.get('/api/experiments/trial-1').json() == saved
    page = client.get('/').text
    assert '최적화 실험 이력' in page and 'agent-select' not in page


@pytest.mark.parametrize(('change', 'reason'), [
    (lambda r: r['config'].update(batch=2), 'workload_mismatch'),
    (lambda r: r['hardware'].update(name='H100'), 'gpu_mismatch'),
    (lambda r: r['model'].update(name='different_model'), 'model_mismatch'),
    (lambda r: r['timings'].update(source='nsys'), 'candidate_not_independent'),
])
def test_incompatible_measurements_do_not_publish_speedup(client, change, reason):
    upload_pair(client, change)
    comparison = client.post('/api/experiments', json=attempt()).json()['comparison']
    assert not comparison['eligible'] and reason in comparison['reasons']
    assert comparison['speedup'] == {} and comparison['latency_reduction_pct'] == {}
    assert comparison['candidate']['timings']['total_ms'] == 5


def test_correctness_failure_preserves_measurements_without_speedup(client):
    upload_pair(client)
    saved = client.post('/api/experiments', json=attempt(
        correctness={'status': 'failed', 'details': 'KV cache mismatch.'})).json()
    assert 'correctness_not_passed' in saved['comparison']['reasons']
    assert saved['comparison']['speedup'] == {}
    assert saved['comparison']['baseline']['run_id'] == 'baseline'


def test_regression_and_failed_attempts_remain_in_history(client):
    upload_pair(client, lambda r: r['timings'].update(total_ms=20))
    saved = client.post('/api/experiments', json=attempt()).json()
    assert saved['comparison']['speedup']['total_ms'] == .5
    assert saved['comparison']['latency_reduction_pct']['total_ms'] == -100
    failed = {'id': 'compile-failure', 'series': 'decoder-gemv', 'title': 'Split-K',
              'status': 'failed', 'changes': 'Changed reduction layout.',
              'conclusion': 'Compilation failed.', 'next_steps': 'Fix reduction.'}
    response = client.post('/api/experiments', json=failed)
    assert response.status_code == 201
    assert response.json()['comparison']['candidate'] is None
    assert response.json()['correctness']['status'] == 'not_checked'
    assert [r['id'] for r in client.get('/api/experiments').json()] == ['compile-failure', 'trial-1']
    assert len(client.get('/api/runs').json()) == 2


def test_invalid_references_and_records_do_not_persist(client):
    assert client.get('/api/experiments').json() == []
    assert client.post('/api/experiments', json=attempt()).status_code == 404
    assert client.get('/api/experiments').json() == []
    assert client.post('/api/experiments', json={'title': 'Missing change'}).status_code == 422
    invalid = attempt(id='../unsafe', status='failed', baseline_run_id=None, candidate_run_id=None)
    assert client.post('/api/experiments', json=invalid).status_code == 422
    assert client.post('/api/experiments', json=attempt(candidate_run_id='baseline')).status_code == 422
    assert client.post('/api/experiments', json=attempt(candidate_run_id=None)).status_code == 422
    assert client.get('/api/experiments/missing').status_code == 404
    assert client.post('/api/experiments', json={}, headers={'origin': 'https://other.example'}).status_code == 403


def test_deleted_measurement_keeps_attempt_but_withholds_comparison(client):
    upload_pair(client)
    assert client.post('/api/experiments', json=attempt()).status_code == 201
    (service.store.path('candidate') / 'result.json').unlink()
    saved = client.get('/api/experiments/trial-1').json()
    assert saved['changes'] == 'Fused dequantization and GEMV.'
    assert 'candidate_unavailable' in saved['comparison']['reasons']
    assert saved['comparison']['speedup'] == {}
