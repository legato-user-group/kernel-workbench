"""Exercise an external agent's HTTP workflow without allocating a GPU."""
import asyncio
import json
import tarfile
import time
from fastapi.testclient import TestClient
import kernel_dashboard.app as service
from kernel_dashboard import jobs as worker


def isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'DATA', tmp_path)
    monkeypatch.setattr(service, 'DATA', tmp_path)
    manager = worker.Jobs()
    monkeypatch.setattr(service, 'jobs', manager)
    return manager


def wait_finished(client, ident):
    for _ in range(200):
        record = client.get(f'/api/runs/{ident}').json()
        if record.get('finished_at'):
            return record
        time.sleep(.01)
    raise AssertionError('Worker did not finish')


def test_register_and_collect_via_http(tmp_path, monkeypatch):
    manager = isolated_jobs(tmp_path, monkeypatch)
    source = tmp_path / 'custom.py'
    source.write_text("raise RuntimeError('registration must not import runner')")
    fixture = json.load(open('kernel_dashboard/static/example.json'))

    async def command(record, *args, **kwargs):
        if args[1] == 'exec':
            config = record['config']
            meta = {'config': config, 'gpu': {'name': fixture['hardware']['name']},
                    'model': fixture['model'], 'total_ms': 9.4,
                    'phase_ms': {'prefill': 1.8, 'decode': 7.6},
                    'estimates': fixture['estimates'], 'versions': {}}
            meta_file = tmp_path / 'meta.json'
            status_file = tmp_path / 'status.json'
            meta_file.write_text(json.dumps(meta))
            status_file.write_text(json.dumps({'warnings': [], 'tools': {}}))
            baseline_file=tmp_path/'baseline.json'
            sample={'total_ms':8,'phase_ms':{'prefill':1,'decode':7}}
            baseline_file.write_text(json.dumps({'source':'unprofiled','statistic':'median','sample_count':5,'samples':[sample]*5,**sample,'config':config,'gpu':meta['gpu']}))
            with tarfile.open(tmp_path / record['id'] / 'artifacts.tar.gz', 'w:gz') as archive:
                archive.add(meta_file, arcname='meta.json')
                archive.add(status_file, arcname='status.json')
                archive.add(baseline_file, arcname='baseline.json')
    manager.command = command
    with TestClient(service.app) as client:
        registered = client.post('/api/runners', json={'id': 'custom', 'path': str(source)})
        assert registered.status_code == 201
        assert registered.json()['id'] == 'custom'
        assert {'id': 'custom'} in client.get('/api/runners').json()
        assert client.post('/api/runners', json={'id': 'bad', 'path': str(tmp_path / 'missing.py')}).status_code == 422
        assert client.post('/api/runners', json={'id': 'tiny_decoder', 'path': str(source)}).status_code == 422
        response = client.post('/api/runs', json={'runner': 'custom', 'timeout_sec': 7200})
        assert response.status_code == 202
        run = response.json()
        assert response.headers['location'] == run['links']['status']
        final = wait_finished(client, run['id'])
        assert final['status'] == 'completed' and not final['owns_session']
        result = client.get(run['links']['result']).json()
        assert result['phase_points']['decode']['source'] == 'estimated'
        assert result['timings']['source']=='unprofiled' and result['timings']['sample_count']==5
        assert result['timings']['total_ms']==8 and result['timings']['profiled']['total_ms']==9.4
        assert client.get(f'/api/runs/{run["id"]}/artifacts/result.json').status_code == 200
        assert client.get('/api/health').json()['guide_url'] == '/api/guide'
        assert '/api/runners' in client.get('/openapi.json').json()['paths']
        assert 'agent-form' not in client.get('/').text
        assert 'id="collect"' not in client.get('/').text


def test_http_cancel_is_asynchronous_and_worker_cleans_up(tmp_path, monkeypatch):
    manager = isolated_jobs(tmp_path, monkeypatch)
    stopped = []
    async def command(record, *args, **kwargs):
        if args[1] == 'exec':
            while True:
                await asyncio.sleep(.01)
        if args[1] == 'stop':
            stopped.append(record['id'])
    manager.command = command
    with TestClient(service.app) as client:
        run = client.post('/api/runs', json={}).json()
        assert client.post('/api/runs', json={}).status_code == 409
        assert client.get(run['links']['result']).status_code == 409
        cancel = client.post(run['links']['cancel'])
        assert cancel.status_code == 202 and cancel.json()['cancel_requested']
        final = wait_finished(client, run['id'])
        assert final['status'] == 'cancelled'
        assert not final['owns_session'] and stopped == [run['id']]
        assert not client.post(run['links']['cancel']).json()['cancel_requested']
