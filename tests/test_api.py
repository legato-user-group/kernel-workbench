import json
from fastapi.testclient import TestClient
import kernel_workbench.app as service
from kernel_workbench.reports import ReportStore


def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'store', ReportStore(tmp_path))
    return TestClient(service.app)


def test_api_is_result_store_only(tmp_path,monkeypatch):
    client=isolated(tmp_path,monkeypatch)
    with client:
        assert client.get('/').status_code==200
        assert client.get('/api/health').json()['mode']=='result_store'
        guide=client.get('/api/guide')
        assert guide.status_code==200
        assert guide.text==(service.ROOT/'README.md').read_text()
        assert client.get('/api/runs').json()==[]
        assert client.post('/api/runs',json={'runner':'tiny_decoder'}).status_code==422
        assert client.post('/api/runs',json={},headers={'origin':'https://untrusted.example'}).status_code==403
        schema=client.get('/openapi.json').json()['paths']
        assert '/api/runners' not in schema
        assert '/api/runs/{ident}/cancel' not in schema
        assert '/api/agent-tasks' in schema and 'post' not in schema['/api/agent-tasks']


def test_legacy_results_remain_readable(tmp_path,monkeypatch):
    source=json.load(open('tests/example_result.json'))
    source.pop('phase_points');source.pop('phase_point_reasons')
    directory=tmp_path/'legacy';directory.mkdir()
    (directory/'result.json').write_text(json.dumps(source))
    (directory/'job.json').write_text(json.dumps({'id':'legacy','created_at':1,'status':'completed','config':source['config']}))
    client=isolated(tmp_path,monkeypatch)
    response=client.get('/api/runs/legacy/result').json()
    assert response['point']==source['point'] and response['phase_points']['decode']
    assert len(client.get('/api/runs').json())==1
    (directory/'job.json').unlink()
    assert client.get('/api/runs').json()==[]


def test_source_analysis_enriches_display_without_rewriting_measurement(tmp_path, monkeypatch):
    client = isolated(tmp_path, monkeypatch)
    raw = json.load(open('tests/example_result.json'))
    raw['hardware'] = {'name': 'NVIDIA A100-SXM4-80GB', 'bandwidth': 2039e9, 'peaks': {'bf16_tensor': 312e12}}
    raw['kernels'], raw['estimates'], raw['point'] = [], [], None
    assert client.post('/api/runs', json={'id':'observed','result':raw}).status_code == 201
    original = (tmp_path/'observed/result.json').read_bytes()
    analysis = {'runner_id':'test_runner','model':{'name':raw['model']['name'],'children':[{'id':'head','label':'LM head'}]},
                'estimates':[{'module':'head','phase':'prefill','flops':{'bf16_tensor':1000},'bytes':100}],
                'kernel_plan':[], 'estimate_scope':'projection subset only'}
    path = '/api/runs/observed/artifacts/analysis.json'
    assert client.put(path, json=analysis).status_code == 201
    displayed = client.get('/api/runs/observed/result').json()
    assert displayed['runner_id'] == 'test_runner'
    assert displayed['hardware']['bandwidth'] == 2039e9
    assert displayed['phase_points']['prefill']['source'] == 'estimated'
    assert displayed['timings'] == raw['timings'] and displayed['kernels'] == []
    assert displayed['point'] is None
    assert client.get('/api/runs').json()[0]['runner_id'] == 'test_runner'
    assert client.put(path, json={**analysis, 'timings':{'total_ms':1}}).status_code == 422
    assert client.get(path).json() == analysis
    assert (tmp_path/'observed/result.json').read_bytes() == original


def test_dashboard_assets_revalidate_cached_versions(tmp_path, monkeypatch):
    with isolated(tmp_path, monkeypatch) as client:
        for path in ('/', '/app.js?v=history-2', '/experiment-groups.js?v=history-2', '/style.css'):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers['cache-control'] == 'no-cache'
            cached = client.get(path, headers={'If-None-Match': response.headers['etag']})
            assert cached.status_code == 304
            assert cached.headers['cache-control'] == 'no-cache'
