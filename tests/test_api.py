import json
from fastapi.testclient import TestClient
import kernel_dashboard.app as service
from kernel_dashboard.reports import ReportStore


def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'store', ReportStore(tmp_path))
    return TestClient(service.app)


def test_api_is_result_store_only(tmp_path,monkeypatch):
    client=isolated(tmp_path,monkeypatch)
    with client:
        assert client.get('/').status_code==200
        assert client.get('/api/health').json()['mode']=='result_store'
        assert client.get('/api/guide').status_code==200
        assert client.get('/api/runs').json()==[]
        assert client.post('/api/runs',json={'runner':'tiny_decoder'}).status_code==422
        assert client.post('/api/runs',json={},headers={'origin':'https://untrusted.example'}).status_code==403
        schema=client.get('/openapi.json').json()['paths']
        assert '/api/runners' not in schema
        assert '/api/runs/{ident}/cancel' not in schema
        assert '/api/agent-tasks' in schema and 'post' not in schema['/api/agent-tasks']


def test_legacy_results_remain_readable(tmp_path,monkeypatch):
    source=json.load(open('kernel_dashboard/static/example.json'))
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
