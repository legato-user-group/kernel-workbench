from fastapi.testclient import TestClient
from kernel_dashboard.app import app

def test_api_and_static():
    # Do not run application lifespan: GPU recovery is tested independently.
    c=TestClient(app)
    assert c.get('/').status_code==200
    assert c.get('/example.json').json()['schema_version']==1
    assert c.get('/api/agent-tasks').status_code==200
    assert c.post('/api/agent-tasks',json={'prompt':''}).status_code==422
    assert c.get('/api/runners').json()[0]['id']=='tiny_decoder'
    assert c.get('/api/runs/nonexistent').status_code==404
    assert c.post('/api/runs',json={'batch':0}).status_code==422
    assert c.post('/api/runs',json={'gpu':'H100'}).status_code==422
    assert c.post('/api/runs',json={'runner':'unregistered'}).status_code==409
    assert c.post('/api/runs',json={},headers={'origin':'https://untrusted.example'}).status_code==403


def test_legacy_result_api_adds_phase_points(tmp_path,monkeypatch):
    import json
    import kernel_dashboard.app as module
    source=json.load(open('kernel_dashboard/static/example.json'))
    source.pop('phase_points'); source.pop('phase_point_reasons')
    work=tmp_path/'legacy';work.mkdir();(work/'result.json').write_text(json.dumps(source))
    monkeypatch.setattr(module,'DATA',tmp_path)
    monkeypatch.setattr(module,'record',lambda ident:{'id':ident})
    response=TestClient(app).get('/api/runs/legacy/result').json()
    assert response['point']==source['point']
    assert response['phase_points']['prefill'] is not None
    assert response['phase_points']['decode'] is not None
