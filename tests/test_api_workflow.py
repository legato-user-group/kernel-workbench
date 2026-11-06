"""An agent uploads reviewed results; the web process never executes commands."""
import asyncio
import copy
import json
import subprocess
from fastapi.testclient import TestClient
import kernel_workbench.app as service
from kernel_workbench.reports import ReportStore


def test_upload_query_and_attachments_never_spawn_processes(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Web app executed a command')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(asyncio,'create_subprocess_exec',forbidden)
    monkeypatch.setattr(service,'store',ReportStore(tmp_path))
    result=json.load(open('kernel_workbench/static/example.json'))
    with TestClient(service.app) as client:
        response=client.post('/api/runs',json={'id':'agent-baseline','summary':'Agent checked correctness.','result':result})
        assert response.status_code==201
        report=response.json();assert response.headers['location']==report['links']['status']
        assert report['source']=='agent_upload' and not report['owns_session']
        assert client.get(report['links']['status']).json()['status']=='completed'
        saved=client.get(report['links']['result']).json()
        assert saved['agent_summary']=='Agent checked correctness.'
        assert saved['phase_points']['decode']
        assert saved['artifacts']==['result.json']
        # Arbitrary archive bytes are stored, never opened or executed by the app.
        raw=b'archive bytes, not a valid tar file'
        url='/api/runs/agent-baseline/artifacts/artifacts.tar.gz'
        assert client.put(url,content=raw).status_code==201
        assert client.get(url).content==raw
        assert client.put('/api/runs/agent-baseline/artifacts/commands.log',content=b'agent CLI output').status_code==201
        assert 'commands.log' in client.get(report['links']['result']).json()['artifacts']
        assert client.put('/api/runs/agent-baseline/artifacts/result.json',content=b'override').status_code==422
        assert client.post('/api/runs',json={'id':'agent-baseline','result':result}).status_code==409
        assert client.get('/api/runs').json()[0]['id']=='agent-baseline'
        assert client.get('/api/runs/../result.json').status_code==404
        assert 'agent-form' not in client.get('/').text
        assert 'id="collect"' not in client.get('/').text


def test_invalid_report_is_rejected_atomically(tmp_path,monkeypatch):
    monkeypatch.setattr(service,'store',ReportStore(tmp_path))
    result=json.load(open('kernel_workbench/static/example.json'))
    client=TestClient(service.app)
    bad=copy.deepcopy(result);bad['config']['output_tokens']=0
    assert client.post('/api/runs',json={'result':bad}).status_code==422
    bad=copy.deepcopy(result);bad['kernels'][0]['flops']='not counters'
    assert client.post('/api/runs',json={'result':bad}).status_code==422
    assert client.post('/api/runs',json={'id':'../escape','result':result}).status_code==422
    assert client.get('/api/runs').json()==[]
    assert list(tmp_path.iterdir())==[]
