import asyncio
import json
import pytest
from kernel_dashboard import jobs as module
from kernel_dashboard.app import RunConfig


def test_cross_process_lease_and_refresh(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    one,two=module.Jobs(),module.Jobs()
    one.acquire()
    with pytest.raises(ValueError):two.acquire()
    one.release();two.acquire();two.release()
    one.save({'id':'external','status':'completed','created_at':1})
    two.refresh();assert two.records['external']['status']=='completed'


def test_cancel_cleanup_and_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    async def check():
        j=module.Jobs(); calls=[]
        async def fake(r,*args,**kwargs):
            calls.append(args)
            if args[1]=='new': await asyncio.sleep(10)
        j.command=fake
        r=await j.start(RunConfig().model_dump())
        await asyncio.sleep(.01)
        await j.cancel(r['id'])
        assert r['status']=='cancelled'
        assert any(a[1]=='stop' for a in calls)
        assert not r['owns_session'] and j.lease is None
        # A different process sees the same terminal state.
        assert module.Jobs().records[r['id']]['status']=='cancelled'
    asyncio.run(check())


def test_recovery_does_not_stop_active_agent(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    async def check():
        agent=module.Jobs(); observer=module.Jobs()
        agent.acquire()
        agent.save({'id':'live','session':'kernel-opt-live','status':'profiling','owns_session':True,'created_at':1})
        await observer.recover()
        assert json.loads((tmp_path/'live/job.json').read_text())['status']=='profiling'
        agent.release()
    asyncio.run(check())


def test_profiling_subprocess_uses_requested_timeout(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    async def check():
        manager=module.Jobs();observed=[]
        async def fake(record,*args,**kwargs):
            if args[1]=='exec':
                observed.append((args,kwargs))
                raise RuntimeError('End of command-only test')
        manager.command=fake
        r=await manager.start(RunConfig(timeout_sec=7200).model_dump())
        await manager.tasks[r['id']]
        args,kwargs=observed[0]
        assert args[args.index('--timeout')+1]==7200
        assert kwargs['timeout']==7200
        assert not r['owns_session'] and manager.lease is None
    asyncio.run(check())


def test_colab_subprocess_never_waits_for_server_stdin(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    manager=module.Jobs();record={'id':'stdin-test'};(tmp_path/record['id']).mkdir()
    class Process:
        returncode=0
        async def wait(self):return 0
    async def spawn(*args,**kwargs):
        assert kwargs['stdin']==asyncio.subprocess.DEVNULL
        return Process()
    monkeypatch.setattr(asyncio,'create_subprocess_exec',spawn)
    asyncio.run(manager.command(record,'colab','exec','-f','script.py'))
