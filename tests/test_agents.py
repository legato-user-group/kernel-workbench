import asyncio
import json
from kernel_dashboard import agents as module


def test_headless_argv_response_events_and_cancel(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    seen=[]
    class FakeProcess:
        returncode=0
        async def wait(self):return 0
    async def spawn(*args,**kwargs):
        seen.extend(args)
        from pathlib import Path
        Path(args[args.index('--output-last-message')+1]).write_text('analysis complete')
        kwargs['stdout'].write(b'{"type":"turn.completed"}\n')
        assert kwargs['stdin']==asyncio.subprocess.DEVNULL
        return FakeProcess()
    monkeypatch.setattr(asyncio,'create_subprocess_exec',spawn)
    async def check():
        manager=module.Agents();r=await manager.start('inspect $(literal)\nno shell','http://127.0.0.1:8765',30)
        await manager.tasks[r['id']]
        assert r['response']=='analysis complete' and r['status']=='completed'
        assert seen[:5]==['codex','-a','never','exec','--json']
        assert '$(literal)' in seen[-1] and '\nno shell' in seen[-1]
        assert '--dangerously-bypass-approvals-and-sandbox' not in seen
        assert manager.get(r['id'])['response']=='analysis complete'
        assert (manager.directory/r['id']/'events.jsonl').exists()
    asyncio.run(check())


def test_agent_timeout_terminates_process_group(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    signals=[]
    class Process:
        returncode=None;pid=12345
        async def wait(self):
            await asyncio.sleep(10)
    async def spawn(*args,**kwargs):return Process()
    monkeypatch.setattr(asyncio,'create_subprocess_exec',spawn)
    def terminate(pid,sig):
        signals.append((pid,sig));Process.returncode=-15
    monkeypatch.setattr(module.os,'killpg',terminate)
    # Cleanup wait should acknowledge SIGTERM immediately.
    async def wait(self):
        while self.returncode is None:await asyncio.sleep(.005)
        return self.returncode
    Process.wait=wait
    async def check():
        manager=module.Agents();r=await manager.start('bounded task','http://127.0.0.1:8765',.02)
        await manager.tasks[r['id']]
        assert r['status']=='failed' and r['error']=='TimeoutError'
        assert signals and signals[0][0]==12345
    asyncio.run(check())
