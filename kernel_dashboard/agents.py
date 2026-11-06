"""Non-interactive Codex subprocess: one prompt in, final answer + events out."""
import asyncio
import fcntl
import json
import os
import signal
import time
import uuid
from .jobs import DATA, ROOT

class Agents:
    def __init__(self):
        self.directory=DATA/'agents';self.directory.mkdir(exist_ok=True)
        self.tasks={}
    def records(self):
        return sorted([json.loads(p.read_text()) for p in self.directory.glob('*/task.json')],key=lambda r:r['created_at'],reverse=True)
    def get(self,ident):
        if not ident.isalnum(): raise ValueError('Unknown task')
        return json.loads((self.directory/ident/'task.json').read_text())
    def save(self,r):
        p=self.directory/r['id']/'task.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(r,indent=2));temp.replace(p)
    async def start(self,prompt,api_url,timeout=1800,model=None):
        lease=(self.directory/'agent.lock').open('a')
        try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: lease.close();raise ValueError('A Codex task is already running')
        ident=uuid.uuid4().hex[:12];work=self.directory/ident;work.mkdir()
        r={'id':ident,'status':'queued','created_at':time.time(),'prompt':prompt,'response':None,'error':None,'model':model,'run_ids':[]}
        self.save(r)
        self.tasks[ident]=asyncio.create_task(self.execute(r,api_url,timeout,lease))
        return r
    async def execute(self,r,api_url,timeout,lease):
        work=self.directory/r['id']; before={p.parent.name for p in DATA.glob('*/job.json')};proc=None
        context=f'''You are a non-interactive GPU kernel/model execution agent working in {ROOT}.
Do not ask the user questions. Complete the requested task and return your findings.
Read README.md for the runner contract and analysis definitions. Default GPU: A100.
Local profiling broker API: {api_url}. Use HTTP JSON POST /api/runs and poll GET /api/runs/{{id}}; obtain GET /api/runs/{{id}}/result.
Include runner_options: {{"agent_task_id":"{r['id']}"}} in each profiling request to associate it with this task.
Use the broker for Colab execution: it owns authentication, timeouts and session cleanup. Do not launch Colab VMs directly.
You may register a self-contained runner through the local CLI. Only edit files needed for the requested model/kernel change.
Check numerical correctness before claiming a kernel improvement. Distinguish measured counters, estimates and unavailable data.
If unable to complete, explain the exact failure rather than claiming success. Include run IDs, changed files and validation in the final response.
USER TASK:\n{r['prompt']}'''
        (work/'prompt.txt').write_text(context)
        args=['codex','-a','never','exec','--json','--sandbox','workspace-write',
              '-c','sandbox_workspace_write.network_access=true','-C',str(ROOT),
              '--output-last-message',str(work/'response.txt')]
        if r.get('model'):args += ['--model',r['model']]
        args.append(context) # An argv element, never a shell interpolation.
        parent=asyncio.current_task()
        async def cancel_watcher():
            while True:
                await asyncio.sleep(.5)
                if (work/'cancel').exists():parent.cancel();return
        watcher=asyncio.create_task(cancel_watcher())
        try:
            r['status']='running';self.save(r)
            with (work/'events.jsonl').open('wb') as events,(work/'stderr.log').open('wb') as errors:
                proc=await asyncio.create_subprocess_exec(*args,cwd=ROOT,stdin=asyncio.subprocess.DEVNULL,stdout=events,stderr=errors,start_new_session=True)
                async with asyncio.timeout(timeout):code=await proc.wait()
            r['exit_code']=code
            answer=work/'response.txt';r['response']=answer.read_text() if answer.exists() else None
            r['status']='completed' if code==0 and r['response'] else 'failed'
            if r['status']=='failed':r['error']=f'Codex exited {code}; see stderr.log and events.jsonl'
        except asyncio.CancelledError:r['status']='cancelled'
        except Exception as e:r['status']='failed';r['error']=str(e) or type(e).__name__
        finally:
            watcher.cancel()
            if proc and proc.returncode is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:await asyncio.wait_for(proc.wait(),5)
                except TimeoutError:os.killpg(proc.pid,signal.SIGKILL);await proc.wait()
            r['run_ids']=[p.parent.name for p in DATA.glob('*/job.json') if p.parent.name not in before and json.loads(p.read_text())['created_at']>=r['created_at']]
            if r['status'] in ('cancelled','failed'):
                from .jobs import Jobs
                broker=Jobs()
                for ident in r['run_ids']:
                    if broker.records[ident]['config'].get('runner_options',{}).get('agent_task_id')==r['id']: await broker.cancel(ident)
            r['finished_at']=time.time();self.save(r)
            fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    async def cancel(self,ident):
        r=self.get(ident)
        if r['status'] not in ('completed','failed','cancelled'):(self.directory/ident/'cancel').touch()
    async def close(self):
        for task in self.tasks.values():
            if not task.done():task.cancel()
        await asyncio.gather(*self.tasks.values(),return_exceptions=True)
