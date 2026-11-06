import asyncio
import fcntl
import json
import os
import tarfile
import time
import uuid
from pathlib import Path
from .analysis import analyze, read_ncu, read_nsys

ROOT=Path(__file__).resolve().parent.parent
DATA=Path(os.environ.get('KERNEL_OPT_DATA', ROOT/'.runs'))
STATES=('completed','failed','cancelled')

class Jobs:
    def __init__(self):
        self.tasks={}; self.records={}; self.lock=asyncio.Lock(); self.lease=None
        DATA.mkdir(parents=True,exist_ok=True)
        for p in DATA.glob('*/job.json'):
            record=json.loads(p.read_text()); self.records[record['id']]=record
    def refresh(self):
        for p in DATA.glob('*/job.json'):
            r=json.loads(p.read_text())
            if r['id'] not in self.tasks or self.tasks[r['id']].done(): self.records[r['id']]=r
    def acquire(self):
        lease=(DATA/'collection.lock').open('a')
        try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: lease.close(); raise ValueError('Another headless/API collection is running')
        self.lease=lease
    def release(self):
        if self.lease:
            fcntl.flock(self.lease,fcntl.LOCK_UN); self.lease.close(); self.lease=None
    def runners(self):
        registry=DATA/'runners.json'
        return {'tiny_decoder':str(ROOT/'runners/tiny_decoder.py'), **(json.loads(registry.read_text()) if registry.exists() else {})}
    def save(self,r):
        dest=DATA/r['id']; dest.mkdir(exist_ok=True)
        temp=dest/'job.tmp'; temp.write_text(json.dumps(r,indent=2)); temp.replace(dest/'job.json')
    async def command(self,r,*args,timeout=1800):
        with (DATA/r['id']/'commands.log').open('ab') as out:
            proc=await asyncio.create_subprocess_exec(*map(str,args),stdout=out,stderr=out)
            try:
                async with asyncio.timeout(timeout): code=await proc.wait()
            finally:
                if proc.returncode is None:
                    proc.terminate()
                    try: await asyncio.wait_for(proc.wait(),5)
                    except TimeoutError: proc.kill(); await proc.wait()
        if code: raise RuntimeError(f'{args[0]} {args[1]} exited {code}; see commands.log')
    async def start(self,config):
        if config['runner'] not in self.runners(): raise ValueError('Runner is not registered')
        async with self.lock:
            self.acquire()
            try: await self.recover_owned()
            except BaseException: self.release(); raise
            ident=uuid.uuid4().hex[:12]
            r={'id':ident,'status':'queued','session':'kernel-opt-'+ident,'config':config,
               'created_at':time.time(),'error':None,'cleanup_error':None,'owns_session':False}
            self.records[ident]=r; self.save(r)
            self.tasks[ident]=asyncio.create_task(self.execute(r))
            return r
    async def execute(self,r,allocate=True):
        work=DATA/r['id']; remote='/content/'+r['session']; c=r['config']; session=r['session']
        def state(value): r['status']=value; self.save(r)
        parent=asyncio.current_task()
        async def watch_cancel():
            while True:
                await asyncio.sleep(1)
                if (work/'cancel').exists(): parent.cancel(); return
        watcher=asyncio.create_task(watch_cancel())
        try:
            async with asyncio.timeout(c.get('timeout_sec',1800)):
                state('allocating'); r['owns_session']=True; self.save(r)
                if allocate: await self.command(r,'colab','new','-s',session,'--gpu',c['gpu'])
                state('preparing')
                (work/'config.json').write_text(json.dumps(c))
                # Include only explicitly selected runner and orchestration code.
                with tarfile.open(work/'source.tar.gz','w:gz') as archive:
                    archive.add(self.runners()[c['runner']],arcname='runner.py')
                    archive.add(ROOT/'kernel_dashboard/remote_entry.py',arcname='remote_entry.py')
                    archive.add(ROOT/'kernel_dashboard/bootstrap.py',arcname='bootstrap.py')
                    archive.add(work/'config.json',arcname='config.json')
                await self.command(r,'colab','upload','-s',session,work/'source.tar.gz','/content/'+session+'.tar.gz')
                script=work/'collect.py'
                script.write_text('import tarfile, pathlib, subprocess, sys\n'
                    +f'p=pathlib.Path({remote!r}); p.mkdir(exist_ok=True)\n'
                    +f'with tarfile.open({("/content/"+session+".tar.gz")!r}) as a: a.extractall(p, filter="data")\n'
                    +f'subprocess.run([sys.executable,str(p/"bootstrap.py"),str(p)],check=True)\n')
                state('profiling')
                await self.command(r,'colab','exec','-s',session,'--timeout',c.get('timeout_sec',1800),'-f',script)
                state('downloading')
                await self.command(r,'colab','download','-s',session,remote+'/artifacts.tar.gz',work/'artifacts.tar.gz')
                with tarfile.open(work/'artifacts.tar.gz') as archive: archive.extractall(work/'artifacts',filter='data')
                artifacts=work/'artifacts'; meta=json.loads((artifacts/'meta.json').read_text())
                status=json.loads((artifacts/'status.json').read_text())
                meta.setdefault('warnings',[]).extend(status['warnings'])
                meta['versions'].update(status.get('tools',{}))
                kernels=[]; ncu=None
                try: kernels=read_nsys(artifacts/'trace.sqlite')
                except Exception as e: meta['warnings'].append('Timeline unavailable: '+str(e))
                if (artifacts/'counters.csv').exists():
                    try: ncu=read_ncu(artifacts/'counters.csv')
                    except Exception as e: meta['warnings'].append('Counters unavailable: '+str(e))
                result=analyze(meta,kernels,ncu)
                (work/'result.json').write_text(json.dumps(result,indent=2))
                state('completed')
        except asyncio.CancelledError:
            state('cancelled')
        except Exception as e:
            r['error']=str(e) or type(e).__name__; state('failed')
        finally:
            watcher.cancel()
            if r['owns_session']:
                try:
                    await self.command(r,'colab','stop','-s',session,timeout=120)
                    r['owns_session']=False
                except Exception as e: r['cleanup_error']=str(e)
            r['finished_at']=time.time(); self.save(r); self.release()
    async def recover(self):
        try: self.acquire()
        except ValueError: return
        try: await self.recover_owned()
        finally: self.release()
    async def recover_owned(self):
        self.refresh()
        for r in self.records.values():
            if r['status'] not in STATES: r['status']='failed'; r['error']='Interrupted local process'
            if r.get('owns_session'):
                try:
                    await self.command(r,'colab','stop','-s',r['session'],timeout=120)
                    r['owns_session']=False; r['cleanup_error']=None
                except Exception as e: r['cleanup_error']=str(e)
            self.save(r)
    async def cancel(self,ident):
        self.refresh()
        if ident not in self.records: raise ValueError('Unknown run')
        if self.records[ident]['status'] not in STATES:
            (DATA/ident/'cancel').touch()
        task=self.tasks.get(ident)
        if task and not task.done():
            task.cancel()
            try: await task
            except asyncio.CancelledError:
                r=self.records[ident]; r['status']='cancelled'; self.save(r); self.release()
    async def close(self):
        for ident in list(self.tasks): await self.cancel(ident)
