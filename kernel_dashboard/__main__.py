"""Headless agent entrypoint. Structured output on stdout; logs live in .runs/."""
import argparse
import asyncio
import json
from pathlib import Path
from .jobs import Jobs, DATA
from .analysis import with_phase_points

async def command(args):
    if args.command=='agent':
        import sys,socket,urllib.request
        from .agents import Agents
        url=args.api_url;server=None
        def ready():
            try:
                with urllib.request.urlopen(url+'/api/runners',timeout=1) as response:return response.status==200
            except Exception:return False
        if not await asyncio.to_thread(ready):
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            url=f'http://127.0.0.1:{port}'
            log=(DATA/'agent-broker.log').open('ab')
            server=await asyncio.create_subprocess_exec(sys.executable,'-m','uvicorn','kernel_dashboard.app:app','--host','127.0.0.1','--port',str(port),stdout=log,stderr=log)
            for _ in range(50):
                if await asyncio.to_thread(ready):break
                if server.returncode is not None:raise ValueError('Local profiling broker failed to start')
                await asyncio.sleep(.1)
            else:server.terminate();await server.wait();raise ValueError('Profiling broker did not become ready')
        agents=Agents()
        try:
            r=await agents.start(args.prompt,url,args.timeout,args.model)
            print(json.dumps({'agent_task_id':r['id'],'status':r['status']}),flush=True)
            await agents.tasks[r['id']]
            return r
        finally:
            await agents.close()
            if server:
                server.terminate()
                try:await asyncio.wait_for(server.wait(),150)
                except TimeoutError:server.kill();await server.wait()
    jobs=Jobs()
    if args.command=='register':
        path=Path(args.file).resolve()
        if not path.is_file(): raise ValueError('Runner file does not exist')
        registry=jobs.runners(); registry.pop('tiny_decoder',None)
        registry[args.name]=str(path); (DATA/'runners.json').write_text(json.dumps(registry,indent=2))
        return {'runner':args.name,'path':str(path)}
    if args.command=='cancel': await jobs.cancel(args.id); return {'id':args.id,'cancel_requested':True}
    if args.command=='status': return jobs.records[args.id]
    if args.command=='list': return list(jobs.records.values())
    if args.command=='recover': await jobs.recover(); return list(jobs.records.values())
    if args.command=='result': return with_phase_points(json.loads((DATA/args.id/'result.json').read_text()))
    if args.command=='run':
        from .app import RunConfig
        config=json.loads(Path(args.config).read_text()) if args.config else {}
        config['runner']=args.runner
        # API validation is shared by the headless path.
        config=RunConfig(**config).model_dump()
        r=await jobs.start(config)
        print(json.dumps({'id':r['id'],'status':r['status'],'session':r['session']}),flush=True)
        try: await jobs.tasks[r['id']]
        finally: await jobs.close()
        return r

def main():
    parser=argparse.ArgumentParser(description='Kernel Observatory: headless Colab profiling and local dashboard')
    sub=parser.add_subparsers(dest='command',required=True)
    server=sub.add_parser('serve'); server.add_argument('--port',type=int,default=8765)
    run=sub.add_parser('run'); run.add_argument('--runner',default='tiny_decoder'); run.add_argument('--config')
    register=sub.add_parser('register'); register.add_argument('name'); register.add_argument('file')
    agent=sub.add_parser('agent'); agent.add_argument('--prompt',required=True); agent.add_argument('--model'); agent.add_argument('--timeout',type=int,default=1800); agent.add_argument('--api-url',default='http://127.0.0.1:8765')
    sub.add_parser('list'); sub.add_parser('recover')
    for name in ('result','status','cancel'):
        result=sub.add_parser(name); result.add_argument('id')
    args=parser.parse_args()
    if args.command=='serve':
        import uvicorn
        uvicorn.run('kernel_dashboard.app:app',host='127.0.0.1',port=args.port)
        return
    try:
        value=asyncio.run(command(args)); print(json.dumps(value,ensure_ascii=False))
        if isinstance(value,dict) and value.get('status') in ('failed','cancelled'): raise SystemExit(1)
    except (ValueError,KeyError,FileNotFoundError) as e:
        print(json.dumps({'error':str(e)})); raise SystemExit(1)
if __name__=='__main__': main()
