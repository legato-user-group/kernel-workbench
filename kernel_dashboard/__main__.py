"""Headless agent entrypoint. Structured output on stdout; logs live in .runs/."""
import argparse
import asyncio
import json
from pathlib import Path
from .jobs import Jobs
from .analysis import with_phase_points

async def command(args):
    jobs=Jobs()
    if args.command=='register':
        return jobs.register(args.name,args.file)
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
    parser=argparse.ArgumentParser(description='Kernel Observatory: coding-agent profiling API and results dashboard')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('api','serve'):
        server=sub.add_parser(name);server.add_argument('--port',type=int,default=8765)
    run=sub.add_parser('run'); run.add_argument('--runner',default='tiny_decoder'); run.add_argument('--config')
    register=sub.add_parser('register'); register.add_argument('name'); register.add_argument('file')
    sub.add_parser('list'); sub.add_parser('recover')
    for name in ('result','status','cancel'):
        result=sub.add_parser(name); result.add_argument('id')
    args=parser.parse_args()
    if args.command in ('api','serve'):
        import uvicorn
        uvicorn.run('kernel_dashboard.app:app',host='127.0.0.1',port=args.port)
        return
    try:
        value=asyncio.run(command(args)); print(json.dumps(value,ensure_ascii=False))
        if isinstance(value,dict) and value.get('status') in ('failed','cancelled'): raise SystemExit(1)
    except (ValueError,KeyError,FileNotFoundError) as e:
        print(json.dumps({'error':str(e)})); raise SystemExit(1)
if __name__=='__main__': main()
