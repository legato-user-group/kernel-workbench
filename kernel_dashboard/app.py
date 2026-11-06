from contextlib import asynccontextmanager
from typing import Literal
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .jobs import Jobs, ROOT, DATA
from .agents import Agents
from .analysis import with_phase_points

class RunConfig(BaseModel):
    runner:str='tiny_decoder'
    gpu:Literal['A100']='A100'
    dtype:Literal['bf16','fp16','fp32']='bf16'
    batch:int=Field(1,ge=1,le=64)
    input_tokens:int=Field(128,ge=1,le=32768)
    output_tokens:int=Field(16,ge=1,le=2048)
    warmup:int=Field(1,ge=0,le=10)
    timeout_sec:int=Field(1800,ge=60,le=7200)
    runner_options:dict=Field(default_factory=dict)

jobs=Jobs()
agents=Agents()
@asynccontextmanager
async def lifespan(app):
    await jobs.recover()
    yield
    await agents.close()
    await jobs.close()
app=FastAPI(title='Kernel Observatory',lifespan=lifespan)

@app.middleware('http')
async def local_mutations(request:Request, call_next):
    # Prevent a different website from driving the authenticated local CLI.
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin and origin != str(request.base_url).rstrip('/'):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'Cross-origin mutation rejected'},status_code=403)
    return await call_next(request)
@app.get('/api/runners')
def runners(): return [{'id':name} for name in jobs.runners()]
@app.get('/api/runs')
def runs():
    jobs.refresh()
    return sorted(jobs.records.values(),key=lambda r:r['created_at'],reverse=True)
@app.post('/api/runs',status_code=202)
async def create(config:RunConfig):
    try: return await jobs.start(config.model_dump())
    except ValueError as e: raise HTTPException(409,str(e))
def record(ident):
    jobs.refresh()
    if ident not in jobs.records: raise HTTPException(404,'Unknown run')
    return jobs.records[ident]
@app.get('/api/runs/{ident}')
def get(ident:str): return record(ident)
@app.get('/api/runs/{ident}/result')
def result(ident:str):
    record(ident); path=DATA/ident/'result.json'
    if not path.exists(): raise HTTPException(409,'Result not ready')
    return with_phase_points(json.loads(path.read_text()))
@app.post('/api/runs/{ident}/cancel')
async def cancel(ident:str):
    record(ident); await jobs.cancel(ident); return record(ident)
@app.get('/api/runs/{ident}/artifacts/{name}')
def artifact(ident:str,name:str):
    record(ident)
    if name not in ('artifacts.tar.gz','commands.log','result.json'): raise HTTPException(404)
    path=DATA/ident/name
    if not path.exists(): raise HTTPException(404)
    return FileResponse(path,filename=name)
class AgentConfig(BaseModel):
    prompt:str=Field(min_length=1,max_length=32000)
    timeout_sec:int=Field(1800,ge=30,le=7200)
    model:str|None=None
@app.get('/api/agent-tasks')
def agent_tasks(): return agents.records()
@app.post('/api/agent-tasks',status_code=202)
async def agent_start(config:AgentConfig,request:Request):
    try:return await agents.start(config.prompt,str(request.base_url).rstrip('/'),config.timeout_sec,config.model)
    except ValueError as e:raise HTTPException(409,str(e))
@app.get('/api/agent-tasks/{ident}')
def agent_task(ident:str):
    try:return agents.get(ident)
    except (ValueError,FileNotFoundError):raise HTTPException(404,'Unknown agent task')
@app.post('/api/agent-tasks/{ident}/cancel')
async def agent_cancel(ident:str):
    agent_task(ident);await agents.cancel(ident);return {'id':ident,'cancel_requested':True}
@app.get('/api/agent-tasks/{ident}/events')
def agent_events(ident:str):
    agent_task(ident);path=agents.directory/ident/'events.jsonl'
    if not path.exists():return {'events':[]}
    # Bounded live tail; full events are retained in the task directory.
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-65536));text=f.read().decode('utf-8',errors='replace')
    return {'tail':text}
app.mount('/',StaticFiles(directory=ROOT/'kernel_dashboard/static',html=True),name='dashboard')
