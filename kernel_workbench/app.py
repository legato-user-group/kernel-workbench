"""Result ingestion and visualization only. Never starts or stops GPU work."""
import json
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .paths import ROOT
from .reports import ReportStore, ReportUpload, ARTIFACT_NAMES
from .agents import AgentHistory
from .experiments import ExperimentStore, ExperimentUpload, compare
from .analysis import with_phase_points

store = ReportStore()
agents = AgentHistory()
experiments = ExperimentStore()
app = FastAPI(title='Kernel Workbench Results API')


@app.middleware('http')
async def local_mutations(request: Request, call_next):
    origin = request.headers.get('origin')
    if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin and origin != str(request.base_url).rstrip('/'):
        return JSONResponse({'detail': 'Cross-origin mutation rejected'}, status_code=403)
    return await call_next(request)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'execution': 'agent_owned', 'mode': 'result_store',
            'openapi_url': '/openapi.json', 'guide_url': '/api/guide'}


@app.get('/api/guide')
def guide():
    return FileResponse(ROOT / 'AGENT_API.md', media_type='text/markdown')


@app.get('/api/runs')
def runs():
    return store.records()


@app.post('/api/runs', status_code=201)
def upload_report(upload: ReportUpload):
    try:
        record = store.save_upload(upload)
    except FileExistsError as error:
        raise HTTPException(409, str(error))
    base = f'/api/runs/{record["id"]}'
    return JSONResponse({**record, 'links': {'status': base, 'result': base + '/result',
                                           'artifacts': base + '/artifacts/{name}'}},
                        status_code=201, headers={'Location': base})


def record(ident):
    try:
        return store.get(ident)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, 'Unknown report')


@app.get('/api/runs/{ident}')
def get(ident: str):
    return record(ident)


@app.get('/api/runs/{ident}/result')
def result(ident: str):
    record(ident)
    path = store.path(ident) / 'result.json'
    if not path.exists():
        raise HTTPException(409, 'This legacy run has no saved result')
    data = with_phase_points(json.loads(path.read_text()))
    data['artifacts'] = record(ident)['artifacts']
    return data


@app.put('/api/runs/{ident}/artifacts/{name}', status_code=201)
async def upload_artifact(ident: str, name: str, request: Request):
    record(ident)
    if name not in ARTIFACT_NAMES:
        raise HTTPException(422, 'Unsupported artifact name')
    directory = store.path(ident)
    temp = directory / ('.artifact-' + uuid.uuid4().hex)
    try:
        with temp.open('wb') as output:
            async for chunk in request.stream():
                output.write(chunk)
        temp.replace(directory / name)
    finally:
        temp.unlink(missing_ok=True)
    return {'id': ident, 'artifact': name}


@app.get('/api/runs/{ident}/artifacts/{name}')
def artifact(ident: str, name: str):
    record(ident)
    if name not in ARTIFACT_NAMES | {'result.json'}:
        raise HTTPException(404)
    path = store.path(ident) / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=name)


def experiment_result(saved):
    return {**saved, 'comparison': compare(saved, store)}


@app.get('/api/experiments')
def list_experiments(series: str | None = None):
    return [experiment_result(saved) for saved in experiments.records(series)]


@app.post('/api/experiments', status_code=201)
def upload_experiment(upload: ExperimentUpload):
    # References are checked before publishing; failed attempts need no run.
    for ident in (upload.baseline_run_id, upload.candidate_run_id):
        if ident:
            record(ident)
            if not (store.path(ident) / 'result.json').is_file():
                raise HTTPException(422, 'Linked run has no result')
    try:
        saved = experiments.save(upload)
    except FileExistsError as error:
        raise HTTPException(409, str(error))
    return JSONResponse(experiment_result(saved), status_code=201,
                        headers={'Location': f'/api/experiments/{saved["id"]}'})


@app.get('/api/experiments/{ident}')
def get_experiment(ident: str):
    try:
        return experiment_result(experiments.get(ident))
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, 'Unknown experiment')


@app.get('/api/agent-tasks')
def agent_tasks():
    return agents.records()


@app.get('/api/agent-tasks/{ident}')
def agent_task(ident: str):
    try:
        return agents.get(ident)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, 'Unknown agent task')


@app.get('/api/agent-tasks/{ident}/events')
def agent_events(ident: str):
    agent_task(ident)
    path = agents.directory / ident / 'events.jsonl'
    if not path.exists():
        return {'events': []}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        text = stream.read().decode('utf-8', errors='replace')
    return {'tail': text}


app.mount('/', StaticFiles(directory=ROOT / 'kernel_workbench/static', html=True), name='dashboard')
