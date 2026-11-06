"""File-backed storage for results prepared and uploaded by a coding agent."""
import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .paths import DATA

ARTIFACT_NAMES = {'artifacts.tar.gz', 'commands.log', 'baseline.json', 'analysis.json'}
ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$'


class Report(BaseModel):
    model_config = ConfigDict(extra='allow')
    schema_version: int = 1
    model: dict
    config: dict
    hardware: dict
    timings: dict
    kernels: list[dict]
    estimates: list[dict] = Field(default_factory=list)
    point: dict | None = None
    coverage: dict
    warnings: list[str] = Field(default_factory=list)
    versions: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_dashboard_fields(self):
        # Require the fields consumed by the view; never execute uploaded content.
        if not isinstance(self.model.get('name'), str) or not isinstance(self.model.get('children'), list):
            raise ValueError('model needs name and children')
        for key in ('runner', 'dtype'):
            if not isinstance(self.config.get(key), str):
                raise ValueError(f'config.{key} must be a string')
        for key in ('batch', 'input_tokens', 'output_tokens'):
            value = self.config.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'config.{key} must be positive')
        if not isinstance(self.hardware.get('name'), str):
            raise ValueError('hardware.name must be a string')
        for key in ('total_ms', 'prefill', 'decode'):
            value = self.timings.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f'timings.{key} must be a nonnegative number')
        # Separate passes can recover different numbers of samples after interruption.
        if 'expected_samples' in self.timings or 'sample_counts' in self.timings:
            expected = self.timings.get('expected_samples')
            counts = self.timings.get('sample_counts')
            if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
                raise ValueError('timings.expected_samples must be a positive integer')
            if not isinstance(counts, dict) or not all(
                isinstance(counts.get(key), int) and not isinstance(counts[key], bool)
                and counts[key] >= 1 for key in ('total_ms', 'prefill', 'decode')
            ):
                raise ValueError('timings.sample_counts needs positive counts for each measured phase')
        for kernel in self.kernels:
            if not all(key in kernel for key in ('name', 'phase', 'module', 'duration_ms', 'bound')):
                raise ValueError('Each kernel needs name, phase, module, duration_ms and bound')
            if not all(isinstance(kernel[k],str) for k in ('name','phase','module')) or kernel['bound'] not in ('compute','memory','unknown'):
                raise ValueError('Invalid kernel identity or bound')
            if not isinstance(kernel['duration_ms'],(int,float)) or kernel['duration_ms']<0:
                raise ValueError('Invalid kernel duration')
            if kernel.get('flops') is not None and (not isinstance(kernel['flops'],dict) or not all(isinstance(v,(int,float)) and v>=0 for v in kernel['flops'].values())):
                raise ValueError('Invalid kernel FLOPs')
            if kernel.get('bytes') is not None and (not isinstance(kernel['bytes'],(int,float)) or kernel['bytes']<0):
                raise ValueError('Invalid kernel traffic')
        for estimate in self.estimates:
            if not all(key in estimate for key in ('module', 'phase', 'flops', 'bytes')):
                raise ValueError('Each estimate needs module, phase, flops and bytes')
            if not isinstance(estimate['flops'],dict) or not all(isinstance(v,(int,float)) and v>=0 for v in estimate['flops'].values()) or not isinstance(estimate['bytes'],(int,float)) or estimate['bytes']<0:
                raise ValueError('Invalid estimated work/traffic')
        if not all(isinstance(self.coverage.get(k), int) for k in ('measured', 'total')):
            raise ValueError('coverage needs measured and total counts')
        if not 0<=self.coverage['measured']<=self.coverage['total']:
            raise ValueError('Invalid coverage counts')
        # JSON NaN/Infinity would also break the dashboard's numeric scales.
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class ReportUpload(BaseModel):
    id: str | None = Field(default=None, pattern=ID_PATTERN)
    summary: str = ''
    result: Report


class AnalysisSupplement(BaseModel):
    """Source-derived display data, kept separate from immutable measurements."""
    model_config = ConfigDict(extra='forbid')
    runner_id: str = Field(min_length=1)
    model: dict
    estimates: list[dict]
    kernel_plan: list[dict]
    estimate_scope: str
    phase_estimate_reasons: dict[str, str] = Field(default_factory=dict)
    analysis_notes: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)


class ReportStore:
    def __init__(self, directory=None, raw_repository=None):
        self.directory = Path(directory) if directory is not None else DATA
        self.raw = raw_repository

    def is_raw(self,ident):
        return self.raw is not None and self.raw.contains(ident)

    def has_result(self,ident):
        return self.is_raw(ident) or (self.path(ident)/'result.json').is_file()

    def path(self, ident):
        if not re.fullmatch(ID_PATTERN, ident):
            raise ValueError('Invalid report id')
        return self.directory / ident

    def get(self, ident):
        if self.is_raw(ident):return self.raw.record(ident)
        path = self.path(ident)
        record = json.loads((path / 'job.json').read_text())
        record['artifacts'] = [name for name in ('result.json', *sorted(ARTIFACT_NAMES)) if (path / name).is_file()]
        analysis = path / 'analysis.json'
        if analysis.is_file():
            record['runner_id'] = json.loads(analysis.read_text())['runner_id']
        return record

    def result(self, ident):
        if self.is_raw(ident):return self.raw.result(ident)
        path = self.path(ident)
        result = json.loads((path / 'result.json').read_text())
        analysis = path / 'analysis.json'
        if analysis.is_file():
            supplement = AnalysisSupplement(**json.loads(analysis.read_text())).model_dump()
            result.update(supplement)
        return result

    def records(self):
        # Read afresh so files removed outside the app also disappear from the list.
        records = [self.get(path.parent.name) for path in self.directory.glob('*/job.json') if not path.parent.name.startswith('.')]
        if self.raw:records.extend(self.raw.records())
        return sorted(records, key=lambda record: (record['created_at'],record['config']['input_tokens']), reverse=True)

    def save_upload(self, upload):
        ident = upload.id or uuid.uuid4().hex[:12]
        target = self.path(ident)
        if target.exists() or self.is_raw(ident):
            raise FileExistsError('Report id already exists')
        result = upload.result.model_dump()
        result['agent_summary'] = upload.summary
        now = time.time()
        record = {'id': ident, 'status': 'completed', 'config': result['config'],
                  'created_at': now, 'finished_at': now, 'source': 'agent_upload',
                  'error': None, 'cleanup_error': None, 'owns_session': False}
        # Publish complete reports atomically, without accepting an archive or code.
        self.directory.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.upload-', dir=self.directory))
        try:
            (staging / 'result.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
            (staging / 'job.json').write_text(json.dumps(record, indent=2))
            staging.rename(target)
        except OSError:
            if target.exists():
                raise FileExistsError('Report id already exists')
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return self.get(ident)
