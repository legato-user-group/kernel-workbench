"""Immutable optimization attempts linked to uploaded measurements."""
import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .paths import DATA
from .reports import ID_PATTERN


class Correctness(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['passed', 'failed', 'not_checked'] = 'not_checked'
    details: str = ''


class ExperimentUpload(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    id: str | None = Field(default=None, pattern=ID_PATTERN)
    series: str = Field(default='default', min_length=1)
    title: str = Field(min_length=1)
    hypothesis: str = ''
    changes: str = Field(min_length=1)
    code_revision: str = ''
    environment: str = ''
    status: Literal['completed', 'failed', 'inconclusive'] = 'completed'
    baseline_run_id: str | None = Field(default=None, pattern=ID_PATTERN)
    candidate_run_id: str | None = Field(default=None, pattern=ID_PATTERN)
    correctness: Correctness = Field(default_factory=Correctness)
    conclusion: str = ''
    next_steps: str = ''

    @model_validator(mode='after')
    def completed_needs_measurement(self):
        if self.status == 'completed' and not self.candidate_run_id:
            raise ValueError('A completed attempt needs a candidate_run_id; use failed/inconclusive when no result exists')
        if self.baseline_run_id and self.baseline_run_id == self.candidate_run_id:
            raise ValueError('Baseline and candidate must be different runs')
        return self


class ExperimentStore:
    def __init__(self, directory=None, raw_repository=None):
        self.directory = Path(directory) if directory is not None else DATA / 'attempts'
        self.raw=raw_repository

    def path(self, ident):
        if not re.fullmatch(ID_PATTERN, ident):
            raise ValueError('Invalid experiment id')
        return self.directory / ident

    def get(self, ident):
        if self.raw:
            try:return self.raw.attempt(ident)
            except FileNotFoundError:pass
        return json.loads((self.path(ident) / 'experiment.json').read_text())

    def records(self, series=None):
        records = [self.get(path.parent.name) for path in self.directory.glob('*/experiment.json')
                   if not path.parent.name.startswith('.')]
        if self.raw:records.extend(self.raw.attempts())
        return sorted([r for r in records if series is None or r['series'] == series],
                      key=lambda r: r['created_at'], reverse=True)

    def save(self, upload):
        ident = upload.id or uuid.uuid4().hex[:12]
        target = self.path(ident)
        if target.exists() or (self.raw and any(r['id']==ident for r in self.raw.attempts())):
            raise FileExistsError('Experiment id already exists')
        record = {**upload.model_dump(), 'id': ident, 'created_at': time.time()}
        self.directory.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.upload-', dir=self.directory))
        try:
            (staging / 'experiment.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
            staging.rename(target)
        except OSError:
            if target.exists():
                raise FileExistsError('Experiment id already exists')
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return record


def compare(record, reports):
    """Only publish speedups for correct, independently timed matching workloads."""
    reasons, measurements = [], {}
    for role in ('baseline', 'candidate'):
        ident = record[role + '_run_id']
        if not ident:
            reasons.append(role + '_missing')
            measurements[role] = None
            continue
        try:
            reports.get(ident)
            data = reports.result(ident)
        except (ValueError, FileNotFoundError):
            reasons.append(role + '_unavailable')
            measurements[role] = None
            continue
        measurements[role] = {'run_id': ident, 'timings': data['timings'],
                              'runner_id': data.get('runner_id', data['config']['runner']),
                              'config': data['config'], 'hardware': data['hardware'],
                              'model_name': data['model']['name']}
        if data['timings'].get('source') != 'unprofiled':
            reasons.append(role + '_not_independent')
    baseline, candidate = measurements['baseline'], measurements['candidate']
    if baseline and candidate:
        if baseline['hardware']['name'] != candidate['hardware']['name']:
            reasons.append('gpu_mismatch')
        if baseline['model_name'] != candidate['model_name']:
            reasons.append('model_mismatch')
        # Runner names may change when implementing a kernel; workload must not.
        ignored = {'runner', 'gpu', 'warmup', 'baseline_repeats', 'timeout_sec'}
        configs = [{k: v for k, v in r['config'].items() if k not in ignored}
                   for r in (baseline, candidate)]
        if configs[0] != configs[1]:
            reasons.append('workload_mismatch')
    if record['correctness']['status'] != 'passed':
        reasons.append('correctness_not_passed')
    if record['status'] != 'completed':
        reasons.append('attempt_not_completed')
    speedup, reduction, metric_reasons = {}, {}, {}
    if not reasons:
        for key in ('total_ms', 'prefill', 'decode'):
            incomplete = [role for role in ('baseline', 'candidate')
                          if (timings := measurements[role]['timings']).get('expected_samples') is not None
                          and timings['sample_counts'].get(key) != timings['expected_samples']]
            if incomplete:
                metric_reasons[key] = [role + '_incomplete_samples' for role in incomplete]
                continue
            old, new = baseline['timings'].get(key), candidate['timings'].get(key)
            if old is not None and new is not None and old > 0 and new > 0:
                speedup[key] = old / new
                reduction[key] = (old - new) / old * 100
    return {'eligible': not reasons, 'reasons': reasons, **measurements,
            'speedup': speedup, 'latency_reduction_pct': reduction,
            'metric_reasons': metric_reasons}
