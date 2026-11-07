"""Read-only dashboard view of experiments/<name>/<date>/results evidence."""
import json
import hashlib
import sqlite3
from threading import RLock
from datetime import datetime
from pathlib import Path

from .evaluation_data import build
from .profile_data import attach_profile


def _records(root, directory, revision):
    uploads, attempts = build(root)
    timestamp = root.stat().st_mtime
    observed = root / 'session_status.json'
    if observed.is_file():
        value = json.loads(observed.read_text()).get('observed_at_utc')
        if value:
            timestamp = datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    runs = {}
    for payload, source in uploads:
        result = {**payload['result'], 'agent_summary': payload['summary']}
        artifacts = {'baseline.json': root / source, 'contract.json': root / 'results/contract.json',
                     'correctness.json': root / 'results/correctness.json', 'inputs.json': root / 'results/inputs.json'}
        try:
            artifacts.update(attach_profile(root, result, json.loads((root / source).read_text())))
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
            result['warnings'].append('프로파일 해석 실패: ' + str(error))
        result['evidence_revision'] = revision
        record = {'id': payload['id'], 'status': 'completed', 'config': result['config'],
                  'runner_id': result['runner_id'], 'created_at': timestamp, 'finished_at': timestamp,
                  'source': 'experiment_raw', 'owns_session': False, 'error': None, 'cleanup_error': None,
                  'artifacts': ['result.json', *[name for name, path in artifacts.items() if path.is_file()]],
                  'data_directory': str(root.relative_to(directory))}
        record['evidence_revision'] = revision
        result['raw_source'] = str((root / source).relative_to(directory))
        runs[payload['id']] = {'record': record, 'result': result, 'artifacts': artifacts}
    attempts = {p['id']: {**p, 'created_at': timestamp, 'source': 'experiment_raw',
                           'data_directory': str(root.relative_to(directory))} for p in attempts}
    return runs, attempts


class RawExperiments:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._fingerprint = None
        self._runs = {}
        self._attempts = {}
        self.issues = []
        self._lock = RLock()

    def _refresh(self):
        with self._lock:
            self._read()

    def _read(self):
        contracts = sorted(self.directory.glob('*/*/results/contract.json'))
        files = sorted({p for contract in contracts for pattern in ('**/*.json', 'profile*/trace.sqlite', 'profile*/*.csv')
                        for p in contract.parent.parent.glob(pattern)})
        fingerprint = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.is_file())
        if fingerprint == self._fingerprint:
            return
        runs, attempts, issues = {}, {}, []
        for contract in contracts:
            root = contract.parent.parent
            try:
                revision = hashlib.sha256(repr([entry for entry in fingerprint if Path(entry[0]).is_relative_to(root)]).encode()).hexdigest()
                new_runs, new_attempts = _records(root, self.directory, revision)
                duplicate = (new_runs.keys() & runs.keys()) | (new_attempts.keys() & attempts.keys())
                if duplicate:
                    raise ValueError('Duplicate experiment/run ID: ' + ', '.join(sorted(duplicate)))
                runs.update(new_runs)
                attempts.update(new_attempts)
            except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                issues.append({'directory': str(root.relative_to(self.directory)), 'error': str(error)})
        self._runs, self._attempts, self.issues = runs, attempts, issues
        self._fingerprint = fingerprint

    def contains(self, ident):
        self._refresh()
        return ident in self._runs

    def record(self, ident):
        self._refresh()
        if ident not in self._runs:
            raise FileNotFoundError(ident)
        return dict(self._runs[ident]['record'])

    def result(self, ident):
        self.record(ident)
        return dict(self._runs[ident]['result'])

    def records(self):
        self._refresh()
        return [dict(value['record']) for value in self._runs.values()]

    def attempts(self):
        self._refresh()
        return [dict(value) for value in self._attempts.values()]

    def attempt(self, ident):
        self._refresh()
        if ident not in self._attempts:
            raise FileNotFoundError(ident)
        return dict(self._attempts[ident])

    def artifact(self, ident, name):
        self.record(ident)
        path = self._runs[ident]['artifacts'].get(name)
        if path is None or not path.is_file():
            raise FileNotFoundError(name)
        return path
