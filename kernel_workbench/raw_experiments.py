"""Read-only dashboard view of experiments/<name>/<date>/results evidence."""
import json
from datetime import datetime
from pathlib import Path

from .evaluation_data import build


def _records(root, directory):
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
        record = {'id': payload['id'], 'status': 'completed', 'config': result['config'],
                  'runner_id': result['runner_id'], 'created_at': timestamp, 'finished_at': timestamp,
                  'source': 'experiment_raw', 'owns_session': False, 'error': None, 'cleanup_error': None,
                  'artifacts': ['result.json', *[name for name, path in artifacts.items() if path.is_file()]],
                  'data_directory': str(root.relative_to(directory))}
        result['raw_source'] = str((root / source).relative_to(directory))
        runs[payload['id']] = {'record': record, 'result': result, 'artifacts': artifacts}
    attempts = {p['id']: {**p, 'created_at': timestamp, 'source': 'experiment_raw'} for p in attempts}
    return runs, attempts


class RawExperiments:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._fingerprint = None
        self._runs = {}
        self._attempts = {}
        self.issues = []

    def _refresh(self):
        contracts = sorted(self.directory.glob('*/*/results/contract.json'))
        files = sorted({p for contract in contracts for p in contract.parent.parent.rglob('*.json')})
        fingerprint = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.is_file())
        if fingerprint == self._fingerprint:
            return
        runs, attempts, issues = {}, {}, []
        for contract in contracts:
            root = contract.parent.parent
            try:
                new_runs, new_attempts = _records(root, self.directory)
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
