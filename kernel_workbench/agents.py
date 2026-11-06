"""Read-only access to records from the former embedded Codex worker.

New work is driven by an external coding agent that runs remote commands and uploads results.
"""
import json
from .paths import DATA


class AgentHistory:
    def __init__(self):
        self.directory = DATA / 'agents'

    def records(self):
        return sorted(
            [json.loads(path.read_text()) for path in self.directory.glob('*/task.json')],
            key=lambda record: record['created_at'], reverse=True,
        )

    def get(self, ident):
        if not ident.isalnum():
            raise ValueError('Unknown task')
        return json.loads((self.directory / ident / 'task.json').read_text())
