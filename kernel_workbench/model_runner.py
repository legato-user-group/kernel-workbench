"""One collection entrypoint for all configured models, locally or in a bundle."""
from importlib import import_module
from pathlib import Path
import sys

# The collector imports this file as runner.py; a local invocation uses the repo root.
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here if (_here / 'models').is_dir() else _here.parent))


def _implementation(config):
    model = config['model']
    if not model.isidentifier() or model.startswith('_'):
        raise ValueError('Invalid model package')
    return import_module(f'models.{model}.experiment')


def describe(config):
    return _implementation(config).describe(config)


def run(config, trace):
    return _implementation(config).run(config, trace)
