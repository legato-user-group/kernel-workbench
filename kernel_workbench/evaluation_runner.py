"""Prepare and execute an evaluation from a reusable experiment definition."""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .configuration import CollectionConfig, load_experiment
from .paths import ROOT


def prepare_evaluation(definition, models, output=None, *, warmup=None, repeats=None,
                       new_tokens=None, resume=False):
    definition = Path(definition).resolve()
    if definition.is_dir():
        definition /= 'config.json'
    experiment = load_experiment(definition)
    overrides = {key: value for key, value in {
        'warmup': warmup, 'baseline_repeats': repeats, 'output_tokens': new_tokens,
    }.items() if value is not None}
    config = CollectionConfig(**{**experiment.collection_config(), **overrides})
    if config.output_tokens < 2:
        raise ValueError('The HF evaluation requires at least two output tokens')
    model_directory = ROOT / 'models' / experiment.model
    suite = config.runner_options.get('suite', 'evaluation')
    if not suite.isidentifier() or suite.startswith('_') or not (model_directory / f'{suite}.py').is_file():
        raise ValueError(f'{experiment.model} does not provide an HF evaluation suite')
    models = Path(models).resolve()
    model_data = models.read_bytes()
    output = Path(output or definition.parent / date.today().isoformat()).resolve()
    command = [sys.executable, '-m', f'models.{experiment.model}.{suite}',
               '--models', str(models), '--output', str(output / 'results'),
               '--lengths', str(config.input_tokens), '--new-tokens', str(config.output_tokens),
               '--warmup', str(config.warmup), '--repeats', str(config.baseline_repeats), '--skip-hf-timing']
    if suite == 'checkpoint_evaluation':
        command += ['--dtype', config.dtype, '--expected-gpu', config.gpu,
                    '--reference-device', config.runner_options.get('reference_device', 'cuda'),
                    '--kernels', config.runner_options.get('kernels', 'torch')]
    plan = {'name': experiment.name, 'model': experiment.model, 'input_tokens': config.input_tokens,
            'models_manifest': str(models), 'models_sha256': hashlib.sha256(model_data).hexdigest(),
            'output': str(output), 'command': command}
    files = [*model_directory.rglob('*.py'), *model_directory.glob('*.json'),
             *(ROOT / 'models/common').glob('*.py')]
    source = {'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
    source['files']['experiment.json'] = hashlib.sha256(definition.read_bytes()).hexdigest()
    plan_path, source_path = output / 'invocation.json', output / 'source_manifest.json'
    if plan_path.exists():
        if not resume:
            raise FileExistsError('Use a new evaluation date/directory or --resume')
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Resume workload/model identity differs from the saved invocation')
        if not source_path.is_file() or json.loads(source_path.read_text()) != source:
            raise ValueError('Resume source differs from the saved invocation')
    elif resume or (output.exists() and any(output.iterdir())):
        raise FileExistsError('Existing evidence cannot be reused without a saved invocation; use a new directory')
    output.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    source_path.write_text(json.dumps(source, indent=2) + '\n')
    (output / 'models.json').write_bytes(model_data)
    return plan


def execute_evaluation(plan, *, resume=False, timeout=7200):
    command = [*plan['command'], *(['--resume'] if resume else [])]
    subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                   check=True, timeout=timeout)
