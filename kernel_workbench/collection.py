"""Agent-side file preparation and analysis, independent of remote transport."""
import json
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from .paths import ROOT
from .analysis import analyze, read_ncu, read_nsys


from .configuration import CollectionConfig, load_experiment


def prepare(runner, config, output, remote_directory=None):
    runner = Path(runner).resolve()
    if not runner.is_file() or runner.suffix != '.py':
        raise ValueError('Runner must be an existing Python file')
    output = Path(output).resolve()
    remote = remote_directory or '/tmp/kernel-workbench-' + uuid.uuid4().hex[:12]
    remote_path = PurePosixPath(remote)
    if not remote_path.is_absolute() or remote_path == PurePosixPath('/') or '..' in remote_path.parts:
        raise ValueError('Remote directory must be an absolute POSIX working path')
    remote = str(remote_path)
    manifest = runner.with_suffix('.bundle.json')
    bundle = json.loads(manifest.read_text()) if manifest.is_file() else {}
    merged = {**bundle.get('config_defaults', {}), **config}
    for key, value in bundle.get('fixed_config', {}).items():
        if key in merged and merged[key] != value:
            raise ValueError(f'This runner requires {key}={value}')
        merged[key] = value
    config = CollectionConfig(**merged).model_dump()
    # Declarative dependencies keep prepare read-only with respect to runner code:
    # never import a runner just to discover its package requirements.
    dependencies = []
    if manifest.is_file():
        for name in bundle.get('directories', []):
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                raise ValueError('Runner dependency must be a relative directory')
            dependency = (runner.parent / name).resolve()
            if not dependency.is_relative_to(runner.parent) or not dependency.is_dir():
                raise ValueError('Runner dependency must remain inside the runner directory')
            if any(p.is_symlink() for p in dependency.rglob('*')):
                raise ValueError('Runner dependencies must not contain symlinks')
            dependencies.append((dependency, relative.as_posix()))
    model_names = list(bundle.get('model_directories', []))
    if config['model']:
        model_names.extend(['common', config['model']])
    model_root = (ROOT / 'models').resolve()
    for name in dict.fromkeys(model_names):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Model dependency must be a relative directory')
        dependency = (model_root / name).resolve()
        if not dependency.is_relative_to(model_root) or not dependency.is_dir():
            raise ValueError('Model dependency missing or outside models')
        if any(p.is_symlink() for p in dependency.rglob('*')):
            raise ValueError('Model dependencies must not contain symlinks')
        dependencies.append((dependency, 'models/' + relative.as_posix()))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    archive_path = output / 'source.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(runner, arcname='runner.py')
        archive.add(ROOT / 'kernel_workbench/remote_entry.py', arcname='remote_entry.py')
        archive.add(ROOT / 'kernel_workbench/bootstrap.py', arcname='bootstrap.py')
        archive.add(output / 'config.json', arcname='config.json')
        if model_names:
            archive.add(ROOT / 'models/__init__.py', arcname='models/__init__.py')
        for dependency, name in dependencies:
            archive.add(dependency, arcname=name,
                        filter=lambda info: None if '__pycache__' in PurePosixPath(info.name).parts else info)
    script = output / 'collect.py'
    shutil.copyfile(ROOT / 'kernel_workbench/collector.py', script)
    return {'source_archive': str(archive_path), 'collect_script': str(script),
            'remote_directory': remote, 'remote_source': remote + '.tar.gz',
            'remote_artifacts': remote + '/artifacts.tar.gz', 'config': config,
            'collector_args': ['--archive', remote + '.tar.gz', '--directory', remote]}



def prepare_experiment(definition, output, remote_directory=None, overrides=None):
    experiment = load_experiment(definition)
    config = {**experiment.collection_config(), **(overrides or {})}
    if config['model'] != experiment.model or config['input_tokens'] != experiment.input_tokens:
        raise ValueError('An experiment fixes its model and input token count')
    return prepare(ROOT / 'kernel_workbench/model_runner.py', config, output, remote_directory)


def analyze_directory(directory):
    directory = Path(directory)
    meta_path = directory / 'meta.json'
    if not meta_path.exists():
        raise ValueError('No meta.json; inspect the agent-collected runner/profiler logs')
    meta = json.loads(meta_path.read_text())
    status_path = directory / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    meta.setdefault('warnings', []).extend(status.get('warnings', []))
    meta.setdefault('versions', {}).update(status.get('tools', {}))
    kernels, ncu, baseline = [], None, None
    try:
        kernels = read_nsys(directory / 'trace.sqlite')
    except Exception as error:
        meta['warnings'].append('Timeline unavailable: ' + str(error))
    if (directory / 'counters.csv').exists():
        try:
            ncu = read_ncu(directory / 'counters.csv')
        except Exception as error:
            meta['warnings'].append('Counters unavailable: ' + str(error))
    if (directory / 'baseline.json').exists():
        try:
            baseline = json.loads((directory / 'baseline.json').read_text())
        except (ValueError, OSError) as error:
            meta['warnings'].append('Baseline unavailable: ' + str(error))
    meta['timing_source'] = 'nsys' if (directory / 'trace.nsys-rep').exists() else 'runner_single'
    return analyze(meta, kernels, ncu, baseline)


def analyze_artifacts(path):
    path = Path(path)
    if path.is_dir():
        return analyze_directory(path)
    with tempfile.TemporaryDirectory(prefix='kernel-workbench-analysis-') as temp:
        with tarfile.open(path) as archive:
            archive.extractall(temp, filter='data')
        return analyze_directory(temp)
