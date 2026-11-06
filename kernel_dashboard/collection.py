"""Agent-side file preparation and analysis, independent of remote transport."""
import json
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Literal
from pydantic import BaseModel, Field
from .paths import ROOT
from .analysis import analyze, read_ncu, read_nsys


class CollectionConfig(BaseModel):
    runner: str = 'tiny_decoder'
    gpu: str = 'A100'
    dtype: Literal['bf16', 'fp16', 'fp32'] = 'bf16'
    batch: int = Field(1, ge=1, le=64)
    input_tokens: int = Field(128, ge=1, le=32768)
    output_tokens: int = Field(16, ge=1, le=2048)
    warmup: int = Field(1, ge=0, le=10)
    baseline_repeats: int = Field(5, ge=1, le=30)
    timeout_sec: int = Field(1800, ge=60, le=7200)
    runner_options: dict = Field(default_factory=dict)


def prepare(runner, config, output, remote_directory=None):
    runner = Path(runner).resolve()
    if not runner.is_file() or runner.suffix != '.py':
        raise ValueError('Runner must be an existing Python file')
    output = Path(output).resolve()
    remote = remote_directory or '/tmp/kernel-opt-' + uuid.uuid4().hex[:12]
    remote_path = PurePosixPath(remote)
    if not remote_path.is_absolute() or remote_path == PurePosixPath('/') or '..' in remote_path.parts:
        raise ValueError('Remote directory must be an absolute POSIX working path')
    remote = str(remote_path)
    config = CollectionConfig(**config).model_dump()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    archive_path = output / 'source.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(runner, arcname='runner.py')
        archive.add(ROOT / 'kernel_dashboard/remote_entry.py', arcname='remote_entry.py')
        archive.add(ROOT / 'kernel_dashboard/bootstrap.py', arcname='bootstrap.py')
        archive.add(output / 'config.json', arcname='config.json')
    script = output / 'collect.py'
    script.write_text('import tarfile, pathlib, subprocess, sys\n'
                      + f'p=pathlib.Path({remote!r}); p.mkdir(parents=True,exist_ok=False)\n'
                      + f'with tarfile.open({(remote + ".tar.gz")!r}) as archive: archive.extractall(p, filter="data")\n'
                      + 'subprocess.run([sys.executable,str(p/"bootstrap.py"),str(p)],check=True)\n')
    return {'source_archive': str(archive_path), 'collect_script': str(script),
            'remote_directory': remote, 'remote_source': remote + '.tar.gz',
            'remote_artifacts': remote + '/artifacts.tar.gz', 'config': config}


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
    with tempfile.TemporaryDirectory(prefix='kernel-opt-analysis-') as temp:
        with tarfile.open(path) as archive:
            archive.extractall(temp, filter='data')
        return analyze_directory(temp)
