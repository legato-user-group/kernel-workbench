"""Run a bundled experiment in an existing GPU runtime and archive its evidence.

KWB_ARCHIVE, KWB_DIRECTORY and KWB_EXPERIMENT select input files; no credentials
are copied. Runtime creation and package installation belong to the caller.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


def main():
    archive = Path(os.environ['KWB_ARCHIVE'])
    root = Path(os.environ['KWB_DIRECTORY'])
    definition = Path(os.environ['KWB_EXPERIMENT'])
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(root, filter='data')
    sys.path.insert(0, str(root))
    os.environ.setdefault('HF_HOME', str(root / 'models/cache/huggingface'))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(root / 'models/cache/inductor'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from huggingface_hub import snapshot_download
    import torch
    from kernel_workbench.configuration import load_experiment
    from kernel_workbench.evaluation_runner import prepare_evaluation, execute_evaluation

    config = load_experiment(root / definition)
    gpu = torch.cuda.get_device_properties(0)
    print(json.dumps({'gpu': gpu.name, 'memory_gib': gpu.total_memory / 2**30,
                      'compute_capability': [gpu.major, gpu.minor], 'torch': torch.__version__}), flush=True)
    if config.gpu not in gpu.name:
        raise ValueError(f'Expected {config.gpu}; got {gpu.name}')
    options = config.runner_options
    repo, revision = options['checkpoint'], options['revision']
    checkpoint = root / 'models/checkpoints' / repo.split('/')[-1] / revision
    print(f'Downloading pinned checkpoint: {repo}@{revision}', flush=True)
    snapshot_download(repo, revision=revision, local_dir=checkpoint,
                      token=False,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*'])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(checkpoint.iterdir()) if p.is_file()}
    manifest = root / 'models/checkpoints/evaluation_models.json'
    record = {'repo': repo, 'revision': revision, 'path': str(checkpoint), 'files': hashes}
    manifest.write_text(json.dumps({'reference': record, 'candidate': record}, indent=2))
    plan = prepare_evaluation(root / definition, manifest)
    output = Path(plan['output'])
    try:
        execute_evaluation(plan, timeout=config.timeout_sec)
    finally:
        result = root.parent / (root.name + '-results.tar.gz')
        with tarfile.open(result, 'w:gz') as target:
            target.add(output, arcname=output.relative_to(root))
        print(json.dumps({'result_archive': str(result), 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
