import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from kernel_workbench.collection import prepare_experiment


def test_imported_model_sources_match_recorded_hashes():
    root = Path('models/qwen3_5')
    manifest = json.loads((root / 'provenance.json').read_text())
    assert manifest['source_revision'] == 'faf3de5cf9f23c2ee2f29ec94ed80721609dc58f'
    for entry in manifest['files']:
        assert hashlib.sha256((root / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    assert not Path('models/_vendor').exists()


@pytest.mark.parametrize('length', [128, 512, 1024, 4096])
def test_model_and_experiment_bundle_are_self_contained(tmp_path, length, experiment_definition):
    definition = experiment_definition(length)
    bundle = prepare_experiment(definition, tmp_path / 'bundle')
    assert bundle['config']['input_tokens'] == length
    assert Path(bundle['collect_script']).read_bytes() == Path('kernel_workbench/collector.py').read_bytes()
    target = tmp_path / 'isolated'
    target.mkdir()
    with tarfile.open(bundle['source_archive']) as archive:
        assert 'models/qwen3_5/model.py' in archive.getnames()
        assert 'models/common/timing.py' in archive.getnames()
        assert not any('_vendor' in n for n in archive.getnames())
        archive.extractall(target, filter='data')
    result = subprocess.run([sys.executable, '-c',
                             'import json,runner; c=json.load(open("config.json")); '
                             'assert runner.describe(c)["layers"] == 64'], cwd=target,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-m', 'models.qwen3_5.evaluation', '--help'], cwd=target,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    wrong = 512 if length == 128 else 128
    with pytest.raises(ValueError, match='fixes its model and input'):
        prepare_experiment(definition, tmp_path / 'wrong', overrides={'input_tokens': wrong})
