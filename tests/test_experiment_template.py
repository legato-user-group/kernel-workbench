import json
from pathlib import Path

import pytest

from kernel_workbench.analysis import display_analysis
from kernel_workbench.evaluation_runner import prepare_evaluation
from kernel_workbench.raw_experiments import RawExperiments
from scripts.new_experiment import TEMPLATE, create_experiment


def test_scaffold_is_invisible_until_collected_and_accepts_common_runner(tmp_path):
    root = tmp_path / 'experiments'
    created = create_experiment('test_run', root, '2026-11-08')
    output = Path(created['output'])
    assert not list(output.iterdir())
    reader = RawExperiments(root)
    assert reader.records() == [] and reader.issues == []
    models = tmp_path / 'models.json'
    models.write_text('{}')
    plan = prepare_evaluation(created['config'], models, output)
    assert plan['command'][2] == 'models.qwen3_5.checkpoint_evaluation'
    assert plan['output'] == created['output']
    assert (output / 'invocation.json').is_file()


def test_scaffold_preserves_settings_and_never_overwrites_a_run(tmp_path):
    created = create_experiment('test_run', tmp_path, '2026-11-08')
    settings = Path(created['config'])
    config = json.loads(settings.read_text())
    config['input_tokens'] = 512
    settings.write_text(json.dumps(config))
    before = settings.read_bytes()
    create_experiment('test_run', tmp_path, '2026-11-09')
    assert settings.read_bytes() == before
    with pytest.raises(FileExistsError):
        create_experiment('test_run', tmp_path, '2026-11-08')
    for name, day in [('../escape', '2026-11-08'), ('x'*30, '2026-11-08'), ('new', 'bad-date')]:
        with pytest.raises(ValueError):
            create_experiment(name, tmp_path, day)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['test_run']


def test_filled_template_is_readable_by_dashboard(tmp_path):
    """Synthetic values only: exercise the raw contract, not GPU performance."""
    root = tmp_path / 'experiments'
    created = create_experiment('test_run', root, '2026-11-08')
    output = Path(created['output'])
    template = TEMPLATE.parent
    for source in (template / 'results').glob('*.json.template'):
        destination = output / 'results' / source.name.removesuffix('.template')
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(source.read_bytes())

    def edit(name, mutate):
        path = output / 'results' / name
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))

    edit('contract.json', lambda c: c.update(repeats=2, runner_id='test_run'))
    edit('environment.json', lambda e: e.update(gpu='Tesla T4', memory_gib=14.56))
    def fill_checks(data):
        check, reset = data['checks']
        check.update(prefill_kl=.001, decode_mean_kl=.001, top1_agreement=1,
                     strong_mismatches=0, passed=True)
        reset.update(passed=True, bitwise_equal_to_first=True)
    edit('correctness.json', fill_checks)
    def fill_timing(data):
        for phase in ('prefill_forward', 'engine_ttft', 'decode', 'generation'):
            stage = data[phase]
            sample = stage['samples'][0]
            sample.update(wall_ms=4, cuda_event_ms=3)
            stage.update(source='unprofiled', sample_count=2,
                         samples=[sample, {**sample, 'wall_ms': 6}])
    edit('timing_fp16_plain_cudagraph_1024.json', fill_timing)
    model = Path('experiments/qwen35_08b_prose_1024/2026-11-07/results/model_config.json')
    (output / 'results/model_config.json').write_bytes(model.read_bytes())
    reader = RawExperiments(root)
    records = reader.records()
    assert reader.issues == [] and len(records) == 1
    assert records[0]['data_directory'] == 'test_run/2026-11-08'
    result = display_analysis(reader.result(records[0]['id']))
    assert result['timings']['prefill'] == 5
    assert result['config']['quantization'] is None and not result['config']['mtp_enabled']
    assert reader.attempts()[0]['correctness']['status'] == 'passed'
    assert result['logical_bounds']['prefill']['bound'] == 'compute'
    assert result['logical_bounds']['decode']['bound'] == 'memory'
