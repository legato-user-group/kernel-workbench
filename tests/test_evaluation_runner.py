import json
from datetime import date
from pathlib import Path

import pytest

from kernel_workbench.evaluation_runner import prepare_evaluation
from kernel_workbench.evaluation_summary import summarize


def test_evaluation_uses_config_and_rejects_changed_resume_before_writing(tmp_path, experiment_definition):
    models = tmp_path / 'models.json'
    models.write_text('{}')
    definition = experiment_definition()
    plan = prepare_evaluation(definition, models)
    output = definition.parent / date.today().isoformat()
    assert plan['output'] == str(output)
    assert plan['command'][1:3] == ['-m', 'models.qwen3_5.evaluation']
    assert plan['command'][plan['command'].index('--lengths') + 1] == '512'
    assert '--skip-hf-timing' in plan['command']
    assert {p.name for p in output.iterdir()} == {'invocation.json', 'source_manifest.json', 'models.json'}
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert prepare_evaluation(definition, models, resume=True) == plan
    with pytest.raises(FileExistsError):
        prepare_evaluation(definition, models)
    with pytest.raises(ValueError, match='identity differs'):
        prepare_evaluation(definition, models, repeats=2, resume=True)
    models.write_text('{"revision": "changed"}')
    with pytest.raises(ValueError, match='identity differs'):
        prepare_evaluation(definition, models, resume=True)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_preparation_preserves_existing_evidence_without_an_invocation(tmp_path, experiment_definition):
    models = tmp_path / 'models.json'
    models.write_text('{}')
    output = tmp_path / 'previous'
    output.mkdir()
    original = output / 'source_manifest.json'
    original.write_text('{"original": true}')
    with pytest.raises(FileExistsError, match='without a saved invocation'):
        prepare_evaluation(experiment_definition(128), models, output)
    assert original.read_text() == '{"original": true}'
    assert list(output.iterdir()) == [original]


def test_summary_computes_raw_statistics_and_gates_each_metric(evidence):
    report = summarize(evidence)
    assert len(report['rows']) == 7
    assert report['validated_decode_comparisons'][0]['input_tokens'] == 1024
    assert report['validated_decode_comparisons'][0]['speedup'] == 2
    mtp = next(row for row in report['rows'] if row['input_tokens'] == 1024 and 'mtp3' in row['engine'])
    assert mtp['prefill_ms'] == 1  # Stored summary is deliberately stale in the fixture.
    assert mtp['metric_eligible']['decode']
    assert not mtp['metric_eligible']['generation']
    path = evidence / 'results/timing_int8_mtp3_cudagraph_1024.json'
    timing = json.loads(path.read_text())
    timing['decode']['sample_count'] += 1
    path.write_text(json.dumps(timing))
    with pytest.raises(ValueError, match='sample count mismatch'):
        summarize(evidence)


def test_plain_checkpoint_plan_keeps_fp16_t4_and_cpu_bf16_reference(tmp_path):
    models = tmp_path / 'models.json'
    models.write_text('{}')
    plan = prepare_evaluation('experiments/qwen35_08b_prose_1024', models, tmp_path / 'result')
    command = plan['command']
    assert command[2] == 'models.qwen3_5.checkpoint_evaluation'
    for flag, value in [('--dtype', 'fp16'), ('--expected-gpu', 'T4'), ('--reference-device', 'cpu'),
                        ('--lengths', '1024'), ('--new-tokens', '256')]:
        assert command[command.index(flag) + 1] == value
