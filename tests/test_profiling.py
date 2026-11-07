import json
import sqlite3

import pytest

from scripts.summarize_profile import summarize


def test_profile_summary_counts_graph_kernels_without_cpu_ranges(tmp_path):
    db = sqlite3.connect(tmp_path / 'trace.sqlite')
    db.executescript('''
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'gemv'),(2,'reduce');
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT);
        INSERT INTO NVTX_EVENTS VALUES(10,90,1,'inference'),(11,89,1,'phase:decode'),
                                      (12,30,1,'step:0'),(31,88,1,'step:1');
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(20,21,1,7),(40,41,1,9),(91,92,1,8);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,shortName INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,1100,7,1),(1100,3100,9,1),(3100,4100,9,2),
                                                       (5000,15000,8,1);
    ''')
    db.close()
    (tmp_path / 'timeline.json').write_text(json.dumps({'input_tokens': 2, 'output_tokens': 3, 'decode_steps': 2}))
    result = summarize(tmp_path)
    decode = result['phases']['decode']
    assert decode['launches'] == 3 and decode['distinct_kernels'] == 2
    assert decode['kernel_device_ms'] == pytest.approx(.004)
    assert decode['unattributed_device_ms'] == pytest.approx(.004)
    assert decode['launches_per_step'] == 1.5
    assert decode['kernel_device_ms_per_step'] == pytest.approx(.002)
    assert len(result['kernels']) == 2
    assert result['kernels'][0]['launches'] == 2
    assert (tmp_path / 'summary.json').is_file()
    assert len((tmp_path / 'kernels.csv').read_text().splitlines()) == 3
    (tmp_path / 'timeline.json').write_text(json.dumps({'decode_steps': 3}))
    with pytest.raises(ValueError, match='decode steps'):
        summarize(tmp_path)


def test_observed_module_tree_keeps_source_labels_and_adds_real_functions():
    from kernel_workbench.profile_data import merge_module_structure
    model = {'name': 'fixture', 'source': 'config + source', 'children': [
        {'id': 'layer.0', 'label': 'Layer 0 · GatedDeltaNet', 'children': [
            {'id': 'layer.0.linear_attn', 'label': 'GatedDeltaNet'}]}]}
    before = json.dumps(model)
    result = merge_module_structure(model, [
        {'id': 'layer.0.linear_attn', 'label': 'Qwen3_5GatedDeltaNet', 'kind': 'nn.Module'},
        {'id': 'layer.0.linear_attn.norm', 'label': 'RMSNorm gated', 'weight_shape': [128], 'kind': 'nn.Module'},
        {'id': 'execution.sampling', 'label': 'Greedy token 선택', 'kind': 'control'}])
    layer = result['children'][0]
    assert layer['label'] == 'Layer 0 · GatedDeltaNet'
    attention = layer['children'][0]
    assert attention['label'] == 'GatedDeltaNet' and attention['profile_observed']
    assert attention['children'][0]['weight_shape'] == [128]
    assert result['children'][1]['id'] == 'execution'
    assert result['children'][1]['children'][0]['id'] == 'execution.sampling'
    assert json.dumps(model) == before


def test_profile_module_ids_match_dashboard_paths():
    from scripts.profile_workload import module_id
    assert module_id('model.language_model.layers.10.mlp.gate_up_proj') == 'layer.10.mlp.gate_up_proj'
    assert module_id('model.language_model.embed_tokens') == 'embedding'
    assert module_id('model.language_model.norm') == 'final_norm'
    assert module_id('lm_head') == 'lm_head'
