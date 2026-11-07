import csv
import json

import pytest

from kernel_workbench.analysis import arithmetic_flops, bounds, flop_metric_plan, hardware, read_ncu
from kernel_workbench.profile_data import attach_counter_bounds


def t4_plan():
    available = '\n'.join(
        [f'smsp__sass_thread_inst_executed_op_{p}{op}_pred_on Counter' for p in ('f', 'h') for op in ('add', 'mul', 'fma')]
        + ['sm__ops_path_tensor_src_fp16_bf16_tf32_dst_fp32 Counter', 'sm__ops_path_tensor_src_fp16_dst_fp16 Counter'])
    return flop_metric_plan(available, 'Tesla T4')


def test_t4_arithmetic_metrics_require_complete_paths_and_count_fma_twice():
    plan = t4_plan()
    assert plan['missing'] == []
    metrics = {name: 0 for name in plan['required']}
    for prefix in ('f', 'h'):
        for op, count in [('add', 1), ('mul', 2), ('fma', 3)]:
            metrics[f'smsp__sass_thread_inst_executed_op_{prefix}{op}_pred_on.sum'] = count
    metrics['sm__ops_path_tensor_src_fp16_bf16_tf32_dst_fp32.sum'] = 20
    metrics['sm__ops_path_tensor_src_fp16_dst_fp16.sum'] = 10
    assert arithmetic_flops(metrics, plan['required']) == {'fp32': 9, 'fp16': 9, 'fp16_tensor': 30}
    del metrics[plan['required'][0]]
    assert arithmetic_flops(metrics, plan['required']) is None
    assert flop_metric_plan('', 'Tesla T4')['missing']
    assert flop_metric_plan('', 'A100-SXM4-80GB')['missing']
    assert bounds({'fp32': 0}, 0, hardware('Tesla T4'))['bound'] == 'unknown'


def test_wide_counter_units_are_normalized(tmp_path):
    path = tmp_path / 'counters.csv'
    path.write_text('"ID","Kernel Name","dram__bytes_read.sum","dram__bytes_write.sum"\n'
                    '"","","Mbyte","Kbyte"\n"0","copy","1.5","2"\n')
    metrics = read_ncu(path)[0]['metrics']
    assert metrics['dram__bytes_read.sum'] == 1500000
    assert metrics['dram__bytes_write.sum'] == 2000


def test_counter_bounds_are_per_sample_and_never_extrapolated(tmp_path):
    directory = tmp_path / 'profile_bounds'
    directory.mkdir()
    metadata = {'engine': 'fp16_plain_cudagraph', 'input_tokens': 1024, 'correctness': {'passed': True},
                'matches_unprofiled_generation': True, 'counter_phase': 'decode'}
    (directory / 'counters-decode.json').write_text(json.dumps(metadata))
    plan = t4_plan()
    status = directory / 'status.json'
    status.write_text(json.dumps({'flop_metric_plan': plan}))
    metrics = {name: 0 for name in plan['required']}
    metrics.update({'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 8100000000,
                    'dram__bytes_read.sum': 100000000, 'dram__bytes_write.sum': 0})
    with (directory / 'counters-decode.csv').open('w') as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
        writer.writerow(['ID', 'Kernel Name', 'Grid Size', 'Block Size', 'Metric Name', 'Metric Unit', 'Metric Value'])
        for name, value in metrics.items():
            writer.writerow([0, 'gemv', '(1, 1, 1)', '(128, 1, 1)', name, 'byte' if name.startswith('dram') else 'inst', value])
        # A second specimen must not produce independent medians which would
        # break the displayed identity Final=max(Compute, Memory).
        other = {**metrics, 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 405000000,
                 'dram__bytes_read.sum': 900000000}
        for name, value in other.items():
            writer.writerow([1, 'gemv', '(1, 1, 1)', '(128, 1, 1)', name, 'byte' if name.startswith('dram') else 'inst', value])
    row = {'phase': 'decode', 'name': 'gemv', 'grid': [1, 1, 1], 'block': [128, 1, 1],
           'calls': 255000, 'duration_ms': 1000, 'flops': None, 'bytes': None, 'source': 'partial'}
    result = {'config': {'runner': 'fp16_plain_cudagraph', 'input_tokens': 1024},
              'hardware': {'name': 'Tesla T4'}, 'profiling': {}, 'kernels': [row]}
    import gzip
    report = directory / 'counters-decode.ncu-rep.gz'
    report.write_bytes(gzip.compress(b'raw report'))
    artifacts = attach_counter_bounds(tmp_path, result, [status])
    assert artifacts['counters-decode.ncu-rep.gz'] == report
    assert row['compute_ms'] == pytest.approx(2)
    assert row['memory_ms'] == pytest.approx(1 / 3)
    assert row['lower_ms'] == pytest.approx(2)
    assert row['bound'] == 'compute' and row['bounds_reference_step'] == 0
    assert row['flops'] is None and row['bytes'] is None and row['duration_ms'] == 1000
    assert row['counter_samples'][0]['flops'] == {'fp32': 16200000000}
    assert result['profiling']['bounds_groups'] == 1
    assert result['profiling']['bounds_samples'] == 2
    assert row['lower_ms'] == max(row['compute_ms'], row['memory_ms'])
    metadata['counter_phase'] = 'prefill'
    (directory / 'counters-decode.json').write_text(json.dumps(metadata))
    result['kernels'] = [{**row, 'compute_ms': None, 'lower_ms': None}]
    attach_counter_bounds(tmp_path, result, [status])
    assert result['kernels'][0]['lower_ms'] is None
