"""Summarize raw Nsight kernels without treating profiled time as a baseline."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_workbench.analysis import read_nsys


def summarize(directory):
    directory = Path(directory)
    kernels = read_nsys(directory / 'trace.sqlite')
    if not kernels:
        raise ValueError('No inference kernels matched launch-side NVTX ranges')
    metadata = json.loads((directory / 'timeline.json').read_text())
    steps = {k['step'] for k in kernels if k['phase'] == 'decode' and k['step'] is not None}
    if steps != set(range(metadata['decode_steps'])):
        raise ValueError('Timeline decode steps differ from the completed workload')
    groups, phases = defaultdict(list), defaultdict(list)
    for kernel in kernels:
        groups[kernel['phase'], kernel['name']].append(kernel)
        phases[kernel['phase']].append(kernel)
    rows = []
    for (phase, name), items in groups.items():
        total = sum(k['duration_ms'] for k in items)
        rows.append({'phase': phase, 'name': name, 'launches': len(items),
                     'device_ms': total, 'mean_us': total * 1000 / len(items)})
    rows.sort(key=lambda r: (r['phase'], -r['device_ms']))
    summary = {'source': 'nsys_kernel_activity', 'scope': metadata,
               'counter_scope': 'separate bounded sample; never whole-generation FLOPs or DRAM traffic',
               'notes': ['Profiled durations are diagnostic; independent latency samples are in results/.',
                         'Device time sums kernels; span includes gaps; overlapping kernels can make sums exceed spans.',
                         'Module ownership uses launch-side NVTX or graph construction node IDs; unmatched calls stay unattributed.',
                         'Memcpy and memset activities are excluded from these kernel totals.'],
               'phases': {}, 'kernels': rows}
    for phase, items in phases.items():
        start = min(k['start_ns'] for k in items)
        end = max(k['start_ns'] + k['duration_ms'] * 1e6 for k in items)
        total = sum(k['duration_ms'] for k in items)
        steps = metadata['decode_steps'] if phase == 'decode' else 1
        summary['phases'][phase] = {
            'launches': len(items), 'distinct_kernels': len({k['name'] for k in items}),
            'kernel_device_ms': total, 'kernel_span_ms': (end - start) / 1e6,
            'launches_per_step': len(items) / steps,
            'kernel_device_ms_per_step': total / steps,
            'attributed_launches': sum(k['module'] != 'unattributed' for k in items),
            'module_count': len({k['module'] for k in items if k['module'] != 'unattributed'}),
            'unattributed_device_ms': sum(k['duration_ms'] for k in items if k['module'] == 'unattributed')}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (directory / 'kernels.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['phase', 'name', 'launches', 'device_ms', 'mean_us'])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Nsight 커널 프로파일', '',
             f'입력 {metadata["input_tokens"]}토큰 · 출력 {metadata["output_tokens"]}토큰 · decode {metadata["decode_steps"]}회.',
             '시간은 프로파일 진단값이며 독립 지연시간 표본은 `../results/`에 있다.', '',
             '| 구간 | Launch 수 | 커널 이름 수 | 커널 합계 ms | GPU 구간 ms |',
             '|---|---:|---:|---:|---:|']
    for phase, values in summary['phases'].items():
        lines.append(f'| {phase} | {values["launches"]} | {values["distinct_kernels"]} | '
                     f'{values["kernel_device_ms"]:.3f} | {values["kernel_span_ms"]:.3f} |')
    lines += ['', '모듈 귀속 미확인 커널은 unattributed로 보존한다. 아래 이름은 표시용으로 잘랐으며 전체 이름은 `kernels.csv`에 있다.', '',
              '| 구간 | 커널 | Launch 수 | 합계 ms | 구간 커널 시간 비중 |', '|---|---|---:|---:|---:|']
    for phase, values in summary['phases'].items():
        for row in [r for r in rows if r['phase'] == phase][:5]:
            share = row['device_ms'] / values['kernel_device_ms'] * 100
            name = row['name'][:120].replace('|', '\\|')
            lines.append(f'| {phase} | `{name}` | {row["launches"]} | {row["device_ms"]:.3f} | {share:.2f}% |')
    lines += ['', 'Memcpy·memset은 커널 합계에서 제외한다. Nsight Compute 표본은 별도 진단이며 전체 생성의 FLOPs·DRAM 트래픽으로 환산하지 않는다.', '']
    (directory / 'report.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    print(json.dumps(result['phases'], indent=2))
