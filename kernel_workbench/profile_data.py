"""Read completed profile evidence and aggregate launches for dashboard display."""
from functools import lru_cache
from copy import deepcopy
import json
from pathlib import Path

from .analysis import arithmetic_flops, bounds, hardware, iter_nsys, read_ncu


def merge_module_structure(model, observed):
    """Keep source labels and add the module/function scopes actually observed."""
    model = deepcopy(model)
    nodes = {}
    def index(children):
        for node in children:
            nodes[node['id']] = node
            index(node.get('children', []))
    index(model['children'])
    def ensure(identifier, label=None):
        if identifier not in nodes:
            parent = identifier.rsplit('.', 1)[0] if '.' in identifier else None
            if identifier.startswith('layer.') and identifier.count('.') == 1:
                parent = None
            children = ensure(parent).setdefault('children', []) if parent else model['children']
            node = {'id': identifier, 'label': label or ('생성 제어' if identifier == 'execution' else identifier.rsplit('.', 1)[-1])}
            nodes[identifier] = node
            children.append(node)
        return nodes[identifier]
    for info in observed:
        node = ensure(info['id'], info['label'])
        node.update(profile_observed=True, **{k: info[k] for k in ('kind', 'torch_path', 'weight_shape') if k in info})
    model['source'] += '; observed nn.Module/function NVTX call scopes'
    return model


@lru_cache(maxsize=4)
def _timeline(path, mtime_ns, size):
    groups, phases = {}, {}
    for kernel in iter_nsys(path):
        key = (kernel['phase'], kernel['module'], kernel['name'],
               tuple(kernel['grid']), tuple(kernel['block']))
        if key not in groups:
            groups[key] = {**kernel, 'step': None, 'calls': 0, 'duration_ms': 0,
                           'steps': {}, 'source': 'partial', 'timing_source': 'nsys',
                           **bounds(None, None, {})}
            groups[key].pop('start_ns')
        row = groups[key]
        row['calls'] += 1
        row['duration_ms'] += kernel['duration_ms']
        if kernel['step'] is not None:
            step = row['steps'].setdefault(kernel['step'], {'step': kernel['step'], 'calls': 0, 'duration_ms': 0})
            step['calls'] += 1
            step['duration_ms'] += kernel['duration_ms']
        phase = phases.setdefault(kernel['phase'], {'launches': 0, 'kernel_device_ms': 0})
        phase['launches'] += 1
        phase['kernel_device_ms'] += kernel['duration_ms']
    for index, row in enumerate(groups.values()):
        row['id'] = index
        row['steps'] = sorted(row['steps'].values(), key=lambda s: s['step'])
    return list(groups.values()), phases


def _matches(metadata, result):
    config = result['config']
    return (metadata.get('engine') == config['runner']
            and metadata.get('input_tokens') == config['input_tokens']
            and metadata.get('correctness', {}).get('passed') is True
            and metadata.get('matches_unprofiled_generation') is True
            and all(metadata.get(key, config.get(key)) == config.get(key)
                    for key in ('checkpoint', 'checkpoint_revision')))


def attach_counter_bounds(root, result, statuses):
    """Match separate specimens by phase/name/launch shape, never multiply them by timeline calls."""
    samples, artifacts = {}, {}
    hw = hardware(result['hardware']['name'])
    for status_path in statuses:
        directory = status_path.parent
        status = json.loads(status_path.read_text())
        plan = status.get('flop_metric_plan', {})
        required = plan.get('required', []) if not plan.get('missing') else []
        for phase in ('prefill', 'decode', 'unknown'):
            metadata, csv = directory / f'counters-{phase}.json', directory / f'counters-{phase}.csv'
            if not metadata.is_file() or not csv.is_file():
                continue
            identity = json.loads(metadata.read_text())
            if not _matches(identity, result) or identity.get('counter_phase', phase) != phase:
                continue
            if any(c['returncode'] for c in status.get('commands', []) if c.get('log') == f'counters-{phase}.log'):
                continue
            for sample in read_ncu(csv):
                key = (phase, sample['name'], tuple(sample['grid']), tuple(sample['block']))
                if key in samples and samples[key][0]['file'] != str(csv.relative_to(root)):
                    continue
                metrics = sample['metrics']
                flops = arithmetic_flops(metrics, required)
                rd, wr = metrics.get('dram__bytes_read.sum'), metrics.get('dram__bytes_write.sum')
                traffic = rd + wr if rd is not None and wr is not None else None
                estimate = bounds(flops, traffic, hw)
                # Partial data still supports one independent lower bound.
                if flops is not None:
                    estimate['compute_ms'] = bounds(flops, 0, hw)['compute_ms']
                if traffic is not None and hw.get('bandwidth'):
                    estimate['memory_ms'] = traffic / hw['bandwidth'] * 1000
                samples.setdefault(key, []).append({**estimate, 'flops': flops, 'dram_bytes': traffic,
                    'file': str(csv.relative_to(root)), 'id': sample['id'], 'reference_step': 0 if phase == 'decode' else None})
            artifacts[f'counters-{phase}.csv'] = csv
            artifacts.setdefault('counter-status.json', status_path)
            for name, filename in [('counter-source-manifest.json', 'source_manifest.json'),
                                   ('counter-source-updates.json', 'source_updates.json')]:
                if (directory / filename).is_file():
                    artifacts.setdefault(name, directory / filename)
            report = directory / f'counters-{phase}.ncu-rep'
            if not report.is_file():
                report = report.with_suffix(report.suffix + '.gz')
            if report.is_file():
                artifacts[report.name] = report
    matched = 0
    for row in result['kernels']:
        key = (row['phase'], row['name'], tuple(row['grid']), tuple(row['block']))
        evidence = samples.get(key, [])
        row['bounds_reason'] = '커널 이름·launch 형상에 대응하는 카운터 표본 미수집'
        if not evidence:
            continue
        row['counter_samples'] = evidence
        row['bounds_source'] = 'measured_sample'
        row['bounds_reference_step'] = 0 if row['phase'] == 'decode' else None
        # Keep all three displayed values from the same measured invocation.
        # Independent medians could violate Final=max(Compute, Memory).
        for field in ('compute_ms', 'memory_ms', 'lower_ms', 'bound'):
            row[field] = evidence[0][field]
        row['bounds_sample_selection'] = 'first matching specimen'
        row['bounds_reason'] = ('FP add/mul/FMA·Tensor 및 DRAM 표본의 이론 하한; SFU·정수 연산·캐시·launch 비용 제외'
                                if row['lower_ms'] is not None else 'FLOPs 메트릭 미수집 또는 불완전: 최종 하한·분류 산출 불가')
        matched += int(row['lower_ms'] is not None)
    if samples:
        result['profiling']['bounds_samples'] = sum(len(v) for v in samples.values())
        result['profiling']['bounds_groups'] = matched
        result['profiling']['bounds_scope'] = 'Separate first-invocation specimens per phase/kernel/launch shape; decode step 0 only. No whole-run FLOPs/DRAM extrapolation.'
        result['profiling']['bounds_formula'] = 'Compute=max(FLOPs[path]/peak[path]); Memory=DRAM bytes/bandwidth; Final=max(Compute,Memory). Datasheet peaks, not measured ceilings.'
    return artifacts


def attach_profile(root, result, timing):
    """Mutate the in-memory view only; independent timing and raw files stay intact."""
    root = Path(root)
    artifacts = {}
    statuses = sorted(root.glob('profile*/status.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in statuses:
        status = json.loads(path.read_text())
        directory = path.parent
        if not status.get('timeline_complete') or not all((directory / n).is_file() for n in ('timeline.json', 'trace.sqlite')):
            continue
        metadata = json.loads((directory / 'timeline.json').read_text())
        hashes = {sample.get('output_ids_sha256') for sample in timing['generation']['samples']}
        if (not _matches(metadata, result) or metadata.get('output_tokens') != result['config']['output_tokens']
                or metadata.get('decode_steps') != result['config']['output_tokens'] - 1
                or metadata.get('output_ids_sha256') not in hashes):
            continue
        trace = directory / 'trace.sqlite'
        stat = trace.stat()
        kernels, phases = _timeline(str(trace.resolve()), stat.st_mtime_ns, stat.st_size)
        steps = {s['step'] for k in kernels if k['phase'] == 'decode' for s in k['steps']}
        if steps != set(range(metadata['decode_steps'])):
            continue
        launches = sum(p['launches'] for p in phases.values())
        result['kernels'] = [dict(k) for k in kernels]
        result['coverage'] = {'measured': 0, 'total': launches, 'timed': launches, 'kernel_groups': len(kernels)}
        result['profiling'] = {'source': 'nsys', 'directory': str(directory.relative_to(root)),
                               'decode_steps': metadata['decode_steps'], 'phases': phases,
                               'module_attribution': metadata.get('module_attribution'),
                               'counter_scope': 'No whole-workload FLOPs or DRAM counters; separate bounded sample only.'}
        if metadata.get('module_structure'):
            result['model'] = merge_module_structure(result['model'], metadata['module_structure'])
            result['profiling']['attributed_launches'] = sum(k['calls'] for k in kernels if k['module'] != 'unattributed')
            result['profiling']['module_count'] = len({k['module'] for k in kernels if k['module'] != 'unattributed'})
            result['profiling']['graph_attributed_launches'] = sum(k['calls'] for k in kernels if k.get('module_source') == 'graph_creation_nvtx')
            result['profiling']['layer_attributed_launches'] = sum(k['calls'] for k in kernels if k['module'].startswith('layer.'))
        result['warnings'] = [w for w in result['warnings'] if '커널 프로파일과 FLOPs·DRAM 카운터 미수집' not in w]
        result['warnings'].append('Nsight 커널 시간·호출 수 수집. FLOPs·DRAM 카운터는 전체 커널에 수집하지 않음. 모듈 귀속 미확인은 unattributed.')
        result['analysis_notes'] = [n for n in result.get('analysis_notes', []) if '커널별 시간·호출·DRAM 카운터는 미수집' not in n]
        result['analysis_notes'].append('커널 시간·호출은 별도 Nsight 프로파일 실측. Roofline 연산량·트래픽은 기존 추정치.')
        result['versions']['nsys'] = (directory / 'nsys-version.txt').read_text().strip() if (directory / 'nsys-version.txt').is_file() else 'unrecorded'
        for name, filename in {'trace.sqlite': 'trace.sqlite', 'trace.nsys-rep': 'trace.nsys-rep',
                               'profile-status.json': 'status.json', 'profile-timeline.json': 'timeline.json',
                               'profile-report.md': 'report.md'}.items():
            if (directory / filename).is_file():
                artifacts[name] = directory / filename
        break
    if 'profiling' in result:
        artifacts.update(attach_counter_bounds(root, result, statuses))
        for path in statuses:
            directory = path.parent
            csv = directory / 'counters-base.csv'
            if not csv.is_file():
                csv = directory / 'counters.csv'
            metadata = directory / 'counters.json'
            if not csv.is_file() or not metadata.is_file() or not _matches(json.loads(metadata.read_text()), result):
                continue
            result['profiling']['counter_sample'] = {'launches': len(read_ncu(csv)),
                                                     'scope': 'separate bounded sample; not whole-generation traffic'}
            artifacts['counters.csv'] = csv
            artifacts['counters.ncu-rep'] = directory / 'counters.ncu-rep'
            break
    return {name: path for name, path in artifacts.items() if path.is_file()}
