"""Unit-normalized, provenance-preserving roofline analysis."""
import csv
import io
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

# Dense (not sparse) peaks. Datasheet units are decimal bytes and FLOPs.
A100 = {
    'A100-SXM4-40GB': 1555e9, 'A100-SXM4-80GB': 2039e9,
    'A100-PCIE-40GB': 1555e9, 'A100-PCIE-80GB': 1935e9,
}
PEAKS = {'fp32': 19.5e12, 'fp64': 9.7e12, 'bf16_tensor': 312e12,
         'fp16_tensor': 312e12, 'tf32_tensor': 156e12}


def hardware(name):
    key = name.upper().removeprefix('NVIDIA ').replace(' ', '-')
    if key in ('T4', 'TESLA-T4'):
        return {'name': name, 'bandwidth': 300e9, 'peaks': {'fp32': 8.1e12, 'fp16': 16.2e12, 'fp16_tensor': 65e12},
                'source': 'NVIDIA T4 datasheet; theoretical, not measured ceilings',
                'source_url': 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf',
                'supported': True}
    bw = A100.get(key)
    return {'name': name, 'bandwidth': bw, 'peaks': PEAKS if bw else {},
            'source': 'NVIDIA A100 dense datasheet', 'supported': bool(bw)}


def bounds(flops, traffic, hw):
    if not flops or traffic is None or not hw.get('bandwidth'):
        return {'compute_ms': None, 'memory_ms': None, 'lower_ms': None, 'bound': 'unknown'}
    if any(v is None or v < 0 or k not in hw['peaks'] for k, v in flops.items()) or traffic < 0:
        return {'compute_ms': None, 'memory_ms': None, 'lower_ms': None, 'bound': 'unknown'}
    compute = max(v / hw['peaks'][k] for k, v in flops.items()) * 1000
    memory = traffic / hw['bandwidth'] * 1000
    return {'compute_ms': compute, 'memory_ms': memory, 'lower_ms': max(compute, memory),
            'bound': ('compute' if compute >= memory else 'memory') if max(compute, memory) > 0 else 'unknown'}


def flop_metric_plan(available, gpu):
    """Arithmetic counters verified for Turing; never assume absent metrics are zero."""
    t4 = gpu.upper().removeprefix('NVIDIA ').replace('TESLA ', '') == 'T4'
    if not t4:
        return {'required': [], 'missing': ['Arithmetic metric plan is currently verified on T4 only']}
    names = {line.split()[0] for line in available.splitlines() if line.strip()}
    scalar = [f'smsp__sass_thread_inst_executed_op_{prefix}{op}_pred_on.sum'
              for prefix in ('f', 'h') for op in ('add', 'mul', 'fma')]
    tensor32 = 'sm__ops_path_tensor_src_fp16_dst_fp32.sum'
    if tensor32.rsplit('.', 1)[0] not in names:
        # T4 cannot execute BF16 or TF32; its combined metric is FP16 only.
        tensor32 = 'sm__ops_path_tensor_src_fp16_bf16_tf32_dst_fp32.sum'
    required = scalar + [tensor32, 'sm__ops_path_tensor_src_fp16_dst_fp16.sum']
    return {'required': required, 'missing': [m for m in required if m.rsplit('.', 1)[0] not in names]}


def arithmetic_flops(metrics, required):
    if not required or any(name not in metrics for name in required):
        return None
    values = {}
    for path, prefix in [('fp32', 'f'), ('fp16', 'h')]:
        names = [f'smsp__sass_thread_inst_executed_op_{prefix}{op}_pred_on.sum' for op in ('add', 'mul', 'fma')]
        if not all(n in metrics for n in names):
            return None
        # CUPTI's PerfWorks mapping counts add + mul + 2*fma for both precisions.
        values[path] = metrics[names[0]] + metrics[names[1]] + 2 * metrics[names[2]]
    tensor = [n for n in required if n.startswith('sm__ops_path_tensor_')]
    if not tensor:
        return None
    values['fp16_tensor'] = sum(metrics[n] for n in tensor)
    return {path: value for path, value in values.items() if value > 0} or {'fp32': 0}


def read_nsys(path):
    return list(iter_nsys(path))


def iter_nsys(path):
    """Read launch-side NVTX attribution; never join GPU timestamps to CPU ranges."""
    db = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {'StringIds', 'NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_KERNEL'}
    if not required <= tables:
        db.close()
        raise ValueError('Unsupported Nsight SQLite schema: missing kernel/NVTX/string tables')
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    ranges = [dict(r) for r in db.execute('SELECT * FROM NVTX_EVENTS') if r['end'] and r['end'] > r['start']]
    for r in ranges:
        r['label'] = r.get('text') or strings.get(r.get('textId'), '')
    launches = []
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:
            launches += [dict(r) for r in db.execute(f'SELECT * FROM {table}')]
    apis = defaultdict(list)
    for api in launches:
        apis[api['correlationId']].append(api)
    module_ranges = defaultdict(list)
    for r in ranges:
        if r['label'].startswith('module:'):
            module_ranges[r.get('globalTid')].append(r)
    graph_modules, originals = {}, {}
    if 'CUDA_GRAPH_NODE_EVENTS' in tables:
        for event in db.execute('SELECT * FROM CUDA_GRAPH_NODE_EVENTS ORDER BY start'):
            event = dict(event)
            node = event['graphNodeId']
            if event.get('originalGraphNodeId') is not None:
                originals[node] = event['originalGraphNodeId']
                continue
            containing = [r for r in module_ranges[event.get('globalTid')]
                          if r['start'] <= event['start'] <= r['end']]
            if containing:
                graph_modules[node] = max(containing, key=lambda r: (r['start'], -r['end']))['label'][7:]

    def graph_module(node):
        visited = set()
        while node is not None and node not in visited:
            if node in graph_modules:
                return graph_modules[node]
            visited.add(node)
            node = originals.get(node)
        return None

    attribution = {}
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
        k = dict(row)
        key = k['correlationId'], k.get('globalPid')
        if key not in attribution:
            candidates = apis[k['correlationId']]
            # All nodes of a graph replay share its launch-side annotation.
            candidates = [a for a in candidates if not k.get('globalPid') or
                          a.get('globalTid', 0) >> 24 == k['globalPid'] >> 24]
            matched = []
            for api in candidates:
                matched += [r for r in ranges if r.get('globalTid') == api.get('globalTid')
                            and r['start'] <= api['start'] and r['end'] >= api['end']]
            attribution[key] = [r['label'] for r in sorted(matched, key=lambda r: r['start'])]
        labels = attribution[key]
        if 'inference' not in labels:
            continue
        phase = next((v[6:] for v in labels if v.startswith('phase:')), 'unknown')
        module = next((v[7:] for v in reversed(labels) if v.startswith('module:')), 'unattributed')
        captured_module = graph_module(k.get('graphNodeId'))
        if captured_module:
            module = captured_module
        module_source = 'graph_creation_nvtx' if captured_module else 'launch_nvtx' if module != 'unattributed' else 'unavailable'
        step = next((int(v[5:]) for v in labels if v.startswith('step:')), None)
        yield {'name': strings.get(k.get('demangledName'), strings.get(k.get('shortName'), '?')),
                       'phase': phase, 'module': module, 'step': step,
                       'module_source': module_source,
                       'grid': [k.get('gridX'), k.get('gridY'), k.get('gridZ')],
                       'block': [k.get('blockX'), k.get('blockY'), k.get('blockZ')],
                       'duration_ms': (k['end'] - k['start']) / 1e6,
                       'start_ns': k['start'], 'flops': None, 'bytes': None, 'source': 'unavailable'}
    db.close()


def read_ncu(path):
    text = Path(path).read_text()
    lines = text.splitlines()
    start = next((i for i,line in enumerate(lines) if '"Kernel Name"' in line and '"ID"' in line), None)
    if start is None: raise ValueError('No Nsight Compute CSV header')
    items = {}
    wide_units = {}
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    for r in reader:
        if not r.get('ID','').isdigit():
            if 'Metric Name' not in r:
                wide_units.update({k: v for k, v in r.items() if '__' in k and v})
            continue
        item = items.setdefault(r['ID'], {'id': r['ID'], 'name':r['Kernel Name'],'metrics':{},'units':{},
            'grid':[int(v) for v in re.findall(r'\d+',r.get('Grid Size',''))],
            'block':[int(v) for v in re.findall(r'\d+',r.get('Block Size',''))]})
        pairs = [(r['Metric Name'],r['Metric Value'])] if 'Metric Name' in r else r.items()
        for name,raw in pairs:
            if '__' not in name or raw is None: continue
            try: value=float(raw.replace(',',''))
            except (ValueError,TypeError): continue
            unit=r.get('Metric Unit','') if 'Metric Name' in r else wide_units.get(name, '')
            scale={'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'Kinst':1e3,'Minst':1e6,'Ginst':1e9}.get(unit,1)
            item['metrics'][name]=value*scale
            item['units'][name]=unit
    return list(items.values())


def counters(metrics):
    paths = {}
    for dtype, prefix in [('fp32', 'f'), ('fp64', 'd')]:
        names = [f'smsp__sass_thread_inst_executed_op_{prefix}{op}_pred_on.sum' for op in ('add', 'mul', 'fma')]
        if all(n in metrics for n in names):
            paths[dtype] = metrics[names[0]] + metrics[names[1]] + 2 * metrics[names[2]]
    for dtype in ('bf16', 'fp16', 'tf32'):
        key = f'sm__ops_path_tensor_src_{dtype}_dst_fp32.sum'
        if key in metrics:
            paths[dtype + '_tensor'] = metrics[key]
    rd, wr = metrics.get('dram__bytes_read.sum'), metrics.get('dram__bytes_write.sum')
    complete = all(f'sm__ops_path_tensor_src_{k}_dst_fp32.sum' in metrics for k in ('bf16','fp16','tf32'))
    return (paths if complete else None), None if rd is None or wr is None else rd + wr


def timing_summary(meta, baseline):
    profiled = {'total_ms': meta['total_ms'], **meta['phase_ms'],
                'source': meta.get('timing_source', 'nsys')}
    fallback = {**profiled, 'statistic': 'single', 'sample_count': 1}
    if not baseline:
        return fallback, 'Separate unprofiled baseline unavailable; timings are from the profiling run.'
    try:
        values = [baseline['total_ms'], baseline['phase_ms']['prefill'], baseline['phase_ms']['decode']]
        if baseline.get('source') != 'unprofiled' or baseline['config'] != meta['config'] or baseline['gpu']['name'] != meta['gpu']['name']:
            raise ValueError('Baseline workload/GPU/source does not match profiling run')
        if baseline['sample_count'] != len(baseline['samples']) or baseline['sample_count'] < 1:
            raise ValueError('Baseline sample count is invalid')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or values[0] <= 0:
            raise ValueError('Baseline timings are invalid')
        return {'total_ms': values[0], 'prefill': values[1], 'decode': values[2],
                'source': 'unprofiled', 'statistic': baseline['statistic'],
                'sample_count': baseline['sample_count'], 'samples': baseline['samples'],
                'profiled': profiled}, None
    except (KeyError, TypeError, ValueError) as error:
        return fallback, 'Separate baseline rejected: ' + str(error)


def analyze(meta, kernels, ncu=None, baseline=None):
    timings, timing_warning = timing_summary(meta, baseline)
    hw = hardware(meta['gpu']['name'])
    # Order matching is intentionally all-or-nothing; never silently match repeated names after a gap.
    matched = bool(ncu) and len(ncu) == len(kernels) and all(a['name'] == b['name'] and (not b.get('grid') or a['grid']==b['grid']) and (not b.get('block') or a['block']==b['block']) for a,b in zip(kernels,ncu))
    if matched:
        for k, n in zip(kernels, ncu):
            k['flops'], k['bytes'] = counters(n['metrics'])
            k['source'] = 'measured' if k['flops'] is not None and k['bytes'] is not None else 'partial'
    for i, k in enumerate(kernels):
        k['id'] = i
        k.update(bounds(k['flops'], k['bytes'], hw))
    complete = bool(kernels) and all(k['source'] == 'measured' for k in kernels)
    point = None
    if complete:
        f = sum(sum(k['flops'].values()) for k in kernels)
        d = sum(k['bytes'] for k in kernels)
        point = {'flops': f, 'bytes': d, 'source': 'measured', 'scope': 'all profiled kernels'}
    elif meta.get('estimates') and not meta.get('phase_estimate_reasons'):
        f = sum(sum(o['flops'].values()) for o in meta['estimates'])
        d = sum(o['bytes'] for o in meta['estimates'])
        point = {'flops': f, 'bytes': d, 'source': 'estimated',
                 'scope': meta.get('estimate_scope', 'runner modeled operations; logical tensor traffic, not measured DRAM')}
    if point:
        point['timing_source'] = timings['source']
        point['intensity'] = point['flops'] / point['bytes'] if point['bytes'] else None
        point['tflops'] = point['flops'] / (timings['total_ms'] / 1000) / 1e12 if timings['total_ms'] else None
    estimates = [{**o, **bounds(o['flops'], o['bytes'], hw), 'source': 'estimated'} for o in meta.get('estimates', [])]
    return with_phase_points({'schema_version': 1, 'hardware': hw, 'model': meta['model'], 'config': meta['config'],
            'timings': timings, 'point': point,
            'kernels': kernels, 'estimates': estimates, 'steps': meta.get('steps', []),
            'coverage': {'measured': sum(k['source'] == 'measured' for k in kernels), 'total': len(kernels)},
            'warnings': meta.get('warnings', []) + ([timing_warning] if timing_warning else []) + ([] if matched else ['NCU counters unavailable or cross-run kernel sequence mismatch.']),
            'versions': meta.get('versions', {}),
            **{key: meta[key] for key in ('runner_id', 'kernel_plan', 'estimate_scope', 'analysis_notes', 'phase_estimate_reasons')
               if key in meta}})


def display_analysis(result):
    """Normalize display metadata without changing saved timing/kernel evidence."""
    result = dict(result)
    hw = {**hardware(result['hardware']['name']), **result['hardware']}
    result['hardware'] = hw
    result['estimates'] = [{**e, **bounds(e['flops'], e['bytes'], hw), 'source': 'estimated'}
                           for e in result.get('estimates', [])]
    phases = {e['phase'] for e in result['estimates']}
    if result.get('point') is None and phases == {'prefill', 'decode'}:
        f = sum(sum(e['flops'].values()) for e in result['estimates'])
        traffic = sum(e['bytes'] for e in result['estimates'])
        duration = result['timings']['total_ms']
        if f > 0 and traffic > 0 and duration > 0:
            result['point'] = {'flops': f, 'bytes': traffic, 'source': 'estimated',
                               'scope': result.get('estimate_scope', 'modeled operations; logical traffic'),
                               'timing_source': result['timings'].get('source'),
                               'intensity': f/traffic, 'tflops': f/(duration/1000)/1e12}
    return result


def with_phase_points(result):
    """Add phase-specific points, including for results saved before phase support."""
    points, reasons = {}, {}
    for phase in ('prefill', 'decode'):
        kernels = [k for k in result['kernels'] if k['phase'] == phase]
        estimates = [e for e in result.get('estimates', []) if e['phase'] == phase]
        duration = result['timings'].get(phase)
        point = None
        if kernels and all(k.get('source') == 'measured' and k.get('flops') is not None
                           and k.get('bytes') is not None for k in kernels):
            rows, source, scope = kernels, 'measured', f'{phase} profiled kernels'
        elif estimates:
            rows, source, scope = estimates, 'estimated', f'{phase} modeled operations; logical tensor traffic, not measured DRAM'
            if result.get('estimate_scope'):
                scope += '; ' + result['estimate_scope']
        else:
            rows = []
        if rows and duration is not None and duration > 0:
            flops = sum(sum(row['flops'].values()) for row in rows)
            traffic = sum(row['bytes'] for row in rows)
            if flops > 0 and traffic > 0:
                point = {'flops': flops, 'bytes': traffic, 'source': source, 'scope': scope,
                         'duration_ms': duration, 'timing_source': result['timings'].get('source', 'legacy_profiled'), 'intensity': flops / traffic,
                         'tflops': flops / (duration / 1000) / 1e12}
        points[phase] = point
        if point:
            reasons[phase] = None
        elif not kernels and not estimates:
            reasons[phase] = ('not_executed' if phase == 'decode' and result['config'].get('output_tokens') == 1
                              else 'missing_data')
            if result.get('phase_estimate_reasons', {}).get(phase):
                reasons[phase] = result['phase_estimate_reasons'][phase]
        elif duration is None or duration <= 0:
            reasons[phase] = 'missing_duration'
        else:
            reasons[phase] = 'missing_counters_or_nonpositive_work'
    return {**result, 'phase_points': points, 'phase_point_reasons': reasons}
