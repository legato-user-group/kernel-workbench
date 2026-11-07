"""Project computed evidence into dashboard reports and optimization attempts."""
import hashlib
import importlib

from .evaluation_source import EvaluationSource, PHASES, PLAIN, MTP, VERIFY
from .evaluation_summary import summarize_source

TIMING_FIELDS = {'total_ms': 'generation', 'prefill': 'prefill_forward', 'decode': 'decode'}


def _run_id(prefix, engine, length):
    return f'{prefix}-{length}-{engine}'


def _hardware(source, environment):
    """GPU limits come only from the agent's record for the GPU that actually ran."""
    hardware = {'name': environment['gpu'], 'memory_gib': environment['memory_gib']}
    spec = source.read('results/hardware.json')
    if spec is None:
        return {**hardware, 'spec_issue': 'results/hardware.json 미수집'}
    if spec.get('gpu') != environment['gpu']:
        return {**hardware, 'spec_issue': f'results/hardware.json의 GPU({spec.get("gpu")})가 실행 GPU와 다름'}
    peaks = spec.get('peaks')
    if (not isinstance(peaks, dict) or not peaks
            or not all(isinstance(v, (int, float)) and v > 0 for v in (spec.get('bandwidth'), *peaks.values()))):
        return {**hardware, 'spec_issue': 'results/hardware.json의 bandwidth·peaks 값이 올바르지 않음'}
    return {**hardware, **{key: spec[key] for key in ('bandwidth', 'peaks', 'source', 'source_url') if key in spec}}


def _cost_model(source, config):
    """Each model owns its cost analysis; without one the dashboard shows no estimates."""
    name = source.read('invocation.json', {}).get('model', '')
    reason = None
    if config['batch'] is None:
        reason = 'batch 미기록: 연산 비용을 계산할 수 없습니다.'
    else:
        try:
            if not name.isidentifier():
                raise ImportError(name)
            return importlib.import_module(f'models.{name}.analysis').supplement(config)
        except (ImportError, AttributeError):
            reason = f'모델 비용 모형 없음: {name or "invocation.json의 model 미기록"}'
    return {'runner_id': source.contract.get('runner_id', config['runner']),
            'model': {'name': config['checkpoint'], 'children': []}, 'estimates': [], 'kernel_plan': [],
            'phase_estimate_reasons': {'prefill': reason, 'decode': reason}}


def _measurement(source, report, row, prefix):
    contract, environment = source.contract, report['environment']
    engine, length = row['engine'], row['input_tokens']
    raw = source.timings[engine, length]
    counts = {key: raw[phase]['sample_count'] for key, phase in TIMING_FIELDS.items()}
    inputs = source.read('results/inputs.json', {})
    config = {'runner': engine, 'dtype': contract.get('candidate_dtype'), 'batch': source.setting('batch'),
              'input_tokens': length, 'output_tokens': contract['new_tokens'],
              'prompt_category': inputs.get('category'), 'chat_template': inputs.get('chat_template'),
              'add_special_tokens': inputs.get('add_special_tokens'), 'stop_on_eos': source.setting('stop_on_eos'),
              'sampling': source.setting('sampling'), 'cache_capacity': contract['cache_capacity'],
              'warmup': contract['warmup'], 'baseline_repeats': contract['repeats'],
              'checkpoint': contract['models']['candidate']['repo'],
              'checkpoint_revision': contract['models']['candidate']['revision']}
    if contract.get('model_config_file'):
        model_config = source.read(contract['model_config_file'])
        config.update(model_config=model_config['text_config'],
                      model_config_sha256=hashlib.sha256((source.directory / contract['model_config_file']).read_bytes()).hexdigest(),
                      model_name=contract['models']['candidate']['repo'], runner_id=contract['runner_id'],
                      quantization=contract['quantization'], mtp_enabled=contract['mtp_enabled'], kernels=contract['kernels'])
    quality = source.check(engine, length)
    warnings = ['커널 프로파일과 FLOPs·DRAM 카운터 미수집.',
                '구간별 별도 pass의 wall 중앙값. 전체 시간은 구간 중앙값의 합이 아님.',
                'KV 및 recurrent state 값 미비교; logits와 cache 위치 검사.']
    if not source.quality_ok(engine, length):
        warnings.append('정확성·출력 안정성 검증 미통과 또는 미완료: 시간은 진단값.')
    if not row['complete']:
        warnings.append(f'Generation {counts["total_ms"]}/{contract["repeats"]}회: 해당 구간 비교 보류.')
    source_name = source.registration.get('source_manifest', 'source_manifest.json')
    result = {
        **_cost_model(source, config), 'schema_version': 1, 'config': config,
        'hardware': _hardware(source, environment),
        'timings': {
            **{key: raw[phase]['wall_ms']['median'] for key, phase in TIMING_FIELDS.items()},
            'engine_ttft': raw['engine_ttft']['wall_ms']['median'],
            'decode_ms_per_token': row['decode_ms_per_token'],
            'source': 'unprofiled' if all(raw[p]['source'] == 'unprofiled' for p in PHASES) else 'profile_collection',
            'statistic': 'median', 'clock': 'wall', 'expected_samples': contract['repeats'],
            'sample_counts': counts, 'sample_count': counts['total_ms'],
            'cuda_event_ms': {key: raw[phase]['cuda_event_ms']['median'] for key, phase in TIMING_FIELDS.items()},
            'separate_passes': True, 'initial_mtp_drafts_in_decode': engine == MTP},
        'kernels': [], 'point': None, 'coverage': {'measured': 0, 'total': 0}, 'warnings': warnings,
        'versions': {'torch': environment['torch'], 'cuda': environment['cuda'], **environment['packages']},
        'evaluation': {'contract': contract, 'teacher_forcing': quality,
                       'mtp_plain_parity': source.parity.get(length) if engine == MTP else None,
                       'complete': row['complete'], 'stable_output_hashes': row['stable_output_hashes'],
                       'session_status': report['session_status'], 'source': source.timing_path(engine, length),
                       'source_manifest': source.read(source_name, {})}}
    summary = (f'{environment["gpu"]} · {config["prompt_category"] or "입력"} {length} → {contract["new_tokens"]}토큰 · {engine}. '
               f'Decode {row["decode_ms_per_token"]:.3f} ms/token. '
               f'Prefill/decode/generation {counts["prefill"]}/{counts["decode"]}/{counts["total_ms"]}회.')
    return {'id': _run_id(prefix, engine, length), 'summary': summary, 'result': result}, source.timing_path(engine, length)


def _correctness(source, length):
    """Publish each recorded check's scalar fields; per-step data stays in the raw file."""
    rows = []
    for engine in source.engines:
        check = source.check(engine, length)
        if check is None:
            continue
        row = {key: value for key, value in check.items()
               if isinstance(value, (int, float, str)) and not key.endswith('_sha256')}
        reset = next((c for c in source.reset_checks if c['engine'] == check['engine']), None)
        if reset:
            row['reset_bitwise_equal'] = reset.get('bitwise_equal_to_first')
        if check['engine'] == VERIFY and length in source.parity:
            row['plain_token_agreement'] = source.parity[length]['token_agreement']
        rows.append(row)
    return {'status': source.correctness_status(length), 'details': '', 'checks': rows,
            'gate': source.correctness.get('gate', source.contract.get('gate', {})),
            'state_comparison': source.contract.get('state_comparison')}


def _description(source, length, conclusion):
    """Agents record what an attempt means; only computed numbers have a default."""
    registration = source.registration
    return {'title': registration.get('title', source.directory.parent.name),
            'hypothesis': registration.get('hypothesis', ''),
            'changes': registration.get('changes', source.contract.get('candidate', '')),
            'conclusion': registration.get('conclusions', {}).get(str(length), conclusion),
            'next_steps': registration.get('next_steps', '')}


def _checkpoint_attempt(source, report, length, prefix, series):
    engine = source.engines[0]
    row = next((r for r in report['rows'] if r['engine'] == engine and r['input_tokens'] == length), None)
    correctness = _correctness(source, length)
    conclusion = (f'Prefill {row["prefill_ms"]:.3f} ms · decode {row["decode_ms_per_token"]:.3f} ms/token.'
                  if row else '시간 미수집.')
    if correctness['status'] != 'passed':
        conclusion += ' 정확성 또는 출력 안정성 검증 미통과·미완료: 진단값.'
    return {'id': f'{prefix}-prose-{length}', 'series': series,
            **_description(source, length, conclusion),
            'code_revision': 'source_manifest.json 참조', 'environment': report['environment']['gpu'],
            'status': 'completed' if row and row['complete'] else 'inconclusive',
            'baseline_run_id': None, 'candidate_run_id': _run_id(prefix, engine, length) if row else None,
            'correctness': correctness}


def _attempt(source, report, length, prefix, series):
    pair = {row['engine']: row for row in report['rows'] if row['input_tokens'] == length}
    plain, mtp = pair.get(PLAIN), pair.get(MTP)
    correctness = _correctness(source, length)
    if not plain or not mtp:
        conclusion = '기준 또는 후보 시간 미수집: 비교 보류.'
    elif correctness['status'] != 'passed':
        conclusion = '정확성 또는 출력 안정성 검증 미통과·미완료: 배속 비교 보류.'
    elif plain['metric_eligible']['decode'] and mtp['metric_eligible']['decode']:
        conclusion = (f'Decode {plain["decode_ms_per_token"]:.3f} → {mtp["decode_ms_per_token"]:.3f} ms/token '
                      f'({plain["decode_ms"]/mtp["decode_ms"]:.3f}×).')
    else:
        conclusion = '독립 측정 횟수 부족: 비교 보류.'
    contract, environment = source.contract, report['environment']
    if mtp and not mtp['complete']:
        conclusion += f' Generation {mtp["sample_counts"]["generation"]}/{contract["repeats"]}회: 전체 시간 비교 보류.'
    return {
        'id': f'{prefix}-prose-{length}', 'series': series, **_description(source, length, conclusion),
        'code_revision': source.registration.get('code_revision', '측정 source_manifest 참조'),
        'environment': f'{environment["gpu"]} · torch {environment["torch"]} · wall median',
        'status': 'completed' if plain and mtp else 'inconclusive',
        'baseline_run_id': _run_id(prefix, PLAIN, length) if plain else None,
        'candidate_run_id': _run_id(prefix, MTP, length) if mtp else None,
        'correctness': correctness}


def build(directory):
    source = EvaluationSource(directory)
    report = summarize_source(source)
    prefix = source.registration.get('id_prefix', f'{source.directory.parent.name}-{source.directory.name}')
    series = source.registration.get('series', source.directory.parent.name)
    # Keep enough room for input length, phase and mode suffixes in API IDs.
    if len(prefix) > 40 or not prefix.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use an ID prefix of at most 40 letters, digits, underscores or hyphens')
    runs = [_measurement(source, report, row, prefix) for row in report['rows'] if row['engine'] in source.engines]
    attempt = _attempt if MTP in source.engines else _checkpoint_attempt
    attempts = [attempt(source, report, length, prefix, series) for length in source.contract['lengths']]
    return runs, attempts
