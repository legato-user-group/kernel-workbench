"""Project computed evidence into dashboard reports and optimization attempts."""
import hashlib

from models.qwen3_5.analysis import supplement

from .evaluation_source import EvaluationSource, ENGINES, PHASES, PLAIN, MTP
from .evaluation_summary import summarize_source

TIMING_FIELDS = {'total_ms': 'generation', 'prefill': 'prefill_forward', 'decode': 'decode'}


def _run_id(prefix, engine, length):
    return f'{prefix}-{length}-{"mtp3" if engine == MTP else "plain"}'


def _measurement(source, report, row, prefix):
    contract, environment = source.contract, report['environment']
    engine, length = row['engine'], row['input_tokens']
    raw = source.timings[engine, length]
    counts = {key: raw[phase]['sample_count'] for key, phase in TIMING_FIELDS.items()}
    config = {'runner': engine, 'dtype': contract.get('candidate_dtype', 'w8a16-g128'), 'batch': 1, 'input_tokens': length,
              'output_tokens': contract['new_tokens'], 'prompt_category': 'prose',
              'chat_template': False, 'add_special_tokens': False, 'stop_on_eos': False,
              'sampling': 'greedy', 'cache_capacity': contract['cache_capacity'],
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
        **supplement(config), 'schema_version': 1, 'config': config,
        'hardware': {'name': environment['gpu'], 'memory_gib': environment['memory_gib']},
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
    summary = (f'{environment["gpu"]} · prose {length} → {contract["new_tokens"]}토큰 · {engine}. '
               f'Decode {row["decode_ms_per_token"]:.3f} ms/token. '
               f'Prefill/decode/generation {counts["prefill"]}/{counts["decode"]}/{counts["total_ms"]}회.')
    return {'id': _run_id(prefix, engine, length), 'summary': summary, 'result': result}, source.timing_path(engine, length)


def _correctness_details(source, length):
    pair = [source.check(engine, length) for engine in source.engines]
    details = '; '.join(f'{c["engine"]}: prefill KL={c["prefill_kl"]:.6g}, '
                        f'decode mean KL={c["decode_mean_kl"]:.6g}, strong mismatches={c["strong_mismatches"]}'
                        for c in pair if c is not None)
    gate = source.correctness.get('gate', source.contract.get('gate', {}))
    details += (f'. Gate: KL 각각 ≤ {gate.get("max_mean_kl", "미기록")}, '
                f'HF margin > {gate.get("margin_threshold", "미기록")} mismatch {gate.get("strong_mismatches_allowed", 0)}. '
                'Logits와 cache 위치 검사; state 값 미비교. ')
    parity = source.parity.get(length)
    if parity:
        details += f'Plain/MTP 출력 일치 {parity["token_agreement"]*100:.4g}% ({source.contract["new_tokens"]}토큰).'
    elif MTP in source.engines:
        details += 'MTP 생성 token parity 미검증.'
    return details


def _checkpoint_attempt(source, report, length, prefix, series):
    engine = source.engines[0]
    row = next((r for r in report['rows'] if r['engine'] == engine and r['input_tokens'] == length), None)
    status = source.correctness_status(length)
    conclusion = (f'Prefill {row["prefill_ms"]:.3f} ms · decode {row["decode_ms_per_token"]:.3f} ms/token.'
                  if row else '시간 미수집.')
    if status != 'passed':
        conclusion += ' HF BF16 정확성 또는 출력 안정성 검증 미통과·미완료: 진단값.'
    return {'id': f'{prefix}-prose-{length}', 'series': series,
            'title': source.directory.parent.name,
            'hypothesis': '원본 checkpoint의 일반 decode 성능과 HF BF16 대비 정확성을 검증한다.',
            'changes': f'{source.contract["candidate"]}. HF BF16은 정확성 기준으로만 사용한다.',
            'code_revision': 'source_manifest.json 참조', 'environment': report['environment']['gpu'],
            'status': 'completed' if row and row['complete'] else 'inconclusive',
            'baseline_run_id': None, 'candidate_run_id': _run_id(prefix, engine, length) if row else None,
            'correctness': {'status': status, 'details': _correctness_details(source, length)},
            'conclusion': conclusion, 'next_steps': ''}


def _attempt(source, report, length, prefix, series):
    pair = {row['engine']: row for row in report['rows'] if row['input_tokens'] == length}
    plain, mtp = pair.get(PLAIN), pair.get(MTP)
    status = source.correctness_status(length)
    if not plain or not mtp:
        conclusion = '기준 또는 후보 시간 미수집: 비교 보류.'
    elif status != 'passed':
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
        'id': f'{prefix}-prose-{length}', 'series': series, 'title': f'Prose {length} · INT8 plain → MTP k=3',
        'hypothesis': '같은 INT8 checkpoint의 MTP k=3로 출력 품질을 보존하며 decode 시간을 줄인다.',
        'changes': 'INT8 plain CUDA Graph → INT8+MTP k=3 parallel verification CUDA Graph. '
                   f'입력 {length} · 출력 {contract["new_tokens"]} · 반복 {contract["repeats"]}회.',
        'code_revision': source.registration.get('code_revision', '측정 source_manifest 참조'),
        'environment': f'{environment["gpu"]} · torch {environment["torch"]} · wall median',
        'status': 'completed' if plain and mtp else 'inconclusive',
        'baseline_run_id': _run_id(prefix, PLAIN, length) if plain else None,
        'candidate_run_id': _run_id(prefix, MTP, length) if mtp else None,
        'correctness': {'status': status, 'details': _correctness_details(source, length)},
        'conclusion': source.registration.get('conclusions', {}).get(str(length), conclusion),
        'next_steps': source.registration.get('next_steps', '실패·미완료 항목은 별도 실행으로 점검한다.')}


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
