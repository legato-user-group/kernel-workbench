"""Render computed evaluation summaries for human review."""
import json


def render_markdown(report):
    contract = report['contract']
    rows = report['rows']
    correctness = report['correctness']
    comparisons = report['validated_decode_comparisons']
    session_status = report['session_status']
    complete = report['complete']
    gate = correctness.get('gate', contract.get('gate', {}))
    lines=[f'# {contract["models"]["candidate"]["repo"]} 평가', '',
           '새 평가의 raw wall/CUDA Event 표본과 teacher-forced HF BF16 비교를 요약한다.', '',
           f"- 완료: {report['complete']}",
           f'- HF reference: {contract.get("reference", "구현 미기록")}.',
           f'- 입력: prose, batch 1; 출력 {contract["new_tokens"]}토큰; warmup {contract["warmup"]}회 + 측정 목표 {contract["repeats"]}회. 아래에 실제 회수 횟수를 표시한다.',
           '- 각 구간은 별도 pass이며 generation 시간은 구간 중앙값의 합이 아니다.', '',
           '| 입력 | 엔진 | Prefill ms | 엔진 TTFT ms | Decode ms/token | Generation ms | 회수 횟수 P/T/D/G |',
           '|---:|---|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['input_tokens']} | {row['engine']} | {row['prefill_ms']:.3f} | "
                     f"{row['engine_ttft_ms']:.3f} | {row['decode_ms_per_token']:.3f} | "
                     f"{row['generation_ms']:.3f} | {'/'.join(str(row['sample_counts'][k]) for k in ('prefill_forward','engine_ttft','decode','generation'))} |")
    paths = ('Plain 1-token graph와 MTP parallel 4-token verification 경로를 각각 검사한다.'
             if contract.get('mtp_enabled', True) else 'MTP와 양자화를 끈 일반 decode 경로를 검사한다.')
    lines += ['', '## 정확성', '',
              f'HF BF16 cached greedy token을 고정한다. 계약상 검사 위치: {contract.get("teacher_forcing_positions", "미기록")}개.',
              paths, '',
              '| 입력 | 경로 | Prefill KL | Decode 평균 KL | Top-1 일치 | Strong mismatch | 통과 |',
              '|---:|---|---:|---:|---:|---:|---|']
    for check in correctness['checks']:
        if check.get('reset_repeat'): continue
        lines.append(f"| {check['input_tokens']} | {check['engine']} | {check['prefill_kl']:.6g} | "
                     f"{check['decode_mean_kl']:.6g} | {check['top1_agreement']:.4f} | "
                     f"{check['strong_mismatches']} | {check['passed']} |")
    lines += ['', f'사전 gate: prefill KL 및 decode 평균 KL 각각 ≤ {gate.get("max_mean_kl", "미기록")}, '
              f'HF margin > {gate.get("margin_threshold", "미기록")}인 argmax 불일치 '
              f'{gate.get("strong_mismatches_allowed", "미기록")}개 이하, finite logits와 올바른 cache 위치.',
              ('MTP 생성 token parity는 `results/mtp_parity_*.json` 및 `recovered_parity.json`에 기록한다.'
               if contract.get('mtp_enabled', True) else 'HF BF16은 정확성 기준이며 성능 비교 대상에 포함하지 않는다.'),
              '검증 실패 또는 MTP parity 실패 행의 시간은 진단값이며 품질 보존 배속으로 해석하지 않는다.', '',
              '## 검증 범위 내 decode 비교', '']
    for comparison in comparisons:
        lines.append(f"- 입력 {comparison['input_tokens']}: {comparison['plain_ms_per_token']:.3f} → "
                     f"{comparison['mtp_ms_per_token']:.3f} ms/token, {comparison['speedup']:.3f}×. "
                     f"양쪽 decode {comparison['samples_per_engine']}회, HF teacher-forcing gate와 plain/MTP 생성 token parity 통과.")
    lines += ['', '## 미완료 항목', '']
    if session_status:
        lines += ['세션 상태: ' + json.dumps(session_status, ensure_ascii=False)]
    if not complete:
        lines += ['미완료 표본과 검증 범위는 위 회수 횟수 및 원본 상태 파일에 기록한다.']
    lines += ['',
              '## 범위', '', *['- '+limit for limit in report['limits']],
              '- raw 표본·step별 오차·출력 hash는 `results/`의 원본 파일에서 확인한다.', '']
    if isinstance(report['profiling'], list):
        for profile in report['profiling']:
            lines.append(f'- [{profile["directory"]}]({profile["status_file"]}): '
                         f'타임라인 완료 {profile["timeline_complete"]}, 별도 카운터 표본 {profile["counter_report_available"]}.')
        lines.append('- 프로파일 시간은 별도 진단값이다. 제한된 카운터 표본을 전체 생성의 FLOPs·DRAM 트래픽으로 환산하지 않는다.')
    else:
        lines.append('- Nsight/커널 프로파일과 FLOPs·DRAM 카운터는 아직 수집하지 않았다.')
    lines.append('')
    return '\n'.join(lines)
