# Qwen/Qwen3.5-0.8B 평가

새 평가의 raw wall/CUDA Event 표본과 teacher-forced HF BF16 비교를 요약한다.

- 완료: True
- HF reference: official HF BF16 cached eager; SDPA; StaticCache; cpu.
- 입력: prose, batch 1; 출력 256토큰; warmup 3회 + 측정 목표 10회. 아래에 실제 회수 횟수를 표시한다.
- 각 구간은 별도 pass이며 generation 시간은 구간 중앙값의 합이 아니다.

| 입력 | 엔진 | Prefill ms | 엔진 TTFT ms | Decode ms/token | Generation ms | 회수 횟수 P/T/D/G |
|---:|---|---:|---:|---:|---:|---|
| 1024 | fp16_plain_cudagraph | 175.248 | 176.118 | 15.054 | 4138.976 | 10/10/10/10 |

## 정확성

HF BF16 cached greedy token을 고정한다. 계약상 검사 위치: 256개.
MTP와 양자화를 끈 일반 decode 경로를 검사한다.

| 입력 | 경로 | Prefill KL | Decode 평균 KL | Top-1 일치 | Strong mismatch | 통과 |
|---:|---|---:|---:|---:|---:|---|
| 1024 | fp16_plain_cudagraph | -3.71755e-08 | 2.59793e-06 | 1.0000 | 0 | True |

사전 gate: prefill KL 및 decode 평균 KL 각각 ≤ 0.005, HF margin > 0.5인 argmax 불일치 0개 이하, finite logits와 올바른 cache 위치.
HF BF16은 정확성 기준이며 성능 비교 대상에 포함하지 않는다.
검증 실패 또는 MTP parity 실패 행의 시간은 진단값이며 품질 보존 배속으로 해석하지 않는다.

## 검증 범위 내 decode 비교


## 미완료 항목

세션 상태: {"observed_at_utc": "2026-11-06T17:49:12.858413+00:00", "state": "evaluation_process_exited"}

## 범위

- not checked; logits and cache position checked
- Nsight/커널 프로파일과 FLOPs·DRAM 카운터는 아직 수집하지 않았다.
- raw 표본·step별 오차·출력 hash는 `results/`의 원본 파일에서 확인한다.
