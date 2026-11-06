# Kernel Observatory

코딩 에이전트용 HTTP API와 조회용 로컬 대시보드를 제공하는 Colab A100 추론 프로파일러입니다. 모델 구조, prefill/decode, 전체 roofline 위치, 커널별 compute/memory 병목과 이론적 하한을 연결합니다.

## 시작

Python 3.12+, `uv`, 인증된 `colab` CLI가 필요합니다. 로컬 GPU는 필요하지 않습니다.

```sh
uv sync --group dev
uv run python -m kernel_dashboard api
```

브라우저: http://127.0.0.1:8765 · API 문서: http://127.0.0.1:8765/docs

화면 검증용 예제가 기본 제공됩니다. 예제는 실제 GPU 측정값이 아니며 화면에서 명확히 표시합니다. 로컬 전용이며 외부 호스팅은 하지 않습니다.

## 코딩 에이전트 워크플로

코딩 에이전트가 HTTP API를 직접 호출합니다. 이 서비스는 에이전트나 LLM을 실행하지 않습니다. Codex, Claude Code 등 어떤 코딩 에이전트든 같은 API를 사용할 수 있습니다. 대시보드는 조회 전용입니다.

연동 절차와 curl 예제는 [AGENT_API.md](AGENT_API.md), 머신용 명세는 `/openapi.json`에 있습니다. `GET /api/health`로 서비스와 문서 주소를 확인하고, `GET /api/guide`로 에이전트용 가이드를 읽을 수 있습니다.

```sh
curl --fail-with-body --silent --show-error http://127.0.0.1:8765/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"runner":"tiny_decoder","input_tokens":128,"output_tokens":16,"timeout_sec":7200}'
```

202 응답의 `id`와 `links.status`를 저장하고 상태를 조회한 뒤 `links.result`에서 분석 결과를 받습니다. 커널 개선 코드는 호출한 코딩 에이전트가 작성·검증하고, 같은 조건으로 수집한 전후 결과를 비교합니다. GPU 수집 제한 시간만 적용되며 중첩된 에이전트 프로세스나 에이전트 전체 제한 시간은 없습니다.

기존 `agent --prompt` 명령과 `POST /api/agent-tasks`는 제거했습니다. 과거 `.runs/agents/`의 응답·이벤트 기록은 조회 API와 대시보드에서 계속 확인할 수 있습니다. 새로운 대화 및 응답 기록은 호출한 코딩 에이전트가 관리합니다. `serve` 명령은 `api`의 기존 호환 이름으로 유지됩니다.

HTTP 호출 외에 다음 저수준 CLI도 사용할 수 있습니다.

```sh
uv run python -m kernel_dashboard run --runner tiny_decoder
uv run python -m kernel_dashboard list
uv run python -m kernel_dashboard status RUN_ID
uv run python -m kernel_dashboard result RUN_ID
uv run python -m kernel_dashboard cancel RUN_ID
uv run python -m kernel_dashboard recover
```

`run`은 첫 줄에 작업 ID JSON을 출력하고, 완료까지 기다린 후 마지막 줄에 최종 작업 JSON을 출력합니다. 실패·취소 시 종료 코드는 1입니다. CLI 자체 로그와 프로파일러 출력은 `.runs/RUN_ID/commands.log`에 보관합니다. 취소 요청은 실행 프로세스가 감지하고 원격 세션을 종료합니다. 로컬 프로세스가 강제 종료됐다면 `recover`로 소유 세션을 정리합니다. 새 작업과 서버 시작에서도 잠금이 없는 중단 작업을 정리합니다. 정리 실패 시 `cleanup_error`를 확인하고 `recover`를 다시 실행하세요.

설정 파일은 선택 사항입니다. 예: `config.json`:

```json
{
  "gpu": "A100", "dtype": "bf16", "batch": 1,
  "input_tokens": 128, "output_tokens": 16,
  "warmup": 1, "baseline_repeats": 5, "timeout_sec": 1800,
  "runner_options": {}
}
```

```sh
uv run python -m kernel_dashboard run --runner tiny_decoder --config config.json
```

결과는 `.runs/RUN_ID/result.json`, 원본 보고서는 `artifacts.tar.gz`에 저장합니다. API와 CLI는 같은 저장소와 프로세스 잠금을 사용합니다. `KERNEL_OPT_DATA`로 저장 경로를 변경할 수 있습니다. 커널을 개선한 runner를 다시 실행하고 두 결과 JSON의 **같은 입력 설정·GPU·정밀도·수집 출처**를 비교하세요. 최적화 작업과 검증 응답은 호출한 코딩 에이전트가 관리합니다.

## 사용자 runner

단일 Python 파일로 작성하고 등록합니다. 필요한 라이브러리는 Colab 런타임에 설치되어 있어야 합니다. 예제는 런타임 기본 PyTorch만 사용합니다.

```sh
uv run python -m kernel_dashboard register optimized /absolute/path/runner.py
uv run python -m kernel_dashboard run --runner optimized --config config.json
```

[runners/tiny_decoder.py](runners/tiny_decoder.py)가 전체 실행 예제입니다. 계약:

- `describe(config) -> dict`: `name`, `children`을 반환합니다. 노드는 `id`, `label`, 선택적인 `kind`, `children`을 가집니다.
- `run(config, trace) -> dict`: 동일한 입력과 시드로 반복 가능해야 합니다. 모델 로딩과 워밍업을 수집 범위에서 제외합니다.
- `trace.range('inference')`, `trace.phase('prefill'|'decode')`, `trace.module(node_id)`, `trace.step(index)` 컨텍스트로 구간을 표시합니다. 추론 구간 종료 전 CUDA를 동기화합니다.
- 반환값에는 `config`, `gpu: {name}`, `total_ms`, `phase_ms: {prefill, decode}`, `versions`, `warnings`를 포함합니다.
- 선택적 `estimates`는 `{module, phase, flops: {fp32|bf16_tensor|fp16_tensor|tf32_tensor|fp64: number}, bytes}` 목록입니다. `bytes`는 논리적 트래픽 추정이며 DRAM 카운터와 구분합니다. 원시 커널에 추정치를 분배하지 않습니다.

예제 모델은 4개 레이어, hidden 256, head 4, vocab 4096의 synthetic decoder입니다. KV cache를 사용하며 첫 출력 토큰은 prefill에서 생성하므로 출력 16토큰의 decode 호출은 15회입니다. dropout·샘플링 없이 시드를 고정합니다.

## 분석 정의

- Nsight Systems: CUDA API correlation과 CPU NVTX 포함 관계로 커널의 모델 연산·단계를 매핑합니다. CPU 구간과 GPU 실행 시각을 직접 겹쳐 매핑하지 않습니다.
- 기준 시간: Nsight 수집이 끝난 뒤 프로파일러와 NVTX를 끈 별도 Python 프로세스에서 같은 runner를 기본 5회 실행하고, 동기화된 추론 시간의 중앙값을 사용합니다. 각 표본은 runner를 다시 초기화하므로 로딩·워밍업은 시간이 아닌 전체 작업 소요 시간에만 포함됩니다. 큰 모델은 `baseline_repeats`를 조절할 수 있습니다. 원본 표본은 `baseline.json`, 프로파일링 중 시간은 결과의 `timings.profiled`에 보존합니다. 기준 측정 실패·구성/GPU 불일치 시 수집 중 시간으로 돌아가며 출처와 경고를 표시합니다.
- Nsight Compute: 동일한 입력을 application replay로 별도 실행합니다. 카운터 이름은 설치된 버전에서 조회합니다. cache/clock 제어는 `none`으로 설정합니다.
- 커널 FLOPs는 지원되는 scalar FP32/FP64 instruction(FMA=2) 및 Tensor Core operation counters에서 계산합니다. DRAM bytes는 read + write입니다.
- 하한(ms): `t_compute = 1000 × max_path(F_path / P_path)`, `t_memory = 1000 × bytes / bandwidth`, `t_lower = max(t_compute, t_memory)`. 병목은 큰 쪽으로 분류합니다. 병목 판정은 DRAM roofline 모델 기준이며 런치 지연·캐시·점유율 진단을 대체하지 않습니다.
- 전체 점: `x=ΣFLOPs/Σbytes`, `y=ΣFLOPs/(total_ms/1000)/1e12`. 커널 시간 합계는 스트림 겹침과 무관하게 전체 추론 시간으로 사용하지 않습니다. 전체 비교에서는 전체·prefill·decode 점을 함께 표시합니다. Prefill/Decode를 선택하면 해당 구간의 FLOPs·DRAM bytes와 구간 실행 시간으로 계산한 점을 표시하며, 커널 표도 같은 단계로 필터링합니다. 모델 선택은 roofline 집계 범위를 바꾸지 않습니다. 출력 1토큰처럼 실제 decode가 없으면 점 대신 미실행 사유를 표시합니다. 기존 저장 결과에도 구간별 점을 계산해 제공하며 `point` 필드는 전체 집계점으로 유지합니다.
- 데이터가 부족하면 커널 하한은 N/A, 전체 점은 runner의 모델 연산 추정치로 표시합니다. 추정치가 없으면 전체 점도 표시하지 않습니다. 추정점은 실제 DRAM roofline 측정점과 직접 비교하면 안 됩니다.
- 정밀도별 dense peak를 사용합니다. 혼합 정밀도 집계점은 개별 정밀도 ceiling 대비 효율 수치로 환산하지 않습니다. MIG 및 미등록 GPU의 roofline은 계산하지 않습니다.
- 반복 커널 표의 실측 시간은 누적 시간, 하한은 첫 호출 기준입니다. 행을 펼치면 모든 호출의 FLOPs·DRAM bytes·하한을 확인할 수 있습니다.
- Decode 상세 필터는 Nsight 커널 단계에 적용됩니다. runner의 `host_submit_ms`는 CPU 제출 시간이며 GPU 토큰 지연으로 표시하지 않습니다.

카운터 권한이 없는 Colab 환경에서는 시간·모델 구조와 비용 추정이 유지됩니다. GPU 사양·도구 버전·원본 로그·누락 범위는 진단 패널과 JSON에 기록됩니다. 도구가 없으면 해당 Colab VM의 CUDA 저장소에서 Nsight 패키지 설치를 시도합니다. GPU 할당 실패 시 다른 GPU로 자동 전환하지 않습니다.

## API

- `GET /api/health`, `GET /api/guide`, `GET /openapi.json`
- `GET /api/agent-tasks`, `GET /api/agent-tasks/{id}`, `GET /api/agent-tasks/{id}/events` — 과거 기록 조회 전용
- `GET/POST /api/runners` — 조회 및 서버 파일 경로로 runner 등록
- `GET /api/runs`
- `POST /api/runs` — 설정 JSON, 202와 작업 ID 반환
- `GET /api/runs/{id}` — 단계·오류·세션 정리 상태
- `GET /api/runs/{id}/result` — 정규화 분석 결과
- `POST /api/runs/{id}/cancel` — 202 즉시 반환, 상태 조회로 종료 확인
- `GET /api/runs/{id}/artifacts/{result.json|artifacts.tar.gz|commands.log}`

로컬 서버만 사용하세요. 등록한 runner는 신뢰하는 코드여야 합니다. 다른 웹사이트의 cross-origin 변경 요청은 거부합니다. Colab 인증 정보는 기존 CLI가 관리하며 브라우저나 보고서에 복사하지 않습니다.

## 프로젝트 측정 스킬

[`.agents/skills/infer-opt-benchmark/SKILL.md`](.agents/skills/infer-opt-benchmark/SKILL.md)에 infer-opt 방식의 측정 절차를 정리했습니다. 이 저장소에서 `$infer-opt-benchmark`로 사용할 수 있습니다. 정확성 검증, 모델을 재사용하는 반복 측정, wall/CUDA Event 시간 구분, 별도 프로파일링과 캐시 조건을 다룹니다.

## 검증

```sh
uv run pytest -q
node --check kernel_dashboard/static/app.js
```

계산, 비동기 NVTX 귀속, CSV 단위, 수집 간 매칭 실패, API 검증, 프로세스 잠금, 취소·세션 정리를 테스트합니다.

이전 버전의 실제 통합 검증: Codex `exec`가 프롬프트를 받아 broker로 A100 작업을 요청하고 결과를 읽어 최종 응답을 반환했습니다. A100-SXM4-40GB, BF16, batch 1, 입력 32 / 출력 1에서 커널 43개 중 43개 카운터를 매핑했습니다. 별도의 입력 128 / 출력 16 실행에서 748개 커널의 prefill/decode 귀속을 확인했습니다. 세션 자동 종료와 활성 세션 없음까지 확인했습니다. 데스크톱·390px 모바일 화면에서 브라우저 오류와 가로 넘침이 없음을 확인했습니다.

구간별 roofline 추가 검증: Codex가 입력 128 / 출력 16으로 실행한 A100 수집에서 748/748 커널 카운터를 매핑했고, prefill 43개·decode 705개 커널로 각 구간의 측정점을 계산했습니다. 출력 1토큰 실행의 decode 미실행 안내와 구간 비교/선택을 브라우저에서 확인했으며 회귀 테스트 16개가 통과했습니다.
