# Kernel Observatory

로컬 대시보드와 headless 에이전트용 Colab A100 추론 프로파일러입니다. 모델 구조, prefill/decode, 전체 roofline 위치, 커널별 compute/memory 병목과 이론적 하한을 연결합니다.

## 시작

Python 3.12+, `uv`, 인증된 `colab` CLI가 필요합니다. 로컬 GPU는 필요하지 않습니다.

```sh
uv sync --group dev
uv run python -m kernel_dashboard serve
```

브라우저: http://127.0.0.1:8765 · API 문서: http://127.0.0.1:8765/docs

화면 검증용 예제가 기본 제공됩니다. 예제는 실제 GPU 측정값이 아니며 화면에서 명확히 표시합니다. 로컬 전용이며 외부 호스팅은 하지 않습니다.

## Headless 에이전트 워크플로

Codex CLI를 비대화형으로 호출합니다. `--prompt` 인자를 한 번 전달하고, 질문 응답 없이 최종 응답과 JSONL 이벤트를 회수합니다. OpenAI/Codex 인증은 기존 CLI 설정을 사용합니다.

```sh
uv run python -m kernel_dashboard agent --prompt "tiny_decoder를 A100에서 프로파일링하고 병목과 하한 시간을 분석해줘"
uv run python -m kernel_dashboard agent --prompt "등록된 optimized runner의 수치 정확성을 확인하고 느린 커널을 개선한 뒤 전후 결과를 비교해줘" --timeout 3600
```

내부 호출은 `codex -a never exec --json --sandbox workspace-write ... PROMPT`입니다. 로컬 수집 API 접근을 위해 작업 sandbox의 네트워크 접근을 활성화하며 승인·sandbox 우회 옵션은 사용하지 않습니다. 프롬프트는 shell을 거치지 않고 단일 argv로 전달합니다. Codex 기본 모델을 사용하고 `--model`로 명시할 수 있습니다.

서버가 없으면 해당 CLI가 로컬 수집 broker를 자동 기동하고 종료 시 정리합니다. 기존 대시보드 서버가 있으면 재사용합니다. 에이전트는 이 API를 통해 Colab 작업을 시작하고 결과를 읽습니다. 작업 ID·상태를 먼저 출력하고 완료 후 `response`, `run_ids`, 종료 코드가 담긴 JSON을 출력합니다. `.runs/agents/TASK_ID/`에 `prompt.txt`, `response.txt`, `events.jsonl`, `stderr.log`, `task.json`을 보관합니다. 한 번에 Codex 작업 하나만 실행합니다. 웹에서 에이전트 프롬프트 제출·진행 확인·응답 조회·취소도 가능합니다.

아래 `run` 명령은 에이전트가 사용하는 저수준 모델 수집 도구입니다. 대시보드와 무관하게 실행할 수 있습니다.

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
  "warmup": 1, "timeout_sec": 1800,
  "runner_options": {}
}
```

```sh
uv run python -m kernel_dashboard run --runner tiny_decoder --config config.json
```

결과는 `.runs/RUN_ID/result.json`, 원본 보고서는 `artifacts.tar.gz`에 저장합니다. API와 CLI는 같은 저장소와 프로세스 잠금을 사용합니다. `KERNEL_OPT_DATA`로 저장 경로를 변경할 수 있습니다. 커널을 개선한 runner를 다시 실행하고 두 결과 JSON의 **같은 입력 설정·GPU·정밀도·수집 출처**를 비교하세요. 최적화 작업은 `agent --prompt`로 명시적으로 요청하며, 실제 변경과 검증 내용은 Codex의 최종 응답과 이벤트 로그에 남습니다.

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

- `GET/POST /api/agent-tasks`, `GET /api/agent-tasks/{id}`, `GET /api/agent-tasks/{id}/events`, `POST /api/agent-tasks/{id}/cancel`
- `GET /api/runners`, `GET /api/runs`
- `POST /api/runs` — 설정 JSON, 202와 작업 ID 반환
- `GET /api/runs/{id}` — 단계·오류·세션 정리 상태
- `GET /api/runs/{id}/result` — 정규화 분석 결과
- `POST /api/runs/{id}/cancel`
- `GET /api/runs/{id}/artifacts/{result.json|artifacts.tar.gz|commands.log}`

로컬 서버만 사용하세요. 등록한 runner는 신뢰하는 코드여야 합니다. 다른 웹사이트의 cross-origin 변경 요청은 거부합니다. Colab 인증 정보는 기존 CLI가 관리하며 브라우저나 보고서에 복사하지 않습니다.

## 검증

```sh
uv run pytest -q
node --check kernel_dashboard/static/app.js
```

계산, 비동기 NVTX 귀속, CSV 단위, 수집 간 매칭 실패, API 검증, 프로세스 잠금, 취소·세션 정리를 테스트합니다.

실제 통합 검증: Codex `exec`가 프롬프트를 받아 broker로 A100 작업을 요청하고 결과를 읽어 최종 응답을 반환했습니다. A100-SXM4-40GB, BF16, batch 1, 입력 32 / 출력 1에서 커널 43개 중 43개 카운터를 매핑했습니다. 별도의 입력 128 / 출력 16 실행에서 748개 커널의 prefill/decode 귀속을 확인했습니다. 세션 자동 종료와 활성 세션 없음까지 확인했습니다. 데스크톱·390px 모바일 화면에서 브라우저 오류와 가로 넘침이 없음을 확인했습니다.

구간별 roofline 추가 검증: Codex가 입력 128 / 출력 16으로 실행한 A100 수집에서 748/748 커널 카운터를 매핑했고, prefill 43개·decode 705개 커널로 각 구간의 측정점을 계산했습니다. 출력 1토큰 실행의 decode 미실행 안내와 구간 비교/선택을 브라우저에서 확인했으며 회귀 테스트 16개가 통과했습니다.
