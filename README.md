# Kernel Workbench

여러 GPU 실행 환경에서 코딩 에이전트가 수집한 결과를 저장·시각화하는 HTTP API와 로컬 대시보드입니다. 모델 구조, prefill/decode, 전체 roofline 위치, 커널별 compute/memory 병목과 이론적 하한을 연결합니다.

## 시작

대시보드는 Python 3.12+와 `uv`만 있으면 실행됩니다. GPU와 원격 CLI 인증은 코딩 에이전트의 실행 환경에서 준비합니다. 기본 수집 설정은 A100이며 Colab, SSH GPU 서버, 클라우드·컨테이너 CLI를 사용할 수 있습니다.

```sh
uv sync --group dev
uv run python -m kernel_workbench api
```

브라우저: http://127.0.0.1:8765 · API 문서: http://127.0.0.1:8765/docs

화면 검증용 예제가 기본 제공됩니다. 예제는 실제 GPU 측정값이 아니며 화면에서 명확히 표시합니다. 로컬 전용이며 외부 호스팅은 하지 않습니다.

## 코딩 에이전트 워크플로

**코딩 에이전트 → GPU 환경의 CLI 실행·출력 확인 → 결과 분석·취합 → 대시보드 업로드** 구조입니다. 웹앱은 결과 저장과 조회만 담당합니다. 프로세스 실행, GPU 할당·종료, runner 등록, 작업 시작·취소 기능은 웹앱에 없습니다.

전체 명령과 업로드 예제는 [AGENT_API.md](AGENT_API.md)에 있습니다. `GET /api/guide`로 같은 가이드를 읽을 수 있습니다.

1. 에이전트가 runner와 설정을 작성합니다. `prepare` CLI로 전송 파일을 만들 수 있습니다.
2. 에이전트가 Colab CLI, SSH, 컨테이너·클라우드 CLI 중 환경에 맞는 도구를 직접 호출하고 출력·오류를 확인합니다.
3. `analyze` CLI 또는 에이전트 코드가 로컬에서 원본 보고서를 분석합니다.
4. 에이전트가 분석 요약과 결과 JSON을 `POST /api/runs`에 업로드합니다.
5. 대시보드가 업로드된 결과를 표시합니다.

```sh
uv run python -m kernel_workbench prepare --runner runners/tiny_decoder.py --output /tmp/tiny-collection --remote-dir /tmp/tiny-measure
# 에이전트에서 GPU 환경의 CLI로 실행하고 artifacts.tar.gz를 회수한 다음:
uv run python -m kernel_workbench analyze --artifacts /tmp/tiny-collection/artifacts.tar.gz --output /tmp/tiny-collection/result.json
```

서버는 실행 환경이나 CLI 종류를 알 필요가 없고, 원격 인증도 보유하지 않습니다. 원격 실행 시간 제한, 중단, 세션 정리는 에이전트가 관리합니다. 기존 자동 실행용 `run/register/cancel/recover` 명령과 시작·취소 API는 제거했습니다. 기존 `.runs/` 결과와 과거 에이전트 기록은 보존합니다.

## Runner와 결과

[runners/tiny_decoder.py](runners/tiny_decoder.py)는 self-contained runner 예제입니다. 필요한 GPU 라이브러리와 모델 가중치는 에이전트가 실행 환경에서 준비합니다.

- `describe(config) -> dict`: 모델 이름과 `children` 구조를 반환합니다.
- `run(config, trace) -> dict`: 설정·GPU·전체 및 단계별 시간·버전·비용 추정치를 반환합니다.
- NVTX의 `inference`, `phase:prefill/decode`, `module:...`, `step:...` 구간으로 모델과 커널을 연결합니다.
- 추론 타이머는 모델 로딩·워밍업을 제외하고 GPU 완료를 동기화해야 합니다.

예제는 4개 레이어, hidden 256, head 4, vocab 4096의 synthetic decoder입니다. 출력 16토큰 중 첫 토큰은 prefill에서 생성되어 decode 호출은 15회입니다.

업로드 형식은 `{"id":"고유-ID", "summary":"에이전트의 분석", "result":{...}}`입니다. `id`는 생략할 수 있습니다. `result`는 `analyze`가 생성하는 정규화 JSON이며 모델 구조·설정·하드웨어·시간·커널·비용 추정·수집 출처를 포함합니다. `.runs/<ID>/result.json`과 `job.json`에 원자적으로 저장하며 기존 ID는 덮어쓰지 않습니다. `KERNEL_WORKBENCH_DATA`로 저장 위치를 변경할 수 있습니다. 기존 `KERNEL_OPT_DATA`도 호환 설정으로 읽습니다.

## 최적화 실험 이력

대시보드의 실험 이력에서 시리즈별로 시도를 모아 보고, 변경 내용·가설·코드 버전·실행 환경·정확성 검증·해석·다음 시도를 확인합니다. 기준·후보 결과를 열면 해당 측정의 모델 구조, roofline, 커널과 원본 파일을 볼 수 있습니다.

에이전트는 측정 결과를 먼저 업로드한 뒤 `POST /api/experiments`로 `baseline_run_id`와 `candidate_run_id`를 연결합니다. 실험은 `.runs/experiments/<ID>/experiment.json`에 저장하며 기존 ID를 덮어쓰지 않습니다. 성능이 떨어진 시도도 그대로 기록합니다. 컴파일·실행 실패로 결과가 없다면 `status: failed` 또는 `inconclusive`로 기록할 수 있습니다.

개선율은 완료된 실험에서 에이전트가 정확성 검증 통과를 기록하고, 두 실행의 모델 이름·GPU 이름·작업 설정과 독립 측정 출처가 맞을 때 계산합니다. runner 이름, GPU 선택 힌트, 워밍업·반복 횟수·시간 제한은 작업 설정 비교에서 제외합니다. 그 밖의 설정이 다르거나 프로파일링 중 시간만 있으면 전후 시간을 보존하고 개선율은 보류합니다. 양수는 시간 단축, 음수는 시간 증가이며 `배속 = 기준 시간 / 후보 시간`입니다. 과거 에이전트 응답·이벤트 파일은 보존하며 기존 조회 API로 읽을 수 있습니다.

기록 예제와 필드 설명은 [AGENT_API.md](AGENT_API.md#record-an-optimization-attempt)에 있습니다.
