# Kernel Observatory

여러 GPU 실행 환경에서 코딩 에이전트가 수집한 결과를 저장·시각화하는 HTTP API와 로컬 대시보드입니다. 모델 구조, prefill/decode, 전체 roofline 위치, 커널별 compute/memory 병목과 이론적 하한을 연결합니다.

## 시작

대시보드는 Python 3.12+와 `uv`만 있으면 실행됩니다. GPU와 원격 CLI 인증은 코딩 에이전트의 실행 환경에서 준비합니다. 기본 수집 설정은 A100이며 Colab, SSH GPU 서버, 클라우드·컨테이너 CLI를 사용할 수 있습니다.

```sh
uv sync --group dev
uv run python -m kernel_dashboard api
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
uv run python -m kernel_dashboard prepare --runner runners/tiny_decoder.py --output /tmp/tiny-collection --remote-dir /tmp/tiny-measure
# 에이전트에서 GPU 환경의 CLI로 실행하고 artifacts.tar.gz를 회수한 다음:
uv run python -m kernel_dashboard analyze --artifacts /tmp/tiny-collection/artifacts.tar.gz --output /tmp/tiny-collection/result.json
```

서버는 실행 환경이나 CLI 종류를 알 필요가 없고, 원격 인증도 보유하지 않습니다. 원격 실행 시간 제한, 중단, 세션 정리는 에이전트가 관리합니다. 기존 자동 실행용 `run/register/cancel/recover` 명령과 시작·취소 API는 제거했습니다. 기존 `.runs/` 결과와 과거 에이전트 기록은 보존합니다.

## Runner와 결과

[runners/tiny_decoder.py](runners/tiny_decoder.py)는 self-contained runner 예제입니다. 필요한 GPU 라이브러리와 모델 가중치는 에이전트가 실행 환경에서 준비합니다.

- `describe(config) -> dict`: 모델 이름과 `children` 구조를 반환합니다.
- `run(config, trace) -> dict`: 설정·GPU·전체 및 단계별 시간·버전·비용 추정치를 반환합니다.
- NVTX의 `inference`, `phase:prefill/decode`, `module:...`, `step:...` 구간으로 모델과 커널을 연결합니다.
- 추론 타이머는 모델 로딩·워밍업을 제외하고 GPU 완료를 동기화해야 합니다.

예제는 4개 레이어, hidden 256, head 4, vocab 4096의 synthetic decoder입니다. 출력 16토큰 중 첫 토큰은 prefill에서 생성되어 decode 호출은 15회입니다.

업로드 형식은 `{"id":"고유-ID", "summary":"에이전트의 분석", "result":{...}}`입니다. `id`는 생략할 수 있습니다. `result`는 `analyze`가 생성하는 정규화 JSON이며 모델 구조·설정·하드웨어·시간·커널·비용 추정·수집 출처를 포함합니다. `.runs/<ID>/result.json`과 `job.json`에 원자적으로 저장하며 기존 ID는 덮어쓰지 않습니다. `KERNEL_OPT_DATA`로 저장 위치를 변경할 수 있습니다.

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

카운터 권한이나 Nsight 도구가 없는 환경에서도 시간·모델 구조와 비용 추정은 유지되며 누락 경고가 표시됩니다. GPU 사양·도구 버전·원본 로그·누락 범위는 진단 패널과 JSON에 기록됩니다. 제공되는 수집 스크립트는 NVIDIA CUDA/Nsight용이며 시스템 패키지를 자동 설치하지 않습니다. 에이전트가 환경에 맞게 도구를 준비합니다. 다른 GPU의 수집기도 동일한 결과 JSON을 작성해 업로드할 수 있습니다. 미등록 GPU는 사양이 없을 때 roofline·하한을 임의 추정하지 않습니다.

## API

- `GET /api/health`, `GET /api/guide`, `GET /openapi.json`
- `POST /api/runs` — 완성된 분석 결과·요약 업로드, 201 반환
- `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/result`
- `PUT /api/runs/{id}/artifacts/{name}` — 원본 아카이브·로그·기준 시간 파일 업로드
- `GET /api/runs/{id}/artifacts/{name}` — 파일 다운로드
- `GET /api/agent-tasks`, `GET /api/agent-tasks/{id}`, `GET /api/agent-tasks/{id}/events` — 과거 기록 조회

서버 시작 시 자동 복구·GPU 세션 종료를 수행하지 않습니다. 원격 실행과 결과 업로드는 별개의 단계이므로 에이전트가 결과를 검토한 뒤 업로드할 수 있습니다. 구버전 서버는 업로드 전에 반드시 재시작하세요. `POST /api/runs`의 의미가 실행 요청에서 결과 업로드로 변경됐습니다.

## 프로젝트 측정 스킬

[`.agents/skills/kernel-benchmark/SKILL.md`](.agents/skills/kernel-benchmark/SKILL.md)에 infer-opt 방식의 측정 절차를 정리했습니다. 이 저장소에서 `$kernel-benchmark`로 사용할 수 있습니다. 정확성 검증, 모델을 재사용하는 반복 측정, wall/CUDA Event 시간 구분, 별도 프로파일링과 캐시 조건을 다룹니다.

## 검증

```sh
uv run pytest -q
node --check kernel_dashboard/static/app.js
```

계산, 비동기 NVTX 귀속, CSV 단위, 수집 간 매칭 실패, API 검증, 에이전트 측 파일 준비·분석, 업로드 검증·중복 방지·첨부 파일 저장과 서버의 프로세스 미실행을 테스트합니다.

이전 버전의 실제 통합 검증: Codex `exec`가 프롬프트를 받아 broker로 A100 작업을 요청하고 결과를 읽어 최종 응답을 반환했습니다. A100-SXM4-40GB, BF16, batch 1, 입력 32 / 출력 1에서 커널 43개 중 43개 카운터를 매핑했습니다. 별도의 입력 128 / 출력 16 실행에서 748개 커널의 prefill/decode 귀속을 확인했습니다. 세션 자동 종료와 활성 세션 없음까지 확인했습니다. 데스크톱·390px 모바일 화면에서 브라우저 오류와 가로 넘침이 없음을 확인했습니다.

구간별 roofline 추가 검증: Codex가 입력 128 / 출력 16으로 실행한 A100 수집에서 748/748 커널 카운터를 매핑했고, prefill 43개·decode 705개 커널로 각 구간의 측정점을 계산했습니다. 출력 1토큰 실행의 decode 미실행 안내와 구간 비교/선택을 브라우저에서 확인했으며 회귀 테스트 16개가 통과했습니다.
