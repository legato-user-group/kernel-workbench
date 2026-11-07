# Kernel Workbench

LLM 추론 실험의 원시 데이터를 읽어 모델 구조, prefill·decode 성능, 정확성 검증과 roofline을 보여주는 로컬 대시보드입니다. 실험 실행은 코딩 에이전트가 관리하고, 웹앱은 결과 저장과 조회를 담당합니다.

## 시작

대시보드에는 GPU가 필요하지 않습니다. Python 3.12+와 `uv`로 실행합니다.

```sh
uv sync --group dev
uv run python -m kernel_workbench api
```

- 대시보드: http://127.0.0.1:8765
- API 문서: http://127.0.0.1:8765/docs

`experiments/`에 원시 결과가 있으면 자동으로 표시합니다. 별도 업로드나 요약 파일 생성은 필요하지 않습니다. 화면 검증용 합성 예제도 제공하며 실제 측정값과 구분합니다.

## 프로젝트 구조

```text
models/                         모델·커널 구현과 공통 측정 함수
  qwen3_5/                      infer-opt에서 가져온 Qwen3.5 구현과 평가 도구
  checkpoints/                  다운로드한 weight
  cache/                        모델·컴파일 캐시와 기준 logits
experiments/<실험명>/
  config.json                   모델·GPU·입력 길이·측정 조건
  <날짜>/                       실행 manifest와 원시 결과
    results/                    계약·환경·입력·정확성·시간 표본
scripts/                        재사용하는 실행·요약·조회 도구
kernel_workbench/               원시 데이터 해석·API·대시보드
```

실험별 Python 실행 파일을 만들지 않고 공통 구현에 설정을 전달합니다. 입력 길이가 다르면 별도 실험 설정으로 구분합니다. Weight와 캐시는 Git에서 제외합니다. 개발 원칙은 [AGENTS.md](AGENTS.md)에 있습니다.

## 실험 실행

현재 설정은 [Qwen3.5-0.8B · prose 1024토큰](experiments/qwen35_08b_prose_1024/config.json)입니다.

| 조건 | 설정 |
|---|---|
| 체크포인트 | 공식 `Qwen/Qwen3.5-0.8B`, revision 고정 |
| 실행 | T4 · FP16 · MTP 꺼짐 · 양자화 없음 · CUDA Graph decode |
| 작업량 | batch 1 · 입력 1024토큰 · 출력 256토큰 · greedy · EOS 조기 종료 없음 |
| 반복 | 구간별 warmup 3회 + 측정 10회 |
| 정확성 기준 | HF BF16 · CPU · cached eager · teacher forcing |

GPU 환경에는 PyTorch·Triton과 Transformers, safetensors, huggingface_hub, pydantic이 필요합니다. 완료한 T4 실험은 Transformers 5.22.2를 사용했습니다. 대시보드용 의존성과 GPU 실행 환경은 별도로 준비합니다.

이미 모델을 준비한 GPU 환경에서는 [공통 평가 CLI](scripts/evaluate.py)를 실행합니다.

```sh
python scripts/evaluate.py \
  --experiment experiments/qwen35_08b_prose_1024/config.json \
  --models models/checkpoints/evaluation_models.json
```

모델 manifest의 `reference`, `candidate`에는 각각 `repo`, `revision`, 로컬 `path`, 파일별 SHA-256을 담은 `files`를 기록합니다. 원본 체크포인트 실험의 두 역할은 같은 모델 ID·revision을 사용합니다.

기본 결과 경로는 설정 폴더 아래 실행 환경의 현재 날짜입니다. 기존 결과가 있으면 새 `--output` 경로를 지정합니다. 실행 명령과 소스 hash만 확인하려면 별도 임시 경로에 `--prepare-only`를 사용합니다. 현재 원본 체크포인트 평가 경로는 `--resume`을 지원하지 않습니다.

Colab 등 원격 GPU에서는 [공통 GPU 실행 도구](scripts/gpu_evaluate.py)를 재사용합니다. 소스 묶음 경로 `KWB_ARCHIVE`, 새 작업 폴더 `KWB_DIRECTORY`, 묶음 안의 설정 경로 `KWB_EXPERIMENT`를 환경 변수로 전달합니다. 이 도구는 실제 GPU를 확인하고 공개 체크포인트 다운로드, manifest 생성, 평가와 결과 압축을 수행합니다.

에이전트가 GPU 할당, 의존성 설치, 소스 전송, 결과 회수와 세션 종료를 관리합니다. 회수한 데이터를 `experiments/<실험명>/<날짜>/`에 보관하면 대시보드에 반영됩니다.

## 측정과 검증

Prefill, 첫 token 준비, decode, 전체 생성을 각각 별도 pass에서 측정합니다. 모델 로딩·토큰화·cache reset·graph capture는 타이머 밖에 두고, 동기화한 wall time과 CUDA Event 시간을 ms 단위로 저장합니다. 대시보드는 전체 생성, prefill, decode 전체와 decode 토큰당 시간을 표시합니다.

첫 출력 token은 prefill에서 나오므로 출력 256개 중 decode 작업량은 **255개**입니다. `decode ms/token = decode 전체 ms / 255`이며 구간별 중앙값을 더해 전체 생성 시간을 만들지 않습니다.

정확성은 같은 HF 후속 token을 teacher-force하여 prefill 1개와 decode 255개 위치의 전체 vocabulary logits를 비교합니다. 현재 gate는 prefill KL·decode 평균 KL 각각 ≤ 0.005, HF margin > 0.5인 argmax 불일치 0개, finite logits와 올바른 cache 위치입니다. Reset 재현성과 측정 표본의 출력 안정성도 확인합니다. Allclose 오차는 진단값이며 state 값 동등성은 별도 미검증입니다.

HF BF16 CPU 시간은 성능 결과에 포함하지 않습니다. 비교 실험의 배속은 정확성 통과, 같은 작업 조건과 독립 측정 표본이 갖춰진 지표에만 계산합니다. 단일 엔진 실험에는 배속을 표시하지 않습니다.

모델 구조와 커널 구현 경로는 소스·체크포인트 설정에서 읽습니다. Roofline의 모델 연산·논리적 트래픽 추정치와 데이터시트 상한은 실측 커널 시간·DRAM 카운터와 구분합니다.

## 결과 조회

```sh
# 원시 데이터에서 report.json과 report.md를 명시적으로 생성
uv run python scripts/summarize.py experiments/qwen35_08b_prose_1024/2026-11-06

# 대시보드에서 읽을 측정·실험 수와 해석 오류 확인
uv run python scripts/inspect_results.py experiments
```

대시보드는 원시 JSON을 직접 읽으며 조회 중 결과 파일을 쓰지 않습니다. 선택한 runner의 실험에서 가설·결과 해석과 정확성 표를 확인하고 모델 구조, 커널 트리, roofline과 원본 데이터를 함께 볼 수 있습니다.

완료한 [0.8B T4 실험](experiments/qwen35_08b_prose_1024/2026-11-06/report.md)의 10회 wall 중앙값은 prefill **175.248 ms**, decode **15.054 ms/token**, 전체 생성 **4138.976 ms**입니다. HF BF16 정확성 gate와 reset 검증을 통과했으며 256개 위치 top-1 일치율은 100%였습니다. Nsight 커널 프로파일과 FLOPs·DRAM 카운터는 수집하지 않았습니다.

## 프로파일링

프로파일 실행·수집·요약은 `scripts/`의 공통 도구로 관리합니다. 현재는 원본 Qwen3.5 체크포인트의 FP16/BF16/FP32·MTP-off 평가 결과를 지원합니다. 모델 구현은 `models/`에서 재사용하며, 실험별 실행 코드는 만들지 않습니다.

```sh
# 평가를 완료한 동일 GPU 환경에서 실행: 모델 manifest의 checkpoint와 reference 캐시 필요
python scripts/profile.py experiments/qwen35_08b_prose_1024/2026-11-07

# 회수한 SQLite 원시 데이터에서 커널별 시간·호출 횟수 요약
python scripts/summarize_profile.py experiments/qwen35_08b_prose_1024/2026-11-07/profile
```

`profile.py`는 별도 프로세스에서 Nsight Systems 타임라인과 Nsight Compute 진단 표본을 수집합니다. `profile_workload.py`가 저장된 입력·HF reference를 검증하고 모델 로딩, 정확성 검사와 warmup을 수집 구간 밖에서 수행합니다. 타임라인에서는 실제 `nn.Module`과 함수 호출에 NVTX 범위를 넣어 Graph를 다시 capture합니다. 커널은 launch의 NVTX 범위에, Graph replay 노드는 capture 시 생성된 node ID와 clone 관계에 연결합니다. capture 준비 작업은 inference 커널 합계에서 제외합니다. [Nsight Systems의 Graph 추적](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-graph-trace)을 사용하며 커널 이름으로 레이어를 추정하지 않습니다.

Colab의 공통 실행 도구 `scripts/gpu_evaluate.py`에는 `KWB_PROFILE=1`을 전달하면 평가 후 같은 도구를 실행하고 결과를 함께 압축합니다. 기존 검증 자료로 추가 수집할 때는 `KWB_REUSE_EVALUATION=<묶음 내 실행 폴더>`를 지정합니다. 공개 체크포인트를 다시 다운로드해 revision·파일 hash, GPU·PyTorch 버전을 확인하고, 저장된 HF 검증 후보의 전체 logits hash·출력 token과 reset 재현성을 통과해야 수집합니다. 이 경로는 새로운 HF 비교나 독립 지연시간 측정을 수행하지 않습니다.

`--collect timeline` 또는 `--collect counters`로 수집 도구를 선택할 수 있습니다. 재시도 시 `--output <새 폴더>`를 지정해 이전 원시 결과를 보존합니다.

카운터 수집 기본값은 prefill·첫 decode·token 선택/출력 결합을 분리하고 커널 이름·launch 형상별 첫 호출을 수집하는 `--counter-sampling configs`입니다. 기존 첫 32개 launch 진단은 `--counter-sampling prefix`로 선택합니다. 대시보드의 하한은 이 별도 표본에 대한 값이며 전체 호출 수를 곱하지 않습니다. `Compute = max(경로별 FLOPs / 이론 peak)`, `Memory = DRAM bytes / 이론 대역폭`, `최종 = max(Compute, Memory)`입니다. 분류는 roofline 모형에 따른 것이며 실제 stall 판정이 아닙니다. FLOPs는 [NVIDIA의 add/mul/FMA 메트릭 식](https://docs.nvidia.com/cupti/13.2.1/main/main.html#metrics-mapping-table)과 Tensor 연산 카운터를 사용하며 SFU·정수 명령은 포함하지 않습니다.

원시 trace, 도구 버전, 명령·종료 코드, 정확성 검사와 출력 hash는 실행 폴더의 `profile/`에 저장합니다. Nsight Systems는 prefill과 255회 CUDA Graph decode를 수집합니다. Compute는 구간·커널 이름·launch 형상별 별도 표본을 수집하며, 전체 생성의 FLOPs·DRAM 트래픽으로 환산하지 않습니다. 프로파일 시간은 별도 진단값입니다. CUDA Graph 커널에 모듈 annotation이 없으면 `unattributed`로 보존합니다. 카운터 접근 실패는 원본 로그에서 확인합니다.

`experiments/qwen35_08b_prose_1024/2026-11-07/`의 SQLite·CSV·JSON 프로파일과 검증 자료는 Git 관리 예외입니다. `profile_structure/`는 24개 레이어에 귀속된 전체 타임라인, `profile_bounds/`는 195개 카운터 표본이며 이전 수집 시도도 보존합니다. 대시보드는 SQLite·CSV·JSON을 직접 읽습니다. Nsight 원본 보고서(`.nsys-rep`, `.ncu-rep`, 압축본 포함)는 로컬에만 보관하며 Git에서 제외합니다.

## 개발 검사

```sh
.venv/bin/pytest -q
node --check kernel_workbench/static/app.js
```
