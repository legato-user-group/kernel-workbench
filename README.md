# Kernel Workbench

LLM 추론 실험 결과를 읽어 모델 구조, prefill·decode 성능, 정확성과 커널 분석을 보여주는 로컬 대시보드입니다. 실험은 공통 CLI로 실행하고, 대시보드는 `experiments/`의 원시 데이터를 직접 읽습니다.

## 시작

Python 3.12+와 `uv`가 필요합니다. 대시보드 실행에는 GPU가 필요하지 않습니다.

```sh
uv sync --group dev
uv run python -m kernel_workbench api
```

- 대시보드: http://127.0.0.1:8765
- API 문서: http://127.0.0.1:8765/docs

저장된 실험 결과는 자동으로 표시되며 별도 업로드는 필요하지 않습니다.

## 구조

| 경로 | 역할 |
|---|---|
| `models/` | 모델·커널 구현 |
| `experiments/<실험명>/config.json` | 실험 설정 |
| `experiments/<실험명>/<날짜>/` | 실행 메타데이터·원시 결과·요약 |
| `scripts/` | 공통 실행·프로파일링·조회 도구 |
| `kernel_workbench/` | 데이터 해석·API·대시보드 |

개발 원칙과 Git 관리 범위는 [AGENTS.md](AGENTS.md)를 참고합니다.

## 평가·프로파일링

GPU 환경에는 PyTorch·Triton·Transformers·safetensors·huggingface_hub·pydantic이 필요하며, 프로파일링에는 Nsight Systems와 Nsight Compute도 필요합니다.

실험 설정과 모델 manifest를 준비한 뒤 실행합니다. 아래 설정 경로와 결과 날짜는 사용할 실험에 맞게 바꿉니다.

```sh
CONFIG=experiments/qwen35_08b_prose_1024/config.json
MODELS=models/checkpoints/evaluation_models.json
RUN_DIR=experiments/qwen35_08b_prose_1024/YYYY-MM-DD

# 평가
python scripts/evaluate.py --experiment "$CONFIG" --models "$MODELS" --output "$RUN_DIR"

# 평가를 완료한 동일 GPU 환경에서 프로파일링
python scripts/profile.py "$RUN_DIR"
```

모델 manifest의 `reference`, `candidate`에는 각각 `repo`, `revision`, 로컬 `path`, 파일별 SHA-256을 담은 `files`를 지정합니다. 프로파일링은 현재 원본 Qwen3.5 체크포인트의 FP16/BF16/FP32·MTP-off 평가를 지원합니다.

`profile.py --collect timeline` 또는 `--collect counters`로 수집 대상을 선택합니다. 기존 결과를 보존하려면 `--output <새 폴더>`를 지정합니다. 각 도구의 전체 옵션은 `--help`로 확인합니다.

## 결과 조회·요약

```sh
# 대시보드가 읽는 실험과 데이터 오류 확인
uv run python scripts/inspect_results.py experiments

# 평가 결과 요약
uv run python scripts/summarize.py "$RUN_DIR"

# 커널 프로파일 요약
uv run python scripts/summarize_profile.py "$RUN_DIR/profile"
```

실험별 설정·측정 결과·수집 기록은 해당 `experiments/` 폴더에서 확인합니다.

## 개발 검사

```sh
.venv/bin/pytest -q
node --check kernel_workbench/static/app.js
```
