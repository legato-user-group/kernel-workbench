# 대시보드 실험 자료 템플릿

대시보드는 에이전트가 이 형식으로 수집한 파일만 표시하며, 실험·환경별 값을 코드에 두지 않는다.
다른 모델은 `models/<모델>/`에 평가 도구와 `analysis.py`의 `supplement(config)`를 추가한다.
비용 모형이 없는 모델은 성능·정확성만 표시한다.

## 사용법

저장소 루트에서 실행한다. GPU를 할당하거나 실험을 실행하지 않는다.

```sh
uv run python scripts/new_experiment.py my_experiment --date 2026-11-08
```

생성된 `experiments/my_experiment/config.json`에서 모델·revision·GPU·입출력 길이를 수정한다.
기본 설정은 Qwen3.5-0.8B, T4, FP16, batch 1, 입력 1024·출력 256, MTP-off다.
날짜 폴더는 빈 상태로 생성한다. 같은 실험명에 다른 날짜를 지정하면 기존 설정을 보존한다.
기존 날짜 폴더는 덮어쓰지 않는다.

모델 manifest는 `models.json.template` 형식을 참고한다.
`reference`와 `candidate`의 실제 로컬 checkpoint 경로와 파일별 SHA-256을 기록한다.
설정의 checkpoint/revision과 manifest를 일치시킨 뒤, 준비된 GPU 환경에서 공통 수집기로 실행한다.

```sh
uv run python scripts/evaluate.py \
  --experiment experiments/my_experiment/config.json \
  --models models/checkpoints/evaluation_models.json \
  --output experiments/my_experiment/2026-11-08

# 결과를 experiments/로 가져온 후 로컬에서 확인
uv run python scripts/inspect_results.py experiments
```

평가 도구가 아래 결과 파일을 작성한다. 별도 HTTP 업로드나 DB 등록은 필요 없다.
대시보드는 실험명 아래 날짜별 이력으로 표시한다. 새 결과는 기본 Git 제외 규칙을 적용한다.

## 파일 형식

```text
experiments/<실험명>/
  config.json
  <YYYY-MM-DD>/
    invocation.json             # 실행 명령과 설정 출처
    source_manifest.json        # 실행 소스 해시
    models.json                 # checkpoint 식별과 해시
    results/
      contract.json            # 실행·정확성 검증 조건
      environment.json         # 실제 GPU·라이브러리 버전
      hardware.json            # 실제 GPU의 대역폭·최대 연산 성능과 출처
      model_config.json        # 사용한 checkpoint의 원본 config.json
      inputs.json              # 실제 입력과 tokenizer 조건
      correctness.json         # 정확성 검사와 reset 재현성 검사
      timing_<engine>_<N>.json # 구간별 개별 시간 표본
      exit.json                # 실행 완료 상태
```

이 디렉터리의 `*.json.template`는 필드 구조 참고용이다. `null`과 `<...>`는 미수집 값이며,
결과로 복사하기 전에 실제 값으로 채운다. 모델 config는 예제 숫자를 만들어 쓰지 않고
checkpoint의 원본을 저장한다. 실험별 실행 스크립트를 만들지 않는다.

| 파일 | 작성 규칙 |
|---|---|
| `contract.json` | `lengths`, `new_tokens`, engine, dtype, quantization, MTP, checkpoint revision이 실제 실행과 일치해야 한다. 기본 형식은 원본 checkpoint의 일반 CUDA Graph decode다. |
| `environment.json` | GPU 이름과 표시 가능한 VRAM, PyTorch·CUDA·의존성 버전을 기록한다. |
| `hardware.json` | 실행 후 `environment.json`의 GPU를 기준으로 에이전트가 작성한다. `gpu`는 같은 이름, `bandwidth`는 bytes/s, `peaks`는 연산 경로별 FLOP/s다. 경로 이름은 비용 모형의 `flops` 키와 같아야 한다. 실측이 아니면 `source`에 데이터시트 등 출처를 적는다. 없으면 Roofline과 이론 하한을 표시하지 않는다. |
| `inputs.json` | 원문, 실제 input token IDs와 해시, chat template·특수 토큰 적용 여부를 보존한다. |
| `correctness.json` | engine·입력 길이별 검사와 engine별 reset 검사를 기록한다. 검사를 하지 않았으면 `checks: []`로 두며 통과 값을 만들지 않는다. |
| `timing_<engine>_<N>.json` | `<N>`은 실제 입력 토큰 수다. `prefill_forward`, `engine_ttft`, `decode`, `generation` 각각에 `source`, `sample_count`, 원시 `samples`를 기록한다. |
| `exit.json` | 실제 완료 상태를 기록한다. 미완료 실행에 `success: true`를 쓰지 않는다. |

시간 단위는 **ms**다. `wall_ms`와 `cuda_event_ms`에는 실제 개별 표본을 넣는다.
중앙값은 대시보드가 표본에서 다시 계산한다. 부분 수집은 실제 `sample_count`를 유지한다.
`source: unprofiled`는 프로파일러 없이 측정한 경우에만 사용한다.
Decode와 generation 표본에는 실제 생성 토큰의 `output_ids_sha256`을 기록한다.
출력 256토큰 중 첫 토큰이 prefill에서 나오므로 일반 decode는 255회다.

`engine_ttft`는 첫 토큰 선택까지, `prefill_forward`는 prefill forward와 마지막 위치 logits까지다.
Decode는 사전 prefill 이후의 생성 구간이며, generation은 prefill부터 전체 생성 완료까지다.
각 구간을 별도로 실행하므로 중앙값을 더해 전체 시간을 대체하지 않는다.

입력 유형은 `inputs.json`, batch·sampling·EOS 설정은 `contract.json`에서 읽어 그대로 표시한다.
기록되지 않은 값은 `미기록`으로 표시하며 기본값으로 채우지 않는다.

## 선택 자료

`scripts/profile.py <날짜 폴더>`는 동일 GPU 환경에서 `profile*/`에
`status.json`, `timeline.json`, `trace.sqlite`, 카운터 CSV와 메타데이터를 만든다.
평가만 있어도 현재 대시보드의 실행 조건·정확성·성능·이론 Roofline을 표시할 수 있다.
이론 FLOPs·논리적 bytes는 저장한 모델 구조에서 계산하며 GPU 카운터와 구분한다.

날짜 폴더의 `registration.json`(`registration.json.template` 참고)에 에이전트가 제목·가설·변경 내용·
결과 해석을 기록한다. 없으면 `기록 없음`으로 표시하며, 결과 해석은 측정값 요약으로 대체한다.
이력 ID를 지정할 때는 `id_prefix`를 쓴다. 기본 접두사는 `<실험명>-<날짜>`이며 최대 40자다.
수집 후 `contract.json`에 없던 실행 설정(`batch`, `sampling`, `stop_on_eos`)은 근거를 확인해 여기에 기록할 수 있다.
원시 결과를 재사용할 때는
이전 실험의 `registration.json`, 모델 식별자, 해시, 측정값을 그대로 복사하지 않는다.
