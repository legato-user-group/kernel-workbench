# Agent-owned collection and report upload

The coding agent runs the GPU environment through Colab CLI, SSH, a cloud CLI
or another remote command tool, reads its output and raw reports, performs the
analysis, and uploads a finished report. The web app only stores and displays
results. The server does not know the remote transport or hold its credentials.
Starting/stopping the web app never starts/stops GPU work.
No runner registration, remote command execution, job queue or cancellation API
is provided by the web app.

## Start the result store

```sh
uv run python -m kernel_workbench api
```

Default URL: `http://127.0.0.1:8765`. `GET /api/health` reports
`execution: agent_owned`, `mode: result_store`. Request schemas: `/openapi.json`.
**Restart an older server before using these upload examples:** `POST /api/runs`
now accepts a completed report, rather than a request to execute a model.

## Prepare files on the agent's machine

Write a self-contained runner, or use `runners/tiny_decoder.py`. The helper only
creates local files; it does not execute remote commands or import the runner.

```sh
uv run python -m kernel_workbench prepare \
  --runner runners/tiny_decoder.py \
  --output /tmp/tiny-collection \
  --remote-dir /tmp/tiny-measure
```

The output directory must be new. For custom settings pass `--config config.json`:

```json
{"runner":"tiny_decoder","gpu":"A100","dtype":"bf16","batch":1,
 "input_tokens":128,"output_tokens":16,"warmup":1,"baseline_repeats":5,
 "timeout_sec":1800,"runner_options":{}}
```

## Choose a remote transport in the coding agent

The remote directory is an absolute POSIX path, defaulting to a unique path
under `/tmp`. The helper assumes Python 3.12+, CUDA and the runner dependencies
on the GPU host. Prepare Nsight tools in the agent when needed; the script never
installs system packages. It streams execution logs to the agent and also saves
them in the artifacts. Missing Nsight tools produce explicit warnings.
The NVIDIA collector is optional: agents using other collectors can produce
the same normalized report and upload it directly.

### SSH GPU host

```sh
scp /tmp/tiny-collection/source.tar.gz gpu-host:/tmp/tiny-measure.tar.gz
scp /tmp/tiny-collection/collect.py gpu-host:/tmp/collect.py
ssh gpu-host 'timeout 1800 python3 -u /tmp/collect.py'
scp gpu-host:/tmp/tiny-measure/artifacts.tar.gz /tmp/tiny-collection/artifacts.tar.gz
```

Use a task-specific remote path when sharing a host. For containers or cloud
CLIs, transfer the same bundle and execute the same script through that tool.

### Colab example

```sh
colab new -s tiny-a100 --gpu A100
colab upload -s tiny-a100 /tmp/tiny-collection/source.tar.gz /tmp/tiny-measure.tar.gz
colab exec -s tiny-a100 --timeout 1800 -f /tmp/tiny-collection/collect.py
colab download -s tiny-a100 /tmp/tiny-measure/artifacts.tar.gz /tmp/tiny-collection/artifacts.tar.gz
colab stop -s tiny-a100
```

Read command output and exit codes in the agent. Arrange cleanup even on failure:
release sessions created for this work, and respect ownership when reusing an
existing session. There is no server-side cleanup timer. Bound your remote
execution and the agent's tool timeout explicitly. Cancel with the environment's
CLI or process controls, not by asking the dashboard. Retain the source, configuration, raw output and reports.

The supplied remote script collects Nsight Systems/Compute reports, then runs a
plain Python process without profilers or NVTX for separate timing samples.
Baseline defaults to five runner invocations; each currently reinitializes the
runner and excludes its model loading/warmup from reported inference time.
For steady-state experiments follow the project `kernel-benchmark` skill and
reuse a prepared model across samples when adapting the runner.

## Inspect and aggregate results in the agent

```sh
uv run python -m kernel_workbench analyze \
  --artifacts /tmp/tiny-collection/artifacts.tar.gz \
  --output /tmp/tiny-collection/result.json
```

`--artifacts` accepts an extracted directory or a tar archive. All parsing and
archive extraction happen locally in this CLI, never in the web app.
Inspect `status.json`, `nsys.log`, `ncu.log`, `baseline.log`, `meta.json` and
`baseline.json` when collection fails or measurements are unavailable.

Before uploading, check correctness, GPU/workload identity, counter coverage and
`timings.source`. `unprofiled` identifies independent timings; fallback warnings
must be reported honestly. Keep counter-based and analytic estimates distinct.
Check `phase_points.prefill` and `.decode`; one output token has no decode call.
For optimization, compare matching inputs/dtypes/modes and validate numerics.
Write the conclusions in `summary`; the service does not generate them.

## Upload a completed report

Prepare an upload envelope using a JSON library, not shell interpolation:

```sh
python3 - <<'PY'
import json
from pathlib import Path
result = json.loads(Path('/tmp/tiny-collection/result.json').read_text())
upload = {'id': 'tiny-baseline', 'summary': 'Agent analysis and validation findings.', 'result': result}
Path('/tmp/tiny-collection/upload.json').write_text(json.dumps(upload))
PY
curl --fail-with-body --silent --show-error http://127.0.0.1:8765/api/runs \
  -H 'Content-Type: application/json' --data-binary @/tmp/tiny-collection/upload.json
```

The response is **201**, with an ID, a `Location` header and read links.
Omit `id` to generate one, or provide a unique ID (letters/digits, `_`, `-`,
maximum 64 characters). Existing IDs return **409**, never overwrite results.
Malformed reports return **422**. An execution configuration alone is rejected
and never launches a model.

Optional artifacts can be streamed to the returned report ID:

```sh
curl --fail-with-body --silent --show-error -X PUT \
  http://127.0.0.1:8765/api/runs/tiny-baseline/artifacts/artifacts.tar.gz \
  --data-binary @/tmp/tiny-collection/artifacts.tar.gz
```

Supported attachment names: `artifacts.tar.gz`, `commands.log`, `baseline.json`.
The server stores attachments as bytes and never extracts or executes them.
`PUT` replaces an existing attachment; result JSON remains immutable.

Read `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/result`, or
`GET /api/runs/{id}/artifacts/{name}`. Saved reports use `.runs/{id}/`.
The dashboard picks up new uploads on its next refresh and shows `summary`, model
structure, prefill/decode roofline points and kernel analysis. Prior local results
and historical agent logs remain readable.

## Record an optimization attempt

After uploading measurements, record the optimization attempt separately. This
does not execute code or start a session. Use `series` to group related attempts.

```json
{
  "id": "gemv-attempt-01",
  "series": "decoder-gemv",
  "title": "Fuse dequantization and GEMV",
  "hypothesis": "Reduce intermediate memory traffic.",
  "changes": "Load and dequantize weights inside the GEMV kernel.",
  "code_revision": "candidate commit or source hash",
  "environment": "SSH GPU host, A100",
  "status": "completed",
  "baseline_run_id": "baseline-report-id",
  "candidate_run_id": "candidate-report-id",
  "correctness": {
    "status": "passed",
    "details": "Full logits and KV cache checked against the reference; record actual errors and tolerances."
  },
  "conclusion": "Record the measured outcome, including regressions.",
  "next_steps": "Tune split-K under the same workload."
}
```

Save this payload as a JSON file and send it:

```sh
curl --fail-with-body --silent --show-error http://127.0.0.1:8765/api/experiments \
  -H 'Content-Type: application/json' --data-binary @experiment.json
```

`title` and `changes` are required. Omit `id` for a generated ID; `series` defaults
to `default`. A `completed` attempt requires a candidate report. Both report IDs
must exist and contain results, and must differ. A first baseline attempt can
link just its candidate report without a previous baseline.

For compilation/execution failures without a measurement, use `status: failed`
or `inconclusive`, omit candidate/baseline IDs as needed, and record the error and
next step. Correctness defaults to `not_checked`; use `passed` or `failed` with
the actual validation evidence. The service stores the agent's validation claim;
it does not perform numerical checks. Do not claim validation that was not run.

The API returns **201** and a `Location` header. Duplicate IDs return **409**,
missing linked reports **404**, and invalid input **422**. Records are immutable:
make a new attempt rather than rewriting history. Failed checks, compile failures
and slower candidates should remain in the series.

`GET /api/experiments` lists newest first. Filter with `?series=decoder-gemv`;
`GET /api/experiments/{id}` reads one attempt. Results include a server-derived
`comparison` with linked timing/configuration/hardware, reasons for withholding
comparison, `speedup` and `latency_reduction_pct` for total/prefill/decode.
Speedups require completed status, correctness `passed`, matching model/GPU names and
workload configuration, and both timing sources `unprofiled`. Runner names, GPU
selection hints, warmup/repetition counts and timeout are excluded from the
configuration comparison. All other configuration fields must match.
Zero-duration phases have no ratio. `speedup = baseline / candidate`; negative
latency reduction indicates a regression. If a linked result is later removed,
the attempt remains visible and comparison becomes unavailable.

Records live at `.runs/experiments/{id}/experiment.json` (or below
`KERNEL_WORKBENCH_DATA`). The dashboard shows experiment history instead of the former
agent-response/event panel; historical agent files and read APIs remain intact.
