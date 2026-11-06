---
name: infer-opt-benchmark
description: "Measure GPU kernels and model inference using the infer-opt correctness-first workflow: warmed independent latency runs, CUDA Event and wall-time separation, controlled kernel microbenchmarks, and separate profiling. Use for inference baselines, kernel tuning, and performance comparisons; not for ordinary model execution without a measurement task."
---

# Infer-opt GPU measurement

Separate three questions: is the implementation correct, how fast does the
unprofiled workload run, and where does a separate profile attribute its work?
A profiling duration is not an independent latency measurement.

## Define and prepare the workload

- Fix the target GPU, model/checkpoint revision, dtype/quantization, seed, batch,
  prompt length, actual decode steps, attention/cache backend and execution mode.
  Compare candidates under the same conditions on the target GPU.
- Define whether the target is kernel time, forward-only latency, or complete
  request/generation latency. State whether logits for all prompt positions or
  only the last position, sampling and token selection are included.
- For steady-state inference, load and prepare the model once, then reuse it
  across samples. Put model loading, transfers, cache allocation, RoPE setup,
  compilation, autotuning and graph capture outside the measured region.
  Reload per sample only when explicitly measuring that condition; do not
  silently turn steady-state measurements into cold-start measurements.
- Complete warmups before measuring. The infer-opt model baseline defaults to
  3 warmups and 10 samples; its shared kernel timer defaults to 5 and 20.
  These are starting points, not mandatory values for every workload. Record
  actual settings. If clocks are still ramping, warm the GPU before comparing.
- Use the project's cache/weight location conventions. In infer-opt specifically,
  downloaded model weights belong under `/tmp/infer-opt-models`, per AGENTS.md.

## Correctness gate

Validate before publishing performance comparisons. Use the established eager/HF
reference and project tolerances. For BF16 infer-opt baselines, the existing gate
uses `atol=rtol=0.02`; do not apply that tolerance indiscriminately to other dtypes.

- Check full logits and each layer's valid K/V, final cache position, and
  preservation of an existing cache prefix when the implementation exposes them.
- Feed identical tokens to reference and candidate for per-step decode checks;
  free-running sequences can diverge and conceal where an error began.
- For quantization, record the reference precision, logit errors, argmax agreement
  and KL checks appropriate to the experiment. Quantized self-consistency is not
  evidence of agreement with an unquantized reference.
- Record failed conditions and diagnostics, but withhold their speedup claims.

## Independent model timing

Run latency samples without Nsight, torch.profiler or other profiling collectors
attached. Moving a timer outside an NVTX range inside `nsys profile` does not
produce an unprofiled baseline. A plain separate process is suitable when the
collection pipeline would otherwise leave a profiler attached.

- Synchronize CUDA before the timed region; use `perf_counter` and synchronize
  completion before stopping the wall timer. This is request/forward elapsed time
  including the host dispatch and synchronization in the stated scope.
- Save CUDA Event elapsed time alongside wall time when supported. Record events
  on the measured stream and synchronize before reading them. For multi-stream
  work, join its dependencies before the end event. Event intervals may include
  host launch gaps; they are not the sum of kernel-active durations.
- Reset/prepare cache state outside each sample. For decode-only latency, finish
  the prerequisite prefill outside the decode timer and start at the same context
  state for each sample. Keep growing-cache and fixed-shape experiments distinct.
- Time the complete sequential decode region, then divide by its actual step
  count for ms/token. State whether the first generated token comes from prefill.
  Do not divide blindly by the requested output length.
- Preserve raw samples. Report median and variability (e.g. standard deviation,
  minimum, or repetition p90). A p90 across request/sequence averages is not an
  individual-token p90. Confirm no compilation/recompilation occurred in samples.

## Kernel implementation and microbenchmarks

For a kernel task, implement candidates at the actual model shapes. Infer-opt
used Triton kernels integrated as `torch.library.triton_op`, including fused
weight dequantization plus GEMV, then tuned tile sizes, warps, stages or split-K.
Use the project's existing implementation boundary rather than forcing Triton.

- Check candidate numerics before comparing speed. Warm compile, autotune,
  allocator and library heuristics outside the capture/timer.
- CUDA Event timing of repeated CUDA Graph replay helps measure kernel throughput
  with reduced Python launch gaps. Report graph replay separately from eager
  request latency; a graph microbenchmark alone does not prove an end-to-end gain.
- Distinguish hot-L2 and HBM-streaming measurements. For the latter, rotate weight
  copies or real modules whose combined working set exceeds the target GPU's L2.
  Infer-opt's synthetic linear benchmark used a configurable 512 MiB minimum
  footprint and divided replay time by the number of calls. Adjust to hardware
  memory capacity; never label repeated cache hits as sustained HBM bandwidth.
- Tune on the intended GPU, dtype, shapes and row counts within the task's compute
  budget. Verify the selected settings again in the full model's ms/token path.
  Keep the BF16/control measurement in the same session where practical.

## Separate profiling and attribution

Use a separate execution with the same workload for torch.profiler or Nsight.
Retain its timing and hardware counters as diagnostic evidence with their own
provenance; do not replace independent latency samples with them. If independent
measurement fails, mark the fallback explicitly instead of calling it baseline.

- Attribute operations through launch correlation and region annotations. Avoid
  adding both CPU annotation rows and duplicate device range rows in profiler
  summaries. Account for nested/overlapping regions when summing durations.
- `kernel-opt`'s `unattributed` module means a measured kernel has no matched model
  operation annotation. Keep its time/counters in totals. A kernel-name-derived
  label such as cat/copy can help, but mark it as inferred; do not invent a module.
- Infer-opt's `StageBreakdown.unattributed_ms` is a different quantity:
  `max(0, wall_ms - attributed_device_ms)`. It diagnoses a residual often involving
  host dispatch or launch gaps, not the time of unassigned kernels and not an
  exact CPU timer. Missing attribution and overlap also affect interpretation.
- Keep counter-based FLOPs/DRAM traffic separate from analytic operation costs.
  Label datasheet peaks versus measured bandwidth/compute ceilings. Do not
  combine different cache/precision/traffic assumptions into one claimed bound.
- Report launch counts, distinct kernel identities and calls per decode step
  separately; a total over many decode steps is not a count of kernel types.

## Artifacts and execution integration

Save configuration, correctness checks, raw wall/Event samples, summary statistics,
GPU/software versions, source commit/hash, compile/setup time and profiling
artifacts. Keep checkpoint identity and measurement scope with each result.

Use the execution resources and compute budget authorized for the task. When
working in kernel-opt, follow its `AGENT_API.md`: the coding agent calls the
profiling API directly, the service owns Colab collection/cleanup, and the
dashboard observes results. Do not add a nested coding-agent process just to
perform measurements. This skill does not require a particular coding agent.

## Existing infer-opt implementations

These are examples to inspect when `/home/jake/infer-opt` is available; the
workflow above is usable without that repository:

- `infer_opt/profiling.py`: `timeit`, CUDA Event median/std/min and hardware probes.
- `infer_opt/baseline.py`: correctness-gated model baseline and wall/Event reports.
- `infer_opt/opprofile.py`: independent timing plus separate region profiling.
- `scripts/bench_int8_linear.py`: graph microbenchmarks with rotated weights.
- `scripts/tune_qwen3_5_group_gemv.py`: shape tuning followed by full-model decode.
