# Evolution Pipeline Report

Generated: 2026-06-04

## Executive Summary

The current evolution pipeline is mechanically healthy: the local unit suite passes, and the implementation includes planning, mutation, review, compile repair, staged device evaluation, archiving, and hypothesis memory. The reason it is unlikely to produce consistently successful evolved candidates is not one obvious broken function. The larger issue is that the feedback loop is too weak and too permissive: it treats a single scalar benchmark result as truth, accepts loosely parsed scores, has no statistical guard against noisy device results, and does not feed real profiler/hotspot evidence into planning.

In short: the pipeline can create, repair, and rank candidates, but it does not yet prove that a candidate is a real performance win, semantically safe, and worth building on.

## What Was Checked

- Repository contents: `agentic_evolver.py`, `agentic_evolver_README.md`, `agentic_evolver.example.json`, `test_agentic_evolver.py`.
- No concrete run artifacts were present in this workspace, such as `run.log`, `history.jsonl`, `strategy_memory.json`, or a real `agentic_evolver.json`; findings are based on the pipeline implementation and example configuration.
- Local tests: `C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest -v test_agentic_evolver.py`.
- Result: 35 tests passed.
- Git could not be inspected because `git` is not available on PATH in this shell.

## Main Reasons The Pipeline Is Not Producing Reliable Success

### 1. Success is ranked from one scalar score, usually from one run

The archive selects elites and global best using only `candidate.score` (`agentic_evolver.py:1488-1503`). Candidate evaluation is performed once per candidate (`agentic_evolver.py:3608-3617`), and the baseline is also evaluated once unless loaded from history (`agentic_evolver.py:3199-3208`). That is fragile for FPS, power, thermals, and device benchmarks because run-to-run noise can easily look like evolution progress.

Impact:

- Random thermal/device variance can become the new best.
- A candidate with a lucky benchmark run can become a parent.
- The learning step can promote a hypothesis that did not actually improve the system.

Recommended fix:

- Add repeated evaluation for baseline and top candidates.
- Store `mean`, `median`, `stddev`, `min`, `max`, and sample count.
- Promote only if score improvement exceeds a configured threshold, for example `min_score_delta` or `min_confident_delta`.
- Re-run the current best every N generations to detect drift.
- Use paired/interleaved evaluation order: baseline, candidate, baseline, candidate.

### 2. Score parsing is too permissive

`_parse_score` first looks for `SCORE=...` or JSON `score`, but then falls back to the last floating-point number in stdout (`agentic_evolver.py:2676-2699`). That means unrelated command output can accidentally become the optimization score.

Impact:

- A compiler version, timing log, memory count, or unrelated numeric line can be interpreted as a successful score.
- Missing evaluator output can produce misleading progress instead of a hard failure.

Recommended fix:

- Remove the last-float fallback by default.
- Require structured JSON or explicit `SCORE=...`.
- Add a config flag such as `allow_score_fallback_float=false`.
- Fail evaluation when required primary metrics are absent.

### 3. Primary metrics explain outcomes but do not gate promotion

The README explicitly says the scalar `score` ranks candidates while `primary_metric_keys` only explain why a child helped or hurt (`agentic_evolver_README.md:94-100`). The implementation follows that: generation summaries display configured metrics, but best selection is still by scalar score (`agentic_evolver.py:3848-3857`, `agentic_evolver.py:3919-3930`).

Impact:

- The pipeline can promote a higher score that violates the real goal, such as lower power with unacceptable FPS loss, or better FPS with too much power increase.
- Missing `fps_1pct_low` or `power_w` does not necessarily block promotion.

Recommended fix:

- Add a candidate acceptance gate:
  - `compile_ok == true`.
  - Required metrics exist.
  - `target_fps_metric >= target_fps`, unless running in recovery mode.
  - Power/energy regression stays below a configured limit.
  - Score delta is significant relative to measurement noise.
- Keep scalar score for ranking only after the gate passes.

### 4. Planning is mostly prompt-guided, not profiler-guided

The index uses regex/static hints for hot paths, constants, and model/data clues (`agentic_evolver.py:826-895`). That is useful, but it is not the same as knowing where the benchmark actually spends time. The README itself identifies the next high-value upgrade as "hotspot-guided planning from profiler or benchmark traces" (`agentic_evolver_README.md:186-188`).

Impact:

- The LLM can spend many generations editing plausible-looking but low-impact files.
- Algorithmic plans may be creative without touching the true bottleneck.
- Constant tuning can waste budget when constants are not on measured hot paths.

Recommended fix:

- Ingest profiler traces, flamegraphs, device counters, or benchmark annotations into the project index.
- Attach hotspot weight to files/functions.
- Bias file selection by measured cost, not only static hints and prior candidate deltas.
- Require every plan to cite either profiler evidence or a known measured metric hypothesis.

### 5. Compile repair can hide low-quality mutation behavior

The default `max_compile_repairs` is high (`agentic_evolver.py:516-517`; README at `agentic_evolver_README.md:159-163`). Repair is valuable, but if many candidates need repair, the search is spending its budget on getting back to compile success instead of generating valid performance experiments.

Impact:

- Compile success becomes the main achievement.
- Repaired candidates may drift away from the original hypothesis.
- LLM budget and wall time are spent rescuing weak edits.

Recommended fix:

- Track `repair_attempts` per candidate in history.
- Penalize candidates that required repair unless they produce a strong measured win.
- Lower `max_compile_repairs` after initial stabilization, for example from 20 to 3-5.
- Add mutation-quality metrics: no-change rate, invalid JSON rate, incomplete content rate, compile failure rate, repaired success rate, and post-repair score delta.

### 6. Semantic correctness is delegated to the external evaluator

The prompts repeatedly ask the model to preserve behavior, and review/repair check compile risks, but the code only treats evaluator return code plus score as success (`agentic_evolver.py:2908-2928`, `agentic_evolver.py:2824-2875`). If the evaluator is mostly a performance benchmark, it may not catch behavior regressions.

Impact:

- A candidate can improve score by deleting work that was actually required.
- Mutations can change quality, accuracy, or output semantics while still passing the benchmark.

Recommended fix:

- Split evaluation into correctness and performance stages.
- Run deterministic functional tests before benchmark.
- Add output diff checks, image/trace comparisons, prediction accuracy thresholds, or domain invariants.
- Store correctness metrics separately from performance metrics.

### 7. Generation 0 can start before baseline score is known

With `pipeline_first_generation_during_baseline=true`, generation 0 planning and mutation starts while baseline evaluation is still running (`agentic_evolver.py:3866-3888`). This improves throughput, but it means generation 0 cannot use actual baseline score or measured baseline metrics.

Impact:

- First generation plans may be less grounded.
- If baseline evaluation fails or reports bad/missing metrics, generation 0 work is already spent.

Recommended fix:

- Keep this optimization only after the evaluator is stable.
- For diagnosis mode, disable `pipeline_first_generation_during_baseline`.
- Require baseline metrics to be valid before planning generation 0 when benchmarking is noisy or newly configured.

## Improvement Roadmap

### Phase 1: Make success trustworthy

1. Require explicit JSON or `SCORE=...`; remove last-number score fallback.
2. Require configured primary metrics to be present.
3. Add acceptance gates for target FPS, power/energy regression, and minimum score delta.
4. Re-evaluate baseline and top candidates multiple times.
5. Save score distributions, not just one score.

### Phase 2: Reduce wasted search

1. Track compile failures, repair attempts, no-change candidates, and invalid LLM responses as first-class run metrics.
2. Penalize or filter candidates that need repeated repair.
3. Lower repair budget once mutation quality improves.
4. Add duplicate-content or duplicate-diff detection, not just changed-file signature.

### Phase 3: Make planning evidence-driven

1. Add profiler/hotspot ingestion.
2. Weight files/functions by measured cost.
3. Require each child plan to cite hotspot evidence, prior successful hypothesis, or a specific metric tradeoff.
4. Let constant tuning operate only on constants located in or feeding measured hot paths.

### Phase 4: Protect real behavior

1. Add a correctness stage before performance scoring.
2. Add domain-specific invariants and quality thresholds.
3. Store correctness, performance, and power as separate metrics.
4. Reject candidates that trade correctness/quality for benchmark score.

## Suggested Config Changes For The Next Run

Start with a diagnosis-oriented configuration:

```json
{
  "pipeline_first_generation_during_baseline": false,
  "children_per_generation": 6,
  "max_compile_repairs": 5,
  "primary_metric_keys": ["fps_1pct_low", "power_w"],
  "keep_failed_candidates": true,
  "keep_successful_candidates": true,
  "command_timeout_sec": 300
}
```

Then update the evaluator so it emits strict JSON:

```json
{
  "score": 123.4,
  "fps_1pct_low": 118.2,
  "power_w": 54.7,
  "correctness_pass": 1
}
```

## High-Value Code Changes To Implement

1. Add config fields:
   - `required_metric_keys`
   - `min_score_delta`
   - `benchmark_repeats`
   - `baseline_repeats`
   - `allow_score_fallback_float`
   - `acceptance_gate`

2. Replace `score: float` with measured score stats:
   - `score_mean`
   - `score_median`
   - `score_stddev`
   - `score_samples`

3. Add a promotion function:

```python
def candidate_is_promotable(candidate, parent, cfg) -> tuple[bool, list[str]]:
    ...
```

4. Persist rejection reasons in `history.jsonl`.

5. Add profiler input support:
   - `profile_json_path`
   - function/file hotspot weights
   - hotspot snippets in planner prompts

## Bottom Line

The current pipeline is capable of producing candidates, but it is not yet strict enough to know whether a candidate is truly successful. Strengthen measurement first, then add profiler-guided planning. Without those two changes, the loop can keep evolving toward benchmark noise, accidental score parsing, compile repairs, or prompt-plausible edits rather than durable FPS/power improvements.
