# Research plan

## Organizing question

When, where, and how does TimesFM 3.0 benefit from cross-variate information, and can a
much smaller native-multivariate student preserve that behavior while substantially reducing
measured inference cost?

This question binds the dissection, distillation, and systems work. All three use the same
window representation, data provenance, metrics, run records, and correctness gates.

## Scope and non-goals

- Training data: the official `Salesforce/GiftEvalPretrain` interface, sampled without
  committing dataset files.
- Zero-shot evaluation: the official `Salesforce/GiftEval` protocol and aggregation.
- Teacher: `google/timesfm-3.0-pytorch` at a recorded immutable revision.
- Student: an independently implemented, native-multivariate transformer trained from
  scratch. Initial target: 20M–40M parameters, selected from computed counts and FLOPs.
- Hardware: two RTX PRO 6000 Blackwell GPUs. Latency uses one GPU; independent evaluation
  and cache shards may use two replicas; training uses DDP.
- No custom dataset, financial scraping, test-label tuning, teacher-weight redistribution,
  derivative-checkpoint distribution, or unsupported novelty claims.

## Shared experimental contract

Each run records its Git commit, config, seed, model and dataset revisions, UTC start/end
times, hardware snapshot, metrics, status, and log location. Large raw outputs remain local;
small aggregate JSON/CSV artifacts and reproducible plot code are committed.

The primary benchmark metrics are official GIFT-Eval MASE and its official normalized
quantile/CRPS metric, reported per dataset and under the official aggregation protocol.
Convenience metrics may supplement but never replace them.

## Workstream A: multivariate microscope

1. Reproduce teacher multivariate and univariate baselines.
2. Add seeded interventions that keep target history fixed: auxiliary permutation, patch-safe
   temporal shifts, channel dropout, controlled irrelevant channels, and supported covariate
   masks.
3. Introduce research-only hooks that disable the intended variate-attention operation in
   coarse layer groups without changing weights; prove the untouched path matches upstream.
4. Relate measured multivariate gain to descriptive (not causal) dataset characteristics:
   variate count, contemporaneous and selected lagged correlations, effective rank, and
   missingness.

## Workstream B: multivariate-aware distillation

Build a patched, normalized, non-autoregressive student with temporal and cross-variate
mixing and nine quantile outputs. Compare ground-truth-only, ordinary output KD, Dual-View
KD, and Cross-Variate Response Distillation (CVRD) under identical architecture,
initialization, data order, optimizer, training budget, validation split, precision, and
evaluation. For a full-context (`mv`) and target-only (`uv`) view of the same example,
CVRD matches the teacher and student forecast responses:

\[
R_T(x) = T_{mv}(x) - T_{uv}(x), \qquad
R_S(x) = S_{mv}(x) - S_{uv}(x), \qquad
\mathcal{L}_{CVRD} = d(R_S(x), R_T(x)).
\]

Dual-View KD supplies both teacher views independently. It is the mandatory supervision-matched
control: if CVRD cannot outperform it, the response-difference term has not added useful forecast
accuracy in this setup. Proper pinball supervision against ground truth remains present in every
KD run. Historical configs and artifacts may retain the internal `relkd` alias, but new public
documentation, figures, and claims use CVRD.

Teacher forecasts are cached ahead of training. Two independent GPU workers write separate,
checksummed shards from a versioned, stratified sampler. A pilot establishes output precision,
space per window, and a bounded cache size before any large run.

## Workstream C: Blackwell inference engineering

Start from unmodified official PyTorch inference and benchmark representative variate,
context, horizon, and batch shapes. Synchronize CUDA timing, separate cold start from warmed
steady state, and report latency distribution, throughput, peak allocated/reserved memory,
utilization, and power where measured.

Profile before optimizing. Test batching, BF16, semantically equivalent SDPA, compilation,
shape bucketing, CUDA graphs, and preprocessing/layout changes in that order of increasing
specialization. A speed result is accepted only after output-drift and benchmark-quality
gates; failed attempts remain in the experiment log.

## Milestone gates

| Gate | Pass condition |
| --- | --- |
| M0 Environment | Repository/package initialized; immutable machine snapshot committed; tests pass. |
| M1 Teacher | Pinned teacher loads; reference forecast works; tiny official evaluation protocol is verified. |
| M2 Microscope | MV/UV baselines, three interventions, layer-group hook, plots, and audit complete. |
| M3 Student | Forward/loss tests and tiny overfit pass; Student-GT trains and evaluates zero-shot. |
| M4 Standard KD | Cache round-trip validated; Student-KD run reproducibly completes. |
| M5 CVRD | Dual-View KD and CVRD behavioral-fidelity diagnostics complete under matched supervision and compute. |
| M6 Systems | Reference profile, justified optimizations, correctness gates, and performance table complete. |
| M7 Artifact | Claims and licenses audited; figures regenerate; clean-clone smoke run; repository clean/pushed. |

Major student training does not begin before M1. Full per-layer teacher ablation is attempted
only if coarse groups are informative. Large cache generation follows a tiny verified shard
and disk estimate.

## Required tests before expensive runs

- Patches, masks, normalization/inversion, quantile ordering, and forecast shapes.
- No-future-leakage invariants and deterministic multivariate interventions.
- Teacher-cache serialization/checksums and deterministic stratified sampling.
- Official metric aggregation and MV/UV conversion.
- Unmodified-vs-hooked teacher equality and optimized-vs-reference agreement.
- Student tiny-batch overfit before long training.

## Reporting policy

README numbers are generated from result artifacts. Every headline statement is mirrored in
`CLAIMS.md`; preliminary findings are not presented as established. Per-dataset results and
negative or failed runs are retained alongside aggregate summaries. Correlation analyses are
explicitly exploratory and are not interpreted causally.
