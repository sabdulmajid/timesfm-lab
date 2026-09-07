# Performance-recovery protocol

**Frozen protocol:** `timesfm3-performance-recovery-v1.2`, 2026-09-06. The
machine-readable authority is
[`configs/performance_recovery/targets.yaml`](../configs/performance_recovery/targets.yaml).
Commit `85be358` is the boundary of the preserved controlled GT/KD/Dual-View/CVRD
study. Recovery runs are a separate result track and must not overwrite or
retroactively reinterpret it.

Version 1.1 is a pre-candidate correctness correction: the pinned TimesFM-3
forecaster's effective context cap is 15,360, not 16,384. No recovery candidate
metric had been observed when this was corrected. The shared quality and latency
protocol therefore caps both models at 15,360; the target ratios, tasks,
aggregation, candidate limits and compute budget are unchanged.

Version 1.2 makes one predeclared screen-slot amendment before either compact
screen reached its first step-5,000 development evaluation. The production-path
audit measured strong elementwise/domain imbalance, so S5 is now a conditional
equal-window, target-blind domain-balanced training screen rather than the
reserved representation-distillation screen. This changes neither the success
criteria nor the data, selection, candidate, finalist, or GPU-hour limits. It
also does not activate S5 automatically: the exact development-only gate below
must pass first.

## Success criteria

A result passes only when one checkpoint, with one deployment configuration,
passes every gate below. An ensemble, a different speed checkpoint, or a
teacher-derived calibration component cannot be combined into a joint pass.

1. **Size:** at most **30,000,000** inference-time trainable parameters. Count
   every unique trainable scalar used by preprocessing, backbone, adapters,
   calibration and decoding. Always report the exact count. The historical
   student has 29,499,264 parameters; that establishes only the size gate.
2. **Quality:** in native multivariate mode, student MASE and MWQL must each be
   no more than 1.06 times the corresponding TimesFM-3 value. This must hold
   independently on (a) the pinned 55-configuration complete short-horizon
   GIFT-Eval scope and (b) the pinned 19-configuration complete short-horizon
   multivariate subset. Both use the unchanged nine quantiles and an unweighted
   geometric mean across configurations.
3. **Speed:** the equal-weight geometric mean of paired, per-shape p50
   end-to-end speedups must be at least 6.0x on the frozen workload matrix.
   Speedup is optimized-teacher latency divided by student latency. Model-only
   latency, p95, every shape, raw samples, throughput, memory and cold start are
   also reported but cannot replace the end-to-end gate.
4. **Capability:** the checkpoint must forecast `V > 1` target histories jointly
   and emit all nine quantiles per target. A fixed auxiliary-history
   intervention must demonstrate a native multivariate path, while regrouping
   unrelated requests must not change a request's forecast.

Quantization changes storage or arithmetic, not parameter count. A larger
capacity probe is diagnostic and cannot satisfy the 30M gate.

### Frozen quality baselines

The authoritative 19-configuration TimesFM-3 multivariate artifact reports
MASE `0.824912283105205` and MWQL `0.10092307269811338`. Therefore the exact
limits are:

| Scope | Teacher MASE | Maximum student MASE | Teacher MWQL | Maximum student MWQL |
| --- | ---: | ---: | ---: | ---: |
| 19-config complete short-horizon multivariate subset | 0.824912283105205 | **0.8744070200915173** | 0.10092307269811338 | **0.10697845706000018** |
| 55-config complete short-horizon scope | 0.9556088126014846 | **1.0129453413575737** | 0.10323988539119257 | **0.10943427851466413** |

The 55-task thresholds were derived once from the completed teacher artifact
after its status, zero-failure count, 55 entries, revisions and config
fingerprint matched the target file. This derivation did not change the frozen
ratio or scope.

For scale, seed-42 CVRD scored MASE `1.201957803874046` and MWQL
`0.14838934866279552` on the 19-task scope. It requires reductions of 27.25%
and 27.91%, respectively, to reach these limits. A small development gain is
not performance recovery.

## Correctness gate before candidate training

The production path—not a pilot path—must pass a recorded parity audit. For
each fixed real-data request, teacher and student must receive the same
forecast origin, target-channel order, available target history, historical
features, missingness and trailing context (capped at 15,360 for both). The
audit must cover target alignment, horizon slicing, quantile order, padding,
normalization and inverse normalization. No future target may enter an input.

Forecasts must be invariant, within the declared numerical tolerance, to
batching the same request alongside unrelated requests. This catches padding,
masking, positional-index and accidental cross-request mixing errors. A small
fixed batch must also establish finite gradients and the ability to overfit
ground truth and teacher targets. Candidate results are ineligible until these
checks pass.

The audit also records which context lengths, positional indices, horizons and
head rows receive updates. Evaluation beyond trained positional support is a
diagnosis, not a reason to shorten only the student's input. Any future shared
context change requires a new protocol version and applies equally to teacher
and student.

## Exploration and selection

Benchmark results are not selection data. The 19 configurations have been
inspected repeatedly, and portions of the 55-task scope were inspected before
this freeze. Neither may be described as never touched. Recovery candidates
are ranked using the leakage-resistant pretraining development data; a final
series/block-disjoint confirmation manifest must be hashed before screening
and its metrics remain embargoed until finalists are frozen.

At most six substantive configurations may be screened: lower-LR continuation,
diagnosis-scaled output KD, one compact future-query decoder under GT and KD,
one conditional equal-window/domain-balanced recipe, and one conditional
capacity probe.
The capacity slot activates only if learning curves and diagnostics support a
capacity limitation. Micro-probes that only establish correctness do not count
as substantive configurations; training alternatives or selecting among their
forecast metrics does.

S5 is the only domain-balancing screen. At the first common step 5,000, select
the lower `balanced_forecast_error` of S3 and S4 (exact tie: S3). S5 must copy
that base's exact GT/KD coefficients; it may not retune KD and balancing in one
candidate. Activate S5 only when (a) at least one compact candidate improves its
overall score from step zero and (b) the geometric mean of its per-domain
step-5,000/step-zero pinball and normalized-median-MAE ratios over Econ/Fin,
Energy, Healthcare, Sales, and Web/CloudOps is at least 1.03 times the analogous
ratio over Nature and Transport. The frozen weights and source identities are
recorded in `configs/performance_recovery/candidates.yaml`. Training changes
from observed-element averaging to per-window averaging followed by those
weights; validation and model selection remain unweighted geometric aggregates
across datasets. Its estimated cost may not exceed 29.1 physical GPU-hours
without another pre-run amendment. If the gate fails, S5 is not implemented or
launched.

Every screen declares its hypothesis, exact initialization, data access,
examples, updates and estimated GPU-hours before launch. Comparable candidates
receive the same screening budget. The score is a weighted geometric mean of
ratios to the frozen incumbent:

- 30% overall development MASE-like error;
- 30% overall development MWQL-like error;
- 15% true-multivariate development MASE-like error;
- 15% true-multivariate development MWQL-like error;
- 10% aggregate end-to-end latency.

Lower is better. Ties use overall development MWQL, parameter count, then
candidate ID. The incumbent is the seed-42 GT checkpoint at step 170,000,
selected because it has the lowest historical seed-42 production-validation
pinball loss (`1.4507755934485174`), not because of GIFT-Eval. Its checkpoint
SHA-256 is
`959dd679274f08e3f4377dbbf3fa48db011208baa735620bac06ab1ede9a0b24`.
The parity audit must populate its remaining score terms before comparisons.

No more than two recipes advance to full convergence training. If the new
decoder wins, its GT and KD screens form the minimum attribution control.
Finalists preserve best-development and final checkpoints. The selected recipe
is confirmed with three seeds. Its deployment checkpoint is the seed/checkpoint
with the lowest frozen final-confirmation score; ties use latency then integer
seed. All seed results are reported.

## Compute boundary

Recovery work after this freeze has a hard cap of **120 physical GPU-hours**,
computed as the sum of `GPU count × elapsed wall hours`. The completed
controlled study is sunk cost and excluded. The provisional budget is 16 hours
to finish the reference, 24 for correctness and screens, 60 for finalists, and
20 for final evaluation and systems work; allocation may move, but the total
may not. Before a substantial job, its measured throughput, GPU-hour estimate,
cumulative spend and resume checkpoint are recorded. Raising the cap requires
a new protocol version before—not after—the additional compute.

## Frozen latency experiment

Teacher and student run serially, without contention, on the same RTX PRO 6000
Blackwell GPU. Each receives the same real GIFT-Eval inputs, historical
features, context cap, batch, horizon and nine-quantile output requirement.
There are ten warmups and fifty synchronized measured calls per shape. The
end-to-end clock starts before CPU validation/packing and host-to-device
transfer and stops after quantiles are materialized on CPU. Stock and reasonably
optimized teacher paths are both reported; the optimized teacher is the gating
denominator. Reduced precision or approximation belongs to the deployment
configuration and must pass the full quality gates.

The fixed matrix spans univariate and true-multivariate requests, context 29 to
15,360, horizon 6 to 60, 1 to 21 variates, and the exact batches listed in the
target YAML. The primary statistic is the geometric mean of the nine paired
p50 speedups. Peak speedup on one shape is never the aggregate claim.

## Reporting boundary

The final claim-validation record is `reports/target_validation.md`. It names
the checkpoint and code hashes, deployment configuration, revisions, exact
scope, per-seed evidence and pass/partial/fail status. Controlled-study results
and performance-recovery results remain separate. Test labels, teacher outputs,
restricted weights and datasets are not published.

Only a full joint pass supports: “Within 6% of teacher MASE and MWQL on 55
short-horizon GIFT-Eval configurations,” followed by the measured end-to-end
speedup and exact parameter count. “94% teacher quality” and “6× lower
inference cost” remain prohibited because neither phrase defines the measured
error or latency experiment.
