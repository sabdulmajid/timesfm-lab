# Representation-distillation screen (S5)

**Decision (2026-09-06):** test one **forecast-origin hidden-state alignment**
loss, and only against the matched compact output-KD control. This is a
training-only addition to the 28,860,480-parameter compact TimesFM-3-shaped
student; it adds no inference-time parameters or operations. It is a bounded
screen, not a claim that representation transfer will close the quality gap.

## Why this is the smallest defensible candidate

[DistilTS](https://arxiv.org/abs/2601.12785) identifies architecture mismatch
as a distillation problem and maps one variate-wise student state into a
teacher's time-indexed states with learned projections and time embeddings.
Its reported factorized temporal-alignment ablation improves long-horizon ETT
errors, but that setting uses iTransformer/DLinear students and point-wise
teachers, not TimesFM-3 on GIFT-Eval. The official DistilTS implementation at
commit
[`0f6982a`](https://github.com/itsnotacie/DistilTS-ICASSP2026/tree/0f6982a9606245747f82e5101884fdfb6e3ecafd)
was inspected for interpretation only: that repository has no license in the
audited snapshot, so no source is to be copied.

The recovery student's situation is simpler. The pinned
[TimesFM source](https://github.com/google-research/timesfm/blob/aa480150652811e732d87a3c5344b235234104e3/src/timesfm3/model.py)
exposes the final transformer state, and the compact student uses the same
32-point input patches, masked-horizon decode layout, stitching, and
time/variate token coordinates. With the frozen short-horizon contract
(`H <= 64`), stitching reads the output head from the last context-patch token.
Teacher and student therefore already have an exact semantic token alignment;
reconstructing every time step as DistilTS does would add an unnecessary
alignment problem.

The design borrows only well-supported minimal pieces:

- [MiniLM](https://arxiv.org/abs/2002.10957) shows that selecting only the
  teacher's last transformer layer avoids an arbitrary layer map.
- [TinyBERT](https://arxiv.org/abs/1909.10351) aligns different hidden widths
  with a learned linear student-to-teacher projection.
- [DistilBERT](https://arxiv.org/abs/1910.01108) supports directional/cosine
  hidden-state alignment alongside output supervision.

This does **not** implement or claim DistilTS, MiniLM, TinyBERT, or DistilBERT.
In particular, original MiniLM attention/value-relation losses are a poor first
screen here: the teacher has 16 attention heads, the compact student has 6,
and full token-relation matrices add head-mapping and quadratic storage/compute.

## Exact candidate

For the identical teacher and student decode request, obtain final transformer
states from `__call__:transformer_output`:

```text
H_T: [B, V, N, 1280]
H_S: [B, V, N,  384]
i0 = number_of_context_patches - 1
h_T = stop_gradient(H_T[:, :, i0, :])
h_S = H_S[:, :, i0, :]
```

Exclude a variate only when its entire context is unobserved. Apply the same
context, left padding, missing-value mask, channel order, horizon, and target
role to both models. The teacher remains in deterministic evaluation mode.

Use one bias-free training adapter
`A: R^384 -> R^1280` and a bounded directional loss:

```text
L_origin = mean || normalize(A h_S) - normalize(h_T) ||^2
L_total  = L_GT + alpha * L_output_KD + lambda * L_origin
```

`A` has exactly `384 * 1280 = 491,520` parameters. The total train-time
parameter count is 29,352,000; the adapter is discarded, not folded into the
student, so the deployment model remains exactly 28,860,480 parameters. Start
with `lambda = 0.1` and the output-KD weight selected for the matched S4 compact
control. Do not add CVRD, UV-view, multi-layer, attention-map, or horizon-state
losses to this screen. Log each unweighted loss and its student-backbone
gradient norm so a null result can be distinguished from a numerically
inactive term.

## Cache and teacher-compute budget

Do not cache full sequences. Cache only `h_T`, in FP16 after a recorded FP32
directional-error check, keyed by the existing five-field window identity and
teacher/code revisions. Raw storage is:

```text
bytes = sum_windows(V) * 1280 * 2
```

The production manifest contains 2,935,038 variate-windows. A full 1,048,576
window origin-state cache would therefore be 7,513,697,280 bytes (7.00 GiB)
before metadata; it is explicitly forbidden before the small screen succeeds.
A composition-matched 65,536-window screen is about 469,606,080 bytes
(448 MiB); even the `V=11` worst case is 1.72 GiB.

The measured production teacher-cache workers consumed 33,313.75 GPU-seconds
(9.25 GPU-hours) for 1,048,576 windows, including the extra UV teacher view on
true-MV examples. Linear extrapolation gives 0.58 GPU-hours for 65,536 windows.
Budget **0.75 GPU-hours maximum** for the one-view origin-state extraction,
including startup and aux-output overhead; first measure one shard and abort
or revise the predeclared budget if that bound is not credible. Raw states stay
local under the same restrictions as teacher outputs; commit only checksums,
counts, finite/error summaries, and provenance.

## Falsifiable screen and stop rule

Use S4 (compact output KD) as the control and S5 (the identical run plus
`L_origin`) as the treatment:

1. Select a deterministic, domain/context/view-stratified 65,536-window nested
   subset of production **training** coordinates without replacement. Record
   its manifest and prove no overlap with development or final confirmation.
2. Before training, verify on a fixed real batch that the extracted `i0` state
   is the state consumed by the stitched forecast head; teacher outputs with
   and without aux capture must match exactly. Require finite states and
   gradients. FP16-cache acceptance requires max directional loss versus FP32
   below `1e-5`; otherwise store FP32.
3. Train S4 and S5 from identical initialization with identical example order,
   optimizer, schedule, precision, batch sizes, and the same 50,000-step
   screening ceiling. Validation and early stopping follow the frozen recovery
   protocol. GIFT-Eval is not consulted.
4. Promote S5 only if its frozen development screening score is at least 1.0%
   lower than S4's and neither overall nor true-MV MASE-like/MWQL-like error is
   more than 0.5% worse. Higher hidden-state cosine similarity is diagnostic,
   never a promotion criterion.
5. Otherwise reject representation alignment for this recovery round. Do not
   try more layers, larger caches, attention relations, or several loss weights
   inside the six-configuration budget.

If S5 passes, a full origin-state cache and convergence run require a new
measured throughput/storage record, but no inference benchmark variant: the
deployed graph and parameter count are exactly those of the corresponding
compact student checkpoint.
