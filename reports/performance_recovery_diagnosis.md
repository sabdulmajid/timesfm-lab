# Production-path diagnosis for performance recovery

Date: 2026-09-06

Controlled-study baseline audited: commit `85be358`

Machine-readable evidence:
`results/reproduction/distillation/performance-recovery-production-path-audit.json`

## Bottom line

The historical 29.50M student is not simply one longer continuation away from the target. Its
pretrain validation number is poorly aligned with GIFT-Eval, its effective training weights vary
strongly with physical batch size, half of its absolute positional table never received an update,
and its GIFT adapter omits past-only covariates that the official teacher uses on nearly the entire
19-configuration multivariate workload. The last-token linear decoder is also a credible capacity
bottleneck, although it is not a correctness bug.

The highest-value recovery experiment is therefore a compact revision with a target/covariate-aware
input interface, a context representation trained over the declared 15,360-point shared context,
and future-query or future-patch decoding. It should be selected with domain-balanced, MASE-like and
probabilistic development metrics. Merely extending the existing 200k schedule is low value.

## 1. Input and evaluation parity

### What is aligned

- Production contexts and targets are reconstructed from the same official source row and forecast
  origin recorded in each cache shard. Training slices `[end-context, end)` and `[end, end+horizon)`;
  the future target is never passed into the student.
- Source channel order is preserved through cache generation, training materialization, student
  inference, and the final `(V,H,Q) -> (Q,H,V)` GluonTS conversion.
- Teacher cache quantiles are sorted at levels 0.1 through 0.9. The student produces those same nine
  levels in monotonic order by construction. The forecast start is the input start plus the observed
  context length.
- All production horizons are 8, 16, 32, or 64. The frozen short-horizon benchmark requires at most
  60, so its horizon is supported and trained.

### Material parity differences

| Operation | Official teacher benchmark | Historical student benchmark |
|---|---|---|
| Context cap | 15,360 points (`timesfm3_forecaster._MAX_CONTEXT_LENGTH`) | 16,384 points |
| Past-only covariates | Passes `past_feat_dynamic_real` | Ignores the field |
| Missing history | Removes leading all-missing time steps, then linearly interpolates | Excludes non-finite values from statistics and supplies a binary observed mask |
| Symmetric averaging | Two logical views, `T(x)` and `T(-x)`, then sign/quantile reversal average | One forward pass |
| Quantile order | Explicit sort | Monotonic parameterization; no post-sort needed |
| Nonnegative history | Conditionally clamps forecasts to zero | No nonnegative clamp |
| Evaluation compute dtype | Teacher checkpoint/model path is float32 | BF16 autocast |

The cache was generated through `TimesFM3Evaluator` defaults, so its KD labels include symmetric
averaging, explicit quantile sorting, and conditional nonnegative clipping. It did **not** pass
past-only covariates. The student therefore learns a postprocessed teacher forecast from target-only
history, then is compared against an official teacher that can use additional covariate channels.

This is large on the primary multivariate scope:

- Full 55: 10 configurations and 36,046 / 282,363 instances (12.77%) contain past-only covariates.
- MV19: 8 configurations and 35,356 / 35,487 instances (99.63%) contain them.
- Covariate width ranges from 2 to 35 channels. TimesFM's evaluator caps a forward at 32 total
  variates and deterministically subsamples excess past covariates where required; the student uses
  zero of them.

This does not justify weakening the teacher or removing those configurations. A recovery student
intended to meet the frozen target needs a distinct target-versus-conditioning-variate interface and
must emit forecasts only for target channels. A target-only teacher result may be reported as a
diagnostic, never as the primary reference.

The context protocol must also be corrected before screening candidates. The official teacher cap is
15,360, not 16,384. On the full scope, 24,740 instances (8.76%) exceed both caps; on MV19, 120
instances (0.34%) do. The frozen joint comparison should right-truncate both models to the same
15,360 observed points. Historical student results retain their original 16,384 behavior and remain
versioned as such.

## 2. Training coverage

### Positional coverage

The student owns 512 learned absolute position rows (16,384 / 32) and right-aligns them. Production
training has no context longer than 8,192, so rows 0--255 received no gradients. Exposure per corpus
pass is highly nonuniform: row 256 first appears in 91,827 windows, while row 511 appears in all
927,346 post-split training windows.

This affects the declared benchmark rather than only a hypothetical long-horizon feature:

- Full 55: 51,184 / 282,363 instances (18.13%) have context longer than 8,192.
- MV19: 16,102 / 35,487 instances (45.37%) have context longer than 8,192.

Every such request activates at least one never-trained absolute position. Relative/RoPE positions or
explicit training through the shared 15,360 cap should remove this discontinuity.

### Horizon-head coverage

The 256-step output head has trained rows 0--63 and completely untouched rows 64--255. Within the
trained region, rows 0--7 see all 927,346 windows, rows 8--15 see 791,758, rows 16--31 see 704,356,
and rows 32--63 see 698,444. This is sufficient for the frozen 55 short-horizon scope (maximum 60),
but it means the model's nominal 256-step support is not an empirically trained capability.

### Resolution and seasonality

The corpus covers all seven recorded broad domains and contexts from 8 to 8,192, but the student is
not given frequency, calendar, or domain identity. Fixed context choices are associated with whole
datasets, so a position row's exposure is also confounded with domain and resolution. This is not a
leak, but it weakens extrapolation to unseen context-length/resolution combinations.

## 3. Validation and effective optimization weights

The global validation score is a target-element-weighted normalized pinball mean, not MASE, MWQL, or
an unweighted configuration aggregate. At GT step 200k, `cdc_fluview_ilinet` alone supplies 86.38%
of the scalar loss; the two flu datasets together supply 89.30%. Consequently, the student's
near-teacher pretrain pinball does not demonstrate near-teacher benchmark quality.

Physical batch size also silently changes the optimized distribution. Every batch loss is averaged
and receives one AdamW update, while batch size ranges from 32 to 256. Per corpus pass:

| Context | Window share | Optimizer-step share | Relative weight/window |
|---:|---:|---:|---:|
| 8 | 12.71% | 5.08% | 0.40x |
| 512 | 15.90% | 6.35% | 0.40x |
| 2,048 | 30.97% | 24.62% | 0.79x |
| 4,096 | 17.27% | 27.32% | 1.58x |
| 8,192 | 9.90% | 31.29% | 3.16x |

Thus “each window once per epoch” does not imply equal example weight. For example, Healthcare is
10.81% of windows but only 4.54% of steps; Transport is 37.52% of windows and 44.30% of steps. A
constant effective-example update or explicit predeclared domain/series weighting is needed for a
clean recovery screen.

## 4. Prediction head and architecture

The exact student size is 29,499,264 trainable inference-time parameters:

- transformer layers: 28,389,888;
- output head: 887,040;
- absolute position table: 196,608;
- patch projection: 24,960;
- final norm: 768.

For every target variate, only the final valid historical patch state is normalized and linearly
projected to all `256 x 9` outputs. Earlier history can reach that state through self-attention, so
this is not erroneous, but a single 384-dimensional bottleneck must encode every horizon and
quantile. The teacher instead appends masked future patches, produces a representation per future
patch, and uses stitching, iterative CPM RevIN, and linear detrending. A compact future-query or
future-patch decoder is therefore the strongest single architectural hypothesis to test. It should
replace, not stack on top of, expensive inference machinery where possible.

## 5. Optimization evidence

Validation behavior over 150k--200k steps is:

| Variant | Best step | Best pinball | Last-50k change |
|---|---:|---:|---:|
| GT | 170k | 1.450776 | 0.005% worse by 200k |
| KD | 200k | 1.453125 | 0.100% better |
| Dual-View | 200k | 1.451782 | 0.093% better |
| CVRD | 200k | 1.454971 | 0.098% better |

The late improvements are real but far too small to explain the roughly 27% reduction in historical
MV19 student error required by the target. Lower-LR continuation remains a cheap bounded screen, not
the primary recovery strategy.

Under the nominal unit weights, recorded run-average component scales show weak teacher pressure:

- KD: GT 0.42482 and MV-KD 0.06187 (KD is 12.7% of the summed objective).
- Dual-View: GT 0.42510, MV-KD 0.05992, UV-KD 0.03217.
- CVRD: GT 0.42701, MV-KD 0.06046, UV-KD 0.03116, CVRD 0.00968 (CVRD is 1.8%).

This motivates a small validation-selected GT/KD balance screen. It does not justify assuming a
larger KD coefficient will help.

Historical records do not contain interval training curves, per-domain gradient norms, quantile-wise
validation, median-only error, or clipping counts. `clip_grad_norm_` is called at 1.0 but its returned
pre-clip norm is discarded, so clipping frequency cannot be recovered after the fact. Add those
measurements to recovery runs before classifying the model as optimization- or capacity-limited.

## 6. Focused correctness probes

The CPU audit used the validation-selected GT checkpoint
`959dd679274f08e3f4377dbbf3fa48db011208baa735620bac06ab1ede9a0b24` and real `ett1/W` inputs.

- Putting a series beside an unrelated, longer request changes its forecast by at most
  `2.31e-6` context-standard-deviation units (`6.57e-7` mean). This is ordinary batched matrix-multiply
  rounding, not cross-request information flow. Attention never crosses the batch dimension.
- Variable-length left padding changes it by at most `1.84e-6` normalized units.
- Permuting target channels and undoing the permutation changes it by at most `2.08e-6` normalized
  units, consistent with native permutation-equivariant variate mixing.
- A context with internal missing values yields finite forecasts; all nine output quantiles remain
  ordered.
- The committed normalization audit is finite and reversible, including extreme Bitcoin-like values;
  its maximum relative round-trip error on that source is `1.11e-7`.
- On eight fixed real four-variate flu windows, a 54,120-parameter wiring probe has finite gradients.
  In 100 CPU updates, GT pinball falls from 6.8828 to 4.2616 and teacher-imitation Smooth-L1 falls
  from 1.4691 to 0.02105. This proves the target/loss wiring can learn; it is not evidence that the
  production architecture is adequate.

No target leakage, channel-order bug, quantile-order bug, normalization failure, or material
cross-request batching dependence was found.

## 7. Recovery priorities implied by the evidence

1. Freeze a shared 15,360-point context protocol and make the student consume official past-only
   covariates as conditioning channels while forecasting target channels only.
2. Replace absolute undertrained positions and the last-token whole-horizon head in one near-30M
   architecture revision; future queries/patches plus RoPE are the most direct controlled change.
3. Screen with domain/series-balanced effective updates and development metrics that include
   MASE-like median error, pinball/MWQL-like error, and multivariate slices.
4. Test inexpensive recipe changes: conventional KD without CVRD privilege, a stronger but bounded
   KD weight, median-focused loss, EMA, and the deployment-time nonnegative clamp/symmetric average
   only if their quality-versus-latency tradeoff is measured.
5. Instrument gradient norms/clipping, per-component and per-domain losses, quantile/median errors,
   and context/horizon slices. Stop continuations when those curves plateau.

These are recovery hypotheses, not validated improvements. Candidate selection must stay on the
leakage-resistant development split; the repeatedly inspected 19/55 GIFT results are final frozen
confirmation evidence only.
