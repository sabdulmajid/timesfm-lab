# Performance-recovery target validation

Protocol: `timesfm3-performance-recovery-v1.1`. This is a live evidence ledger,
not a claim of success. The controlled study ends at commit `85be358`; recovery
results are recorded separately. The 19-task and portions of the 55-task
benchmark were inspected before this protocol was frozen.

| Proposed claim | Exact definition | Checkpoint / code | Scope and revision | Measured result | Repeatability | Status | Established |
|---|---|---|---|---|---|---|---|
| Approximately 30M inference-time parameters | One deployed checkpoint has at most 30,000,000 unique trainable scalars, including every inference component | Historical GT: `959dd679…`; compact candidate code `9d37c67` | Model definition only | Historical student: 29,499,264; compact candidate: 28,860,480 | Exact deterministic count | **Pass (size only)** | 2026-09-06 |
| Within 6% of teacher error | The same student has MASE and MWQL no greater than `1.06 ×` teacher on both frozen short55 and MV19, native-MV mode | Historical seed-42 CVRD best: see controlled artifacts | GIFT-Eval `30841734…`; short55 and MV19 | Short55 teacher: 0.955609 MASE / 0.103240 MWQL, giving limits 1.012945 / 0.109434. MV19 teacher: 0.824912 / 0.100923, giving limits 0.874407 / 0.106978. Historical CVRD: 1.201958 / 0.148389 on MV19; recovery candidate pending | Teacher short55 and MV19 are exact fingerprint-validated artifacts; controlled three-seed CVRD remains far above target | **Fail (current); recovery pending** | 2026-09-06 |
| At least 6× lower end-to-end latency | Geometric mean of optimized-teacher/student paired p50 speedups over the frozen real-data workload is at least 6.0×; same quality checkpoint and deployment | Pending | One RTX PRO 6000 Blackwell, serial matched calls, nine quantiles | Claim-grade measurement pending | 50 timed calls/shape required | **Pending** | — |
| Native multivariate support | One forward jointly consumes `V>1` targets, emits nine quantiles/target, responds to an auxiliary-history intervention, and is invariant to unrelated request batching | Historical student `959dd679…`; compact interface code `0967276` | Fixed real-data probes | Historical permutation/batch invariance max normalized error `2.31e-6`; compact covariate-aware confirmation pending trained checkpoint | Deterministic probes | **Partial** | 2026-09-06 |
| Honest zero-shot protocol | No evaluation targets, per-dataset checkpoint choice, or GIFT-driven tuning; disclose earlier benchmark exposure | Protocol/config `c595b45`, corrected by `21a22a6` | Pinned pretrain and GIFT revisions | Development-only recovery selection frozen; MV19 repeatedly inspected and short55 partially inspected previously | Auditable manifests/checksums required | **Partial / disclosed exposure** | 2026-09-06 |

## Current evidence boundary

The controlled study established that output KD helps in some comparisons and
that CVRD improves response fidelity, but it did not close the forecasting gap
or reliably recover the teacher's multivariate advantage. No “94% teacher
quality” or “6× lower inference cost” statement is currently supported.

The recovery row will be replaced only by a single checkpoint/deployment pair
that has passed the frozen size, short55, MV19, speed, and native-MV gates.
