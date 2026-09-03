# timesfm-lab

**Dissecting, distilling, and accelerating multivariate time-series foundation models.**

`timesfm-lab` asks when and how TimesFM 3.0 benefits from cross-variate information,
whether a much smaller native-multivariate student can preserve that behavior, and what
the measured accuracy/efficiency trade-off looks like on NVIDIA Blackwell GPUs.

> Results are being reproduced. No forecasting-quality or performance claim is made yet.

## Research program

The repository has three connected workstreams built on one data and evaluation layer:

1. **Dissect:** compare native multivariate behavior with univariate inference and
   controlled, leakage-safe cross-variate interventions.
2. **Distill:** train an independently implemented compact multivariate student from
   scratch on `Salesforce/GiftEvalPretrain`, with ground truth and cached teacher outputs.
3. **Accelerate:** profile reference TimesFM inference, then evaluate optimizations behind
   numerical and benchmark-quality correctness gates.

The zero-shot benchmark is `Salesforce/GiftEval`; its official metrics and aggregation
remain authoritative. Negative results and failed optimization attempts are retained.

## Status

The project is in the environment and teacher-reproduction phase (M0/M1). See
[`docs/research_plan.md`](docs/research_plan.md) for milestone gates and [`CLAIMS.md`](CLAIMS.md)
for the evidence ledger. Result summaries will be generated from committed machine-readable
artifacts rather than entered manually.

## Quick start

```bash
python3 -m pip install 'uv>=0.8,<0.9'
uv sync --extra dev
uv run pytest
```

The lock file pins exact dependency artifacts and immutable upstream Git commits. Install the
teacher and benchmark stacks only when needed with `uv sync --extra teacher --extra data
--extra benchmark`. Downloaded datasets, teacher caches, model weights, and distilled
checkpoints are intentionally excluded from Git.

## Reproducibility and integrity

- Every experiment is driven by a committed config and emits a structured run record.
- Raw environment snapshots are append-only and include actual GPU compute capability.
- GIFT-Eval test labels are never used for training or hyperparameter selection.
- Locally reproduced, externally reported, exploratory, and verified results are labeled
  separately.
- No TimesFM weights or distilled derivative weights are distributed here.

See [`LICENSE_NOTES.md`](LICENSE_NOTES.md) before using the teacher or producing a distilled
checkpoint. Original repository code is licensed under Apache-2.0; upstream models, code,
and datasets retain their own terms.
