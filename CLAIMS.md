# Claims ledger

No forecasting-quality or performance claim is verified yet. One infrastructure claim has
passed its reproducibility gate.

## Verified infrastructure

### CLAIM: the pinned TimesFM 3.0 evaluator executes a valid multivariate reference forecast

- **Status:** verified
- **Command:** `PYTHONPATH=src:$TIMESFM_SRC/src HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_teacher.py configs/reproduction/reference_forecast.yaml`
- **Config:** `configs/reproduction/reference_forecast.yaml`
- **Evidence:** `results/reproduction/teacher_reference_smoke.json`
- **Git commit:** `156b26c4dd5f403eeea75410c85523c64ac73bf7`
- **Runs/seeds:** 1 run, seed 20260903
- **Notes:** The check verifies model identity, parameter count, expected shapes, finite values,
  quantile ordering, and median consistency. It is synthetic protocol input and is not a
  benchmark-quality, latency, or speed claim.

Every prominent result in the README or reports must have an entry below. A claim may be
`preliminary`, `verified`, or `rejected`. Verification requires a committed config and result
artifact, the producing Git commit, and the number of runs or seeds.

## Entry template

### CLAIM: concise, falsifiable statement

- **Status:** preliminary | verified | rejected
- **Command:** `command used to reproduce the result`
- **Config:** path to committed configuration
- **Evidence:** path to committed aggregate result
- **Git commit:** full commit SHA
- **Runs/seeds:** count and seed list
- **Notes:** limitations, external values, or failure details

## Preliminary measurements

### CLAIM: native multivariate inference is better than target-only inference on the pinned `ett1/W/short` smoke slice

- **Status:** preliminary
- **Command:** `CUDA_VISIBLE_DEVICES={0,1} .venv/bin/python scripts/run_gift_smoke.py --data-root $GIFT_EVAL --mode {multivariate,univariate}`
- **Config:** `configs/reproduction/teacher_smoke.yaml`
- **Evidence:** `results/reproduction/teacher_smoke/ett1-w-short-{multivariate,univariate}-seed42.json`
- **Git commit:** `882fa2fb1fb8dc13478932b15b2c63372298d815`
- **Runs/seeds:** 1 paired run, seed 42, two test instances
- **Notes:** Multivariate MASE/MWQL are 1.4723008/0.2496274; univariate values are
  1.5054522/0.2533636. Relative reductions are 2.20% and 1.47%. The multivariate values
  match the pinned Salesforce artifact within 2.4e-8 relative error. This non-reportable
  integration slice does not support a general forecasting claim.
