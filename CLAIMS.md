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
