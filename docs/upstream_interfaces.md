# Pinned upstream interfaces

Verified on 2026-09-03. These are the immutable inputs recommended for the
first teacher reproduction. Moving branch names such as `master` and `main`
must not appear in run records.

| Component | Recommended immutable revision | License/access |
| --- | --- | --- |
| TimesFM source | [`aa480150652811e732d87a3c5344b235234104e3`](https://github.com/google-research/timesfm/tree/aa480150652811e732d87a3c5344b235234104e3) | Apache-2.0 source repository |
| TimesFM package | `3.0.1` from the source commit above | Python >=3.10 |
| TimesFM 3.0 weights/config | [`google/timesfm-3.0-pytorch@43046b85ec22d584a13f8098c2ed39c889e129c2`](https://huggingface.co/google/timesfm-3.0-pytorch/tree/43046b85ec22d584a13f8098c2ed39c889e129c2) | TimesFM Non-Commercial License v1.0 |
| GIFT-Eval code | [`e8887e4bae7a8f236af0df41d48e505d075dffbd`](https://github.com/SalesforceAIResearch/gift-eval/tree/e8887e4bae7a8f236af0df41d48e505d075dffbd) | Apache-2.0 |
| GIFT-Eval data | [`Salesforce/GiftEval@30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`](https://huggingface.co/datasets/Salesforce/GiftEval/tree/30841734ac5cfddbd0c3bad6d09d2b6b32becbb0) | Apache-2.0, public/ungated |
| Pretraining data | [`Salesforce/GiftEvalPretrain@6830b624de7ed2b3d3e5b85bb6959d81dcc5d874`](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain/tree/6830b624de7ed2b3d3e5b85bb6959d81dcc5d874) | Apache-2.0, public/ungated |

The model repository reports 330,710,976 float32 parameters and a
1,322,898,824-byte `model.safetensors`. Its current
[`config.json`](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/config.json)
specifies 20 layers, model/hidden width 1280, 16 heads, input patch length 32,
output patch length 64, variate attention, SDPA, stitching, linear detrending,
and iterative CPM RevIN. The config, rather than constructor defaults, is the
authority when loading from the Hub.

## Environment and installation contract

Use a dedicated environment and pinned checkouts outside this repository. The
official TimesFM GIFT runner assumes that the GIFT-Eval checkout is named
`gift-eval` directly below the TimesFM checkout.

```bash
export UPSTREAM_ROOT=/absolute/path/outside/timesfm-lab/upstream
export TIMESFM_SRC="$UPSTREAM_ROOT/timesfm"
export GIFT_EVAL_SRC="$TIMESFM_SRC/gift-eval"

git clone https://github.com/google-research/timesfm.git "$TIMESFM_SRC"
git -C "$TIMESFM_SRC" checkout --detach aa480150652811e732d87a3c5344b235234104e3
git clone https://github.com/SalesforceAIResearch/gift-eval.git "$GIFT_EVAL_SRC"
git -C "$GIFT_EVAL_SRC" checkout --detach e8887e4bae7a8f236af0df41d48e505d075dffbd

python3.12 -m venv "$UPSTREAM_ROOT/venv"
source "$UPSTREAM_ROOT/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "${TIMESFM_SRC}[torch]"
python -m pip install -e "$GIFT_EVAL_SRC"
```

Upstream does not supply one jointly locked environment. At the pinned commits:

- TimesFM requires Python >=3.10, NumPy >=1.26.4,
  `huggingface-hub>=0.28.0`, `safetensors>=0.5.3`, and optionally
  `torch>=2.0.0`.
- GIFT-Eval requires Python >=3.10, `numpy~=1.26.0`,
  `gluonts~=0.15.1`, `datasets~=2.17.1`, `scipy~=1.11.3`, pandas >=2,
  and several analysis dependencies. Its `baseline` extra raises the Torch
  floor to 2.1 but is not needed for the TimesFM wrapper.
- The safe NumPy intersection is 1.26.4. Record the resolved lock before a
  reportable run because Torch and several transitive packages have no exact
  upstream pin.
- This repository's `data` extra now matches the pinned GIFT-Eval NumPy,
  `datasets`, `gluonts`, and SciPy constraints. `uv.lock` resolves the combined
  teacher/benchmark environment; keep that lock unchanged for reportable runs.

## TimesFM 3.0 loading and prediction API

The public entry points are `timesfm3.ModelConfig`,
`timesfm3.TimesFM3Forecaster`, and the benchmark-specific
`timesfm3.TimesFM3Evaluator`. Pin the Hub revision through `ModelConfig.revision`:

```python
import numpy as np
from timesfm3 import ModelConfig, TimesFM3Evaluator

model = TimesFM3Evaluator(
    ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        revision="43046b85ec22d584a13f8098c2ed39c889e129c2",
        per_core_batch_size=1,
        device="cuda",
    )
)

context = np.stack([
    np.linspace(0, 1, 64, dtype=np.float32),
    np.sin(np.linspace(0, 8, 64, dtype=np.float32)),
])
output = next(model.predict_batch(contexts=[context], horizon=8))
```

Interface details established from the pinned
[`timesfm3_forecaster.py`](https://github.com/google-research/timesfm/blob/aa480150652811e732d87a3c5344b235234104e3/src/timesfm3/timesfm3_forecaster.py)
and
[`evaluator.py`](https://github.com/google-research/timesfm/blob/aa480150652811e732d87a3c5344b235234104e3/src/timesfm3/evaluator.py):

- A univariate context is `(time,)`; a multivariate context is
  `(target_variates, time)`. Every item in one call must have the same target
  variate count, though context lengths may differ.
- Past-only covariates are `(covariates, context)` and known-future covariates
  are `(covariates, context + horizon)`.
- Output shapes are `(horizon,)` and `(horizon, 9)` for 1-D input, or
  `(target_variates, horizon)` and `(target_variates, horizon, 9)` for 2-D
  input. The quantiles are 0.1 through 0.9; index 4 is the point/median
  forecast.
- Context is dynamically left-padded to a patch boundary and truncated to the
  most recent 15,360 points. Horizon is internally patch-rounded and sliced
  back to the request.
- `TimesFM3Evaluator` defaults to quantiles, symmetric sign averaging,
  nonnegative clamping for nonnegative histories, sorted quantiles,
  `use_znorm=False`, and `padding_mode="none"`.
- Evaluator multivariate mode permits at most 32 total target and covariate
  channels per forward. Above that, it deterministically subsamples excess
  covariates with RNG seed 42 and chunks target channels. This behavior is part
  of the official benchmark path and must be recorded.
- `univariate=True` flattens target channels into independent forecasts and
  discards covariates. This is the correct official target-only comparison,
  not merely `use_variate_attention=False`.

### Important configuration trap

For a Hub repo ID, `TimesFM3Forecaster._init_model()` calls
`TimesFM3Torch.from_pretrained()` without forwarding the top-level
`ModelConfig.use_variate_attention` or `ModelConfig.use_sdpa` values, then
synchronizes the forecaster config from the downloaded model config. Therefore
setting either top-level field does **not** override the pinned Hub model.
Reference reproduction should accept the checkpoint defaults. Research
ablations and SDPA comparisons must alter/rebuild the relevant loaded modules
deliberately and pass an untouched-path parity test.

## GIFT-Eval evaluation contract

The official data class is
[`gift_eval.data.Dataset`](https://github.com/SalesforceAIResearch/gift-eval/blob/e8887e4bae7a8f236af0df41d48e505d075dffbd/src/gift_eval/data.py).
It reads a Hugging Face `save_to_disk` layout below the path in `GIFT_EVAL`:

```python
dataset = Dataset(name="ett1/W", term="short", to_univariate=False)
dataset.training_dataset
dataset.validation_dataset
dataset.test_data
dataset.prediction_length
dataset.freq
```

The canonical TimesFM wrapper is in Google's
[`evaluate_all_gift.py`](https://github.com/google-research/timesfm/blob/aa480150652811e732d87a3c5344b235234104e3/timesfm3-usage/benchmarks/gift_eval/evaluate_all_gift.py)
and Salesforce's
[`timesfm3.ipynb`](https://github.com/SalesforceAIResearch/gift-eval/blob/e8887e4bae7a8f236af0df41d48e505d075dffbd/notebooks/timesfm3.ipynb).
It converts model quantiles from `(V,H,Q)` to GluonTS `(Q,H,V)` and calls
`evaluate_model` with:

```python
axis=None
mask_invalid_label=True
allow_nan_forecast=False
seasonality=get_seasonality(dataset.freq)
```

Authoritative columns include `MASE[0.5]` and
`mean_weighted_sum_quantile_loss` at quantiles 0.1 through 0.9. The upstream
runner informally prints the latter as "CRPS (MWQL)"; retain the exact metric
name in artifacts. It computes per-configuration metrics over all dimensions
and windows and then reports a geometric mean over the 97 configurations.
`MSE[mean]` is not an actual model mean: `QuantileForecast` falls back to its
median when a mean forecast is absent.

### Dataset download

Full benchmark download:

```bash
export GIFT_DATA=/absolute/path/to/gift-eval-data
huggingface-cli download Salesforce/GiftEval \
  --repo-type dataset \
  --revision 30841734ac5cfddbd0c3bad6d09d2b6b32becbb0 \
  --local-dir "$GIFT_DATA"
export GIFT_EVAL="$GIFT_DATA"
```

The minimum multivariate smoke only needs the three `ett1/W` files:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Salesforce/GiftEval",
    repo_type="dataset",
    revision="30841734ac5cfddbd0c3bad6d09d2b6b32becbb0",
    allow_patterns=["ett1/W/*"],
    local_dir=os.environ["GIFT_DATA"],
)
PY
```

Validate the committed smoke contract with:

```bash
timesfm-lab validate-config configs/reproduction/teacher_smoke.yaml
```

The config selects `ett1/W/short`: two official test instances, seven target
variates, and horizon eight. It is intentionally marked non-reportable. The
smallest reportable benchmark claim still requires a preserved run artifact,
and an overall GIFT-Eval claim requires all 97 configurations and the official
aggregation.

Until the project wrapper consumes that config, the following is the smallest
standalone evaluation command verified against the pinned interfaces. It reuses
Google's predictor adapter and evaluates all two official instances of the
selected configuration with the two primary metrics; it does not write a
project result artifact.

```bash
cd "$TIMESFM_SRC"
GIFT_EVAL="$GIFT_DATA" python - <<'PY'
import os
import runpy
from pathlib import Path

from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gift_eval.data import Dataset
from timesfm3 import ModelConfig, TimesFM3Evaluator

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
root = Path(os.environ["TIMESFM_SRC"])
upstream = runpy.run_path(
    str(root / "timesfm3-usage/benchmarks/gift_eval/evaluate_all_gift.py"),
    run_name="timesfm3_gift_smoke",
)
dataset = Dataset(name="ett1/W", term="short", to_univariate=False)
teacher = TimesFM3Evaluator(ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    revision="43046b85ec22d584a13f8098c2ed39c889e129c2",
    per_core_batch_size=2,
    device="cuda",
))
metrics = [
    MASE(),
    MeanWeightedSumQuantileLoss(quantile_levels=quantiles),
]
result = evaluate_model(
    upstream["TimesFm3Predictor"](teacher, dataset.prediction_length, batch_size=2),
    test_data=dataset.test_data,
    metrics=metrics,
    batch_size=2,
    axis=None,
    mask_invalid_label=True,
    allow_nan_forecast=False,
    seasonality=get_seasonality(dataset.freq),
)
assert len(dataset.test_data) == 2
print(result[["MASE[0.5]", "mean_weighted_sum_quantile_loss"]])
PY
```

## Current blockers and upstream defects

1. The weights are public and ungated; no Hugging Face token was required in
   this audit. Unauthenticated downloads receive lower rate limits. Downloading
   or using the model still binds the user to the
   [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE).
   It permits non-commercial testing/evaluation/research, prohibits commercial
   and production use, and prohibits distributing the model or a derivative.
   Do not publish distilled checkpoints. This is a usage/license blocker for
   commercial work, not an authentication blocker.
2. The pinned full runner's diagnostic line indexes `dataset.test_data[0]`.
   With GIFT-Eval's pinned GluonTS 0.15.1, `TestData` is iterable but not
   subscriptable, so the runner fails before evaluation. The semantics-preserving
   correction is `next(iter(dataset.test_data.input))`; our harness should use
   the public iterator and must not silently change the metric protocol.
3. Google's runner README says `pip install -e . datasets gluonts`, but the
   script also imports `gift_eval` and reads
   `gift-eval/notebooks/dataset_properties.json`. The GIFT checkout/install and
   expected nested path are therefore additional undocumented requirements.
4. The published Salesforce `results/TimesFM-3/config.json` records float32 but
   no model, source, data, Torch, or CUDA revision. Google's committed result
   CSV also differs from the Salesforce leaderboard CSV. Treat both as
   externally reported reference artifacts, not exact provenance targets.

## Locally verified smoke

The pinned source and model revision were actually loaded on one RTX PRO 6000
Blackwell Max-Q GPU (compute capability 12.0, driver 580.159.03) using Python
3.12.3, Torch 2.9.0+cu128, NumPy 1.26.4, huggingface-hub 0.36.2, and
safetensors 0.5.3. A two-variate, 64-point context with horizon eight produced
a finite float32 point forecast of shape `(2, 8)` and finite, ordered float32
quantiles of shape `(2, 8, 9)`. No latency was instrumented, so no timing or
speed claim is made.

The `ett1/W/short` two-instance GluonTS path was also executed successfully
with MASE and `MeanWeightedSumQuantileLoss`. Its values are deliberately not
recorded here: this was a protocol smoke, not reportable benchmark evidence.
