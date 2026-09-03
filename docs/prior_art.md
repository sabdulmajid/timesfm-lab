# Prior-art and claim-boundary audit

**Audit cutoff:** 2026-09-03. **Project question:** when, where, and how does
TimesFM 3.0 use cross-variate information, and can a substantially smaller
native-multivariate student preserve that behavior at lower measured inference cost?

This is an early literature audit, not evidence that a proposed contribution is new. It
prioritizes papers, official repositories, model/dataset cards, and publisher records. Author
benchmark claims below are attributed and remain external until reproduced in this repository.
Searches can miss unpublished, newly released, or differently named work; the audit should be
refreshed before any paper submission. Bibliographic and immutable software identifiers are in
[`CITATIONS.md`](../CITATIONS.md), and use/distribution constraints are in
[`LICENSE_NOTES.md`](../LICENSE_NOTES.md).

## What the public record establishes

| Area | Evidence available at the cutoff | What that evidence does **not** establish |
| --- | --- | --- |
| TimesFM lineage | The ICML 2024 [TimesFM paper](https://proceedings.mlr.press/v235/das24c.html) establishes a patched decoder-style univariate foundation forecaster trained on real and synthetic time series. | It predates the TimesFM 3.0 multivariate architecture. |
| TimesFM 3.0 | Google's [release article](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [repository](https://github.com/google-research/timesfm), and [model card](https://huggingface.co/google/timesfm-3.0-pytorch) document native multiple-target, past-only-covariate, and known-future-covariate forecasting. | We found no TimesFM-3-specific archival paper or preprint. The release article and committed evaluation artifacts are not a substitute for a locally reproduced benchmark. |
| Aggregate multivariate benefit | Google's release plots report that TimesFM-3's full multivariate mode improves average rank over its univariate mode for point and probabilistic metrics on GIFT-Eval, FEV-Bench, and TIME. | An average rank does not identify which tasks benefit, whether the change is practically material, or whether the model uses genuine cross-series structure rather than shortcuts. |
| General multivariate TSFMs | [Chronos-2](https://arxiv.org/abs/2510.15821) uses group attention for related series and covariates; [Moirai](https://proceedings.mlr.press/v235/woo24a.html) is an any-variate masked encoder; [Toto](https://arxiv.org/abs/2505.14766) uses factorized space-time attention for multivariate observability data. | These architectures do not answer how TimesFM-3 itself responds to missing, stale, shuffled, or irrelevant auxiliary series. |
| Representation analysis and interventions | Wiliński et al.'s ICML 2025 [representation study](https://proceedings.mlr.press/v267/wilinski25a.html) analyzes layer similarity and learned concepts in TSFMs, uses the observed redundancy for pruning, and steers forecasts through latent interventions. | The study predates TimesFM-3 and does not isolate its variate-attention path or measure multivariate/covariate behavioral fidelity. |
| Foundation-model distillation | [DistilTS](https://arxiv.org/abs/2601.12785) presents horizon-weighted losses and temporal feature alignment for distilling TSFM teachers into smaller forecasting models. [TimeDistill](https://arxiv.org/abs/2502.15016) transfers multi-scale and multi-period information from non-foundation-model teachers to an MLP. | Neither source evaluates distillation of TimesFM-3's cross-variate response or a native-multivariate student pretrained on GiftEvalPretrain. |
| Zero-shot TSFM distillation artifact | The public [goia-forecast repository](https://github.com/gredio/goia-forecast) reports a 4.73M-parameter model trained from scratch, with ground truth throughout and auxiliary Chronos-2 quantile targets on a subset of GiftEvalPretrain batches, then evaluated on GIFT-Eval. | This is a single, non-peer-reviewed, author-reported artifact. Its current model is univariate and lists multivariate/covariate support as future work. It is prior art for zero-shot output distillation, not validation of TimesFM-3 relational distillation. |
| Efficient TSFMs | [Tiny Time Mixers](https://arxiv.org/abs/2401.03955), [Moirai 2.0](https://arxiv.org/abs/2511.11698), [Toto 2.0](https://arxiv.org/abs/2605.20119), [Reverso](https://arxiv.org/abs/2602.17634), and [Kairos](https://arxiv.org/abs/2509.25826) show several routes to smaller or more efficient pretrained forecasters. | Parameter count and paper-reported speed are not substitutes for matched hardware, shapes, precision, preprocessing, and quality gates. None demonstrates preservation of a TimesFM-3 teacher's multivariate delta. |
| TimesFM inference optimization | [TimesFM issue 457](https://github.com/google-research/timesfm/issues/457) and merged [PR 459](https://github.com/google-research/timesfm/pull/459) publicly identify and fix a no-op `torch.compile` path for TimesFM 2.5. The issue reports a 1.48x end-to-end speedup for one A100 shape after compiling `forward` directly. | This is a community report for one autoregressive TimesFM 2.5 configuration, not a general performance result and not evidence about non-autoregressive TimesFM-3 or Blackwell GPUs. |

## TimesFM 3.0 technical record

The official sources consistently describe a 330M-parameter model pretrained on more than one
trillion real and synthetic time points. The model card identifies a 20-layer stacked Mixing
Transformer with model dimension 1280, 16 attention heads, input patches of length 32, horizon
patches of length 64, and nine quantiles from 0.1 through 0.9 (median at index 4). It names
GiftEvalPretrain with FEV-Bench overlaps removed, cutoff-bounded Wikipedia pageviews and Google
Trends data, plus synthetic and augmented data. Those are release disclosures, not an independently
audited inventory of the pretraining corpus.

The [release article](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
describes tokens on a variate-by-time-patch grid. Within each layer, causal temporal attention acts
within a series, then full attention acts across variates at a fixed patch position. Contiguous
patch masking exposes known-future covariates while masking target and past-only-covariate horizons,
allowing a horizon to be decoded non-autoregressively. These details make cross-variate behavior a
natural object of study, but they do not by themselves show which information the trained model
actually uses.

Source inspection at TimesFM commit `aa480150652811e732d87a3c5344b235234104e3` adds two useful
boundaries:

- [`MixingTransformer`](https://github.com/google-research/timesfm/blob/aa480150652811e732d87a3c5344b235234104e3/src/timesfm3/transformer.py)
  applies sequence attention, optional variate attention, and a feed-forward block in each layer.
- `StackedMixingTransformer` passes one `use_variate_attention` boolean to every layer. The public
  configuration therefore exposes an all-layers switch, not a supported per-layer selector.
  Layer-group experiments require research hooks plus an untouched-path equality test.
- The current repository includes an official
  [TimesFM-3 GIFT-Eval notebook](https://github.com/google-research/timesfm/blob/aa480150652811e732d87a3c5344b235234104e3/timesfm3-usage/benchmarks/gift_eval/gift_eval_timesfm3.ipynb)
  and result artifacts. They are the implementation reference; their reported results remain
  external until rerun under pinned dependencies and revisions here.

As of the cutoff, the TimesFM-3 model card still cites the original TimesFM paper rather than a
TimesFM-3 methods paper. Searches of Google Research, arXiv, the official repository, and the model
card did not locate a separate TimesFM-3 paper. Consequently, claims beyond the code, model card,
and release disclosures should be labeled as observations from local experiments, not as details
established by a TimesFM-3 publication.

## GIFT-Eval protocol and terminology

[GIFT-Eval](https://arxiv.org/abs/2410.10393) provides a heterogeneous forecasting benchmark and a
separate [GiftEvalPretrain corpus](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain).
The paper/dataset card describes 23 datasets, while the current official repository requires 97
dataset configurations (98 CSV lines including the header). These counts refer to different units
and have also changed across public versions; reports must name the unit and pin the revision.

At repository commit `e8887e4bae7a8f236af0df41d48e505d075dffbd`, compatible evaluation uses
GluonTS `evaluate_model` with `axis=None`, invalid target labels masked, NaN forecasts rejected, and
frequency-appropriate seasonality. The canonical result fields include
`eval_metrics/MASE[0.5]` and `eval_metrics/mean_weighted_sum_quantile_loss`. Official notebooks and
the leaderboard colloquially label the latter CRPS. For an auditable report, retain the raw field,
then compute leaderboard-normalized MASE and probabilistic score relative to the pinned seasonal
naive baseline with the official aggregation; do not silently substitute another CRPS estimator.

The repository's current submission rules reserve the `zero-shot` label for models with no test
leakage that were not trained on a GIFT-Eval training split. GiftEvalPretrain is explicitly published
as a non-leaking pretraining corpus, but that statement is about benchmark construction rather than
licensing or proof against every possible source overlap. Record the complete training-data
declaration and confirm the label against the pinned rules before any leaderboard submission.

## Multivariate and compact-model context

- **Chronos-2** is a 120M-parameter encoder-only model whose group attention shares information
  among related targets and covariates. Its paper also describes a separately trained 28M variant;
  this is compact-model training, not knowledge distillation. The paper discloses partial overlap
  between the main model's training corpus and training portions of some GIFT-Eval datasets, while
  distinguishing a synthetic-only strict zero-shot variant. Comparisons must preserve those labels.
- **Moirai 1.x** established universal any-variate pretraining on LOTSA. **Moirai 2.0** replaces the
  masked encoder and mixture output with a simpler decoder, quantile loss, and multi-token
  prediction; its paper reports a small model around 11M parameters and better size/speed tradeoffs
  than the earlier large model. Treat those numbers as author-reported until rebenchmarked.
- **Toto 1.0** is a 151M decoder-only multivariate model using proportional factorized space-time
  attention. **Toto 2.0** scales one recipe from 4M to 2.5B parameters. This family is especially
  useful for a size/quality scaling control, while its observability-heavy data and architecture
  differ from TimesFM-3.
- **Tiny Time Mixers** starts near 1M parameters and supports multivariate zero/few-shot transfer;
  its cross-channel and exogenous mechanisms are not the same as TimesFM-3 variate attention.
- **Reverso** reports 0.2M/0.55M/2.6M hybrid convolutional-linear-RNN variants. **Kairos** reports
  dynamic patching and instance-adaptive positional encoding in 10M/23M/50M variants. Both are
  current preprints. Release availability and exact checkpoint license must be rechecked before a
  baseline is run.

Together these works establish that compact zero-shot forecasters and native multivariate
foundation models already exist. A smaller parameter count alone is therefore not a contribution;
the scientifically relevant comparison is matched quality, behavioral fidelity, and measured
latency/memory under a transparent protocol.

## Explicit claim boundary

### Already established in public sources

- Google reports that TimesFM-3's multivariate mode improves aggregate benchmark rank over its own
  univariate mode. This project must reproduce and disaggregate that result before treating it as a
  local finding.
- Generic TSFM representation analysis, layer-redundancy analysis, pruning, and latent steering
  exist (Wiliński et al., 2025).
- Forecasting-model and TSFM distillation exist, including horizon-aware objectives and temporal
  alignment (DistilTS; TimeDistill is adjacent cross-architecture forecasting KD).
- At least one public artifact reports zero-shot TSFM output distillation from Chronos-2
  (goia-forecast). Its evidence status and univariate scope must be stated.
- Native-multivariate and compact TSFMs exist (Chronos-2, Moirai, Toto, TTM, and newer efficiency
  preprints).
- TimesFM inference optimization through compilation has prior public work for TimesFM 2.5.

### Potential gaps to investigate, not priority claims

In the public sources searched through 2026-09-03, we did not find work that reports all of the
following for TimesFM-3. Absence from this search is not proof of priority.

- Controlled TimesFM-3-specific auxiliary-series interventions: permutation, patch-safe temporal
  shifting, channel dropout, irrelevant-channel injection, and supported covariate masking.
- Per-layer or layer-group attribution of TimesFM-3's variate-attention contribution.
- Robustness of TimesFM-3's multivariate gain to missing, shuffled, stale, or irrelevant series,
  with target history held fixed.
- A compact native-multivariate student trained to preserve TimesFM-3's response to cross-series
  context and covariates, including the relational delta between full-context and target-only
  forecasts.
- Blackwell-specific TimesFM-3 multivariate scaling over controlled variate/context/horizon/batch
  shapes with output-drift and forecast-quality gates.

These are questions for falsifiable experiments. Acceptable paper language is, for example,
“In the sources searched through 2026-09-03, we did not find prior public work evaluating …”. Do
not turn that absence-of-evidence statement into a precedence or superiority claim.

## Consequences for this project

1. Reproduce TimesFM-3 univariate and multivariate baselines before interpreting interventions.
2. Separate author-reported numbers, locally reproduced numbers, and new experimental results in
   every table and claims ledger entry.
3. Compare ground-truth-only, ordinary output KD, and multivariate-relational KD under matched
   student capacity and training budget. Ground-truth pinball loss remains present in every KD run.
4. Treat cross-series statistics as descriptive correlates, not causal explanations.
5. Benchmark the unmodified official implementation before optimizing. Do not generalize the TimesFM 2.5 A100
   compilation result to TimesFM-3 or RTX PRO 6000 Blackwell.
6. Pin code, model, dataset, seasonal-naive, and environment revisions. Preserve the official 97-row
   result contract and all failed/negative runs.
7. Keep teacher caches and trained student artifacts local under the conservative license policy in
   [`LICENSE_NOTES.md`](../LICENSE_NOTES.md).
