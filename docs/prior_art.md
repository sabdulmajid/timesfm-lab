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
| Foundation-model distillation | [DistilTS](https://arxiv.org/abs/2601.12785) presents horizon-weighted losses and temporal feature alignment for distilling TSFM teachers into smaller forecasting models. [TimeDistill](https://arxiv.org/abs/2502.15016) transfers multi-scale and multi-period information from non-foundation-model teachers to an MLP. [Guard](https://arxiv.org/abs/2606.19363) routes among TSFM teachers and gates their supervision by uncertainty for domain-specific scientific forecasting. | None of these sources reports explicitly matching a teacher's forecast change when cross-variate context is supplied versus removed. |
| Zero-shot TSFM distillation artifact | The public [goia-forecast repository](https://github.com/gredio/goia-forecast) reports a 4.73M-parameter model trained from scratch, with ground truth throughout and auxiliary Chronos-2 quantile targets on a subset of GiftEvalPretrain batches, then evaluated on GIFT-Eval. | This is a single, non-peer-reviewed, author-reported artifact. Its current model is univariate and lists multivariate/covariate support as future work. It is prior art for zero-shot output distillation, not validation of TimesFM-3 cross-variate response distillation. |
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

## Distillation audit and CVRD claim boundary

### Method definition and name

The public/research name for this project's intervention-response objective is **Cross-Variate
Response Distillation (CVRD)**. “RelKD” is not used publicly because it collides with Park et al.'s
established Relational Knowledge Distillation (RKD). Old configuration keys, checkpoint paths, and
result records may retain `relkd` as a backward-compatible historical identifier; they do not name
the method in new claims.

For the same multivariate example $x$, let `mv` mean that the forecaster receives all available
variates and `uv` mean that each target is forecast using only its own history. Define

\[
R_T(x) = T_{mv}(x) - T_{uv}(x), \qquad
R_S(x) = S_{mv}(x) - S_{uv}(x),
\]

and

\[
\mathcal{L}_{CVRD} = d\!\left(R_S(x), R_T(x)\right).
\]

Thus CVRD matches a **finite, structured intervention contrast in forecast space**. It is not a
claim that all causal effects have been identified: removing the other variates changes the
model's available context, but observational cross-series dependence and model behavior alone do
not establish a real-world causal relationship.

CVRD must be compared with a Dual-View KD control. Ordinary output KD sees only $T_{mv}$, while
both Dual-View KD and CVRD see $T_{mv}$ and $T_{uv}$. Under matched initialization, batches,
optimizer, compute, precision, validation, and evaluation, the relevant objectives are:

| Variant | Teacher supervision in addition to ground truth |
| --- | --- |
| GT | none |
| KD | $d(S_{mv}, T_{mv})$ |
| Dual-View KD | $d(S_{mv}, T_{mv}) + d(S_{uv}, T_{uv})$ |
| CVRD | the Dual-View KD terms plus $d(S_{mv}-S_{uv}, T_{mv}-T_{uv})$ |

Without Dual-View KD, an apparent CVRD gain is confounded by access to an additional teacher view.
If CVRD does not outperform Dual-View KD, the conclusion is that the explicit response-difference
term did not add useful forecast accuracy under the tested setup. Response NMAE or correlation is
a mechanism-fidelity diagnostic, not a substitute for held-out forecast quality.

### Closest and adjacent objectives

| Work | What is transferred | Why it is not CVRD |
| --- | --- | --- |
| Park et al., [Relational Knowledge Distillation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html), CVPR 2019 | Distance-wise and angle-wise relations among different training examples in the teacher representation. | CVRD uses no inter-example distance or angle geometry. Its relation is the within-example difference between forecasts under two input views. |
| Srinivas and Fleuret, [Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html), ICML 2018 | The teacher's input-output Jacobian, with an analysis connecting Jacobian matching to distillation under input noise. | A Jacobian is a local derivative with respect to input coordinates. CVRD matches a finite forecast contrast produced by one specified cross-variate-context intervention; it does not match gradients. Jacobian matching remains a plausible separate control, not an alias for CVRD. |
| Wu et al., [What Mechanisms Does Knowledge Distillation Distill?](https://proceedings.mlr.press/v243/wu24a.html), UniReps/PMLR 2024 | Shared invariant outputs under counterfactual changes to dataset latent variables; the paper studies Jacobian matching and contrastive representation learning as transfer methods. | This is the closest conceptual motivation for behavioral fidelity, but its mechanism formalism and training objectives are broader and different. CVRD directly supervises one domain-specific two-view forecast response. |
| Liu et al., [TimeKD](https://arxiv.org/abs/2505.02138), ICDE 2025 | A cross-modality teacher uses ground-truth prompts as privileged information; privileged KD includes correlation and feature distillation for multivariate forecasting. | Correlation/feature alignment and future-label privileged information differ from matching the MV-minus-UV response of one TSFM teacher. CVRD's two teacher views use the same historical example; ground truth is a separate supervised loss. |
| Li et al., [Frequency-Aligned Knowledge Distillation](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Frequency-Aligned_Knowledge_Distillation_for_Lightweight_Spatiotemporal_Forecasting_ICCV_2025_paper.html), ICCV 2025 | Multi-scale high- and low-frequency teacher features from a spectral latent space guide a lightweight spatiotemporal student. | It aligns spectral representations, not the output change caused by withholding cross-variate context. |
| Ni et al., [TimeDistill](https://arxiv.org/abs/2502.15016), KDD 2026 | Multi-scale and multi-period temporal/frequency patterns are transferred from Transformer/CNN teachers into an MLP. | Temporal differences, scales, periods, and frequency structure are not the MV-versus-UV intervention contrast. |
| Li et al., [DistilTS](https://arxiv.org/abs/2601.12785), ICASSP 2026 | Horizon-weighted forecasting objectives address unequal difficulty across forecast steps; temporal alignment addresses teacher/student architectural discrepancy. | Horizon weighting changes how forecast steps are supervised, while CVRD changes which teacher behavior is targeted. A DistilTS-inspired horizon-weighted output-KD baseline is complementary and must be independently implemented and cited. |
| Fu et al., [REDNet](https://doi.org/10.1016/j.patrec.2026.08.011), Pattern Recognition Letters 2026 | The publisher record establishes a two-stage KD method for robust edge time-series forecasting. | The accessible publisher metadata does not describe an MV-versus-UV response objective. Because full methodological text was not accessible during this audit, this is a cautious bibliographic boundary rather than a detailed technical comparison. |
| Dey et al., [Guard](https://arxiv.org/abs/2606.19363), KDD 2026 | Instance-wise routing among multiple TSFM teachers plus uncertainty-gated distillation strength for lightweight, domain-specific scientific forecasting. | It decides which teacher supervision to trust under domain shift; it does not explicitly distill one teacher's change under removal of cross-variate context. It is newer, directly relevant evidence that teacher usefulness can be heterogeneous across examples. |
| Den et al., [TIPS](https://arxiv.org/abs/2603.16985), KDD 2026 | Regime-dependent alignment to teacher models specialized for causal, local, and periodic inductive biases in financial forecasting. | TIPS combines multiple teacher priors and studies regime-dependent alignment, rather than an MV/UV view contrast. It is relevant to response heterogeneity and financial forecasting but is not TSFM compression. |

The distinctions above concern the stated objectives, not precedence over an entire design space.
Finite-difference behavior matching, multi-view learning, privileged-information KD, feature and
correlation alignment, temporal/frequency KD, and TSFM compression are established areas. **To our
knowledge, we did not find prior work explicitly distilling the change in a TSFM's forecast induced
by providing versus removing cross-variate context.** This is a bounded search finding, not a claim
that the idea is unprecedented, and it must be refreshed before publication.

## Explicit claim boundary

### Already established in public sources

- Google reports that TimesFM-3's multivariate mode improves aggregate benchmark rank over its own
  univariate mode. This project must reproduce and disaggregate that result before treating it as a
  local finding.
- Generic TSFM representation analysis, layer-redundancy analysis, pruning, and latent steering
  exist (Wiliński et al., 2025).
- Forecasting-model and TSFM distillation exist, including privileged correlation/feature transfer,
  frequency and temporal alignment, horizon-aware objectives, teacher routing, and robust edge
  deployment (TimeKD, frequency-aligned KD, TimeDistill, DistilTS, Guard, and REDNet).
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
  context and covariates, including the CVRD contrast between full-context and target-only
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
3. Compare ground-truth-only, ordinary output KD, Dual-View KD, and CVRD under matched student
   architecture, initialization, sample order, optimizer, compute, precision, validation, and
   evaluation. Ground-truth pinball loss remains present in every KD run. CVRD must beat Dual-View
   KD before attributing a forecast-quality gain to the explicit response-difference term.
4. Treat cross-series statistics as descriptive correlates, not causal explanations.
5. Benchmark the unmodified official implementation before optimizing. Do not generalize the TimesFM 2.5 A100
   compilation result to TimesFM-3 or RTX PRO 6000 Blackwell.
6. Pin code, model, dataset, seasonal-naive, and environment revisions. Preserve the official 97-row
   result contract and all failed/negative runs.
7. Keep teacher caches and trained student artifacts local under the conservative license policy in
   [`LICENSE_NOTES.md`](../LICENSE_NOTES.md).
