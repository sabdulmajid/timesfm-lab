# Citation and source ledger

**Verified through:** 2026-09-03. Prefer the archival/publisher record where one exists; use the
arXiv identifier for newer preprints. URLs below point to primary sources. Recheck moving revisions
before a release, benchmark run, or paper submission.

## Papers and official technical releases

### TimesFM and TimesFM 3.0

- Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. “A decoder-only foundation model for
  time-series forecasting.” ICML 2024, PMLR 235:10148–10167.
  [PMLR](https://proceedings.mlr.press/v235/das24c.html) ·
  [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- Ayush Jain and Rajat Sen. “TimesFM-3: A zero-shot foundation model for multivariate forecasting.”
  Google Research, 2026-08-31. This is an official release article, not an archival paper.
  [Release article](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)

```bibtex
@inproceedings{das2024timesfm,
  title     = {A decoder-only foundation model for time-series forecasting},
  author    = {Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {10148--10167},
  year      = {2024},
  url       = {https://proceedings.mlr.press/v235/das24c.html}
}

@misc{jain2026timesfm3,
  title        = {{TimesFM-3}: A zero-shot foundation model for multivariate forecasting},
  author       = {Jain, Ayush and Sen, Rajat},
  year         = {2026},
  month        = aug,
  howpublished = {Google Research Blog},
  url          = {https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/},
  note         = {Official release article; accessed 2026-09-03}
}
```

### Evaluation and pretraining data

- Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and
  Doyen Sahoo. “GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation.”
  NeurIPS 2024 Workshop on Time Series in the Age of Large Models.
  [arXiv:2410.10393](https://arxiv.org/abs/2410.10393) ·
  [workshop record](https://neurips.cc/virtual/2024/103025) ·
  [OpenReview](https://openreview.net/forum?id=9EBSEkFSje)

```bibtex
@misc{aksu2024gifteval,
  title         = {{GIFT-Eval}: A Benchmark For General Time Series Forecasting Model Evaluation},
  author        = {Aksu, Taha and Woo, Gerald and Liu, Juncheng and Liu, Xu and Liu, Chenghao and
                   Savarese, Silvio and Xiong, Caiming and Sahoo, Doyen},
  year          = {2024},
  eprint        = {2410.10393},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2410.10393}
}
```

### Representation analysis and distillation

- Michał Wiliński, Mononito Goswami, Willa Potosnak, Nina Żukowska, and Artur Dubrawski.
  “Exploring Representations and Interventions in Time Series Foundation Models.” ICML 2025,
  PMLR 267:66861–66886.
  [PMLR](https://proceedings.mlr.press/v267/wilinski25a.html) ·
  [arXiv:2409.12915](https://arxiv.org/abs/2409.12915)
- Yuqi Li, Kuiye Ding, Chuanguang Yang, Szu-Yu Chen, and Yingli Tian. “Distilling Time Series
  Foundation Models for Efficient Forecasting.” ICASSP 2026, paper 9809.
  [DOI:10.1109/ICASSP55912.2026.11460474](https://doi.org/10.1109/ICASSP55912.2026.11460474) ·
  [arXiv:2601.12785](https://arxiv.org/abs/2601.12785) ·
  [conference record](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=9809&bare=1)
- Juntong Ni, Zewen Liu, Shiyu Wang, Ming Jin, and Wei Jin. “TimeDistill: Efficient Long-Term Time
  Series Forecasting with MLP via Cross-Architecture Distillation.” KDD 2026.
  [DOI:10.1145/3770854.3780218](https://doi.org/10.1145/3770854.3780218) ·
  [arXiv:2502.15016](https://arxiv.org/abs/2502.15016)

```bibtex
@inproceedings{wilinski2025representations,
  title     = {Exploring Representations and Interventions in Time Series Foundation Models},
  author    = {Wili\'{n}ski, Micha{\l} and Goswami, Mononito and Potosnak, Willa and
               {\.Z}ukowska, Nina and Dubrawski, Artur},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {66861--66886},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/wilinski25a.html}
}

@inproceedings{li2026distilts,
  title     = {Distilling Time Series Foundation Models for Efficient Forecasting},
  author    = {Li, Yuqi and Ding, Kuiye and Yang, Chuanguang and Chen, Szu-Yu and Tian, Yingli},
  booktitle = {2026 IEEE International Conference on Acoustics, Speech and Signal Processing},
  year      = {2026},
  doi       = {10.1109/ICASSP55912.2026.11460474},
  url       = {https://arxiv.org/abs/2601.12785}
}

@inproceedings{ni2026timedistill,
  title     = {{TimeDistill}: Efficient Long-Term Time Series Forecasting with {MLP} via
               Cross-Architecture Distillation},
  author    = {Ni, Juntong and Liu, Zewen and Wang, Shiyu and Jin, Ming and Jin, Wei},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770854.3780218},
  url       = {https://arxiv.org/abs/2502.15016}
}
```

### Multivariate and compact foundation models

- Abdul Fatir Ansari et al. “Chronos-2: From Univariate to Universal Forecasting.”
  [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo.
  “Unified Training of Universal Time Series Forecasting Transformers.” ICML 2024.
  [PMLR](https://proceedings.mlr.press/v235/woo24a.html) ·
  [arXiv:2402.02592](https://arxiv.org/abs/2402.02592)
- Chenghao Liu et al. “Moirai 2.0: When Less Is More for Time Series Forecasting.”
  [arXiv:2511.11698](https://arxiv.org/abs/2511.11698)
- Ben Cohen et al. “This Time is Different: An Observability Perspective on Time Series Foundation
  Models.” [arXiv:2505.14766](https://arxiv.org/abs/2505.14766)
- Emaad Khwaja et al. “Toto 2.0: Time Series Forecasting Enters the Scaling Era.”
  [arXiv:2605.20119](https://arxiv.org/abs/2605.20119)
- Vijay Ekambaram et al. “Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot
  Forecasting of Multivariate Time Series.” NeurIPS 2024.
  [arXiv:2401.03955](https://arxiv.org/abs/2401.03955)
- Xinghong Fu, Yanhong Li, Georgios Papaioannou, and Yoon Kim. “Reverso: Efficient Time Series
  Foundation Models for Zero-shot Forecasting.” Preprint.
  [arXiv:2602.17634](https://arxiv.org/abs/2602.17634)
- Kun Feng et al. “Kairos: Toward Adaptive and Parameter-Efficient Time Series Foundation Models.”
  Preprint. [arXiv:2509.25826](https://arxiv.org/abs/2509.25826)

```bibtex
@misc{ansari2025chronos2,
  title         = {{Chronos-2}: From Univariate to Universal Forecasting},
  author        = {Ansari, Abdul Fatir and Shchur, Oleksandr and K{\"u}ken, Jaris and Auer, Andreas
                   and Han, Boran and Mercado, Pedro and Rangapuram, Syama Sundar and others},
  year          = {2025},
  eprint        = {2510.15821},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2510.15821}
}

@inproceedings{woo2024moirai,
  title     = {Unified Training of Universal Time Series Forecasting Transformers},
  author    = {Woo, Gerald and Liu, Chenghao and Kumar, Akshat and Xiong, Caiming and
               Savarese, Silvio and Sahoo, Doyen},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {53140--53164},
  year      = {2024},
  url       = {https://proceedings.mlr.press/v235/woo24a.html}
}

@misc{liu2025moirai2,
  title         = {{Moirai 2.0}: When Less Is More for Time Series Forecasting},
  author        = {Liu, Chenghao and Aksu, Taha and Liu, Juncheng and Liu, Xu and Yan, Hanshu and
                   Pham, Quang and Savarese, Silvio and Sahoo, Doyen and Xiong, Caiming and Li, Junnan},
  year          = {2025},
  eprint        = {2511.11698},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2511.11698}
}

@misc{cohen2025toto,
  title         = {This Time is Different: An Observability Perspective on Time Series Foundation Models},
  author        = {Cohen, Ben and Khwaja, Emaad and Doubli, Youssef and Lemaachi, Salahidine and
                   Lettieri, Chris and Masson, Charles and others},
  year          = {2025},
  eprint        = {2505.14766},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2505.14766}
}

@misc{khwaja2026toto2,
  title         = {{Toto 2.0}: Time Series Forecasting Enters the Scaling Era},
  author        = {Khwaja, Emaad and Lettieri, Chris and Woo, Gerald and Belouadah, Eden and
                   Cenac, Marc and Jarry, Guillaume and Paquin, Enguerrand and Zhao, Xunyi and others},
  year          = {2026},
  eprint        = {2605.20119},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.20119}
}

@inproceedings{ekambaram2024ttm,
  title     = {Tiny Time Mixers ({TTMs}): Fast Pre-trained Models for Enhanced Zero/Few-Shot
               Forecasting of Multivariate Time Series},
  author    = {Ekambaram, Vijay and Jati, Arindam and Dayama, Pankaj and Mukherjee, Sumanta and
               Nguyen, Nam H. and Gifford, Wesley M. and Reddy, Chandra and Kalagnanam, Jayant},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024},
  eprint    = {2401.03955},
  url       = {https://arxiv.org/abs/2401.03955}
}

@misc{fu2026reverso,
  title         = {Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting},
  author        = {Fu, Xinghong and Li, Yanhong and Papaioannou, Georgios and Kim, Yoon},
  year          = {2026},
  eprint        = {2602.17634},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2602.17634}
}

@misc{feng2026kairos,
  title         = {Kairos: Toward Adaptive and Parameter-Efficient Time Series Foundation Models},
  author        = {Feng, Kun and Lan, Shaocheng and Fang, Yuchen and He, Wenchao and Lu, Sihan and
                   Gu, Shuqi and Ma, Lintao and Lu, Xingyu and Ren, Kan},
  year          = {2026},
  eprint        = {2509.25826},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2509.25826}
}
```

## Software, model, and dataset records

The snapshot column records the exact public state inspected for this audit. A license in this table
describes that particular artifact, not every associated paper, package, checkpoint, or underlying
dataset.

| Artifact | Official record | Snapshot inspected | Declared license / status |
| --- | --- | --- | --- |
| TimesFM source | [google-research/timesfm](https://github.com/google-research/timesfm) | `aa480150652811e732d87a3c5344b235234104e3` | Apache-2.0 source; README says weights through 2.5 are Apache-2.0 |
| TimesFM 3.0 checkpoint | [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch) | `43046b85ec22d584a13f8098c2ed39c889e129c2` | TimesFM Non-Commercial License v1.0; **not** Apache-2.0 |
| GIFT-Eval code/protocol/results | [SalesforceAIResearch/gift-eval](https://github.com/SalesforceAIResearch/gift-eval) | `e8887e4bae7a8f236af0df41d48e505d075dffbd` | Apache-2.0 |
| GIFT-Eval leaderboard/aggregation | [`Salesforce/GIFT-Eval` Space](https://huggingface.co/spaces/Salesforce/GIFT-Eval) | `b6cb73323064b1c114680e12b4455a262d8d41e8` | Public implementation used to verify seasonal-naive normalization and geometric aggregation; recheck its own repository metadata before copying code |
| GIFT-Eval data | [`Salesforce/GiftEval`](https://huggingface.co/datasets/Salesforce/GiftEval) | `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0` | Card metadata: Apache-2.0; constituent-data terms still require review |
| Pretraining data | [`Salesforce/GiftEvalPretrain`](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) | `6830b624de7ed2b3d3e5b85bb6959d81dcc5d874` | Card metadata: Apache-2.0; constituent-data terms still require review |
| TSFM representation code | [representations-in-tsfms](https://github.com/moment-timeseries-foundation-model/representations-in-tsfms) | `72337c6e4072d4c213588c116633be71a1b87760` | MIT |
| DistilTS code | [DistilTS-ICASSP2026](https://github.com/itsnotacie/DistilTS-ICASSP2026) | `0f6982a9606245747f82e5101884fdfb6e3ecafd` | No license file or GitHub license declaration found; do not copy code |
| goia-forecast code | [gredio/goia-forecast](https://github.com/gredio/goia-forecast) | `b04d3390dbccaf4caa2127875f294d056a2e828b` | MIT; results are repository-author reports |
| goia checkpoint | [`gredio/goia-forecast-nano-v0`](https://huggingface.co/gredio/goia-forecast-nano-v0) | `a3d17d826118003e8e5f54addfc04fe39e966398` | MIT metadata |
| Chronos code | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | `8589d1988e9676817548e9626738ff06b6ca6370` | Apache-2.0 |
| Chronos-2 checkpoint | [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) | `29ec3766d36d6f73f0696f85560a422f50e8498c` | Apache-2.0 metadata |
| Moirai source | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | `cfd46d4510ed8896f263116f32928eede05b0a75` | Apache-2.0 source |
| Moirai 2.0 R-small checkpoint | [`Salesforce/moirai-2.0-R-small`](https://huggingface.co/Salesforce/moirai-2.0-R-small) | `30f43ff08c8494f4943ae1521e9d4e94a0fbb389` | CC-BY-NC-4.0 metadata; do not infer Apache from the code repository |
| Toto code | [DataDog/toto](https://github.com/DataDog/toto) | `46bd92adeeef6b3c2afb21008659f607fa203e38` | Apache-2.0 |
| Toto 2.0 2.5B checkpoint | [`Datadog/Toto-2.0-2.5B`](https://huggingface.co/Datadog/Toto-2.0-2.5B) | `51a2812bbe449437c01b79c0e425ed578f335f5b` | Apache-2.0 metadata |
| Granite TSFM / TTM code | [ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm) | `fe7a35697723e2a2f5246ae979474bfc554e26c0` | Apache-2.0 |
| TTM R2 enterprise checkpoint | [`ibm-granite/granite-timeseries-ttm-r2`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) | `d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f` | Apache-2.0 metadata |
| Reverso code / checkpoint | [code](https://github.com/shinfxh/reverso) · [model](https://huggingface.co/shinfxh/reverso) | code `35f724d3ccf26527f17bc774e897ff0c07d0761a`; model `a0eaa2d3a8e1fa200c7610584cd66dea89f575b1` | MIT |
| Kairos code | [foundation-model-research/Kairos](https://github.com/foundation-model-research/Kairos) | `0322393840ccf6e2bfe9c663f9dcd088a5a7ee07` | Apache-2.0 |
| TimeDistill code | [LingFengGold/TimeDistill](https://github.com/LingFengGold/TimeDistill) | `08dedd3ebdf2cca0a2a9494111186bf090fde1d8` | Apache-2.0 |

## Public implementation/performance records

- TimesFM 2.5 compilation investigation:
  [issue 457](https://github.com/google-research/timesfm/issues/457) and
  [merged PR 459](https://github.com/google-research/timesfm/pull/459). This is useful prior art but
  is a single-shape community measurement, not a paper or a TimesFM-3/Blackwell result.
- TimesFM-3 implementation reference:
  [`src/timesfm3/transformer.py`](https://github.com/google-research/timesfm/blob/aa480150652811e732d87a3c5344b235234104e3/src/timesfm3/transformer.py),
  [`config.json`](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/config.json), and the official
  [GIFT-Eval benchmark directory](https://github.com/google-research/timesfm/tree/aa480150652811e732d87a3c5344b235234104e3/timesfm3-usage/benchmarks/gift_eval).
- goia-forecast is a software artifact rather than a reviewed publication. Cite its immutable
  repository commit and describe all results as self-reported.

## Citation hygiene

- Use “TimesFM 3.0” for the checkpoint and “TimesFM-3” when following Google's release title;
  record the exact model revision either way.
- Distinguish a dataset from a dataset configuration. At the pinned GIFT-Eval repository revision,
  the submission file contains 97 configurations plus one header line.
- Distinguish the raw GluonTS `mean_weighted_sum_quantile_loss` field from the seasonal-naive-
  normalized aggregate that the leaderboard calls CRPS.
- Never transfer a repository's license to an associated checkpoint or dataset without checking
  that artifact's own license. The TimesFM and Moirai records above demonstrate why.
- Label release-blog, model-card, repository, and preprint claims by source; only locally reproduced
  results belong in this project's empirical claims ledger.
