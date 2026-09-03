# License and artifact-handling notes

**Reviewed:** 2026-09-03. This is an engineering audit, not legal advice. It records the public
terms inspected for this non-commercial research project and adopts a conservative operating
policy. Obtain qualified legal review before redistributing any trained artifact or using any part
of this work in a commercial, production, client, or revenue-linked setting.

## The critical license split

The [google-research/timesfm source repository](https://github.com/google-research/timesfm) is
Apache-2.0. Its README states that pretrained weights through TimesFM 2.5 also remain Apache-2.0.
The TimesFM 3.0 checkpoint is different: the official
[`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch) repository uses
Google's [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
Do not describe the TimesFM 3.0 weights as Apache-licensed, open source, or commercially usable.

The exact checkpoint revision reviewed here is
`43046b85ec22d584a13f8098c2ed39c889e129c2`. Re-review the license if the model revision or download
source changes.

## What the TimesFM 3.0 license says

The following is a summary; the linked license controls.

- Accessing, downloading, using, distributing, or creating a “Derivative” triggers the license.
- The grant is limited to non-commercial, non-production testing, evaluation, and research. The
  definition permits internal benchmarking, academic research, and experiments on public or
  private data only when the results are not used for commercial decisions, client deliverables,
  or paid products/services.
- Commercial or production activities are outside the grant. The definition specifically excludes
  training, fine-tuning, or distilling another model **for commercial use**.
- A “Derivative” is defined unusually broadly: it includes customized, fine-tuned, retrained, or
  adapted versions; works incorporating, utilizing, based on, or derived from the TimesFM model,
  its logic, or parameters; and other derivative works regardless of copyright-law status.
- Predictions, scores, probabilities, recommendations, explanations, and other results generated
  from user-supplied inputs are “Outputs,” and Outputs are expressly not Derivatives. However, the
  license does **not** expressly state that a model trained from Outputs is outside its broad
  Derivative definition.
- Distribution of the TimesFM Model or a Derivative is prohibited. Model/Derivative/Output use or
  distribution for commercial or production purposes is also prohibited.
- The grant is non-transferable, non-sublicensable, revocable, limited, and does not grant Google
  trademark rights. Data-protection, privacy, export-control, and applicable AI-law compliance
  remain the user's responsibility.
- The license acknowledges that some elements may be available under separate licenses. The
  Apache-2.0 source repository is one such separately licensed artifact; its license remains
  relevant to that code.

### Consequence for a distilled student

Outputs themselves are excluded from the definition of Derivative, but the separate broad language
covering works based on the model, its logic, or parameters creates material uncertainty for a
student trained under TimesFM-3 supervision. The project specification therefore adopts the safer
rule without asserting a legal conclusion: **any checkpoint, adapter, optimizer state, teacher
cache, or other trained parameter artifact produced with TimesFM-3 supervision is local-only and
must not be committed, released, mirrored, or uploaded.** A from-scratch initialization does not
change this operational rule.

Independent source code, architecture descriptions, experiment configurations, aggregate metrics,
and figures may be published under this repository's Apache-2.0 license when they contain no Google
weights, restricted artifacts, or copied code lacking required notices. Aggregate research results
should be sufficient for the public artifact; raw per-example teacher outputs remain local.

## Permitted and prohibited project artifacts

| Artifact | Repository policy | Reason / condition |
| --- | --- | --- |
| Original project source, tests, configs, plotting code | May commit and publish | Must be independently authored or comply with every incorporated dependency license. The repository's Apache-2.0 license does not relicense third-party material. |
| TimesFM repository code copied or modified here | Avoid when a wrapper or independent implementation suffices; otherwise preserve license/notices and mark changes | Upstream source is Apache-2.0, including attribution and NOTICE obligations where applicable. |
| TimesFM 3.0 checkpoint/config downloaded for a run | Local cache only; never commit or mirror | Separate non-commercial license; pin revision and access only through an authorized researcher/entity. |
| Teacher predictions and precomputed cache shards | Local, access-controlled, non-commercial research only; never commit raw caches | Outputs are not Derivatives, but commercial/production use is prohibited and caches can encode source data. Aggregate metrics are the intended public result. |
| Student weights, adapters, optimizer/scaler state, resumable checkpoints | Never commit, publish, attach to a release, or upload | Conservative treatment under the license's broad Derivative language and the project requirement. Obtain legal review before any external transfer. |
| Student architecture description and independently written implementation | May commit and publish | Must not embed Google parameters or restricted artifacts. The broad “logic” wording is a reason to seek review if implementation closely tracks protected material. |
| Aggregate evaluation CSV/JSON, plots, tables, run metadata | May commit and publish for research | Strip raw teacher outputs and private/source records; label author-reported versus locally reproduced values. |
| GIFT-Eval / GiftEvalPretrain dataset files | Never commit or mirror | Large external datasets; retain local provenance and honor constituent source terms. Commit only identifiers, loaders, checksums/metadata where appropriate. |
| Third-party baseline weights | Local by default; review before redistribution | Source-code and checkpoint licenses may differ. Download only exact audited revisions. |

The repository should ignore at least model weights/checkpoints, adapters, optimizer states,
teacher-cache shards, and raw datasets. A clean-tree check is required before every push and release.

## Other audited assets

| Asset | Declared terms at audited revision | Practical note |
| --- | --- | --- |
| This repository | Apache-2.0 | Applies only to material the repository's contributors have the right to license. |
| TimesFM source, commit `aa480150…` | Apache-2.0 | Preserve copyright, license, attribution/NOTICE material, and mark modified files when redistributing upstream-derived code. |
| TimesFM weights through 2.5 | Apache-2.0 per official repository README | Verify the exact checkpoint card before use; this does not change TimesFM 3.0's license. |
| TimesFM 3.0 model, SHA `43046b85…` | TimesFM Non-Commercial License v1.0 | Non-commercial/non-production use only; no distribution of the model or Derivatives. |
| GIFT-Eval code, commit `e8887e4…` | Apache-2.0 | Preserve notices if code is copied; prefer depending on/pinning upstream. |
| `Salesforce/GiftEval`, SHA `3084173…` | Hugging Face card metadata: Apache-2.0 | The card also frames the release as research-oriented. Audit constituent dataset provenance before redistribution or non-research use. |
| `Salesforce/GiftEvalPretrain`, SHA `6830b62…` | Hugging Face card metadata: Apache-2.0 | “Non-leaking” describes benchmark design, not a license warranty. Do not redistribute a compiled corpus without a source-by-source rights review. |
| DistilTS code, commit `0f6982a…` | **No license found** in the repository snapshot | Publicly visible is not permission to copy, modify, or redistribute. Use the paper as a methodological reference and implement ideas independently. |
| TSFM representation-analysis code, commit `72337c6…` | MIT | Retain its notice if code is incorporated. |
| Chronos-2 code/checkpoint, audited snapshots | Apache-2.0 | Preserve required notices. This permissive teacher is not evidence that TimesFM-3 permits the same uses. |
| goia-forecast code/checkpoint, audited snapshots | MIT | Retain notice; results are author-reported. Its license analysis does not govern this project. |
| Moirai `uni2ts` source | Apache-2.0 | Checkpoint terms differ. The audited Moirai 2.0 R-small card is CC-BY-NC-4.0. |
| Moirai 2.0 R-small weights, SHA `30f43ff…` | CC-BY-NC-4.0 | Non-commercial baseline only; satisfy attribution and do not assume the code repository's Apache license applies. |
| Toto source / audited Toto weights | Apache-2.0 | Verify the exact checkpoint and retain notices. |
| Granite TSFM / audited TTM R2 enterprise weights | Apache-2.0 | The TTM paper distinguishes research and enterprise model releases; use the specifically audited artifact. |
| Reverso code/checkpoint | MIT at audited snapshots | Only the currently available checkpoint is covered by the inspected card; recheck future releases. |
| Kairos and TimeDistill code | Apache-2.0 at audited commits | Preserve license and notice requirements; papers remain the preferred source for methodological ideas. |

Hugging Face metadata and repository license detection are useful evidence, not a warranty that all
training data or bundled dependencies are covered. Keep a per-baseline record of model SHA, code
commit, license text hash or URL, acceptance identity, and data terms.

## Required operating controls

1. **Acceptance authority:** only a person authorized to bind the relevant individual or entity may
   accept the TimesFM license and download/use the checkpoint. Do not accept on an employer's behalf
   without authority.
2. **Research-only environment:** no production endpoint, end-user interaction, paid service,
   client deliverable, commercial decision, or revenue-linked workflow may use TimesFM-3, its
   Derivatives, or its Outputs under the reviewed terms.
3. **Immutable inputs:** record the TimesFM code commit, model SHA, license URL, GIFT-Eval code
   commit, dataset SHAs, dependency lock, and access date in every run.
4. **Local artifact boundary:** store external weights, raw datasets, per-example forecasts, cache
   shards, and student checkpoints outside Git-tracked paths or under explicit ignore rules. Use
   least-privilege filesystem permissions for caches.
5. **Public-output review:** before a push or release, inspect staged files and Git history for
   safetensors, PyTorch/JAX checkpoints, adapters, cache arrays, dataset shards, raw teacher outputs,
   credentials, and third-party notices.
6. **Independent implementation:** cite papers for methodological inspiration. Do not copy from an
   unlicensed repository (notably the audited DistilTS snapshot). For incorporated Apache/MIT code,
   preserve required notices and mark upstream modifications.
7. **Data provenance:** retain a manifest for every GiftEvalPretrain constituent actually sampled.
   The aggregate Apache metadata and non-leakage claim do not replace upstream terms, privacy
   review, or benchmark-contamination checks.
8. **Baseline review:** check code and weight licenses separately before downloading or running any
   new model. Non-commercial checkpoints must not enter a commercial or production workflow.
9. **No trademark implication:** describe compatibility and citations factually; do not imply Google,
   Salesforce, Amazon, Datadog, IBM, or any author endorses this project.
10. **Escalate before distribution:** legal approval and, where needed, a commercial license from
    Google are prerequisites to any external transfer of a TimesFM-3-supervised trained artifact or
    any commercial/production use.

## Release checklist

- [ ] `git diff --cached --name-only` contains no weight, cache, raw dataset, or teacher-output file.
- [ ] Repository history contains no accidentally committed restricted binary.
- [ ] All reported model/dataset revisions match `CITATIONS.md` and run metadata.
- [ ] Each borrowed code fragment has its license, attribution, and modification notice.
- [ ] Public tables contain aggregate research results, not raw per-example teacher outputs.
- [ ] Trained student and resume artifacts remain local and are absent from releases and registries.
- [ ] No wording markets the repository as a production/commercial TimesFM-3 serving system.
- [ ] License terms have been rechecked if an upstream revision, model, or data source changed.
