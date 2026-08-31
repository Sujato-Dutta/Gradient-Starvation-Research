# ICLR 2027 editorial review resolution

## Current status

Updated 2026-08-31 for the user-facing manuscript rewrite. Scientific authority remains `paper/main.tex` and the claim/theorem records under `research_scope/`; this document records editorial resolution and the current build only.

## Review requests and resolutions

| Review request | Status | Resolution |
|---|---|---|
| Keep the title and central thesis | **Resolved** | The paper remains centered on **“Weak Learning Failure Alone Is Not Gradient Starvation.”** |
| Compress the abstract | **Resolved** | The abstract is 182 words and states the causal test, exact dynamics, regime-dependent evidence, and prediction/intervention limits without an exhaustive result inventory. |
| Make the paper read as four findings | **Resolved** | The introduction explicitly presents: matched causal attribution; exact finite-width dynamics; regime-dependent rather than universal starvation; and local or conditional prediction/intervention. |
| Remove audit-report presentation | **Resolved** | Internal study IDs, repository state, hashes, source-history discussion, artifact machinery, software contracts, and implementation incidents were removed from the manuscript. Scientific protocols, assumptions, proofs, failures, and negative/mixed outcomes remain. |
| Do not make the held-out tangent study the centerpiece | **Resolved** | It appears as one restricted event-classification result. The main narrative integrates the controlled linear evidence, exact theory, nonlinear counterexamples, long-lag and generated-cue gate failures, and local control theorem. |
| Preserve negative evidence | **Resolved** | Lag-8, nonlinear gate failures, the failed broad tangent pilot, generated-cue failures, the negative threshold-robust endpoint, and the comparator-restricted CDC result remain explicit. |
| Keep theorem limits precise | **Resolved** | The paper makes no global positive-lag predictor, trained-network tangent exactness, universal prevalence or architecture ranking, canonical comparator parity, or broad CDC-superiority claim. |
| Use a minimal reproducibility statement | **Resolved** | The paper states that code, configurations, and relevant files will be made available upon publication to facilitate reproducibility. |
| Fit the ICLR main-paper limit | **Resolved** | In the official style, `page:main-end` is page 5; references, disclosures, and appendix follow. |

## Scientific boundaries retained

1. A weak response that fails to learn is not, by itself, evidence of gradient starvation.
2. Causal starvation requires outcome suppression and a nondegenerate weak-only first hit under the stated target and horizon.
3. A failed weak-only gate is indeterminate: it proves neither starvation nor its absence.
4. The exact finite-width equations separate cross-entropy weighting from response-specific geometry, but they do not close a general recurrent mean-field theory.
5. The initialization criterion and tube/rank-one results are local or assumption-scoped; the general positive-lag parameter-level predictor remains open.
6. The frozen tangent surrogate is exact only for its own model. Held-out event agreement is not calibrated dynamics, causal prediction, or a kernel-stability proof.
7. CDC has fixed-chart, feasible, uncapped, target-attaining minimum-norm optimality and a local one-step guarantee only. The adapted-comparator evidence is mixed and establishes no broad superiority.
8. Fixed cells and seed grids do not establish universal prevalence or architecture rankings.

## Current build snapshot

- Canonical source: `paper/main.tex`, SHA-256 `9f9b4f94d9eb12af0d070b2f86d5d73b05f5368c53f3ce7442aea06667e92061`.
- Reviewer PDF: `paper/iclr2027/submission.pdf`, 26 pages, 513,075 bytes, SHA-256 `5efb77bdf94786549794ee67b2682779089fe64831fa7ca9652e0ae04eedf3e0`.
- Detailed rendering: `paper/main.pdf`, 24 pages, 551,217 bytes, SHA-256 `b9e21effa326d07a8f99b2ed3aa71ca1929c644d35715f7a7ab71d809bdc4372`.
- Official-style main-text boundary: page 5.
- Build diagnostics: no overfull boxes, unresolved references, or unresolved citations. The remaining notices are non-blocking underfull page/bibliography spacing messages.
- Full software suite: 330 tests passed with four pre-existing plotting warnings.
- Release integrity: four archives, five studies, 67 files, and 3,510,433 bytes verified with Git tracking required.
- PDF audit: required title/disclosure text is present; internal study IDs, provenance terms, local paths, and missing-figure fallback text are absent from both renderings; title metadata is empty and the author presentation is anonymous.
- Independent semantic review: no blocker, major, or minor findings; `BLOCKER VERDICT: NONE`.
- Readiness consistency check: manifest validation passes; external and historical submission blockers remain separately recorded in `research_scope/submission_readiness.yaml`.

The PDF to share for scientific review is `paper/iclr2027/submission.pdf`.