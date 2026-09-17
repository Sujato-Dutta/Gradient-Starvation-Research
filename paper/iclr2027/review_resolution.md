# ICLR 2027 editorial review resolution

## Current status

Updated 2026-09-17 after the frozen eight-block CIFAR-10 confirmation and final clean-clone release audit. Scientific authority remains `paper/main.tex` and the claim/theorem records under `research_scope/`; this document records editorial resolution and the current build only.

## Review requests and resolutions

| Review request | Status | Resolution |
|---|---|---|
| Keep the title and central thesis | **Resolved** | The paper remains centered on **“Weak Learning Failure Alone Is Not Gradient Starvation.”** |
| Compress the abstract | **Resolved** | The abstract is 182 words and states the causal test, exact dynamics, regime-dependent evidence, and prediction/intervention limits without an exhaustive result inventory. |
| Make the paper read as four findings | **Resolved** | The introduction explicitly presents: matched causal attribution; exact finite-width dynamics; regime-dependent rather than universal starvation; and local or conditional prediction/intervention. |
| Remove audit-report presentation | **Resolved** | Internal study IDs, repository state, hashes, source-history discussion, artifact machinery, software contracts, and implementation incidents were removed from the manuscript. Scientific protocols, assumptions, proofs, failures, and negative/mixed outcomes remain. |
| Do not make the held-out tangent study the centerpiece | **Resolved** | It appears as one restricted event-classification result. The main narrative integrates the controlled linear evidence, exact theory, nonlinear counterexamples, long-lag and generated-cue gate failures, and local control theorem. |
| Preserve negative and confirmatory evidence | **Resolved** | Lag-8, nonlinear gate failures, the failed broad tangent pilot, generated-cue failures, the negative threshold-robust endpoint, and the comparator-restricted CDC result remain explicit. The frozen CIFAR result is presented as a controlled existence result, not prevalence or universality. |
| Keep theorem limits precise | **Resolved** | The paper makes no global positive-lag predictor, trained-network tangent exactness, universal prevalence or architecture ranking, canonical comparator parity, or broad CDC-superiority claim. |
| Use a minimal reproducibility statement | **Resolved** | The paper states that code, configurations, and relevant files will be made available upon publication to facilitate reproducibility. |
| Fit the ICLR main-paper limit | **Resolved** | In the official style, the main text ends on page 9; references begin on page 10, followed by disclosures and appendix. |

## Scientific boundaries retained

1. A weak response that fails to learn is not, by itself, evidence of gradient starvation.
2. Causal starvation requires outcome suppression and a nondegenerate weak-only first hit under the stated target and horizon.
3. A failed weak-only gate is indeterminate: it proves neither starvation nor its absence.
4. The exact finite-width equations separate cross-entropy weighting from response-specific geometry, but they do not close a general recurrent mean-field theory.
5. The initialization criterion and tube/rank-one results are local or assumption-scoped; the general positive-lag parameter-level predictor remains open.
6. The frozen tangent surrogate is exact only for its own model. Held-out event agreement is not calibrated dynamics, causal prediction, or a kernel-stability proof.
7. CDC has fixed-chart, feasible, uncapped, target-attaining minimum-norm optimality and a local one-step guarantee only. The adapted-comparator evidence is mixed and establishes no broad superiority.
8. Fixed cells and seed grids do not establish universal prevalence or architecture rankings.
9. The CIFAR-10 confirmation establishes the preregistered certificate for one artificial ribbon cue and one CNN architecture; it does not directly test the continuous-time tail-area theorem.

## Current build snapshot

- Canonical source: `paper/main.tex`, SHA-256 `f1dd153bee7ddd02845d5fe5fcaaad698b0797dfc501ab9d015bef418a24fdba`.
- Reviewer PDF: `paper/iclr2027/submission.pdf`, 16 pages, 508,009 bytes, SHA-256 `efeae58a9a4e9cd227bd7f278468015363cf22852442ed55b639970a38428a00`.
- Detailed rendering: `paper/main.pdf`, 16 pages, 592,185 bytes, SHA-256 `595cde2a3beca5fe69b78dc08553627dc05243692e0c8fdcb7a233fc8f9b6ab6`.
- Official-style main-text boundary: page 9; references begin on page 10.
- Build diagnostics: no overfull boxes, unresolved references, or unresolved citations. The remaining notices are non-blocking underfull page/bibliography spacing messages.
- Full software suite: 370 tests passed with four pre-existing plotting warnings. A fresh exact-lock clone separately passed 362 tests and skipped eight explicitly unavailable historical-data replays.
- Release integrity: four archives, five studies, 67 files, and 3,510,433 bytes verified with Git tracking required.
- PDF audit: required title/disclosure text is present; internal study IDs, provenance terms, local paths, and missing-figure fallback text are absent from both renderings; title metadata is empty and the author presentation is anonymous.
- Anonymous CIFAR supplement: `paper/artifacts/cifar-confirmation-anonymous-v1.zip`, SHA-256 `da4acec99abdfc4994c81fb27cca140a3937120dbf56eb14eacaf00e6b0464b8`; identifying Slurm stdout and model payloads are excluded.
- Independent scientific/statistical review remains an external blocker; no completed independent-review record is asserted.
- Readiness consistency check passes with eight remaining external, historical, or scientific-breadth blockers recorded in `research_scope/submission_readiness.yaml`.

The PDF to share for scientific review is `paper/iclr2027/submission.pdf`.
