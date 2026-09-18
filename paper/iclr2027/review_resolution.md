# ICLR 2027 editorial review resolution

## Current status

Updated 17 September 2026 after the frozen eight-block CIFAR-10 confirmation and the latest scientific review. The scientific authority is `paper/main.tex`; this file records how the review was resolved and the validation state of the resulting build.

## Latest review requests and resolutions

| Review request | Status | Resolution |
|---|---|---|
| Repair the mismatch between Definition 1 and the prospective theorem | **Resolved** | The primary certificate now requires a nondegenerate weak-only hit plus a negative paired gap at some positive time. “At-hit starvation” is a separately named stricter subtype. The prospective theorem proves the broad certificate and states the additional hit-time condition sufficient for the subtype. |
| Make the stronger parameter-level theorem visible | **Resolved with an explicit evidence boundary** | Section 5 now states a general finite-horizon theorem for any finite smooth network with independently certified parameter-region bounds, including suppression, learnability, at-hit, and safe-transfer conclusions. It also gives the proof package's fully trained five-parameter tanh-CNN existence certificate and matched no-suppression control as a short corollary; the exact logits and rational interval arithmetic are in the appendix. The manuscript labels this as a constructed broad certificate with a tiny margin and short horizon, not an at-hit, prevalence, or practical-harm result. |
| Reduce competing contributions | **Resolved** | The introduction has exactly three contributions: causal diagnosis; finite-width mechanism and prospective prediction; and controlled evidence. CDC is supporting mechanism control rather than a separate headline. |
| Replace the obsolete dense-grid classifier figure | **Resolved** | Figure 3 is regenerated directly from the E0 run-level audit. Cell colors use the final broad definition, while labels separately report causal (C) and stricter at-hit (AH) counts. Historical threshold-delay labels are absent. |
| Make CIFAR central | **Resolved** | The frozen CIFAR study is the first empirical result and the first empirical figure. It reports trajectories, neutral/random/consistent/conflict behavior, persistence, and fresh-head recovery before the recurrent scope checks. |
| Remove frozen tangent from the main narrative | **Resolved** | The tangent comparison theorem, proof, and restricted 64-record event heuristic now appear only in the appendix. The table reports zero causal-endpoint coverage rather than mislabeling it as 0% accuracy. |
| Complete CIFAR reproducibility details | **Resolved** | Appendix D records source/environment/data digests, splits, ribbon geometry and coding, reliability and amplitude, augmentation, CNN4-GN architecture, optimizer and schedule, logging interval, balanced Brier-skill definition, gate threshold, and E4-lite head-fitting protocol. |
| Keep CDC but reduce its role | **Resolved** | CDC occupies one short main-text subsection, is explicitly local and chart-dependent, and makes no global optimizer-superiority claim. The full comparator table remains in the appendix. |

## Scientific boundaries retained

1. Weak learning failure alone is not evidence of gradient starvation.
2. A failed or initialization-degenerate weak-only gate is indeterminate, not a negative certificate.
3. At-hit starvation is stronger than the primary any-time outcome condition; the two counts are reported separately.
4. The exact finite-width equations do not establish a universal recurrent-network or positive-lag mean-field theory.
5. The prospective smooth-network theorem is conditional on certified uniform bounds. A constructed five-parameter tanh-CNN certificate is included, but no such theorem-level certificate is claimed for the CIFAR CNN.
6. The frozen tangent surrogate is an event heuristic, not calibrated trained-network dynamics or a causal predictor.
7. CDC has local, fixed-chart, feasible minimum-norm and one-step guarantees only.
8. The CIFAR result establishes one controlled artificial-cue existence case for one architecture, not natural-shortcut prevalence or an architecture ranking.

## Current build snapshot

- Canonical source: `paper/main.tex`, SHA-256 `b1fb927f8af8cf863923a0ba0a44e039a6c0c0b0257e7496efc133671275bdb5`.
- Anonymous reviewer PDF: `paper/iclr2027/submission.pdf`, 18 pages, 539,434 bytes, SHA-256 `f47338be59591a1dcdb7cf691f5df04d5ab0ed3aad48fef7735a424433b1e052`.
- Canonical rendering: `paper/main.pdf`, 19 pages, 617,629 bytes, SHA-256 `6af6d10dc696ef6dacbf976d332b93db627bd438e8281dd9f0839bd5eb0f537d`.
- The official `page:main-end` boundary records page 9; references begin on page 9 and appendices on page 13.
- Both PDFs build without overfull boxes, unresolved references, or unresolved citations. The official rendering has one non-blocking underfull vertical-box notice.
- All 18 official-style pages were rendered to PNG and visually inspected; no clipping, missing figures, broken tables, or unreadable layouts were found.
- Full software suite: **371 tests passed** with four pre-existing Seaborn warnings.
- Submission manifests validate consistently and intentionally remain `overall_submission_ready=False` with eight external, historical, or breadth blockers.
- Independent scientific/statistical review remains external work; this record does not claim reviewer agreement or acceptance readiness.

The PDF to share for the next scientific review is `paper/iclr2027/submission.pdf`.
