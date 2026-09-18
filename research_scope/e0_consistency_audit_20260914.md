# E0 consistency audit — 14 September 2026

Status: **manuscript corrections resolved on 17 September 2026**. This is a reanalysis of preserved local CSV outcomes, not a rerun of the historical training jobs. The raw E1 and E-NL trajectories live under ignored `results/`; a clean clone will not contain them. The historical source provenance limits described in `paper/artifacts/provenance_manifest.json` remain. Code: `scripts/audit_e0_consistency.py` and independent piecewise-linear endpoint implementation in `src/gradient_starvation/causal_audit.py`. Detailed per-run output and input SHA-256 hashes are in `results/e0_consistency_audit-20260914/{run_level.csv,summary.json}`. The corrected manuscript uses the broad any-time certificate as its primary definition, reports at-hit separately, and renders `paper/figures/fig_e0_final_causal_grid.{pdf,png}` directly from the audited rows.

## Definitions used in this audit

- `weak_gate`: W first reaches the fixed target strictly after initialization and within the logged horizon. A target met at initialization is degenerate. A failed gate is indeterminate, not a negative certificate.
- `at_hit_certificate`: the common weak-response gap `B-W` is below `-1e-6` **at W's first hit**. This is the manuscript's stricter at-hit subtype.
- `any_time_certificate`: the gap is below `-1e-6` at any positive logged time (or within the piecewise-linear interpolation). This is the primary causal outcome condition in the corrected Definition 1 and matches E10 in the attached `parameter_prediction_proofs.pdf`.
- Positive signed weak-AUC means the integrated W-minus-B response is positive. It is not equivalent to the at-hit certificate. `regime_class` in historical summaries is a threshold-delay classifier, also a distinct endpoint.
- Times and responses were interpolated linearly between stored checkpoints. Initial responses matched within `1e-5` for all reviewed runs. The audit verifies stored AUCs and W first-hit times against the raw CSVs. It does **not** establish continuous-time events between checkpoints or replay historical training.

## Recomputed results

| Study and lag | Runs | W gate | At-hit certificate | Any-time certificate | Positive signed AUC |
|---|---:|---:|---:|---:|---:|
| Dense linear positive, 0 | 32 | 32 | **31** | 32 | 32 |
| Dense linear positive, 2 | 32 | 32 | 32 | 32 | 32 |
| Dense linear positive, 4 | 32 | 32 | 32 | 32 | 32 |
| Dense linear positive, 8 | 32 | **0** | 0 (indeterminate) | 0 (indeterminate) | 27 |
| Dense linear sign-reversal control, 4 | 8 | 8 | 0 | 2 | 0 |
| Tanh, lag 2 | 8 | 8 | **8** | 8 | 8 |
| GRU, lag 2 | 8 | **0** | 0 (indeterminate) | 0 (indeterminate) | 5 |

The single dense lag-0 exception is `rho=1, seed=11`: W hits `beta=0.25` at `tau≈0.268004`, B at `tau≈0.265811`, and the response gap at W's hit is **+0.001748**. Its later minimum gap is about **−1.154844**, so its AUC is positive and it satisfies the broader any-time definition, but not the paper's at-hit definition. This is not a rounding-edge classification at tolerance `1e-6`.

For tanh, the historical `3/8` count is exactly the **transfer-then-starvation subtype**; the other five seeds are labeled `suppression_throughout` and also pass the W gate and at-hit outcome test. The manuscript's statement that those five exhibit “neither crossing nor starvation” is false under its own certificate definition. GRU has negative response gaps in all eight logged runs but no W first hits, so all eight are causally indeterminate; its five crossing events must not be counted as certificates.

The dense sign-reversal arm is not a clean no-cue negative control. Two of its eight runs have small transient negative gaps (`minimum ≈ -0.000436` and `-0.000145`) although none passes the at-hit test. It should be named a sign/equivariance intervention and accompanied by a true identical-input null and, if affordable, a label-independent cue control in a future study.

## Manuscript and artifact resolution record

1. **Resolved:** the manuscript and replacement figure use the actual strength grid **`{1,2,4,8}`**, separately from lag 8.
2. **Resolved:** dense positive at-hit counts are **`31/32, 32/32, 32/32`** for lags 0, 2, and 4; broad any-time counts and signed AUC remain separately named endpoints. Lag 8 is indeterminate.
3. **Resolved:** tanh is reported as **8/8 causal and at-hit certificates in this fixed cell**, with **3/8** transfer-then-suppression and **5/8** suppression throughout. GRU remains causally indeterminate because 0/8 weak-only controls pass the gate.
4. **Governance retained:** historical label corrections are not treated as new confirmatory outcomes. The frozen claim and readiness records remain fail-closed for genuinely new outcomes.
5. **Resolved:** the dense regime figure was regenerated as `paper/figures/fig_e0_final_causal_grid.{pdf,png}` from the corrected run-level table. It displays any-time and at-hit counts separately and omits obsolete threshold-delay labels.
6. **Resolved:** the manuscript now uses the broad any-time condition as the primary definition and names at-hit as a stricter subtype. The prospective theorem proves the broad endpoint and states the additional timing condition sufficient for at-hit.

The tests in `tests/test_causal_audit.py` cover the logical non-equivalence, noncrossing suppression, failed/degenerate gates, initial mismatch, exact piecewise positive area and invalid inputs. `tests/test_e0_consistency_replay.py` locks the local E1/E-NL counts when ignored raw files are available; it skips clean CI when they are absent. These tests verify the audit calculation, not the original training or proof.

The retained historical `e1_causal_regions.pdf` was visually inspected. Its x-axis correctly shows `rho={1,2,4,8}` and it renders obsolete **threshold-delay** categories. It is no longer used in the main paper. The replacement figure reports primary causal and stricter at-hit counts from the corrected audit table.

Validation state on 14 September: the new audit tests pass. The full suite had **340 passed, 2 failed**; both failures arise from a stale `paper/main.tex` digest in the pre-existing readiness manifest while the paper is already modified. Excluding the readiness test module yields **310 passed**. Do not refresh the digest merely to make tests green before the corrected manuscript and claim ledger are reviewed.
