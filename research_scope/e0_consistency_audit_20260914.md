# E0 consistency audit — 14 September 2026

Status: **needs manuscript revision**. This is a reanalysis of preserved local CSV outcomes, not a rerun of the historical training jobs. The raw E1 and E-NL trajectories live under ignored `results/`; a clean clone will not contain them. The historical source provenance limits described in `paper/artifacts/provenance_manifest.json` remain. Code: `scripts/audit_e0_consistency.py` and independent piecewise-linear endpoint implementation in `src/gradient_starvation/causal_audit.py`. Detailed per-run output and input SHA-256 hashes are in `results/e0_consistency_audit-20260914/{run_level.csv,summary.json}`.

## Definitions used in this audit

- `weak_gate`: W first reaches the fixed target strictly after initialization and within the logged horizon. A target met at initialization is degenerate. A failed gate is indeterminate, not a negative certificate.
- `at_hit_certificate`: the common weak-response gap `B-W` is below `-1e-6` **at W's first hit**. This matches the current `paper/main.tex` Definition 1.
- `any_time_certificate`: the gap is below `-1e-6` at any positive logged time (or within the piecewise-linear interpolation). This matches E10 in the attached `parameter_prediction_proofs.pdf` but is **not** the current paper definition.
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

## Required manuscript and artifact changes before submission

1. `paper/main.tex` experimental setup lists `rho ∈ {0.5,1,2,4}` and table labels `0.5–4.0`; the actual E1 config/raw trajectories use **`{1,2,4,8}`**. Fix text, axis/table ranges, and any affected interpretation. This is separate from the lag index 8.
2. Revise the dense positive at-hit counts from `32/32, 32/32, 32/32` to **`31/32, 32/32, 32/32`** for lags 0, 2, 4, respectively; keep `32/32` positive AUC as a *separate* statement. Do not infer one from the other. Lag 8 remains indeterminate.
3. Revise tanh from “3/8 causal certificates” to **8/8 at-hit certificates in this fixed cell**, of which **3/8** show transfer-then-suppression and **5/8** suppress throughout. This does not imply tanh generally starves. GRU is 0/8 assessable due to gate failure, not evidence of absence.
4. Update the claim ledger and frozen `submission_claim_set.yaml` only through its review/governance process; the current `new_confirmatory_outcomes_allowed_now: false` gate remains until a clean source, locked environment, committed preregistration, and untouched seeds are secured. Historical label corrections are not new confirmatory outcomes.
5. Regenerate the dense regime figure and every table/caption from the corrected run-level table, using explicit separate legend keys for at-hit, any-time, AUC and threshold-delay categories. Audit all retained figure files and PDFs against source rows; this has **not yet been done**.
6. Choose the manuscript's primary definition before adding the new theorem. If it stays *at-hit*, every theorem using an any-time witness needs an additional at-hit hypothesis/conclusion or must be described as proving only the broader endpoint. Do not silently change the definition to preserve counts.

The tests in `tests/test_causal_audit.py` cover the logical non-equivalence, noncrossing suppression, failed/degenerate gates, initial mismatch, exact piecewise positive area and invalid inputs. `tests/test_e0_consistency_replay.py` locks the local E1/E-NL counts when ignored raw files are available; it skips clean CI when they are absent. These tests verify the audit calculation, not the original training or proof.

The retained historical `e1_causal_regions.pdf` was visually inspected. Its x-axis correctly shows `rho={1,2,4,8}`, contradicting the manuscript's `0.5–4.0` range. It renders **threshold-delay** categories (`neutral` in some lag-0/2 cells), not the paper's at-hit certificate. The caption must say so if the old figure is retained; a new at-hit figure would be clearer. This visual check covers that E1 figure only, not every manuscript image.

Validation state on 14 September: the new audit tests pass. The full suite had **340 passed, 2 failed**; both failures arise from a stale `paper/main.tex` digest in the pre-existing readiness manifest while the paper is already modified. Excluding the readiness test module yields **310 passed**. Do not refresh the digest merely to make tests green before the corrected manuscript and claim ledger are reviewed.
