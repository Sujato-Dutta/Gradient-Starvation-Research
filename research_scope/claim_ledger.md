# Claim ledger

Every statement this project might put in a paper, tagged by epistemic status and
tied to the artifact that backs it. The purpose is narrow: to make it impossible to
carry an unproved statement into a manuscript by accident.

Tags are used strictly.

- **PROVED** — a complete argument exists in this repository, and a test exercises
  the implementation it describes.
- **TARGET** — believed, partially measured, proof not written. Never to be phrased
  as established.
- **EMPIRICAL** — measured on specific runs under specific settings. Carries no
  claim beyond those settings.
- **BLOCKED** — cannot be evaluated until a named proof obligation is discharged.
- **RETRACTED** — was believed, and the evidence no longer supports it.

Last updated 2026-08-23, covering commits `f631352`–`5af3d35` on
`codex/corrected-dense-baseline` (unpushed).

---

## 1. Frozen definitions

These are not claims; they are the vocabulary every claim below depends on. Changing
one invalidates the rows that use it.

1. **Causal starvation.** A weak feature is starved only if (a) it demonstrably
   learns in the matched weak-only counterfactual, and (b) its learning is
   measurably delayed or suppressed when the strong feature is present. Condition
   (a) is a gate, not a footnote.
2. **Sequence time `k` versus optimization time `τ`.** Distinct axes. Any kernel or
   order parameter that involves both must say so explicitly.
3. **GSI-5 is a diagnostic**, predicted from the margin-weight distribution. It is
   not the definition of starvation and must never be used as one.
4. **No starvation label outside the learnable region.** A positive AUC gap where
   the weak-only counterfactual never learns is *indeterminate*, not starvation.
5. **`degenerate` is an artifact bucket, not a phase.** It marks runs whose target
   was already met at initialization, before any optimization step.
6. **`closure_reference` is a numerical approximation.** It is a mean over trained
   finite networks pushed through `integrate_projected_flow`. It is not a solved
   theory and must never be called DMFT.

Implemented in `metrics.CAUSAL_REGIMES` and `metrics.classify_causal_regime`;
enforced by `tests/test_causal_regimes.py`.

---

## 2. Theory

| # | Statement | Status | Backing |
|---|---|---|---|
| T1 | Exact signed CE function-space gradient flow | **PROVED** | pre-existing; `theory.projected_statistics` |
| T2 | Exact two-mode projected flow `dm/dτ = G g` | **PROVED** | `tests/test_theory.py::test_projected_ce_identity_matches_direct_gradient_flow`; max abs error `2.86e-06` in the corrected E2 acceptance record |
| T3 | Exact finite-width dense-RNN projected geometry | **PROVED** | `theory.exact_dense_linear_geometry`; agrees with autograd at `rtol=2e-5` |
| T4 | Exact matched weak-drift decomposition into CE gating, geometry shift, cross transport | **PROVED** | `theory.matched_weak_drift_decomposition`; reconstruction error `< 2e-6` |
| T5 | `d_w = t_geom − s_ce` is an exact reparameterization of T4 | **PROVED** | `theory.crossover_decomposition`, a delegating wrapper; agreement with T4 asserted at `rtol=0` |
| T6 | Zero-disorder projected dynamics close exactly on six scalars at `g = 0, lag = 0` | **PROVED** | `dmft.solve_zero_disorder`. The recurrent block is provably inert there, and verified so: `‖W‖` stays bitwise `0.0` at lag 0 but reaches `0.346` by step 20 at lag 1 |
| T7 | CDC Result 1: exact instantaneous strong-drift preservation | **PROVED** | `research_scope/cdc_theorem.md` §Result 1; holds for any `α`; measured `4.77e-07` |
| T8 | CDC Result 2: minimum-norm optimality of the correction | **PROVED** | `cdc_theorem.md` §Result 2, Cauchy–Schwarz with uniqueness |
| T9 | CDC Result 3: `O(η²)` finite-step strong-response deviation | **TARGET** | three missing pieces named in `cdc_theorem.md`: explicit Lipschitz neighbourhood, bound on `‖v′‖`, accumulation over steps |
| T10 | Joint paired CE-RNN mean-field theorem (Theorem A) | **BLOCKED** | `e2_theorem.md` obligations 1–4 |
| T11 | Transfer-to-starvation crossover, sufficient conditions (Corollary A) | **BLOCKED** | needs T10 first |
| T12 | Learnability + starvation boundary `ρ_c` (Theorem B) | **BLOCKED** | needs T10; also see E7 on metric dependence |

**T10–T12 have not advanced this pass, by design.** No derivation was written and no
placeholder document was created.

---

## 3. Empirical results

| # | Statement | Status | Backing run |
|---|---|---|---|
| E1 | tanh shows a transfer-to-starvation crossover in 8/8 seeds, equal-time `τ*` mean `1.6112`, 95% CI `[1.3248, 1.8977]` | **EMPIRICAL** | `enl_tanh_crossover-20260823-100039` (corrected). The matched-state convention gives `1.7412`; the gap is `−0.1300`. Both are logged. The superseded run `-074607` reported only the matched value |
| E2 | Exactly one `d_w` sign change per seed; no oscillation | **EMPIRICAL** | same |
| E3 | The drift-level crossover precedes the outcome-level one by `1.1939` in `τ`, 95% CI `[1.0344, 1.3534]`, strictly positive in every seed | **EMPIRICAL** | `enl_tanh_crossover-20260823-100039`. Now a like-for-like comparison: the drift is measured on the derivative of the same gap whose crossing is compared. The superseded `≈1.06` mixed conventions |
| E4 | GRU shows transfer throughout: 0/8 crossings under both conventions | **EMPIRICAL** | `enl_tanh_crossover-20260823-100039`. Its equal-time geometry term *grows* (`0.082 → 0.972`) while the field term stays small and slightly negative, so the transfer is geometry-sustained. Caveat: equal-time `d_w` decays to `0.021` and is still falling, so a longer horizon might cross |
| E14 | tanh late-time suppression is geometry-dominated, but only modestly | **EMPIRICAL** | same. At the crossover the geometry term is still positive (`0.64`) and the field term negative (`−0.24`), so the crossover is field-driven. The geometry term reverses later (`τ ≈ 2.88`) and ends larger in magnitude (`−0.124` vs `−0.080`). **The matched-state convention inflated this**: it showed `−2.78` versus `0.79`, suggesting overwhelming geometry dominance where the equal-time split gives roughly `1.5×` |
| E15 | Equal-time `d_w` peaks and then decays toward zero (`−0.86` at `τ=4` to `−0.14` at `τ=9.9`) | **EMPIRICAL** | same. The two conditions' weak drifts re-converge late. The matched-state convention showed the opposite, growing monotonically to `−3.57`, so any claim of ever-increasing suppression came from the wrong quantity |
| E5 | Zero-disorder solver is the correct continuous limit; residual is `O(η)` discretization, fitted slope `1.0115` | **EMPIRICAL** | `e2r_solver_checks-20260823-084911` |
| E6 | Frozen-geometry solver agrees with `integrate_projected_flow` to `1.91e-15` | **EMPIRICAL** | same |
| E7 | Under the corrected dense parameterization the AUC gap is monotone increasing in `ρ` at every lag | **EMPIRICAL** | `e1_dense_rerun-20260823-085142` |
| E8 | Corrected E1: 32/32 seeds positive AUC gap at lag 0/2/4; negative control 8/8 `transfer`; lag 8 `unlearnable` at every `ρ` | **EMPIRICAL** | same |
| E9 | CDC Result 3 scaling: fitted log-log slope `2.0000` (dense, low-rank), `1.9961` (tanh) | **EMPIRICAL** | `tests/test_cdc_families.py`. Requires float64; float32 quantization gives a spurious `1.78` |
| E10 | CDC is inactive before the crossover, because no weak-drift deficit exists yet | **EMPIRICAL** | `tests/test_cdc_families.py::test_no_correction_is_applied_without_a_deficit` |
| E11 | All five shadow-based methods are *numerically similar* on the causal weak gap, agreeing to ~4 decimal places | **EMPIRICAL** | `e3_cdc_dense_ablation-20260823-090702`. **Not** statistically indistinguishable: 9 of 10 direct pairwise tests are significant at 0.05 (e.g. CDC − bloop `−0.000720`, `p = 2.9e-07`). Differences are detectable and practically negligible against the `≈ −10.5` effect versus ERM. Equivalence would need a preregistered margin |
| E12 | Only CDC preserves the instantaneous strong drift: `4.77e-07` against `1.5e-01`–`3.8e-01` | **EMPIRICAL** | same |
| E13 | `unconstrained_rescue` and `pcgrad` end with a *higher* final strong response than CDC (`1.295` vs `1.042`) | **EMPIRICAL** | same |

---

## 4. Retractions and narrowings from this pass

| # | Statement | Status | Why |
|---|---|---|---|
| R1 | "`τ*` predicted from `T_geom = S_CE` can be compared against the observed `d_w` sign change" | **RETRACTED** | Degenerate. `d_w = t_geom − s_ce` is an identity, so the two agree by construction; measured gap `1.52e-07`. Emitted only as a self-consistency check |
| R2 | "`T_geom` dominates early and `S_CE` dominates late" | **NARROWED** | Right about the crossover, wrong about the late regime. `T_geom` itself turns negative at `τ ≈ 2.75` and reaches `−2.78` while `S_CE` saturates near `0.79`, so late suppression is geometry-driven, not CE-gating-driven |
| R3 | "The `ΔT_w` and AUC-gap metrics rank the lag axis in opposite directions" | **NARROWED** | Measured on the superseded run. Under the corrected parameterization the two orderings differ but no longer oppose. The requirement that a boundary state its metric stands |
| R4 | E1 per-lag AUC magnitudes `19.20 / 8.96 / 1.49` | **RETRACTED** | Superseded parameterization. The corrected run peaks at lag 2: `5.76 / 8.93 / 6.11` |
| R5 | "BOTH-feature E2 RMSE worsens with width, so the closure is incomplete" | **RETRACTED** | True only of `e2_width_extension-20260821-134952` (`19.862 → 25.975`), which is superseded. The corrected run improves (`0.09084 → 0.02958`, non-monotone at N=128). The solver is still needed, for the epistemic reason that a reference calibrated from trained networks cannot falsify the theory |
| R6 | "CDC beats Bloop and the ablations on the causal weak gap" | **RETRACTED** | It does not; see E11. The defensible claim is narrower: CDC is the unique family member preserving the instantaneous first-order response of a theory-identified feature, and this run does not demonstrate that the property has practical value |
| R7 | The E-NL `τ*` figures and the "drift leads response" lead time | **RETRACTED, pending rerun** | The logged `d_w` was the matched-state deficit, which recomputes the weak-only field at the both-feature `m_w`. That is **not** `d/dτ[m_w^B − m_w^W]`, so comparing its crossing against the response-gap crossing subtracted two different quantities. Measured discrepancy: `1.741` matched versus `1.611` equal-time, mean over 8 seeds. The qualitative 8/8 crossover and the GRU 0/8 non-crossing survive, because both conventions agree on *whether* a sign change occurs. Fixed by logging `d_w_equal_time` alongside `d_w_matched`, and **rerun**: `enl_tanh_crossover-20260823-100039` supersedes `-074607`. The qualitative 8/8 crossover and GRU 0/8 survive; `τ*`, the lead time, and the geometry-versus-field magnitudes all moved |
| R8 | "Waterbirds CDC preserves the instantaneous strong drift" | **RETRACTED as previously implemented** | Two defects, both fixed. The correction wrote gradients only for the head, leaving the backbone at `grad = None` and silently frozen while baselines fine-tuned the whole network. And the optimizer was AdamW with weight decay, whose preconditioned, momentum-carrying, decayed update is not the corrected gradient the guarantee is stated for. The runner now requires SGD with zero decay for *every* arm whenever a CDC arm is present, so the constraint cannot become an optimizer confound |

---

## 5. Software correctness findings

| # | Finding | Status | Backing |
|---|---|---|---|
| S1 | Sequential paired trainer returned aliased parameter snapshots; `states["both"]` actually held the weak-only parameters | **FIXED** | commit `0bee09f`. `tensor.detach().cpu()` shares storage on CPU. No published result affected — every `experiments.py` call site discards states |
| S2 | `.gitignore` negation for the data package overrode `__pycache__`, staging bytecode | **FIXED** | commit `f631352` |
| S3 | `sign_crossing_time` interpolated across a bracketed exact zero, returning `2/3` instead of `1.0` for `[1, 0, −2]`. Symmetric cases were right by luck, which is why the original zero test missed it | **FIXED** | a bracketed zero is now returned directly |
| S4 | `plot_e2` recognized only the obsolete `particle_closure` label, so the renamed `closure_reference` was drawn as `network N=<closure width>` — erasing the epistemic distinction the rename existed to make | **FIXED** | both labels recognized |
| S5 | MSE training was selectable while every logged projected diagnostic stayed cross-entropy specific | **FIXED** | the paired trainers now refuse `objective: mse` rather than emit mislabelled columns. Objective-aware diagnostics remain unimplemented |
| S6 | `tau_max` was unvalidated, and a both-condition crossing *after* the horizon produced a finite delay the run never demonstrated | **FIXED** | non-finite or sub-initial horizons rejected; post-horizon crossings censored |
| S7 | Waterbirds reused one classifier-head initialization across all seeds, so seed variation covered only data order | **FIXED** | the head is re-drawn per seed, preserving within-seed pairing across methods |

---

## 6. References

Both resolve. **Both remain unread**, and no comparative-novelty claim in this
project may rest on them until the author has read them.

- Pezeshki et al., "Gradient Starvation: A Learning Proclivity in Neural Networks",
  NeurIPS 2021. Status: **read** (establishes starvation and Spectral Decoupling in
  a feedforward/linearized setting; the recurrent optimization-time claim is what
  must be new).
- Yu et al., "Gradient Surgery for Multi-Task Learning" (PCGrad), NeurIPS 2020.
  Status: **implemented as a baseline**, primary text not verified this pass.
- Hsieh et al., "Careful with that Scalpel" (Bloop), ICML 2024. Status:
  **implemented as a baseline**, primary text not verified this pass. The
  implementation follows the EMA-smoothed auxiliary-projection idea; whether it
  matches the paper's exact algorithm has not been checked.
- Ger & Barak, "Learning reveals invisible structure in low-rank RNNs",
  arXiv:2605.04115 — <https://arxiv.org/abs/2605.04115>. Status: **UNREAD**.
  Load-bearing for the loss-invisible-overlap argument in obligation 2.
- Clark et al., "Structure, disorder, and dynamics in task-trained recurrent neural
  circuits", bioRxiv 2026.03.02.708943 —
  <https://www.biorxiv.org/content/10.64898/2026.03.02.708943v1>. Status:
  **UNREAD**. Load-bearing for scoping the mean-field contribution.

The Bloop and PCGrad rows matter for E11: the baselines are *our reading* of those
methods, not verified reimplementations, so "CDC does not beat Bloop" should be
stated as "does not beat our Bloop-style baseline" until the primary text is checked.

---

## 7. Statements that must never appear

1. A passing test suite proves software invariants, not the theorem.
2. `e2_acceptance.json` passing is convergence to a held-out empirical closure.
   Only a future `e2r_acceptance.json` carrying a frozen independent solution would
   support "DMFT validation". The current record reports `passed: false` by design.
3. A positive AUC gap outside the learnable region is starvation.
4. CDC preserves the final strong response. It preserves the instantaneous
   first-order drift; the final response is lower than ERM and lower than the
   unconstrained variant.
5. Any quantitative claim sourced from `e2_medium_validation`,
   `e2_high_confidence`, `e2_width_extension`, or the 13:xx E1/E3 batch.
6. Any Waterbirds result. Nothing has been run.
7. Anything about acceptance outcome or oral readiness.
