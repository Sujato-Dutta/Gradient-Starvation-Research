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

Last updated 2026-08-24 on `codex/corrected-dense-baseline`. The final nonlinear
rerun is archived at `paper/artifacts/enl_tanh_crossover-20260824-141207` with a
resolved configuration and the executable-source fingerprint
`a1ce0e85a8c221a13897d18651a8d6234d5f72803c8f29b15dad3480b9729698`. Its
`git_dirty: true` state is therefore content-addressed over `run_experiment.py`
and `src/**/*.py`. Historical E1, E2, E2-R, exploratory E-NL, and E3 artifacts
were also dirty but predate source fingerprinting; their exact executed source
cannot be reconstructed retroactively. See
`paper/artifacts/provenance_manifest.json`.

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
5. **A drift crossing is *suppression*; starvation additionally requires the outcome
   and the learnability gate.** `transfer_then_suppression` means the weak mode's
   rate was suppressed; `transfer_then_starvation` requires both a response-gap
   crossing and weak-only learnability. An unlearnable outcome crossing is emitted
   as `transfer_then_outcome_crossing_unlearnable`, never as starvation.
6. **Geometry-versus-field attribution is ordering-dependent.** Splitting a product
   difference admits two exact orderings. No dominance claim may be made unless both
   agree, which `dominance_ordering_invariant` records.
7. **The Waterbirds "weak coordinate" is an intercept, not an identified feature.**
   Regressing the margin on `(strong, 1)` makes the second coefficient the average
   signed margin unexplained by the strong mode. Nothing identifies a bird-shape
   response, so Waterbirds is a **surrogate** mechanistic probe only.
8. **`degenerate` is an artifact bucket, not a phase.** It marks runs whose target
   was already met at initialization, before any optimization step.
9. **`closure_reference` is a numerical approximation.** It is a mean over trained
   finite networks pushed through `integrate_projected_flow`. It is not a solved
   theory and must never be called DMFT.

Implemented in `metrics.CAUSAL_REGIMES` and `metrics.classify_causal_regime`;
enforced by `tests/test_causal_regimes.py`.

---

## 2. Theory

| # | Statement | Status | Backing |
|---|---|---|---|
| T1 | Universal finite-width signed-CE response flow through the response/logit cross-kernel | **PROVED** | `e2_theorem.md`, Theorem A.1; direct autograd in `theory.projected_statistics` |
| T2 | Closed two-mode flow `dm/dτ=Gg` when the signed logits have an exact parameter-independent two-mode realization | **PROVED WITH ASSUMPTION** | `e2_theorem.md`, Corollary A.2; exact for the linear synthetic model with zero background noise |
| T3 | Exact finite-width dense-linear-RNN projected geometry | **PROVED** | `theory.exact_dense_linear_geometry`; agrees with autograd at `rtol=2e-5` |
| T4 | Exact matched weak-drift decomposition into CE gating, geometry shift, and cross transport under exact two-mode realization | **PROVED WITH ASSUMPTION** | `theory.matched_weak_drift_decomposition`; reconstruction error `<2e-6` |
| T5 | Both equal-time product-difference orderings reconstruct the projected drift exactly; neither attribution is canonical | **PROVED** | `theory.equal_time_drift_difference`; both algebraic reconstructions tested |
| T6 | Tail-area criterion: after a unique transfer-to-suppression drift crossing, an outcome crossing occurs iff accumulated negative drift reaches the positive peak area | **PROVED** | `e2_theorem.md`, Theorem B.1; executable finite-step analogue in `discrete_crossover_certificate` |
| T7 | Noiseless rank-one cue log drift ratio and monotone sufficient condition for a unique **rate** crossover | **PROVED WITH ASSUMPTIONS** | `e2_theorem.md`, Theorem C and Corollary C.1; does not imply outcome starvation without T6 |
| T8 | Transverse first hitting times are stable under uniform trajectory convergence | **PROVED WITH ASSUMPTIONS** | `e2_theorem.md`, Corollary D |
| T9 | Zero-disorder projected dynamics close exactly on six scalars at `g=0, lag=0` | **PROVED** | `dmft.solve_zero_disorder`; `e2_theorem.md`, Theorem E.1 |
| T10 | The zero-disorder six-scalar process converges uniformly on fixed horizons from an explicit `9/(N ε²)` initialization bound and Grönwall propagation | **PROVED WITH LOCAL-LIPSCHITZ TUBE** | `e2_theorem.md`, Theorem E.2; bound evaluators in `theory.py` |
| T11 | CDC Result 1: exact instantaneous strong-drift preservation | **PROVED** | `cdc_theorem.md`, Result 1 |
| T12 | CDC Result 2: unique uncapped minimum-norm target-attaining correction when feasible | **PROVED WITH ASSUMPTIONS** | `cdc_theorem.md`, Result 2; a binding cap loses target-attaining optimality |
| T13 | CDC Result 3: one-step strong-response deviation is `O(η²)` under local `L`-smoothness, bounded velocity/deficit, and `||q||>=q_min` | **PROVED WITH ASSUMPTIONS** | `cdc_theorem.md`, Result 3; `cdc_finite_step_deviation_bound` |
| T14 | General joint paired CE-RNN optimization-time DMFT at positive disorder/lag | **BLOCKED CONJECTURE** | `e2_theorem.md` obligations 1–4; all dependent solver surfaces still raise `NotImplementedError` |
| T15 | Independent tanh/GRU crossover-time prediction and analytic learnability/starvation boundary `rho_c` | **BLOCKED** | requires T14 or another independent closure; trajectory certificates are not predictions |

The established headline is now the **finite-width causal theorem package**
(T1–T13), not the blocked DMFT conjecture. T6 proves precisely what a rate crossover
must additionally satisfy to become an outcome crossing. It does not prove that an
arbitrary RNN develops the required drift sign pattern. T7 supplies sufficient rate-
crossover conditions in the noiseless rank-one setting, but checking those
conditions on the same trajectory is certification rather than independent
prediction.

---

## 3. Empirical results

| # | Statement | Status | Backing run |
|---|---|---|---|
| E1 | Under the final common-probe/direct-autograd protocol, tanh exhibits an exact drift crossing, response crossing, and causal tail-area certificate in **3/8** seeds | **EMPIRICAL** | `paper/artifacts/enl_tanh_crossover-20260824-141207`; conditional on the three crossing seeds, exact drift `tau*` mean `0.1942` and response-crossing mean `0.3890` |
| E2 | The other **5/8 tanh seeds are suppressed from initialization**, not transfer-to-starvation; each of the three crossing seeds has one exact `+ -> -` sign change | **EMPIRICAL** | same; phases are `suppression_throughout|transfer_then_starvation` |
| E3 | In each of the three tanh crossing seeds, exact drift suppression precedes response equality; conditional mean lead `0.1947` in optimization time | **EMPIRICAL** | same; this is conditional descriptive evidence, not an independent `tau*` prediction |
| E4 | GRU has exact drift and response crossings in **5/8** seeds, but weak-only never reaches `beta=0.5` in **8/8** seeds, so causal starvation is certified in **0/8** | **EMPIRICAL** | same; outcome crossings without the learnability gate must not be called causal starvation |
| E14 | ~~tanh late-time suppression is geometry-dominated~~ | **RETRACTED** | See R9. The additive projected attribution is not ordering-invariant and does not replace the exact nonlinear response drift |
| E15 | Direct-versus-projected drift residuals are nonzero and exposed: maximum absolute residual `0.02197` for tanh and `0.002986` for GRU | **EMPIRICAL** | `paper/artifacts/enl_tanh_crossover-20260824-141207`; confirms that projected `Gg` cannot be silently treated as exact for nonlinear probes |
| E16 | GRU's final projected geometry/field sign attribution is ordering-invariant in 8/8 seeds, but it is diagnostic only and does not overcome the failed causal-learnability gate | **EMPIRICAL** | same; no causal-starvation or exact-drift mechanism claim follows |
| E17 | Both projected equal-time orderings reconstruct their projected drift difference to at most `9.54e-7` (tanh) and `5.59e-9` (GRU) at logged points | **EMPIRICAL** | same; per-row reconstruction is asserted in `tests/test_experiment_smoke.py` and `tests/test_training.py` |
| E18 | The exploratory 25-cell tanh grid found **no 4/4 robust condition**; the best cells reached 3/4 causal certificates | **EMPIRICAL NEGATIVE RESULT** | `enl_exact_regime_search-20260824-105003`; exploratory seeds 0–3 only, so no searched cell is confirmatory evidence |
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
| R7 | The old E-NL 8/8 tanh crossover, `tau*=1.6112`, and GRU 0/8 narrative | **RETRACTED AND SUPERSEDED** | Those values used a condition-dependent/projection-based response convention. Under the final common symmetric probe and universal direct-autograd drift, `paper/artifacts/enl_tanh_crossover-20260824-141207` gives tanh 3/8 exact drift/response/causal crossings and GRU 5/8 exact drift/response crossings but 0/8 causal certifications. The old numerical `tau*` is invalid for the final response definition |
| R9 | "Late tanh suppression is geometry-driven" (E14) | **RETRACTED** | Two defects. First the split was algebraically wrong: it paired ordering A's geometry term with ordering B's field term, reconstructing nothing — mismatch up to `0.753`, e.g. a row with `d_w = 0.011` reported `geometry 0.923`, `field −0.159`, summing to `0.764`. Second, and fatally for the claim, splitting a product difference admits two exact orderings and they **disagree**: at the final point only 1/8 tanh seeds agree on which channel dominates. Both orderings are now computed with per-row reconstruction tests, and a `dominance_ordering_invariant` flag gates any dominance statement. The tanh claim is withdrawn; the GRU one survives as E16 because it *is* invariant |
| R10 | "A drift crossing necessarily precedes a response crossing" | **RETRACTED as stated** | A `+ -> -` derivative crossing establishes a local maximum of the response gap, not that the gap later reaches zero. The negative tail-area condition in T6 is necessary and sufficient. In the final tanh run the ordering holds only for the 3/8 crossing seeds; the other 5/8 are suppressed from initialization. In GRU it holds for 5/8 outcome-crossing seeds, but none passes the weak-only learnability gate |
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
| S8 | The Waterbirds modal arm ran a second forward pass in train mode. `no_grad` suppresses gradients but **not** BatchNorm running-statistic updates, so that arm advanced its normalization state twice per batch while every baseline advanced it once | **FIXED** | features are captured by a forward hook on the pass that already happens; a test compares BatchNorm state against a single-forward baseline at `atol=0` |
| S9 | `modal_feature_coordinates` and `modal_estimator_agreement` each called randomized `torch.pca_lowrank` independently, so the logged agreement could describe a different component from the one corrected | **FIXED** | `leading_feature_direction` uses a deterministic full SVD and the direction is computed once and shared |

---

## 6. References

The two 2026 works below were reviewed sufficiently to delimit the novelty and the
remaining mean-field obligations; this is a scope review, not a claim that every
proof or experiment in either work has been independently reproduced.

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
  arXiv:2605.04115 — <https://arxiv.org/html/2605.04115v1>. Status: **SCOPED**.
  Their overlap-space dynamics and loss-invisible memory variables narrow any
  novelty claim based only on projected geometry and motivate obligation 2.
- Clark et al., "Structure, disorder, and dynamics in task-trained recurrent neural
  circuits", bioRxiv 2026.03.02.708943 —
  <https://www.biorxiv.org/content/10.64898/2026.03.02.708943v1.full-text>.
  Status: **SCOPED**. Their long-time Langevin/Gibbs mean-field setting is distinct
  from noiseless optimization-time paired gradient flow, so it narrows but does
  not discharge the blocked DMFT obligations.

Content was rephrased for compliance with licensing restrictions.

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
