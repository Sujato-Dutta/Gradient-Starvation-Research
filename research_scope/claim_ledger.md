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

Last updated 2026-08-31 on `codex/gradient-starvation-research`. The final nonlinear
rerun is archived at `paper/artifacts/enl_tanh_crossover-20260824-141207` with a
resolved configuration and the executable-source fingerprint
`a1ce0e85a8c221a13897d18651a8d6234d5f72803c8f29b15dad3480b9729698`. Its
`git_dirty: true` state is therefore content-addressed over `run_experiment.py`
and `src/**/*.py`. That fingerprint and the unmodified run metadata are frozen at
repository commit `2d0ee83` / tag `theorem-aligned-v1`; the digest identifies the
executed run source, not any later descendant or working tree.

The initialization-frozen empirical-NTK studies are separately sealed. The broad
pilot source evaluation at `results/enl_ntk_pilot_evaluate-20260825-160757`, with
compact checked-in evidence at `paper/artifacts/enl_ntk_pilot-20260825`, verifies
manifest
`f186d587e28f6408b68baf95566d2d1f1c5d5701e205e9a69ce86cbe4f0ffaf5`, frozen
prediction digest `9787d99953f1046b4934aa6e7b3f969e30a4efafb3d02a67cc95926232f9dfaa`,
and executable-source digest
`8d8162558e7d25fff7a5059a4cf49139a354ca211fd6f0759417e75dd87f435c`.
The restricted fresh factorial source evaluation at
`results/enl_ntk_crossing_factorial_evaluate-20260825-162949`, with compact
checked-in evidence at
`paper/artifacts/enl_ntk_crossing_factorial-20260825`, verifies manifest
`bdcf02abbef5b9d8f7ce3979d3363ee8d39ab7817a1d554616bc4dd4a22a0db1`, frozen
prediction digest `c05b31810cb3869e7d051f539741732a8758f6cac542deb448a98d2ada7b6542`,
and executable-source digest
`ca878df376b01c0ea45ac352e353530bf20c567de59bb400617e5dd74d5b6b6b`.
Both evaluations report that all sealed inputs were unchanged. The compact archive
roots and every archived file's digest/size are indexed by
`paper/artifacts/provenance_manifest.json`. They omit large kernel tensors and full
trajectory/prediction tables. More importantly, the exact dirty Python source bytes
corresponding to the two recorded source fingerprints were not archived and are no
longer reconstructible; the later tree fails the source guard. Thus a clean checkout
can audit the compact outcomes and seals but cannot replay either archived study.
E20 remains the submission-primary empirical claim only for its preregistered drift-
and response-crossing classification endpoints. The digests establish artifact
identity and record historical source fingerprints; they do not reconstruct omitted
source, prove kernel stability, or establish scientific validity.

E26--E28 come from externally sealed executions run from clean repository commit
`9ebbc617a9aaad5c3c19e9d5fd4fa08f2d3517d0`. E26 is archived at
`paper/artifacts/semi-real-generated-cue-v1-20260830` with executable-source digest
`02ba4ce4d6aff3ff726d9e798606aed29cff1276a6837e10ff43fb37824ca042`; E27--E28
are archived at `paper/artifacts/expanded-studies-v1-20260830` with executable-source
digest `7f43b602b71199367938c1df51aa201a7d601e57ebb47835bdb5c9282b42f84e`.
Unlike E19--E20, these exact sources are reconstructible from the recorded clean Git
commit. The compact archives are nevertheless not self-contained replay packages:
they omit respectively `192` and `1,792` `records/**` files. The omitted paths,
digests, and sizes remain committed by each `artifact_manifest.json`, but hashes
cannot recover the omitted bytes. This archive incompleteness is distinct from
source reconstructibility and does not weaken the preserved top-level outcomes,
seals, or integrity records.

Historical E1, E2, E2-R, exploratory E-NL, and E3 artifacts were also dirty but
predate source fingerprinting; their exact executed source cannot be reconstructed
retroactively. See `paper/artifacts/provenance_manifest.json`.

---

## 1. Frozen definitions

These are not claims; they are the vocabulary every claim below depends on. Changing
one invalidates the rows that use it.

1. **Causal hierarchy.** `d>0` is relative transfer, `d<0` is relative rate
   suppression, and `Delta<0` is outcome suppression. A response crossing is an
   outcome-suppression event after a positive excursion. A weak feature is called
   causally starved only when outcome suppression occurs and the matched weak-only
   trajectory demonstrably learns under the preregistered first-hit gate. The gate
   is not a footnote.
2. **Sequence time `k` versus optimization time `τ`.** Distinct axes. Any kernel or
   order parameter that involves both must say so explicitly.
3. **GSI-5 is a diagnostic**, predicted from the margin-weight distribution. It is
   not the definition of starvation and must never be used as one.
4. **No starvation label outside the learnable region.** A positive AUC gap or an
   outcome crossing where the weak-only counterfactual never learns is descriptive
   or indeterminate, not causal starvation. A failed weak-only learnability gate is
   indeterminate and establishes neither starvation nor its absence.
5. **A rate crossing is not an outcome or causal certificate.**
   `transfer_then_suppression` means the weak mode's rate was suppressed;
   `transfer_then_starvation` requires both a response-gap crossing and weak-only
   learnability. An unlearnable outcome crossing is emitted as
   `transfer_then_outcome_crossing_unlearnable`, never as starvation.
6. **Geometry-versus-field attribution is ordering-dependent.** Splitting a product
   difference admits two exact orderings. No dominance claim may be made unless both
   agree, which `dominance_ordering_invariant` records.
7. **The Waterbirds "weak coordinate" is an intercept, not an identified feature.**
   Regressing the margin on `(strong, 1)` makes the second coefficient the average
   signed margin unexplained by the strong mode. Nothing identifies a bird-shape
   response. Its CDC-style method is only a minibatch head-coordinate surrogate,
   outside the full-batch matched-shadow theorem, and has **not been run**; thus
   Waterbirds is a surrogate mechanistic probe only.
8. **`degenerate` is an artifact bucket, not a phase.** It marks runs whose target
   was already met at initialization, before any optimization step.
9. **`closure_reference` is a numerical approximation.** It is a mean over trained
   finite networks pushed through `integrate_projected_flow`. It is not a solved
   theory and must never be called DMFT.
10. **Full signed-logit empirical NTK versus two-probe Gram matrix.** For sample
    signed logits `r_i`, `J_r` has rows `grad r_i`, `K=J_r J_r^T`, and the response
    cross-kernel is `C_i=<grad r_i,grad M_w>`. These full sample-level objects drive
    the exact finite-width flow. The two-response Gram matrix `G` is not a substitute
    for `K` or `C` in nonlinear tanh/GRU models.
11. **Frozen-surrogate exactness is internal.** Freezing `K` and `C` at initialization
    gives an exact nonlinear logistic flow, and an exact explicit-Euler recursion,
    for the initialization-linearized tangent surrogate. Agreement with a trained
    nonlinear network is a conditional approximation requiring kernel-movement
    control; it is not an identity for tanh or GRU training.
12. **Crossing-event prediction is not causal prediction.** Correctly classifying a
    drift or response crossing does not predict the weak-only learnability gate,
    certify causal starvation, estimate event prevalence, or validate crossing
    times and trajectory magnitudes.
13. **Manifest verification is an integrity contract.** It establishes that a
    preflight did not train, that evaluation consumed the externally pinned files,
    config and executable source, and that those files remained unchanged. It does
    not establish kernel stability or any scientific approximation assumption.
14. **The terminal weak-only response profile is post-hoc sensitivity evidence.**
    `B_W(H)=M_w^W(H)` is threshold-free at a fixed horizon, but thresholding it is
    not generally equivalent to the preregistered first-hitting-time gate for a
    nonmonotone trajectory. The profile generates no new training outcome and does
    not establish an architecture ranking.

Implemented in `metrics.CAUSAL_REGIMES` and `metrics.classify_causal_regime`;
enforced by `tests/test_causal_regimes.py`. The signed-kernel definitions and frozen
recursion are implemented in `theory.initial_frozen_empirical_kernel` and
`theory.integrate_frozen_logistic_sgd`.

---

## 2. Theory

| # | Statement | Status | Backing |
|---|---|---|---|
| T1 | Universal finite-width signed-CE response flow through the response/logit cross-kernel | **PROVED** | `e2_theorem.md`, Proposition A.1; direct autograd in `theory.projected_statistics` |
| T2 | Closed two-mode flow `dm/dτ=Gg` when the signed logits have an exact parameter-independent two-mode realization | **PROVED WITH ASSUMPTION** | `e2_theorem.md`, Corollary A.2; exact for the linear synthetic model with zero background noise |
| T3 | Exact finite-width dense-linear-RNN projected geometry | **PROVED** | `theory.exact_dense_linear_geometry`; agrees with autograd at `rtol=2e-5` |
| T4 | Exact matched weak-drift decomposition into CE gating, geometry shift, and cross transport under exact two-mode realization | **PROVED WITH ASSUMPTION** | `theory.matched_weak_drift_decomposition`; reconstruction error `<2e-6` |
| T5 | Both equal-time product-difference orderings reconstruct the projected drift exactly; neither attribution is canonical | **PROVED** | `theory.equal_time_drift_difference`; both algebraic reconstructions tested |
| T6 | General cumulative drift balance: for arbitrary sign changes, `Delta(t)=P(t)-N(t)`, so strict outcome suppression occurs exactly when cumulative negative drift exceeds cumulative positive drift at some time; under one `+ -> -` crossing this reduces to the tail-area criterion | **PROVED** | `e2_theorem.md`, Proposition B.1 and Corollary B.2; executable finite-step single-crossing analogue in `discrete_crossover_certificate` |
| T7 | Noiseless rank-one cue log drift ratio and monotone sufficient condition for a unique **rate** crossover | **PROVED WITH ASSUMPTIONS** | `e2_theorem.md`, Theorem C and Corollary C.1; does not imply outcome suppression without T6 or causal starvation without the independent learnability gate |
| T8 | Quantitatively isolated transverse first hitting times are stable under uniform trajectory error, with radius/mesh guards and finite paired hits | **PROVED WITH ASSUMPTIONS** | `e2_theorem.md`, Corollary D; caller establishes first-entry semantics, prehistory margin, derivative bound, and any grid validity |
| T9 | In the dense linear full-batch CE flow with zero lag and zero background input, all pre-final states vanish, so any finite realized recurrent matrix is loss-invisible and the projected dynamics close exactly on six scalars at every width | **PROVED** | `e2_theorem.md`, Theorem E.1 specialized proof plus the arbitrary-`W` extension in `parameter_prediction_theorem.md`, Corollary P.4; `dmft.solve_zero_disorder` is an executable realization with a legacy name, not the proof |
| T10 | With iid `N(0,1/N)` input/readout initialization, a fixed bounded-second-moment final-step cue law, fixed horizon, and deterministic radius `r`, the zero-lag recurrence-invisible six-scalar process converges uniformly with initialization bound `9/(N ε²)` and trajectory bound `9 exp(2L_(H,r)H)/(N min(r,δ)²)`, independently of any finite realized recurrent matrix | **PROVED WITH ASSUMPTIONS** | `e2_theorem.md`, Theorem E.2 and the zero-lag arbitrary-`W` observation; deterministic `A,C,L_(H,r)` and exit-time bootstrap are proved there; helpers only evaluate caller-supplied constants/radius |
| T11 | CDC Result 1: exact instantaneous strong-drift preservation | **PROVED** | `cdc_theorem.md`, Result 1 |
| T12 | CDC Result 2: unique uncapped minimum Euclidean-norm, lower-bound-attaining feasible correction in the fixed implemented Euclidean/Frobenius chart | **PROVED WITH ASSUMPTIONS** | `cdc_theorem.md`, Result 2; exact positive-deficit feasibility is `q!=0`, numerical `feasibility_epsilon` is only a tolerance, and a binding cap loses target-attaining optimality |
| T13 | CDC Result 3: fixed-chart one-step strong-response deviation is `O(η²)` when local `L`-smoothness holds on both straight coordinate segments; the explicit bound also assumes bounded velocity/deficit and `norm(q)>=q_min` | **PROVED WITH ASSUMPTIONS** | `cdc_theorem.md`, Result 3; actual-velocity bound is cap-compatible, explicit `D/q_min` derivation is active uncapped (or reduced by a nonnegative cap), and the helper does not establish smoothness or segment containment |
| T14 | The full finite-width signed-logit/response empirical-NTK flow is exact: `dot r=K(theta) sigma(-r)/n` and `dot M=C(theta)^T sigma(-r)/n` | **PROVED** | Chain rule as in `e2_theorem.md`, Proposition A.1 and Corollary A.4; sample-level signed Jacobians and the initial identity are implemented in `theory.initial_frozen_empirical_kernel` and tested for tanh/GRU |
| T15 | Freezing the full signed-logit NTK `K_0` and response cross-kernel `C_0` defines an exact nonlinear logistic tangent-surrogate ODE; `integrate_frozen_logistic_sgd` is the exact explicit-Euler recursion of that surrogate under the experiment's mean-BCE scaling and `tau=eta k` convention | **PROVED** | `theory.integrate_frozen_logistic_sgd`; initial drift and grid conventions tested. Exactness is for the surrogate, not the trained nonlinear model |
| T16 | Continuous true-flow versus frozen-flow error, and discrete nonlinear-SGD versus frozen-Euler error, are bounded conditionally by uniform instantaneous/secant drift bounds for `K` and `C`; zero movement gives exact agreement | **PROVED WITH ASSUMPTIONS** | Derivation in `oral_theorem_package.md`, “Conditional secant-kernel comparison”; discrete evaluator `theory.frozen_kernel_discrete_error_bound` is tested. The caller must prove the kernel bounds |
| T17 | General joint paired CE-RNN optimization-time DMFT at positive lag, including recurrent disorder | **BLOCKED CONJECTURE** | `e2_theorem.md` obligations 1–4; all dependent positive-lag solver surfaces still raise `NotImplementedError` |
| T18 | Certified or quantitatively accurate tanh/GRU crossover-time prediction and an analytic learnability/starvation boundary `rho_c` | **OPEN / BLOCKED** | The frozen surrogate supplies an empirical event classifier, not certified kernel control or calibrated times; T17 or another independent closure/bound is still required |
| T19 | In the noiseless positive rank-one dense-linear pair, the realized initialization gives the exact rate gap `d_0=s_B(rho K+A)-s_W A` and threshold `K>(A/rho)(s_W/s_B-1)`; because shared initialization has `Delta(0)=0`, strict negative/positive `d_0` respectively implies strict local outcome suppression/transfer for some unspecified sufficiently small positive interval, while a caller-tolerance band is boundary/undetermined; independently proved product-ball bounds then imply unique rate crossing, outcome suppression, nondegenerate weak-only learnability, or finite-horizon safe transfer | **PROVED WITH ASSUMPTIONS** | `parameter_prediction_theorem.md`, Theorems P.1--P.2 and Corollary P.3; `theory.dense_linear_initial_gap_certificate` and `theory.parameter_tube_certificate`; initialization-met targets are recorded as degenerate and cannot certify learnability or causal starvation; no quantitative local-sign horizon or product-ball constants are constructed |
| T20 | With `cue_noise=0`, zero background noise, zero lag, and any finite realized recurrent matrix (hence arbitrary recurrent bulk gain), the deterministic symmetric reference satisfies `bar d(0)=0`, `bar d'(0)=-rho^2/2<0`, and weak-only attainment of every finite positive target; thus suppression begins immediately rather than after transfer, with the corresponding finite-width conclusion requiring T10 witness margins | **PROVED WITH ASSUMPTIONS** | `parameter_prediction_theorem.md`, Corollary P.4; zero lag makes all pre-final states vanish and `W` loss-invisible; positive lag remains blocked |

The findings-first synthesis is that **weak learning failure alone is not causal
gradient starvation**. At finite width, the exact weak-response dynamics are
`dot M_w=c_w^T q(r)/n`: the cross-entropy weights `q(r)` and response-specific
response--logit geometry `c_w` are distinct objects. The matched BOTH/WEAK
intervention then separates relative rate suppression, outcome suppression, and
weak-only gate failure from causal starvation; a failed learnability gate is
indeterminate and establishes neither starvation nor its absence. Pathwise, the
general balance is `Delta=P-N`, so outcome suppression occurs exactly when
accumulated negative relative drift overtakes accumulated positive relative drift,
while causal starvation additionally requires the independent matched weak-only
gate.

The theorem package T1--T16 and T19--T20 supports that causal account without
solving the blocked global predictor. T19's realized-initialization threshold gives
an existential strict local outcome sign outside its caller indifference band, and
higher initialization-jet information such as T20's `d'(0)` is likewise local or
assumption-scoped. Conditional product-ball implications require independently
proved nonvacuous constants; none is currently constructed for general positive
lag. T14--T16 make the time-varying finite-width flow and the frozen surrogate exact
in their stated objects, but using the initialization-frozen surrogate for trained
tanh/GRU networks is an empirical event heuristic unless the unproved kernel-
movement premises are supplied. Thus the global positive-lag parameter-level
predictor from `rho`, lag, and recurrent geometry remains unresolved, as does the
general positive-disorder/positive-lag optimization-time DMFT.

---

## 3. Empirical results

| # | Statement | Status | Backing run |
|---|---|---|---|
| E1 | Under the final common-probe/direct-autograd protocol, tanh exhibits an exact transfer-to-rate-suppression crossing, response crossing, and causal tail-area certificate in **3/8** seeds | **EMPIRICAL** | `paper/artifacts/enl_tanh_crossover-20260824-141207`; conditional on the three crossing seeds, exact drift `tau*` mean `0.1942` and response-crossing mean `0.3890` |
| E2 | The other **5/8 tanh seeds are suppressed from initialization**, not transfer-to-starvation; each of the three crossing seeds has one exact `+ -> -` sign change | **EMPIRICAL** | same; phases are `suppression_throughout` or `transfer_then_starvation` |
| E3 | In each of the three tanh crossing seeds, exact drift suppression precedes response equality; conditional mean lead `0.1947` in optimization time | **EMPIRICAL** | same; this is conditional descriptive evidence, not a certified or quantitatively accurate `tau*` prediction |
| E4 | GRU has exact drift and response crossings in **5/8** seeds, but weak-only never reaches `beta=0.5` in **8/8** seeds, so causal starvation is certified in **0/8** | **EMPIRICAL** | same; outcome crossings without the learnability gate must not be called causal starvation |
| E14 | ~~tanh late-time suppression is geometry-dominated~~ | **RETRACTED** | See R9. The additive projected attribution is not ordering-invariant and does not replace the exact nonlinear response drift |
| E15 | Direct-versus-projected drift residuals are nonzero and exposed: maximum absolute residual `0.02197` for tanh and `0.002986` for GRU | **EMPIRICAL** | `paper/artifacts/enl_tanh_crossover-20260824-141207`; confirms that projected `Gg` cannot be silently treated as exact for nonlinear probes |
| E16 | GRU's final projected geometry/field sign attribution is ordering-invariant in 8/8 seeds, but it is diagnostic only and does not overcome the failed causal-learnability gate | **EMPIRICAL** | same; no causal-starvation or exact-drift mechanism claim follows |
| E17 | Both projected equal-time orderings reconstruct their projected drift difference to at most `9.54e-7` (tanh) and `5.59e-9` (GRU) at logged points | **EMPIRICAL** | same; per-row reconstruction is asserted in `tests/test_experiment_smoke.py` and `tests/test_training.py` |
| E18 | The exploratory 25-cell tanh grid found **no 4/4 robust condition**; the best cells reached 3/4 causal certificates | **EMPIRICAL NEGATIVE RESULT** | `enl_exact_regime_search-20260824-105003`; exploratory seeds 0–3 only, so no searched cell is confirmatory evidence |
| E5 | Zero-disorder solver is the correct continuous limit; residual is `O(η)` discretization, fitted slope `1.0115` | **EMPIRICAL** | `e2r_solver_checks-20260823-084911` |
| E6 | Frozen-geometry solver agrees with `integrate_projected_flow` to `1.91e-15` | **EMPIRICAL** | same |
| E7 | Under the corrected dense parameterization the AUC gap is monotone increasing in `ρ` at every lag | **EMPIRICAL** | `e1_dense_rerun-20260823-085142` |
| E8 | Corrected E1: all 32 `ρ`–seed cells per lag (four strengths by eight seed IDs) have positive AUC gap at lags 0/2/4; the negative control is 8/8 `transfer`; lag 8 is `unlearnable` at every `ρ` | **EMPIRICAL** | same |
| E9 | CDC Result 3 scaling: fitted log-log slope `2.0000` (dense, low-rank), `1.9961` (tanh) | **EMPIRICAL** | `tests/test_cdc_families.py`. Requires float64; float32 quantization gives a spurious `1.78` |
| E10 | CDC is inactive before the crossover, because no weak-drift deficit exists yet | **EMPIRICAL** | `tests/test_cdc_families.py::test_no_correction_is_applied_without_a_deficit` |
| E11 | All five shadow-based methods are *numerically similar* on the causal weak gap, agreeing to ~4 decimal places | **EMPIRICAL** | `e3_cdc_dense_ablation-20260823-090702`. **Not** statistically indistinguishable: 9 of 10 direct pairwise tests are significant at 0.05 (e.g. CDC − bloop `−0.000720`, `p = 2.9e-07`). Differences are detectable and practically negligible against the `≈ −10.5` effect versus ERM. Equivalence would need a preregistered margin |
| E12 | Only CDC preserves the instantaneous strong drift: `4.77e-07` against `1.5e-01`–`3.8e-01` | **EMPIRICAL** | same |
| E13 | `unconstrained_rescue` and `pcgrad` end with a *higher* final strong response than CDC (`1.295` vs `1.042`) | **EMPIRICAL** | same |
| E19 | The sealed broad empirical-NTK pilot **failed** its preregistered acceptance rule: classifier correctness was phase `12/16`, drift crossing `14/16`, response crossing `14/16`, causal certificate `14/16`, and weak-only learnability `12/16`; tanh learnability was only `4/8`, below the `0.625` per-model threshold | **EMPIRICAL NEGATIVE RESULT** | `results/enl_ntk_pilot_evaluate-20260825-160757`; manifest `f186d587...`, predictions `9787d999...`, source `8d816255...` |
| E20 | On the fresh restricted `4` data-seed by `8` model-seed factorial for each architecture (`64` records), the frozen surrogate classified response-crossing events correctly in `64/64` and drift-crossing events in `60/64` (`32/32` tanh, `28/32` GRU); two-way-bootstrap 95% lower bounds were respectively `1.0` and `0.8125` | **EMPIRICAL, RESTRICTED SUCCESS** | `results/enl_ntk_crossing_factorial_evaluate-20260825-162949`; only drift/response crossing were preregistered primary outputs; manifest `bdcf02ab...`, predictions `c05b3181...`, source `ca878df3...` |
| E21 | In that factorial, phase `52/64`, causal-certificate `56/64`, and learnability `49/64` are classifier-correctness counts, not event prevalence, and were explicitly secondary/rejected after the failed broad pilot | **EMPIRICAL, NON-PRIMARY** | same; these endpoints cannot be promoted to causal prediction or used to rescue the broad hypothesis |
| E22 | Restricted-factorial crossing times were systematically early (bias: tanh drift `−0.0593`, tanh response `−0.1380`, GRU drift `−1.4330`, GRU response `−1.8199`), and tanh trajectory magnitudes were poor (`2.455` response-gap RMSE, `2.489` weak-response RMSE, `0.490` drift RMSE) | **EMPIRICAL LIMITATION** | same; times and trajectory magnitudes were secondary and uncalibrated, so quantitatively accurate time prediction remains open |
| E23 | Preflight/evaluation integrity passed for both studies: the evaluator verified `24` pilot and `72` factorial sealed files before training and re-hashed them unchanged afterward | **EMPIRICAL SOFTWARE/PROVENANCE** | each evaluation's `provenance.json`; this does not prove tanh/GRU kernel stability or surrogate accuracy |
| E24 | In a post-hoc audit of the tracked final E-NL summaries, terminal weak-only `B_W(H)` min/mean/max are `4.41313266754 / 4.49664855003 / 4.67605876923` for tanh (8/8 above `beta=0.5`) and `0.0828229486942 / 0.129690139554 / 0.187142759562` for GRU (0/8); the GRU gate failure is not marginal at that operating point | **EMPIRICAL POST-HOC SENSITIVITY** | `paper/artifacts/enl_tanh_crossover-20260824-141207/learnability_profile*`; derived without training from tracked `summary.csv`; terminal reach is not generally a first-hit replacement and supports no architecture ranking |
| E25 | The citable evidence now includes controlled synthetic studies and the semi-real generated-cue intervention E26, which uses real MNIST 3-vs-8 and FashionMNIST 0-vs-6 core images plus an exactly removable generated cue channel; Waterbirds remains unrun, and all reported proportions are study-specific descriptions rather than prevalence estimates | **EMPIRICAL SCOPE LIMITATION** | protocol tables and named artifacts; the semi-real intervention does not establish behavior on an unmodified natural task, and Waterbirds remains outside the full-batch matched-shadow theorem |
| E26 | In the completed semi-real generated-cue study, MNIST had `0/32` weak-only-learnable records and `0/32` causal certificates, with mean weak-AUC gap `−0.0012362842136667493` and 95% CI `[−0.0034405428466004646, 0.0010159591371180453]`; FashionMNIST likewise had `0/32` weak-only-learnable records and `0/32` causal certificates, with mean weak-AUC gap `−0.00010921728159018996` and 95% CI `[−0.004129384520886106, 0.003182670049955049]` | **EMPIRICAL NEGATIVE / INDETERMINATE RESULT** | `paper/artifacts/semi-real-generated-cue-v1-20260830`; zero causal certificates cannot be interpreted as evidence of no starvation because the weak-only learnability gate failed in every record of both datasets |
| E27 | In completed expanded Study A (`192` records), the preregistered beta-robust architecture endpoint was negative for tanh, point estimate `0.2916666666666667`, 95% CI `[0.16666666666666666, 0.4166666666666667]`, Holm-adjusted `p=1`, and for GRU, point estimate `0`, 95% CI `[0, 0]`, Holm-adjusted `p=1` | **EMPIRICAL NEGATIVE RESULT** | `paper/artifacts/expanded-studies-v1-20260830`; neither architecture satisfies the frozen beta-robust claim, and no architecture ranking was performed or is supported |
| E28 | In completed expanded Study B (`256` method-records), against the **Bloop-style shadow-target rescue** (`canonical: false`), CDC minus comparator weak rescue was `0.002198259399210656` with 95% CI `[0.0009779187294930126, 0.003277366267916477]`, comparator minus CDC trajectory deviation was `3.4651361294978416` with 95% CI `[3.3443827215131754, 3.5849715262780952]`, and comparator minus CDC final deviation was `0.6422362388111651` with 95% CI `[0.6288088704226539, 0.6563168084365315]`; the frozen joint tradeoff was true. Against the **PCGrad-style shadow-target rescue** (`canonical: false`), the corresponding estimates were `−0.00015351238407674823` with 95% CI `[−0.00047909348592838785, 0.00013248012419808214]`, `−0.48709889128076556` with 95% CI `[−0.6897378944428417, −0.27379327373499074]`, and `−0.2565436437726021` with 95% CI `[−0.3016841153614223, −0.2087651835754514]`; the frozen joint tradeoff was false | **EMPIRICAL, RESTRICTED MIXED RESULT** | `paper/artifacts/expanded-studies-v1-20260830`; only the Bloop-style comparator satisfies the frozen joint tradeoff, the PCGrad-style comparator does not, neither comparator has canonical parity, and no broad CDC superiority is supported |

---

## 3A. Completed externally sealed execution

Both formerly prospective programs were externally sealed before outcome generation,
authorized, and executed from clean Git commit
`9ebbc617a9aaad5c3c19e9d5fd4fa08f2d3517d0` with untouched frozen inputs. Their
outcomes are E26--E28 above; this section records execution status rather than adding
new claims.

| Program | Frozen and completed scope | Status | Archived evidence |
|---|---|---|---|
| Semi-real generated-cue intervention | `64` records across MNIST 3-vs-8 and FashionMNIST 0-vs-6, using common real core pixels plus an exactly removable label-aligned cue channel and paired BOTH/WEAK probes | **COMPLETED; E26 NEGATIVE / INDETERMINATE** | `paper/artifacts/semi-real-generated-cue-v1-20260830`; external pre-outcome seal and human authorization archived; clean source digest `02ba4ce4d6aff3ff726d9e798606aed29cff1276a6837e10ff43fb37824ca042` |
| Expanded nonlinear and CDC tradeoff | Study A: `192` nonlinear records over frozen `beta` values `[0.25,0.5,1,2]`; Study B: `256` method-records for ERM, CDC, Bloop-style shadow-target rescue, and PCGrad-style shadow-target rescue | **COMPLETED; E27 NEGATIVE AND E28 RESTRICTED MIXED** | `paper/artifacts/expanded-studies-v1-20260830`; external pre-outcome seal and human authorization archived; clean source digest `7f43b602b71199367938c1df51aa201a7d601e57ebb47835bdb5c9282b42f84e` |

The frozen Study-B victory rule was conjunctive: weak rescue plus trajectory-level
and final strong-preservation superiority, with feasibility, cap, cost, and
reuse-aware inference reporting. It was satisfied only against the Bloop-style
shadow-target rescue, not against the PCGrad-style shadow-target rescue. Both
comparators have `canonical: false`; canonical named-baseline parity remains
unaudited, so the restricted mixed result cannot be promoted to broad CDC
superiority.

---

## 4. Retractions and narrowings from this pass

| # | Statement | Status | Why |
|---|---|---|---|
| R1 | "`τ*` predicted from `T_geom = S_CE` can be compared against the observed `d_w` sign change" | **RETRACTED** | Degenerate. `d_w = t_geom − s_ce` is an identity, so the two agree by construction; measured gap `1.52e-07`. Emitted only as a self-consistency check |
| R2 | "`T_geom` dominates early and `S_CE` dominates late" | **NARROWED** | Right about the crossover, wrong about the late regime. `T_geom` itself turns negative at `τ ≈ 2.75` and reaches `−2.78` while `S_CE` saturates near `0.79`, so late suppression is geometry-driven, not CE-gating-driven |
| R3 | "The `ΔT_w` and AUC-gap metrics rank the lag axis in opposite directions" | **NARROWED** | Measured on the superseded run. Under the corrected parameterization the two orderings differ but no longer oppose. The requirement that a boundary state its metric stands |
| R4 | E1 per-lag AUC magnitudes `19.20 / 8.96 / 1.49` | **RETRACTED** | Superseded parameterization. The corrected run peaks at lag 2: `5.76 / 8.93 / 6.11` |
| R5 | "BOTH-feature E2 RMSE worsens with width, so the closure is incomplete" | **RETRACTED** | True only of `e2_width_extension-20260821-134952` (`19.862 → 25.975`), which is superseded. The corrected run improves (`0.09084 → 0.02958`, non-monotone at N=128). The solver is still needed, for the epistemic reason that a reference calibrated from trained networks cannot falsify the theory |
| R6 | "The historical E3 CDC run beats the Bloop-style shadow-target rescue and the ablations on the causal weak gap" | **RETRACTED** | It does not; see E11. The defensible claim for that historical E3 run is narrower: CDC is the unique family member preserving the instantaneous first-order response of a theory-identified feature, and that run does not demonstrate that the property has practical value. This finding is separate from the completed E28 restricted tradeoff test |
| R7 | The old E-NL 8/8 tanh crossover, `tau*=1.6112`, and GRU 0/8 narrative | **RETRACTED AND SUPERSEDED** | Those values used a condition-dependent/projection-based response convention. Under the final common symmetric probe and universal direct-autograd drift, `paper/artifacts/enl_tanh_crossover-20260824-141207` gives tanh 3/8 exact drift/response/causal crossings and GRU 5/8 exact drift/response crossings but 0/8 causal certifications. The old numerical `tau*` is invalid for the final response definition |
| R9 | "Late tanh suppression is geometry-driven" (E14) | **RETRACTED** | Two defects. First the split was algebraically wrong: it paired ordering A's geometry term with ordering B's field term, reconstructing nothing — mismatch up to `0.753`, e.g. a row with `d_w = 0.011` reported `geometry 0.923`, `field −0.159`, summing to `0.764`. Second, and fatally for the universal claim, splitting a product difference admits two exact orderings. At the final logged point they agree on which channel dominates in 7/8 tanh seeds, not all 8/8; across the full trajectories the ordering-invariant fractions are only `0.188–0.426`. Both orderings are now computed with per-row reconstruction tests, and a `dominance_ordering_invariant` flag gates any per-point dominance statement. The universal tanh claim is withdrawn; the GRU final-point statement survives as E16 because it is invariant in 8/8 seeds |
| R10 | "A transfer-to-rate-suppression crossing necessarily precedes a response crossing" | **RETRACTED as stated** | A `+ -> -` derivative crossing establishes a local maximum of the response gap, not that the gap later reaches zero. The negative tail-area condition in T6 is necessary and sufficient. In the final tanh run the ordering holds only for the 3/8 crossing seeds; the other 5/8 are suppressed from initialization. In GRU it holds for 5/8 outcome-crossing seeds, but none passes the weak-only learnability gate |
| R8 | "Waterbirds CDC preserves the instantaneous strong drift" | **RETRACTED as previously implemented; corrected surrogate not run** | The former code froze the backbone by writing only head gradients and used AdamW with decay, so it was outside the theorem. Those implementation defects are fixed, but the remaining CDC-style method is a minibatch head-coordinate surrogate outside the full-batch matched-shadow theorem. No corrected Waterbirds run or result exists |
| R11 | "The broad initialization-frozen empirical-NTK predictor passed held-out evaluation" | **RETRACTED** | The 16-record pilot failed because tanh weak-only-learnability correctness was `4/8`; aggregate correctness on other labels does not override the preregistered per-model failure |
| R12 | "The initialization-frozen surrogate is the exact tanh/GRU training dynamics" or "tanh/GRU kernels are stable" | **RETRACTED / UNPROVED** | The signed-logit/response flow is exact with time-varying kernels, and the frozen recursion is exact for its own tangent surrogate. Relating it to nonlinear training requires uniform instantaneous/secant kernel-movement bounds that have not been proved or certified for either architecture |
| R13 | "The restricted factorial validates phase, causal starvation, or learnability prediction" | **NARROWED** | Only drift- and response-crossing event classification were frozen as primary after the broad pilot failed. Phase, causal certificate, and learnability were explicitly rejected as primary and retained as uncalibrated secondary diagnostics |
| R14 | "High crossing-event accuracy establishes calibrated crossing times, causal prevalence, or universal recurrent behavior" | **RETRACTED** | Classifier correctness is not prevalence. Predictions were systematically early, tanh trajectory magnitudes were poor, the experiment covered one width/task cell, and GRU outcome crossings still failed the causal learnability gate |
| R15 | "Zero causal certificates in E26 show that starvation is absent on MNIST and FashionMNIST" | **RETRACTED / INDETERMINATE** | Weak-only learnability was `0/32` in each dataset, so the causal gate was never evaluable as positive; zero certificates cannot distinguish absence of starvation from an unlearnable weak-only task under this design |
| R16 | "E27 establishes beta robustness or an architecture ranking between tanh and GRU" | **RETRACTED** | Both frozen beta-robust architecture claims were negative with Holm-adjusted `p=1`; no architecture ranking was performed or supported |
| R17 | "The E28 Bloop-style shadow-target rescue and PCGrad-style shadow-target rescue comparators are canonical implementations or parity-audited named baselines" | **RETRACTED / UNVERIFIED** | Both the Bloop-style shadow-target rescue and PCGrad-style shadow-target rescue are explicitly `canonical: false`; the completed study supports claims only against the fully named shadow-target rescue variants actually run |
| R18 | "E28 establishes broad CDC superiority over Bloop-style shadow-target rescue and PCGrad-style shadow-target rescue" | **NARROWED** | The frozen joint tradeoff was true only against the Bloop-style shadow-target rescue and false against the PCGrad-style shadow-target rescue. The result is restricted, mixed, and cannot establish broad superiority |

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

The baseline labels require study-specific wording. In historical E3/E11, the
implementations were the authors' readings of the methods rather than verified
reimplementations, so the result must be stated against the Bloop-style baseline
actually run. In E28, the full labels are **Bloop-style shadow-target rescue** and
**PCGrad-style shadow-target rescue**, and both have `canonical: false`. No E11 or
E28 result establishes canonical Bloop or PCGrad parity; E28 supports the frozen
joint tradeoff only against the Bloop-style comparator and does not support broad
CDC superiority.

---

## 7. Statements that must never appear

1. A passing test suite proves software invariants, not the theorem.
2. `e2_acceptance.json` passing is convergence to a held-out empirical closure.
   Only a future `e2r_acceptance.json` carrying a frozen independent solution would
   support "DMFT validation". The current record reports `passed: false` by design.
3. A positive AUC gap outside the learnable region is starvation.
4. The historical E3 ablation proves CDC preserves the final strong response. It
   establishes only instantaneous first-order drift preservation; in that historical
   run the final response is lower than ERM and lower than the unconstrained variant.
5. Any quantitative claim sourced from `e2_medium_validation`,
   `e2_high_confidence`, `e2_width_extension`, or the 13:xx E1/E3 batch.
6. Any Waterbirds result. Nothing has been run.
7. The broad empirical-NTK pilot passed, or aggregate accuracy rescues its failed
   tanh learnability criterion.
8. The initialization-frozen surrogate is exact for trained tanh/GRU networks, or
   the manifest/preflight establishes nonlinear kernel stability.
9. The restricted factorial predicts phase, weak-only learnability, causal
   starvation, calibrated crossing times, or trajectory magnitudes.
10. Crossing-event classifier correctness is causal-event prevalence or evidence
    of universal recurrent behavior.
11. The manifest hashes prove the scientific approximation rather than artifact,
    configuration, and executable-source integrity.
12. Certified or quantitatively accurate tanh/GRU crossing-time prediction is
    complete; it remains open.
13. E26's zero causal certificates show no starvation on MNIST or FashionMNIST.
    Both datasets had `0/32` weak-only-learnable records, so that inference is
    indeterminate rather than negative evidence for starvation.
14. E27 establishes beta robustness for either architecture or supports an
    architecture ranking between tanh and GRU. Both frozen architecture claims were
    negative, and no ranking was performed.
15. Either E28 comparator is canonical Bloop or canonical PCGrad, or named-method
    parity has been audited. Both shadow-target rescue comparators have
    `canonical: false`.
16. E28 establishes broad CDC superiority. Its frozen joint tradeoff passed only
    against the Bloop-style shadow-target rescue and failed against the PCGrad-style
    shadow-target rescue.
