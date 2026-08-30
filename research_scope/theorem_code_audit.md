# Theorem-to-code audit

Date: 2026-08-29

## Scope and interpretation

This audit maps the proved T1--T16 and T19--T20 finite-width theorem package,
plus blocked T17--T18, to proofs, executable surfaces, and direct tests. A numerical
helper is not treated as a proof of hypotheses supplied by its caller. Passing
tests establish software identities and guards; they do not prove that a trained
trajectory lies in a required tube, that nonlinear kernels remain near
initialization, that a blocked positive-lag closure exists, or that caller-supplied
product-ball constants have been constructed.

The authoritative mathematical sources are `e2_theorem.md`, `cdc_theorem.md`,
`oral_theorem_package.md`, and `parameter_prediction_theorem.md`. The canonical
claim status is in `claim_ledger.md` and `submission_claim_set.yaml`.

## Audit matrix

| ID | Exact scope and assumptions | Proof location | Executable role | Direct test coverage | Unimplemented or caller-supplied obligations |
|---|---|---|---|---|---|
| T1 | Any differentiable finite-width model under unregularized mean logistic-loss gradient flow | `e2_theorem.md`, Proposition A.1 | `projected_statistics(..., compute_direct_drift=True)` evaluates the response/logit cross-kernel contraction | Direct response drift is compared with gradient flow for dense linear models; T14 tests the same contraction through the full signed-logit Jacobian for tanh/GRU | No separate test differentiates the complete logit equation along a parameter step |
| T2 | `dot M=Gg` only when every signed logit has an exact parameter-independent realization in the selected responses | `e2_theorem.md`, Corollary A.2 | `projected_statistics` returns the projected component and an explicit residual | Exact zero residual for dense linear, zero-background-noise data; nonzero nonlinear residual is exposed | No exact two-probe closure for tanh/GRU; the projected component must not be promoted to the universal drift |
| T3 | Dense linear RNN and channel-localized unit temporal probes | `e2_theorem.md`, Corollary A.3 | `exact_dense_linear_geometry` computes the three parameter-block contributions | Compared directly with autograd | Applies neither to low-rank parameter coordinates nor nonlinear recurrent models without a new derivation |
| T4 | Matched-state projected weak-drift identity under T2's exact realization | Algebra implemented in `matched_weak_drift_decomposition`; claim recorded in `claim_ledger.md` | Evaluates CE-gating, geometry-shift, and cross-transport terms | Exact reconstruction test | It is an algebraic identity, not an independently established trajectory theorem; nonlinear direct-drift residual remains outside it |
| T5 | Equal-time projected product difference, with both valid algebraic orderings | `equal_time_drift_difference` and claim ledger | Returns both reconstructions and an ordering-invariance flag | Both orderings are reconstructed in tests | No attribution is canonical; a dominance claim is invalid when the orderings disagree |
| T6 | Absolutely continuous response gap with common initial response; arbitrary-sign cumulative balance, plus stronger one-crossing corollary | `e2_theorem.md`, Proposition B.1 and Corollaries B.2--B.3 | `discrete_crossover_certificate` is only a strict one-crossing finite-step analogue | Tail-area distinction and sign-pattern guards are tested | The helper does not implement the arbitrary-sign continuous theorem, and its legacy `strict_outcome_starvation` name actually denotes outcome suppression unless the separate learnability gate passes |
| T7 | Noiseless positive rank-one cue law, exact two-mode realization, positive geometry factors; uniqueness additionally requires an independently established negative derivative bound | `e2_theorem.md`, Theorem C and Corollary C.1 | `rank_one_drift_ratio` evaluates the pointwise factorization and log ratio | Factorization and sign equivalence are tested | No implementation proves `Psi'`, the primitive monotonicity inequalities, uniqueness, or the crossing-time bound on an interval |
| T8 | Uniform trajectory error plus quantitative prehistory separation, local transversality, radius, finite first hit, and mesh assumptions | `e2_theorem.md`, Corollary D | `transverse_hitting_time_error_bound` evaluates the supplied scalar bound and guards | Radius, mesh, transversality, and separation guards are tested | The helper does not establish first-entry semantics, derivative bounds, tube error, paired finiteness, or grid validity |
| T9 | Dense linear, full-batch CE, zero lag, zero background input, zero initial hidden state, and any finite realized recurrent matrix | `e2_theorem.md`, Theorem E.1, with the arbitrary-`W` zero-lag extension proved in `parameter_prediction_theorem.md`, Corollary P.4 | Legacy-named `dmft.solve_zero_disorder` integrates the exact recurrence-invisible six-scalar closure; `zero_lag_six_scalar_rhs` exposes its vector field | Direct tests use nonzero, differing recurrent matrices and bulk gains to check responses, loss, input/readout gradients, zero recurrent gradients, geometry, solver trajectory, and dispatch invariance | The numerical solver is not the proof; zero background input is an implicit solver premise, and positive lag remains blocked because recurrence is generally visible there |
| T10 | IID `N(0,1/N)` input/readout initialization, fixed bounded-second-moment final-step cue law, fixed horizon, explicit deterministic reference tube, and any finite realized recurrent matrix | `e2_theorem.md`, Theorem E.2 plus T9's arbitrary-`W` extension | `zero_disorder_initialization_failure_bound` and `zero_disorder_trajectory_failure_bound` evaluate the formulas | Scaling, saturation, radius guards, and arbitrary-bulk-gain zero-lag solver invariance are tested | The helpers trust the supplied Lipschitz constant, radius, and deterministic reference; they do not validate interval integration, construct a tube, or extend to positive lag |
| T11 | CDC correction in the fixed Euclidean/Frobenius parameter chart | `cdc_theorem.md`, Result 1 | `counterfactual_drift_correction` / `drift_correction` construct the protected correction | Exact instantaneous strong-drift preservation is tested | This is instantaneous and coordinate-dependent; it does not imply final strong-response superiority |
| T12 | Positive-deficit uncapped CDC with a nonzero projected rescue direction | `cdc_theorem.md`, Result 2 | CDC returns target residual, correction norm, feasibility, and cap status | Target attainment and preservation are tested | There is no direct test comparing against every arbitrary feasible alternative; a binding cap forfeits target-attaining optimality |
| T13 | Fixed-chart local smoothness on both update segments, bounded velocity/deficit, and a projected-direction lower bound | `cdc_theorem.md`, Result 3 | `cdc_finite_step_deviation_bound` evaluates the one-step bound | Quadratic scaling is tested across model families | Local smoothness and segment containment are caller obligations; the helper does not certify them |
| T14 | Any differentiable finite-width model under mean logistic-loss flow, using the complete signed sample-logit tangent space | `e2_theorem.md`, Corollary A.4 | `empirical_logit_jacobian` and `initial_frozen_empirical_kernel` materialize `J`, `Theta`, and the response cross-kernel | Initial response drift is reconstructed for tanh and GRU | The time-varying equations are exact but not closed because the kernels move |
| T15 | Initialization-frozen full signed-logit NTK and response cross-kernel | `e2_theorem.md`, Corollary A.4 | `integrate_frozen_logistic_sgd` is the exact explicit-Euler recursion of the frozen surrogate | Optimization-time grid and initial drift are tested | Exactness is internal to the surrogate, not to trained tanh/GRU dynamics |
| T16 | Uniform instantaneous or secant-kernel movement bounds supplied over the full interval/steps | `e2_theorem.md`, Theorem A.5; `oral_theorem_package.md` | `frozen_kernel_discrete_error_bound` evaluates the discrete formula | Zero movement gives zero bound; perturbed movement gives positive error | No secant-kernel constructor, continuous evaluator, or tanh/GRU movement certificate establishes the premises |
| T17 | General joint paired CE-RNN optimization-time DMFT at positive lag, including recurrent disorder | Blocked on `e2_theorem.md` obligations 1--4 | General positive-lag solver surfaces raise `NotImplementedError` with the unmet obligations | Positive-lag refusal is a tested contract; a direct dense example confirms that changing `W` can change the weak response | No effective process, closure, well-posedness, or concentration proof exists; arbitrary gain at zero lag does not unblock positive lag |
| T18 | Certified tanh/GRU crossover times or a universal analytic learnability/starvation boundary | Open/blocked in the claim ledger | Frozen-kernel code is only a surrogate/predictor | Existing tests establish surrogate identities, not nonlinear kernel control or calibrated time accuracy | T17 or another independent closure/bound remains required |
| T19 | Noiseless positive rank-one dense-linear pair with shared initialization; exact realized `d_0`; product-ball implications require independently proved uniform constants | `parameter_prediction_theorem.md`, Theorems P.1--P.2 and Corollary P.3 | `dense_linear_initial_gap_certificate` returns exact `d_0`, caller tolerance, and strict-local sign implications; `parameter_tube_certificate` evaluates only supplied tube premises and records initialization-met targets as degenerate | Exact autograd agreement, tolerance-band fail-closed behavior, local sign labels, crossing/safe paths, and degenerate target semantics are tested | The local sign theorem has no quantitative horizon; no universal positive-lag sign or constructor for `S_R,lambda,Lambda,nu,J` is claimed |
| T20 | `cue_noise=0`, zero background input, zero lag, symmetric deterministic reference, arbitrary finite realized recurrent matrix; the finite-width outcome-suppression/learnability conclusion additionally needs T10 witness margins | `parameter_prediction_theorem.md`, Corollary P.4 and proof | `zero_lag_six_scalar_rhs` and legacy-named `solve_zero_disorder` implement the recurrence-invisible closure | Tests directly verify arbitrary-`W` invariance, `bar d(0)=0`, `bar d'(0)=-rho^2/2`, the P.12 invariant, and the P.13 speed bound | Floating-point integration is not a proof of witness margins; cue noise is allowed by the general closure but not by the displayed P.11--P.13 rank-one identities; positive lag remains blocked |

## Blocked promotions

1. T1--T16 and T19--T20 do not yield a general positive-lag optimization-time
   DMFT. The general positive-lag solver surfaces correctly fail closed; arbitrary
   recurrent gain is proved only at recurrence-invisible zero lag.
2. No current theorem gives a deterministic universal boundary
   `rho_c(rho, lag, geometry)` for causal starvation. At positive lag the leading
   initialization rate gap contains a centered, seed-dependent cross-geometry term.
3. A rate crossing is not outcome suppression. Outcome suppression is not causal
   starvation without independent, nondegenerate weak-only learnability; a target
   met at initialization is explicitly degenerate.
4. Frozen-NTK event classification is empirical evidence in its frozen study cell,
   not a nonlinear kernel-movement proof or calibrated crossing-time theorem.
5. Waterbirds has not been run and its present second coordinate is an intercept,
   not an identified bird/core response. It cannot support a real-world causal
   starvation claim.
6. Existing `bloop` and `pcgrad` arms are Bloop-style and PCGrad-style motivated
   variants, not audited faithful reproductions.

## Terminology defects to repair

- `DiscreteCrossoverCertificate.strict_outcome_starvation` denotes strict outcome
  suppression; only `causal_starvation_certified` includes learnability.
- `crossover_decomposition` describes a transfer-to-**rate-suppression** crossing,
  not a transfer-to-starvation crossing.
- `metrics.sign_crossing_time` locates a sign crossing; it does not by itself locate
  causal starvation.
- `experiments.run_enl` measures transfer/suppression and outcome/learnability gates;
  its docstring must not collapse them into one mechanism.

## Conclusion

The proved T1--T16 and T19--T20 chain is mathematically grounded at its written
scope. T19 supplies an initialization-conditioned dense-linear formula, its
existential local outcome-sign implication, and conditional parameter-tube logic;
it constructs no tube constants and gives no universal `(rho, lag)` phase boundary.
T20 soundly removes zero recurrent gain at zero lag because `W` is loss-invisible,
but it explicitly requires `cue_noise=0` for P.11--P.13 and does not weaken the
positive-lag block. The main remaining software/theory gaps are premise
construction, nonlinear kernel movement, and general positive-lag closure.
