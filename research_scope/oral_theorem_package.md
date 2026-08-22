# Oral-level theorem package: implementation status

This note separates implemented finite-dimensional identities from theorems
that still require mathematical proofs. It is not itself a proof.

## Implemented: exact causal weak-drift decomposition

For the both-feature projected drift

`F_w^B = G_ws^B g_s(m_s,m_w) + G_ww^B g_w(m_s,m_w)`

and the weak-only geometry evaluated at matched weak state,

`F_w^W = G_ww^W g_w(0,m_w)`,

the code verifies the exact identity

`F_w^W - F_w^B = CE gating + geometry shift + cross transport`,

where

- `CE gating = G_ww^B [g_w(0,m_w) - g_w(m_s,m_w)]`;
- `geometry shift = (G_ww^W - G_ww^B) g_w(0,m_w)`;
- `cross transport = -G_ws^B g_s(m_s,m_w)`.

The implementation is `matched_weak_drift_decomposition` in `theory.py` and
has a numerical reconstruction test.

## Implemented: Counterfactual Drift Correction

Let `v_ERM` be the both-feature parameter velocity, and let `F_w^W` be the
instantaneous weak drift of a matched weak-only shadow model. The correction
uses the part of `grad(m_w)` orthogonal to `grad(m_s)` and chooses the smallest
coefficient that removes a positive weak-drift deficit.

When that protected direction is nonzero and the correction is not capped:

1. the corrected weak drift is at least `F_w^W`;
2. the instantaneous strong drift is unchanged;
3. the correction is the minimum-norm vector satisfying 1 and 2.

The trainer records feasibility and target attainment at every logged step.

### Corrected-scaling validation result

The eight-seed width-96 validation in `configs/e3_cdc_validation.yaml` produced:

- ERM weak-trajectory AUC gap: `3.115968`;
- fixed-susceptibility interaction gap: `3.115618`;
- Spectral Decoupling gap: `3.038149`;
- Counterfactual Drift Correction gap: `0.000001`;
- final accuracy: `1.0` for every method;
- correction feasibility and target attainment: `100%` of logged steps;
- maximum absolute instantaneous strong-drift change: `7.63e-6`.

The paired CDC-versus-ERM gap reduction was `-3.115966`, with a 95% confidence
interval of `[-3.707183, -2.524750]`. CDC reduced the final strong-mode response
by about `0.258`, so publication experiments must report both the exact
instantaneous preservation guarantee and the longer-horizon strong-feature
tradeoff caused by following a different parameter trajectory.

### Nonlinear regime discovery

A short-horizon tanh/GRU scan found no positive causal AUC gap. The strong cue
initially accelerates weak-mode learning in both architectures. Extending tanh
training revealed a later crossover: weak-only learning continues while the
both-feature response plateaus. At `rho=4`, lag 2, and optimization horizon 10,
the two-seed search produced a mean AUC gap of `12.956939`, with final weak
responses `1.208475` (both) and `4.340887` (weak-only).

The four-seed nonlinear mitigation pilot at horizon 7.5 produced:

- ERM gap: `7.560667`;
- interaction gap: `7.220363`;
- Spectral Decoupling gap: `7.669840`;
- Counterfactual Drift Correction gap: `-2.182139`;
- final accuracy: `1.0` for every method;
- CDC feasibility and target attainment: `100%` of logged steps.

The frozen width-64 confirmation in `configs/e3_tanh_cdc_validation.yaml`
then used eight independent seeds and produced:

- ERM gap: `19.728794`, 95% CI `[18.811261, 20.646327]`;
- interaction gap: `19.431932`;
- Spectral Decoupling gap: `16.942663`;
- Counterfactual Drift Correction gap: `-1.933250`, 95% CI
  `[-2.204207, -1.662293]`;
- paired CDC-minus-ERM change: `-21.662044`, 95% CI
  `[-22.337681, -20.986407]`;
- final accuracy: `1.0` for every run;
- CDC feasibility and target attainment: `100%` over all 808 logged checks;
- maximum absolute instantaneous strong-drift change: `7.45e-8`.

A matched-seed width check at widths 32, 64, and 96 gave paired CDC-minus-ERM
changes of `-17.846010`, `-21.852428`, and `-22.091320`, respectively. Every
seed improved at every width, and target attainment remained `100%`. This is
strong evidence that the mitigation effect is not a single-width artifact.

Together these runs establish a two-phase nonlinear phenomenon: early positive
transfer followed by late causal starvation. A trajectory theorem must
therefore allow the drift-deficit sign to change over time; an instantaneous
positive-susceptibility condition is not sufficient. The current GRU setup
remains an explicit negative result and should not be used as a mitigation
benchmark until a positive-starvation GRU regime is found.

### Corrected E1 phase-grid validation

The width-96, eight-seed grid in `configs/e1_corrected_validation.yaml`
completed all 136 paired runs. Every positive-regime cell through lag 4 had a
positive causal weak-trajectory AUC gap for every seed. Mean gaps ranged from
`11.937663` to `23.326406` at lag 0, from `8.170996` to `9.363293` at lag 2,
and from `1.451338` to `1.517044` at lag 4. The matched negative control had a
mean gap of `-0.750387` with a 95% interval of `[-0.876793, -0.623980]`.

Lag 8 is deliberately classified as indeterminate: its weak-only final
response was only `0.001643`, so there was no meaningful counterfactual weak
learning to suppress. The theorem statement must therefore include a
**counterfactual learnability condition**, such as the weak-only trajectory
reaching a preregistered response by the evaluation horizon. Without this
condition, an arbitrarily small positive AUC difference could be mislabeled as
starvation even though neither condition learns the weak feature.

## Proof targets not yet implemented

1. **Trajectory comparison and hitting-time bound.** Convert the pointwise
   drift guarantee into a continuous-time weak-trajectory and hitting-time
   guarantee under explicit smoothness, monotonicity, and feasibility
   assumptions; quantify discrete-step error.
2. **Paired causal DMFT/concentration.** Prove joint finite-horizon convergence
   of both-feature and weak-only order parameters under shared quenched
   disorder, including the order parameters required by the drift
   decomposition.
3. **Starvation phase boundary.** Derive and prove the existence and location
   of `rho_c(lag, bulk_gain)` inside the counterfactually learnable region,
   characterize the separate learnability boundary, then establish
   finite-width boundary consistency.
4. **Non-identifiability of observational GSI.** Construct systems with the
   same both-feature margin-weight statistic but different weak-only causal
   effects, proving that GSI alone cannot identify starvation.

Publication-scale claims must wait for these proofs and for independent
width-convergence experiments; empirical particle averaging is not a proof or
an independent DMFT solver.
