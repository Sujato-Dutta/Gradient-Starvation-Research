# E2 theorem and validation contract

This document distinguishes the exact finite-width result, the large-width
theorem target, and the empirical test. A successful E2 experiment is evidence
for the target; it is not a proof of it.

## Controlled model and scaling

For fixed sequence length `T`, let

`h_{t+1} = W h_t + B x_t`, `f(x) = c^T h_T`,

with independent initialization

- `W_ij(0) ~ N(0, g^2/N)`;
- `B_ia(0) ~ N(0, 1/N)` for each input channel `a`;
- `c_i(0) ~ N(0, 1/N)`.

The random initialization is shared by the both-feature and weak-only members
of each causal pair. `W`, `B`, and `c` are trained by full-batch logistic-loss
gradient flow. Sequence time `t` and optimization time `tau` are distinct.
This is the parameterization implemented by `DenseLinearRNN`.

The `B_ia ~ N(0,1/N)` scaling is essential. Scaling `B` by input dimension
instead makes `||B_a||^2` and the projected geometry grow with width, so there
is no common optimization-time limit.

## Exact finite-width theorem (implemented and tested)

For channel-localized temporal modes with responses `m_a`, define the impulse
vectors and recurrent derivatives `p_a`, `q_a`, and `S_a` as in Theorem 6 of
the blueprint. At every finite width and parameter state,

`d m_a / d tau = sum_b G_ab g_b`,

where

`G_ab = p_a^T p_b + 1[channel_a=channel_b] q_a^T q_b + <S_a,S_b>_F`.

The channel indicator is required because different input columns are
different parameter blocks. The identity is exact, includes the evolving
recurrent geometry, and is checked against autograd and direct loss-gradient
flow.

## Joint paired large-width theorem (oral-level target; not yet proved)

Fix `T`, a finite optimization horizon `tau_max`, bounded feature moments, and
a bulk gain below the finite-horizon stability threshold. Let `Q_N^B(tau)` and
`Q_N^W(tau)` contain the both-feature and weak-only mode responses, projected
geometry, margin law, CE sensitivity, and the recurrent correlation/response
observables needed to close their evolution under shared quenched disorder.

The target theorem is joint convergence in probability

`sup_{tau <= tau_max} ||(Q_N^B,Q_N^W) - (Q^B,Q^W)|| -> 0`,

where `(Q^B,Q^W)` is the unique solution of one coupled recurrent
generating-functional/DMFT system. The limiting projected modes satisfy

`dot m_a^C = sum_b G_ab^C(Q^C) E[z_b^C sigma(-z^C . m^C)]`,

for `C in {B,W}`. The joint law must retain `G_sw` and the corresponding
cross-response observables. Two unrelated one-condition limits are
insufficient because they do not identify the causal difference.

If the weak-only limiting trajectory reaches a preregistered target `beta`
transversally, the finite-width weak-only hitting time converges. If the both
trajectory also reaches `beta` transversally, the paired hitting-time delay
converges. Points where the weak-only limit does not reach `beta` belong to an
unlearnable/indeterminate region, not the starvation phase.

### Proof obligations

1. Derive the effective single-site process and its covariance/response
   kernels from the stated finite-width probability space.
2. List every visible and loss-invisible order parameter required for closure.
3. Prove finite-horizon existence and uniqueness of the closed Volterra/ODE
   system.
4. Prove joint quenched concentration for both causal conditions under shared
   disorder.
5. Establish the hitting-time corollary under transversality and quantify the
   finite-step discretization error.

Until these obligations are complete, the repository calls the numerical
reference a **held-out quenched ensemble closure**, not a proved DMFT solver.

## Falsifiable E2 validation contract

Calibration seeds and evaluated-network seeds must be disjoint. At no fewer
than three widths, E2 reports held-out errors for:

- strong/weak trajectories;
- `G_ss`, `G_sw`, and `G_ww`;
- margin mean, spread, and quantiles;
- GSI-5;
- the exact `dot m = Gg` identity.

The automated acceptance record requires finite diagnostics, exact geometry
and projected-flow errors below tolerance, and decreasing end-to-end mode error
with a negative log-log width slope for every tested point and condition.
Passing this contract supports a deterministic wide-limit claim; it does not
replace the generating-functional derivation or concentration proof.
