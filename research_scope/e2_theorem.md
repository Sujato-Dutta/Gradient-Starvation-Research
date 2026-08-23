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
reference a **held-out quenched ensemble closure**, not a proved DMFT solver. In
new outputs it is labelled `closure_reference`; the older `particle_closure` label
survives only in run directories already on disk, which are not rewritten.

### Status of these obligations as of 2026-08-23

**None of the five has advanced.** No derivation was written, and no placeholder
derivation document was created, deliberately: an empty skeleton of headings looks
like progress while containing none, and invites later edits that fill in plausible
mathematics without proof. This file remains the single home of the obligation list.

`src/gradient_starvation/dmft.py` provides the solver *interface* and two exact
special cases that need no part of the derivation:

| path | status | validated against |
|---|---|---|
| `solve_zero_disorder` (`g = 0`, `lag = 0`) | implemented | `exact_dense_linear_geometry` plus autograd; residual is `O(η)` discretization with fitted slope `1.0115` |
| `solve_frozen_geometry` | implemented | `integrate_projected_flow` with constant `G`, agreeing to `1.91e-15` |
| general solve | **blocked** | obligations 1, 2 |
| weak-only reduction | **blocked** | obligations 1, 2 — a weak-only mean-field theory is itself defined by the undelivered kernels, so there is nothing to reduce *to* |
| MSE solver branch | **blocked** | obligations 1, 2. The MSE *objective* for finite networks is implemented in `losses.py` |
| self-consistency residual / refinement | **blocked** | obligations 2, 3 — a residual is only defined relative to derived equations, so a converging residual would be a false pass |
| finite-width against a frozen prediction | **blocked** | obligations 1–4 |

The zero-disorder reduction deserves one note, because it is stronger than a
mean-field statement and weaker than progress on Theorem A. At `g = 0` and `lag = 0`
the recurrent block is provably inert — `m_a = c·b_a` carries no recurrent
dependence, and the synthetic logits depend on the parameters only through the two
mode responses, so `∂L/∂W` vanishes identically. Verified: `‖W‖` stays bitwise zero
at lag 0 while reaching `0.346` by step 20 at lag 1. The projected system then closes
*exactly, at every width*, on six scalars:

```
dm_a/dτ  = g_a w + Σ_b g_b u_ab
du_ab/dτ = g_a m_b + g_b m_a
dw/dτ    = 2 Σ_b g_b m_b        with  G_ab = u_ab + δ_ab w
```

where `u_ab = b_a·b_b` and `w = c·c`. Width enters only through the initial
conditions. This is a closed finite-dimensional identity, not a large-width limit,
so it does not bear on obligations 1–4.

Every blocked surface raises `NotImplementedError` naming its obligation numbers,
and each has a test asserting the raise rather than skipping it, so a later
placeholder cannot quietly start passing.

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
