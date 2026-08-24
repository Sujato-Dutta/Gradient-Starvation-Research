# Finite-width causal crossover theorem and mean-field boundary

## Epistemic status

This document replaces the former unproved joint-DMFT headline. The established
results below are finite-width identities, deterministic trajectory theorems, and
one exactly solvable large-width limit. They do **not** constitute a general
optimization-time DMFT for a trained recurrent network.

The paper may claim that the transfer-to-suppression-to-starvation implication is
proved **under the stated trajectory conditions**. It may not claim that those
conditions hold for every RNN, that the crossover time is predicted independently
for tanh/GRU, or that an analytic phase boundary `rho_c` has been derived.

Throughout, sequence time is `k` and optimization time is `tau`.

## Setting

For a condition `C` in `{B,W}` (both-feature or weak-only), let a finite-dimensional
parameter vector `theta_C(tau)` follow full-batch logistic-loss gradient flow

```
L_C(theta) = n_C^{-1} sum_i log(1 + exp(-r_i^C(theta))),
r_i^C(theta) = y_i f_C(x_i; theta),
d theta_C / d tau = - grad L_C(theta_C).
```

Let `M^C(theta) = (M_1^C,...,M_k^C)` be differentiable response functionals. The
paired experiment starts from the same parameter initialization and uses the same
labels, weak coordinates, samples, and noise; the weak-only member differs only by
zeroing the strong input channel. A common weak-response functional is used in both
conditions, so shared initialization implies

```
Delta(0) = M_w^B(0) - M_w^W(0) = 0.
```

For the controlled dense linear RNN,

```
h_(k+1) = W h_k + B x_k,       f(x) = c^T h_T,
```

with `W_ij(0) ~ N(0,g^2/N)`, `B_ia(0) ~ N(0,1/N)`, and
`c_i(0) ~ N(0,1/N)`. The `B` scaling is load-bearing: input-dimension scaling
would make the projected metric diverge with width.

## Theorem A — exact finite-width CE response flow

**Theorem A.1 (universal cross-kernel form).** At every finite width and every
optimization time at which the responses are differentiable,

```
d M_a^C / d tau = n_C^{-1} sum_i K_ai^C sigma(-r_i^C),
K_ai^C = <grad M_a^C, grad r_i^C>.
```

**Proof.** For `ell(r)=log(1+exp(-r))`, `ell'(r)=-sigma(-r)`. Therefore

```
d theta_C/d tau
  = -grad L_C
  = n_C^{-1} sum_i sigma(-r_i^C) grad r_i^C.
```

Taking the directional derivative of `M_a^C` along this velocity gives the
formula. This is the chain rule and requires no width limit. □

**Corollary A.2 (closed projected form under exact realization).** Suppose the
signed logits admit the exact, parameter-independent realization

```
r_i^C(theta) = sum_b z_ib^C M_b^C(theta).
```

Then

```
d M_a^C / d tau = sum_b G_ab^C g_b^C,
G_ab^C = <grad M_a^C, grad M_b^C>,
g_b^C = n_C^{-1} sum_i z_ib^C sigma(-z_i^C . M^C).
```

**Proof.** Exact realization gives
`grad r_i^C = sum_b z_ib^C grad M_b^C`. Substitute this into Theorem A.1 and
exchange the finite sums. □

**Residual qualification.** If instead

```
r_i^C = z_i^C . M^C + r_i^perp,
```

then an additional term is present:

```
R_a^C = n_C^{-1} sum_i sigma(-r_i^C)
        <grad M_a^C, grad r_i^perp>.
```

Thus `dot M=Gg` is exact for the linear synthetic model with zero background
noise, but is only the projected component for nonlinear tanh/GRU unit-probe
responses because their logits need not be additive in those two responses.
Nonlinear crossover experiments must use the direct cross-kernel/autograd drift as
the primary derivative and report the projection residual.

**Corollary A.3 (dense linear geometry).** For channel-localized temporal modes in
the dense linear RNN,

```
G_ab = p_a^T p_b
       + 1[channel_a = channel_b] q_a^T q_b
       + <S_a,S_b>_F,
```

where `p_a`, `q_a`, and `S_a` are respectively the input/readout impulse vectors
and recurrent derivative matrices. This follows by differentiating the response
with respect to the disjoint parameter blocks `(c,B,W)` and summing their gradient
inner products. The channel indicator is required because different input columns
are distinct parameter blocks. `theory.exact_dense_linear_geometry` implements
this expression and tests it against autograd.

## Theorem B — transfer, rate suppression, and outcome starvation

Define the equal-time weak-response gap and its exact derivative

```
Delta(tau) = M_w^B(tau) - M_w^W(tau),
d(tau) = Delta'(tau) = F_w^B(tau) - F_w^W(tau).
```

Here `d>0` means relative transfer, `d<0` means relative rate suppression, and
`Delta<0` means the both-feature weak response has fallen below its matched
weak-only counterfactual. The matched-state drift deficit is a different quantity
and cannot replace `d` in this theorem.

**Theorem B.1 (necessary and sufficient tail-area criterion).** Let `Delta` be
absolutely continuous on `[0,H]`, with `Delta(0)=0`. Suppose there is
`tau_d in (0,H)` such that

```
d(tau) > 0  for almost every tau in (0,tau_d),
d(tau) < 0  for almost every tau in (tau_d,H).
```

Define

```
A_+ = integral_0^tau_d d(u) du = Delta(tau_d) > 0,
N(t) = integral_tau_d^t [-d(u)] du.
```

Then, for `t >= tau_d`,

```
Delta(t) = A_+ - N(t).
```

Consequently:

1. a later equality `Delta(t)=0` is reached by `H` iff `N(H) >= A_+`;
2. strict outcome starvation, `Delta(t)<0` for some `t<=H`, occurs iff
   `N(H) > A_+`;
3. if `d<0` on every nontrivial post-crossover interval, the later equality is
   unique whenever it exists.

**Proof.** Absolute continuity and the fundamental theorem of calculus give

```
Delta(tau_d)=Delta(0)+integral_0^tau_d d=A_+,
Delta(t)=Delta(tau_d)+integral_tau_d^t d=A_+-N(t).
```

The sign assumptions make `N` continuous and strictly increasing on every
nontrivial post-crossover interval. The three conclusions follow immediately. □

This theorem repairs the false statement that a drift crossing necessarily causes
a response crossing. The additional tail-area condition is both necessary and
sufficient.

**Corollary B.2 (quantitative crossing bounds).**

- If `d(t)<=-kappa<0` on `[t_0,H]`, where `t_0>tau_d`, then, provided the right
  side lies within the horizon,

  ```
  tau_response <= t_0 + [A_+ - N(t_0)]_+ / kappa
               <= t_0 + A_+/kappa.
  ```

- If `d` is differentiable, `d(tau_d)=0`, and
  `d'(t)<=-lambda<0` on `[tau_d,tau_d+r]`, then

  ```
  N(t) >= lambda (t-tau_d)^2 / 2.
  ```

  Hence, if `A_+ <= lambda r^2/2`,

  ```
  tau_response <= tau_d + sqrt(2 A_+/lambda).
  ```

**Proof.** Integrate the stated lower bounds on `-d` and apply Theorem B.1. □

A causal **starvation** label additionally requires the preregistered learnability
gate: the weak-only trajectory must reach its target. Theorem B by itself proves a
response ordering, not learnability.

## Theorem C — ordering-invariant rank-one drift crossover

Assume the noiseless positive cue law

```
z^B=(rho,1),       z^W=(0,1),       rho>0,
```

and exact two-mode realization. Define

```
u_B = rho M_s^B + M_w^B,       s_B = sigma(-u_B),
u_W = M_w^W,                   s_W = sigma(-u_W),
H_B = rho G_ws^B + G_ww^B,      H_W = G_ww^W,
Q_B = rho^2 G_ss^B + 2 rho G_sw^B + G_ww^B >= 0.
```

The weak drifts factor exactly as

```
F_w^B=s_B H_B,       F_w^W=s_W H_W.
```

Assume `H_B,H_W>0` on an interval. Define the log drift ratio

```
Psi = log(F_w^B/F_w^W)
    = log(s_B/s_W) + log(H_B/H_W).
```

Then `sign(d)=sign(Psi)`, and whenever these quantities are differentiable,

```
Psi'
 = -s_B(1-s_B) Q_B
   +s_W(1-s_W) H_W
   +(log(H_B/H_W))'.
```

**Proof.** Under the rank-one cue law, `g^B=s_B(rho,1)` and
`g^W=s_W(0,1)`. Corollary A.2 gives the factorization and
`u_B'=s_B Q_B`, `u_W'=s_W H_W`. Since
`d log sigma(-u)/du=-(1-sigma(-u))`, differentiating `Psi` yields the displayed
identity. Positivity makes the logarithm monotone, so
`sign(F_w^B-F_w^W)=sign(Psi)`. □

**Corollary C.1 (sufficient condition for a unique rate crossover).** Suppose on
`[t_0,H]` that `H_B,H_W>0`, `Psi(t_0)>0`, and

```
Psi'(t) <= -kappa < 0,
H-t_0 >= Psi(t_0)/kappa.
```

Then there is exactly one drift crossover in `(t_0,H]`, and

```
tau_d <= t_0 + Psi(t_0)/kappa.
```

A sufficient primitive bound is

```
s_B(1-s_B)Q_B >= a,
s_W(1-s_W)H_W <= b,
(log(H_B/H_W))' <= c,
a-b-c >= kappa > 0.
```

**Proof.** Integrating `Psi'<=-kappa` gives
`Psi(t)<=Psi(t_0)-kappa(t-t_0)`. Continuity gives a zero by the stated horizon;
strict decrease makes it unique. The primitive inequalities imply the derivative
bound through Theorem C. □

`Psi` avoids the non-unique additive geometry/field attribution. It predicts a
rate crossover only when its bounds are established independently. Evaluating
`Psi` on the same trained trajectory is an exact diagnostic certificate, not an
independent prediction. Outcome starvation still requires Theorem B's tail area.
For noisy cues, the rank-one factorization generally fails and Theorem A.1 must be
used instead.

## Corollary D — stability of target and crossover hitting times

Let continuous trajectories `x_N,x` on `[0,H]` satisfy
`||x_N-x||_infinity <= epsilon_N`. Let `tau` be the first upward crossing of a
target `beta`, assume it lies in `(0,H)`, is separated from earlier times, and
suppose `x'(t)>=kappa>0` on `[tau-r,tau+r]`. Then, for sufficiently small
`epsilon_N`,

```
|tau_N-tau| <= epsilon_N/kappa.
```

On a grid with spacing `h_N`, add `h_N` to the bound. For a paired delay,

```
|Delta tau_N-Delta tau|
 <= epsilon_B,N/kappa_B + epsilon_W,N/kappa_W + 2h_N.
```

**Proof.** Before `tau`, transversality gives
`x(t)<=beta-kappa(tau-t)` locally; after `tau`, it gives
`x(t)>=beta+kappa(t-tau)`. Uniform error excludes a crossing before
`tau-epsilon_N/kappa` and forces one by `tau+epsilon_N/kappa`. Separation from
earlier times excludes a spurious earlier hit. The paired bound follows by the
triangle inequality. □

This corollary transfers an already-proved uniform trajectory limit into a
hitting-time limit. It does not prove that uniform limit.

## Theorem E — exact zero-disorder closure and deterministic wide limit

Assume `bulk_gain=0`, `lag_separation=0`, zero background noise, and the dense
linear model. Write the two input columns as `b_s,b_w`. The recurrent block is
inert, `M_a=c.b_a`, and the logits depend on the trainable parameters only through
these modes. Define

```
u_ab=b_a.b_b,       w=c.c.
```

**Theorem E.1 (every-width closure).** At every width,

```
d M_a/dtau = g_a w + sum_b g_b u_ab,
d u_ab/dtau = g_a M_b + g_b M_a,
d w/dtau = 2 sum_b g_b M_b,
G_ab = u_ab + delta_ab w.
```

**Proof.** Parameter gradient flow gives
`dot b_a=g_a c` and `dot c=sum_b g_b b_b`; the recurrent derivative vanishes.
Differentiate the six scalar inner products. □

**Theorem E.2 (deterministic finite-horizon limit).** Let all coordinates of
`b_s,b_w,c` be mutually independent `N(0,1/N)`. For

```
q_N=(M_s,M_w,u_ss,u_sw,u_ww,w),
q_0=(0,0,1,0,1,1),
```

and every `epsilon>0`,

```
P(||q_N(0)-q_0||_infinity > epsilon) <= min(1, 9/(N epsilon^2)).
```

Let `q_N(tau)` and `q(tau)` solve the six-scalar ODE from their respective initial
conditions. On every fixed horizon `[0,H]`, the trajectories lie in a compact set.
If `L_H` is a Lipschitz constant of the vector field on a common compact tube, then
while the initial perturbation stays in that tube,

```
sup_(tau<=H) ||q_N(tau)-q(tau)||_infinity
 <= exp(L_H H) ||q_N(0)-q_0||_infinity,
```

and therefore

```
P(sup_(tau<=H)||q_N(tau)-q(tau)||_infinity > delta)
 <= min(1, 9 exp(2L_H H)/(N delta^2)).
```

In particular the convergence is uniform on finite horizons in probability and
has initialization scale `O_P(N^{-1/2})`.

**Proof.** The variances of `(M_s,M_w,u_sw)` are `1/N`; those of
`(u_ss-1,u_ww-1,w-1)` are `2/N`. Chebyshev's inequality and a union bound give
the constant `1+1+1+2+2+2=9`.

For bounded second cue moments, the CE field is locally Lipschitz. The Gram energy
`R=w+tr(u)` satisfies

```
R'=4 g.M = 4 E[X sigma(-X)],       X=z.M.
```

The scalar function `x sigma(-x)` is bounded above, while `R>=0`; hence `R`
cannot blow up on a finite horizon. Cauchy--Schwarz bounds every component of
`q` by `R`, giving a compact forward set and a finite local Lipschitz constant.
Gronwall's inequality gives the trajectory bound. Substituting
`epsilon=delta exp(-L_H H)` into the initialization bound proves the probability
statement. □

The same argument applies jointly to the both/weak-only zero-disorder pair because
both are deterministic flows of the same concentrating initial six scalars; use
the larger of their two finite-horizon Lipschitz constants.

This theorem is a genuine deterministic large-width result at a singular solvable
point. It is not a DMFT and says nothing about positive recurrent disorder or
positive lag.

## General paired CE-RNN DMFT — conjecture, not theorem

For positive recurrent disorder or positive lag, a joint optimization-time DMFT
would need a coupled order-parameter process containing both causal conditions,
projected modes and geometry, margin laws, recurrent covariance/response kernels,
and every loss-invisible overlap required for closure. That remains a conjectural
extension. The unresolved obligations are:

1. derive the effective single-site process and covariance/response kernels from
   the finite-width probability space;
2. enumerate all visible and loss-invisible closure variables;
3. prove finite-horizon existence and uniqueness of the closed system;
4. prove joint quenched concentration under shared disorder;
5. combine the limit with Corollary D and a finite-step discretization bound.

`src/gradient_starvation/dmft.py` therefore keeps the general solver, weak-only
reduction, MSE solver branch, self-consistency residual, and refinement paths
blocked with `NotImplementedError`. No executable certificate may turn those
blocks into a pass.

Ger and Barak (2026) derive overlap-space learning dynamics for low-rank RNNs and
show why loss-invisible overlaps are required. Clark et al. (2026) derive an
equilibrium task-trained RNN DMFT using a long-time Langevin/Gibbs limit; they
explicitly distinguish that limit from noiseless optimization-time gradient flow.
These works narrow the novelty claim and reinforce, rather than remove, obligations
1--4 above.

## Executable theorem contracts

The code can verify algebra and hypotheses; it does not replace the proofs.

- `projected_statistics`: exact cross-kernel drift when direct autograd is enabled,
  plus the `Gg` projected component and its residual;
- `equal_time_drift_difference`: projected equal-time decomposition and, when
  available, the exact direct gap derivative;
- `rank_one_drift_ratio`: Theorem C factorization and log-ratio diagnostic;
- `discrete_crossover_certificate`: exact finite-step analogue of Theorem B using
  realized response increments;
- `transverse_hitting_time_error_bound`: Corollary D's deterministic bound;
- `zero_disorder_initialization_failure_bound` and
  `zero_disorder_trajectory_failure_bound`: Theorem E.2's explicit bounds;
- `cdc_finite_step_deviation_bound`: the conditional one-step CDC theorem in
  `research_scope/cdc_theorem.md`.

Tests establish these software contracts and numerical identities. They are not
evidence that tanh/GRU satisfy the sufficient inequalities in Theorem C or that the
general DMFT conjecture is true.
