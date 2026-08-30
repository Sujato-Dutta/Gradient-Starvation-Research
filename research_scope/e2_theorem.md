# Finite-width causal crossover theorem and mean-field boundary

## Epistemic status

This document replaces the former unproved joint-DMFT headline. The established
results below are exact finite-width identities, conditional finite-width
trajectory-comparison theorems, deterministic trajectory theorems, and one exactly
solvable large-width limit. They do **not** constitute a general optimization-time
DMFT for a trained recurrent network.

The paper may claim that the transfer-to-suppression-to-starvation implication is
proved **under the stated trajectory conditions**, and that a sealed held-out study
supports a restricted frozen-kernel crossing-event classifier in its specified
finite-width regime. It may not claim that kernel-movement hypotheses have been
proved for tanh/GRU, that the conditions hold for every RNN, that crossover times
are accurately predicted, that starvation is universal, or that an analytic phase
boundary `rho_c` has been derived.

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

## Proposition A — exact finite-width CE response flow

**Proposition A.1 (universal cross-kernel form).** At every finite width and every
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
`grad r_i^C = sum_b z_ib^C grad M_b^C`. Substitute this into Proposition A.1 and
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

**Corollary A.4 (exact full empirical-NTK equations and frozen surrogate).** Fix one
condition and suppress its superscript. Let `theta in R^p`, let
`r(theta) in R^n` be the signed-logit vector, and define

```
J(theta)_i = grad r_i(theta)^T,       Theta(theta) = J(theta) J(theta)^T,
c(theta) = J(theta) grad M(theta),    q(r) = sigma(-r)
```

with `q` applied componentwise. Along mean logistic-loss gradient flow,

```
d theta/dtau = J^T q(r)/n,
d r/dtau = Theta q(r)/n,
d M/dtau = c^T q(r)/n.                         (A.4)
```

These are exact at every finite width. Unlike the two-probe `Gg` form, they retain
the complete sample-logit tangent space and therefore have no projection residual.
They are not closed in `(r,M)`, because `Theta(theta)` and `c(theta)` generally move.

Freezing only those two kernels at initialization defines the nonlinear logistic
tangent surrogate

```
d r_bar/dtau = Theta_0 q(r_bar)/n,
d M_bar/dtau = c_0^T q(r_bar)/n,
(r_bar(0),M_bar(0)) = (r(0),M(0)).              (A.5)
```

Its gate remains nonlinear in the evolving margins. Its exact explicit-Euler
recursion at learning rate `eta` is

```
q_l = q(r_bar_l),
r_bar_(l+1) = r_bar_l + eta Theta_0 q_l/n,
M_bar_(l+1) = M_bar_l + eta c_0^T q_l/n.        (A.6)
```

**Proof.** The parameter equation is the logistic-loss calculation in Proposition A.1.
The chain rule gives `dot r=J dot theta` and
`dot M=<grad M,dot theta>`, yielding (A.4). Replacing `Theta(theta),c(theta)` by
their initial values defines (A.5), and explicit Euler gives (A.6). No width limit
or mode-additivity assumption is used. □

**Theorem A.5 (conditional frozen-kernel comparison, continuous and discrete).**
Use Euclidean vector norms and their induced matrix norm. Suppose on `[0,H]`

```
||Theta(tau)-Theta_0|| <= B_Theta,
||c(tau)-c_0|| <= B_c,
||q(r(tau))||, ||q(r_bar(tau))|| <= V,
a = ||Theta_0||/(4n).
```

The universal choice `V=sqrt(n)` is valid. Then

```
||r(t)-r_bar(t)|| <= (B_Theta V/n) Phi_a(t),
|M(t)-M_bar(t)|
  <= B_c V t/n + ||c_0|| B_Theta V Xi_a(t)/(4n^2),             (A.7)
```

where

```
Phi_a(t) = (exp(a t)-1)/a,          Xi_a(t) = (exp(a t)-1-a t)/a^2
```

with continuous extensions `Phi_0(t)=t` and `Xi_0(t)=t^2/2`. The instantaneous
response-drift error also satisfies

```
|dot M(t)-dot M_bar(t)|
 <= B_c V/n + ||c_0|| ||r(t)-r_bar(t)||/(4n).                  (A.8)
```

There is an exact finite-step counterpart for full-batch explicit SGD. Write

```
theta_(l+1) = theta_l + eta J_l^T q(r_l)/n,
J_l^sec = integral_0^1 J(theta_l+s(theta_(l+1)-theta_l)) ds,
m_l^sec = integral_0^1 grad M(theta_l+s(theta_(l+1)-theta_l)) ds,
Theta_l^sec = J_l^sec J_l^T,       c_l^sec = J_l m_l^sec.
```

Thus the true finite increments are exactly

```
r_(l+1)-r_l = eta Theta_l^sec q(r_l)/n,
M_(l+1)-M_l = eta (c_l^sec)^T q(r_l)/n.                        (A.9)
```

For `eta>0`, if, for `0<=l<k`,

```
||Theta_l^sec-Theta_0|| <= B_Theta,
||c_l^sec-c_0|| <= B_c,
||q(r_l)||, ||q(r_bar_l)|| <= V,
```

then

```
||r_k-r_bar_k|| <= B_Theta V phi_k/n,
|M_k-M_bar_k|
 <= k eta B_c V/n + eta ||c_0|| B_Theta V psi_k/(4n^2),        (A.10)
```

where, for `a>0`,

```
phi_k = ((1+eta a)^k-1)/a,
psi_k = (((1+eta a)^k-1)/(eta a)-k)/a,
```

and for `a=0`, `phi_k=k eta` and `psi_k=eta k(k-1)/2`. These are the formulas
implemented by `theory.frozen_kernel_discrete_error_bound`. Under the secant
hypothesis, the per-step response-increment quotient obeys the discrete analogue of
(A.8). The exact current-parameter grid drift obeys (A.8) when the additional bound
`||c(theta_l)-c_0||<=B_c` is supplied.

For a paired experiment apply (A.7), (A.8), or (A.10) separately to `B` and `W`.
Writing any established response and drift bounds as `epsilon_M^C` and
`epsilon_F^C`,

```
|Delta-Delta_bar| <= epsilon_M^B+epsilon_M^W,
|d-d_bar| <= epsilon_F^B+epsilon_F^W.                          (A.11)
```

Consequently, a frozen paired sign transfers wherever its magnitude strictly
exceeds the corresponding right-hand side. Opposite, error-separated signs at two
times bracket a true continuous crossing. A first crossing or target-hitting time
transfers with the bounds of Corollary D only when its prehistory separation,
transversality, and mesh hypotheses also hold; use the summed paired response error
for a response-gap hit, the summed drift error for a drift-gap hit, and the
single-condition response error for weak-only learnability. A starvation claim
still additionally requires Proposition B.1's cumulative-balance condition and the learnability
gate.

**Proof.** The scalar sigmoid is `1/4`-Lipschitz, hence
`||q(u)-q(v)||<=||u-v||/4`. Subtracting (A.5) from (A.4), adding and subtracting
`Theta_0 q(r)`, and taking norms gives, with
`e_r=||r-r_bar||`,

```
e_r'(t) <= a e_r(t) + B_Theta V/n.
```

Gronwall and `e_r(0)=0` give the first inequality in (A.7). Similarly,

```
|dot M-dot M_bar| <= B_c V/n + ||c_0|| e_r/(4n),
```

which is (A.8); integrating the first bound and using
`integral_0^t Phi_a(u)du=Xi_a(t)` proves the second inequality in (A.7).

For SGD, the fundamental theorem of calculus along the update segment gives

```
r_(l+1)-r_l = J_l^sec(theta_(l+1)-theta_l),
M_(l+1)-M_l = <m_l^sec,theta_(l+1)-theta_l>,
```

which proves (A.9). Subtract (A.6). If `e_l=||r_l-r_bar_l||`, then

```
e_(l+1) <= (1+eta a)e_l + eta B_Theta V/n.
```

Solving this recurrence gives `e_k<=B_Theta V phi_k/n`. The response recurrence is

```
E_(l+1) <= E_l + eta B_c V/n + eta ||c_0|| e_l/(4n).
```

Summing and using
`sum_(l=0)^(k-1) phi_l=psi_k` proves (A.10), including the `a=0` limits. The
per-step secant-quotient and current-grid drift statements follow by the same
add-and-subtract argument used for (A.8), with `c_l^sec` and `c(theta_l)`,
respectively. Finally, (A.11) is the triangle inequality. Error-separated signs and
bracketing follow from these uniform bounds and continuity; the first-hit
conclusions are exactly Corollary D applied with the stated single or summed errors.
□

The constants `B_Theta` and `B_c` are caller-supplied uniform movement bounds. No
current result derives them for trained tanh or GRU networks, and the held-out
frozen-kernel experiment is empirical evidence rather than a proof of them.
Corollary A.4 and Theorem A.5 are finite-width comparison statements, not DMFT.

## Proposition B — cumulative drift balance and outcome suppression

Define the equal-time weak-response gap and its exact derivative

```
Delta(tau) = M_w^B(tau) - M_w^W(tau),
d(tau) = Delta'(tau) = F_w^B(tau) - F_w^W(tau).
```

Here `d>0` means relative transfer, `d<0` means relative rate suppression, and
`Delta<0` means the both-feature weak response has fallen below its matched
weak-only counterfactual. The matched-state drift deficit is a different quantity
and cannot replace `d` in this proposition.

**Proposition B.1 (cumulative drift balance).** Let `Delta` be absolutely continuous
on `[0,H]`, with `Delta(0)=0` and `d=Delta'` almost everywhere. Define

```
d_+(t) = max(d(t),0),          d_-(t) = max(-d(t),0),
P(t) = integral_0^t d_+(u) du, N(t) = integral_0^t d_-(u) du.
```

Then, for every `t in [0,H]`,

```
Delta(t) = P(t) - N(t).
```

Consequently, strict outcome suppression occurs by `H` if and only if
`N(t)>P(t)` for some `t in (0,H]`. This statement allows arbitrary sign changes,
including oscillatory drift and suppression from initialization.

**Proof.** The positive and negative parts satisfy `d=d_+-d_-` almost everywhere.
Absolute continuity and the fundamental theorem of calculus give
`Delta(t)=Delta(0)+integral_0^t d=P(t)-N(t)`. The pointwise sign equivalence follows
immediately. □

This is deliberately a bookkeeping proposition: its contribution is the causal
separation it enforces, not proof difficulty.

**Corollary B.2 (single-crossing tail-area criterion).** Suppose additionally that
there is `tau_d in (0,H)` such that

```
d(tau) > 0  for almost every tau in (0,tau_d),
d(tau) < 0  for almost every tau in (tau_d,H).
```

Define

```
A_+ = P(tau_d) = Delta(tau_d) > 0.
```

Then `P(t)=A_+` and

```
Delta(t) = A_+ - N(t),       t >= tau_d.
```

Consequently:

1. a later equality `Delta(t)=0` is reached by `H` iff `N(H) >= A_+`;
2. strict outcome suppression, `Delta(t)<0` for some `t<=H`, occurs iff
   `N(H) > A_+`;
3. if `d<0` on every nontrivial post-crossover interval, the later equality is
   unique whenever it exists.

**Proof.** The sign pattern gives `d_-=0` before `tau_d` and `d_+=0` afterward, so
`P(t)=P(tau_d)=A_+` for `t>=tau_d`. Proposition B.1 gives the displayed identity.
The post-crossing assumptions make `N` continuous and strictly increasing on every
nontrivial post-crossing interval, yielding the three conclusions. □

The corollary repairs the false statement that a drift crossing necessarily causes
a response crossing. The additional tail-area condition is both necessary and
sufficient under its single-crossing hypotheses.

**Corollary B.3 (quantitative crossing bounds).**

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

**Proof.** Integrate the stated lower bounds on `-d` and apply Corollary B.2. □

A causal **starvation** label additionally requires the preregistered learnability
gate: the weak-only trajectory must reach its target. Proposition B.1 and its
corollaries prove response orderings, not learnability.

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
independent prediction. Outcome suppression still requires Proposition B.1's
cumulative balance (or Corollary B.2 in the single-crossing case); causal
starvation additionally requires the independent weak-only first-hit gate.
For noisy cues, the rank-one factorization generally fails and Proposition A.1 must be
used instead.

## Corollary D — stability of target and crossover hitting times

Let continuous `x_N,x:[0,H]->R` satisfy
`||x_N-x||_infinity <= epsilon`. Define

```
tau   = inf{t in [0,H] : x(t)   >= beta},
tau_N = inf{t in [0,H] : x_N(t) >= beta},
```

with `inf(empty)=infinity`. Assume that `x` is absolutely continuous,
`x(0)<beta`, and `tau in (0,H)`. Choose
`0<r<=min(tau,H-tau)` such that `x'(t)>=kappa>0` almost everywhere on
`[tau-r,tau+r]`, and define the quantitative prehistory margin

```
eta = beta - max_(0<=t<=tau-r) x(t) > 0.
```

If `epsilon<eta` and `epsilon/kappa<=r`, then `tau_N` is finite and

```
|tau_N-tau| <= epsilon/kappa.
```

For an ordered grid `0=t_0<...<t_m=H` with maximum mesh
`h=max_j(t_j-t_(j-1))`, define the sampled first hit as
`tau_N^grid=min{t_j:x_N(t_j)>=beta}` and the interpolated first hit as the first
hit of the piecewise-linear interpolant through `(t_j,x_N(t_j))`. If additionally
`epsilon/kappa+h<=r`, each is finite and each obeys

```
|tau_N^grid-tau| <= epsilon/kappa+h,
|tau_N^interp-tau| <= epsilon/kappa+h.
```

For paired trajectories, apply these assumptions separately to both finite first
hits. Then

```
|(tau_N,B-tau_N,W)-(tau_B-tau_W)|
 <= epsilon_B/kappa_B + epsilon_W/kappa_W + h_B + h_W.
```

A common mesh bound `h` gives the earlier `+2h` form. If either hit is infinite,
the paired delay is censored rather than assigned a finite error bound.

**Proof.** Continuity and first-entry semantics give `x(tau)=beta`. Absolute
continuity and the almost-everywhere derivative bound give the local cones

```
x(t) <= beta-kappa(tau-t),   tau-r<=t<=tau,
x(t) >= beta+kappa(t-tau),   tau<=t<=tau+r.
```

On `[0,tau-r]`, the prehistory margin and `epsilon<eta` imply `x_N<beta`.
On `[tau-r,tau-epsilon/kappa)`, the left cone and the uniform error again imply
`x_N<beta`; hence `tau_N>=tau-epsilon/kappa`. At
`t=tau+epsilon/kappa`, the right cone gives `x_N(t)>=beta`, so `tau_N` is finite
and at most that time. This proves the continuous bound.

For the sampled hit, the first grid point at or after
`tau+epsilon/kappa` is no more than `h` later and remains in the right-cone
interval; it is at or above `beta`. No sampled hit can precede the continuous
lower bound. For the interpolated hit, a segment can cross only if an endpoint is
at or above `beta`; its left endpoint is at most one mesh before the corresponding
sampled hit. These facts give the two `epsilon/kappa+h` bounds. The paired result
is the triangle inequality. □

Descending crossings follow by sign reversal, but they must still be localized
after a strict excursion. In particular, the response gap satisfies `Delta(0)=0`,
so a later downward outcome crossover must be treated on `[a,H]` for some
`Delta(a)>0`, with `x=-Delta` and target zero; the unlocalized first hit at time
zero is not the later crossover.

This corollary transfers an already-proved uniform trajectory limit into a
hitting-time limit. It does not prove that limit, first-entry semantics, the
prehistory margin, or the derivative bound.

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

Fix a cue law with bounded second moments and write

```
g_a(M) = E[z_a sigma(-z.M)],
A = max_a E|z_a|,
C = (1/4) max_a sum_b E|z_a z_b|.
```

Then `||g(M)||_infinity<=A` and
`||g(M)-g(M')||_infinity<=C||M-M'||_infinity`.
Let `q(t)` be the deterministic six-scalar solution from `q_0`, and set

```
c_* = W(e^(-1)) = sup_x x sigma(-x),
R_H = 3 + 4 c_* H.
```

Its Gram energy `R=w+u_ss+u_ww` satisfies `R(0)=3`, `R'<=4c_*`, and hence
`||q(t)||_infinity<=R_H` on `[0,H]`. Fix a deterministic tube radius `r>0` and
define

```
B_(H,r) = R_H+r,
L_(H,r) = 4(A+B_(H,r) C).
```

The six-scalar vector field is `L_(H,r)`-Lipschitz in infinity norm on the
radius-`r` tube around the reference path. If `q_N(t)` is the solution from its
random initial condition and

```
||q_N(0)-q_0||_infinity
  <= exp(-L_(H,r) H) min(r,delta),
```

then the path does not exit that tube before `H` and

```
sup_(t<=H) ||q_N(t)-q(t)||_infinity <= delta.
```

Consequently,

```
P(sup_(t<=H)||q_N(t)-q(t)||_infinity > delta)
 <= min(1,
        9 exp(2 L_(H,r) H)/(N min(r,delta)^2)).
```

For fixed `r=1` and `0<delta<=1`, this reduces to the earlier denominator
`delta^2`, now with the deterministic constant `L_(H,1)`. Thus convergence is
uniform on fixed horizons in probability and has initialization scale
`O_P(N^(-1/2))`.

**Proof.** The variances of `(M_s,M_w,u_sw)` are `1/N`; those of
`(u_ss-1,u_ww-1,w-1)` are `2/N`. Chebyshev's inequality and a union bound give
the constant `1+1+1+2+2+2=9`.

The field bound follows directly from `sigma in [0,1]`. Moreover,

```
|g_a(M)-g_a(M')|
 <= (1/4) sum_b E|z_a z_b| ||M-M'||_infinity,
```

because `sigma` is `1/4`-Lipschitz, proving the stated constant `C`. Along the
deterministic flow,

```
R'=4 g.M = 4 E[X sigma(-X)],       X=z.M.
```

Since `x sigma(-x)<=c_*`, integration gives `R(t)<=R_H`. All six coordinates are
inner products or squared norms from the Gram triple `(b_s,b_w,c)`, so
Cauchy--Schwarz gives `||q(t)||_infinity<=R(t)`.

For two states in the cube `||q||_infinity<=B_(H,r)`, put
d=`||q-q'||_infinity`. Each product in the `M` and `u` equations obeys

```
|g_a(M)y-g_a(M')y'| <= (A+B_(H,r) C)d.
```

There are at most three such products in an `M` equation, two in a `u` equation,
and, including its factor two, four in the `w` equation. Therefore the whole
vector field is `L_(H,r)=4(A+B_(H,r)C)`-Lipschitz on the cube, hence on the tube.

Let `T_r` be the first time before `H` at which
`||q_N(t)-q(t)||_infinity` reaches the tube boundary. Up to `T_r`, Gronwall gives

```
||q_N(t)-q(t)||_infinity
 <= exp(L_(H,r)t)||q_N(0)-q_0||_infinity.
```

If `L_(H,r)>0`, the displayed initialization event makes this bound strictly less
than `r` for every `t<H`; hence the boundary cannot be reached before `H`. If
`L_(H,r)=0`, then `A=C=0`, the cue law is zero almost surely, and the six-scalar
vector field vanishes, so the initial discrepancy remains constant and again
cannot exit the closed tube. The endpoint may lie on its boundary. In either case
the same estimate is at most `delta` on `[0,H]`. The complement of the
initialization event is bounded by the first display with
`epsilon=exp(-L_(H,r)H)min(r,delta)`, which proves the probability statement. □

The same bootstrap applies jointly to the both/weak-only zero-disorder pair. Use a
common deterministic radius `r` and the maximum of the two deterministic
`L_(H,r)` constants computed from their two fixed cue laws; the shared random
initial six scalars retain the same initialization bound.

This theorem is a genuine deterministic large-width result at a singular solvable
point. It is not a DMFT and says nothing about positive recurrent disorder or
positive lag.

## General paired CE-RNN DMFT — conjecture, not theorem

The full empirical-NTK equations in Corollary A.4 are exact at finite width, but they
do not close because the sample NTK and response cross-kernel move with the trained
parameters. Theorem A.5 converts independently established movement bounds into a
trajectory comparison; freezing measured initialization kernels without such
bounds is a surrogate, not a derivation of their dynamics. The sealed held-out
crossing-event result is correspondingly empirical and does not supply kernel
stability.

For positive recurrent disorder or positive lag, a joint optimization-time DMFT
would still need a coupled order-parameter process containing both causal
conditions, projected modes and geometry, full margin laws, recurrent
covariance/response kernels, and every loss-invisible overlap required for closure.
That remains a conjectural extension. The unresolved obligations are:

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

The code evaluates theorem formulas and enforces scalar guards represented in its
arguments. It does not establish trajectory, model, smoothness, or first-entry
hypotheses and does not replace the proofs.

- `projected_statistics`: exact cross-kernel drift when direct autograd is enabled,
  plus the `Gg` projected component and its residual;
- `empirical_logit_jacobian` and `initial_frozen_empirical_kernel`: materialize
  Corollary A.4's complete signed sample-logit Jacobian, `Theta_0`, and `c_0` without
  training or populating parameter gradients;
- `integrate_frozen_logistic_sgd`: evaluates the initialization-frozen nonlinear
  logistic recursion (A.6) on the experiment's optimization-time grid;
- `frozen_kernel_discrete_error_bound`: evaluates (A.10) from caller-supplied
  secant-kernel and response-cross-kernel movement bounds; it neither computes nor
  proves those bounds for tanh/GRU;
- `run_enl_preflight` and `run_enl_evaluate`: separate initialization-only
  prediction from training, seal and externally pin prediction artifacts, reject
  source/configuration/seed mismatches before optimization, and verify frozen-input
  immutability after evaluation; their acceptance metrics are empirical tests, not
  kernel-stability certificates;
- `equal_time_drift_difference`: projected equal-time decomposition and, when
  available, the exact direct gap derivative;
- `rank_one_drift_ratio`: Theorem C factorization and log-ratio diagnostic;
- `discrete_crossover_certificate`: exact finite-step analogue of the
  single-crossing Corollary B.2 using realized response increments;
- `transverse_hitting_time_error_bound`: evaluates Corollary D's bound and guards
  supplied uniform error, quantitative pre-hit separation, transversality radius,
  and mesh; the caller establishes directional first-entry semantics, derivative
  facts, and grid validity;
- `zero_disorder_initialization_failure_bound` and
  `zero_disorder_trajectory_failure_bound`: evaluate Theorem E.2's explicit bounds
  from a caller-supplied deterministic reference-tube Lipschitz constant and
  bootstrap radius;
- `cdc_finite_step_deviation_bound`: evaluates the fixed-coordinate Euclidean
  one-step CDC bound from supplied norms and `L`; the caller establishes local
  smoothness and segment containment.

Tests establish these software contracts and numerical identities. They are not
evidence that tanh/GRU kernels remain close to initialization, that tanh/GRU satisfy
the sufficient inequalities in Theorem C, that the semantic hypotheses supplied to
a helper hold, or that the general DMFT conjecture is true.
