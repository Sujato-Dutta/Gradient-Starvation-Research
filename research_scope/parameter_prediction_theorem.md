# Initialization-conditioned parameter prediction theorem

## Epistemic status

**Proved at the scopes stated below.** The first result is an exact finite-width,
initialization-only dense-linear formula with its strongest unconditional local
outcome implication. The second is a deterministic implication from independently
proved parameter-ball bounds. The third is a distribution-only corollary at zero
lag, where recurrence is loss-invisible for every finite realized recurrent
matrix.

Together these are the strongest honest pre-trajectory results currently supported
by the model. They do **not** produce a deterministic general
`(rho, lag_separation)` starvation phase boundary. At positive lag, the realized
cross-geometry has a seed-dependent leading sign, and this repository has no
general positive-lag trajectory closure or interval constructor for the required
ball bounds.

Throughout, sequence lag is `ell=Delta ell` and optimization time is `tau`.

## 1. Exact realized-parameter initialization threshold

Consider the dense linear RNN

```text
h_(k+1) = W h_k + B x_k,        f(x)=c^T h_T,
```

with strong input column `b_s`, weak input column `b_w`, the strong cue at the final
sequence position, and the weak cue `ell` positions earlier. Use the noiseless
positive rank-one cue laws

```text
z^B=(rho,1),        z^W=(0,1),        rho>0,
```

and shared initial parameters in the paired both/weak-only full-batch logistic-loss
gradient flows. Define the common unit-probe responses

```text
M_s = c^T b_s,
M_w = c^T W^ell b_w.
```

Let `G` be the exact finite-width response-gradient Gram matrix from T3 and set

```text
K = G_ws = b_s^T W^ell b_w,
A = G_ww
  = ||W^ell b_w||^2 + ||(W^ell)^T c||^2 + ||S_w||_F^2,
S_w = sum_(k=0)^(ell-1) ((W^k)^T c)(W^(ell-1-k)b_w)^T.
```

The recurrent derivative term is zero when `ell=0`. Define

```text
s_B = sigmoid(-(rho M_s+M_w)),       s_W = sigmoid(-M_w).
```

### Theorem P.1 (exact initial rate-gap criterion)

The initial equal-time weak-response rate gap is exactly

```text
d_0 = dot M_w^B(0)-dot M_w^W(0)
    = s_B(rho K+A)-s_W A.                                      (P.1)
```

Consequently,

```text
initial transfer          iff K > (A/rho)(s_W/s_B-1),
initial rate suppression  iff K < (A/rho)(s_W/s_B-1),          (P.2)
initial boundary          iff equality holds.
```

This criterion uses only `rho`, `ell`, and the realized initialization
`(W,b_s,b_w,c)`; no trained trajectory is observed. It also has an unconditional
local outcome implication. Shared initialization gives `Delta(0)=0` and
`Delta'(0)=d_0`, so

```text
d_0<0  =>  Delta(t)<0 for every sufficiently small t>0,
d_0>0  =>  Delta(t)>0 for every sufficiently small t>0.             (P.2a)
```

Thus a strict negative sign certifies local outcome suppression and a strict
positive sign certifies local outcome transfer. This statement is existential: it
does not provide or estimate a quantitative time horizon. When `d_0=0`, no local
outcome sign follows from first order alone.

**Proof.** Exact mode realization gives `dot M=Gg`. In the both condition,
`g^B=s_B(rho,1)`, so `dot M_w^B=s_B(rho G_ws+G_ww)`. In the weak-only condition,
`g^W=s_W(0,1)`, so `dot M_w^W=s_W G_ww`. Subtraction proves (P.1). Because
`rho s_B>0`, rearranging (P.1) proves (P.2). The displayed formulas for `K` and
`A` are T3 specialized to a zero-lag strong probe and an `ell`-lag weak probe.
Finally, differentiability and `Delta(0)=0` give
`Delta(t)/t -> Delta'(0)=d_0` as `t downarrow 0`; the strict-sign implications in
(P.2a) follow. This limit argument proves existence of a sufficiently small
interval but supplies no numerical endpoint. □

`theory.dense_linear_initial_gap_certificate` evaluates (P.1)--(P.2a) and refuses
nonpositive-rank-one or noisy task specifications. Its caller tolerance is an
absolute band in weak-response-rate units. Values in that band are reported as
`boundary_or_undetermined` and produce neither local transfer nor local suppression
certification; the exact `d_0` and tolerance are both retained. A direct test
compares `d_0` with `paired_initial_response_jet`, which obtains the same derivative
from the universal full-batch CE field by higher-order autograd.

### Why P.1 is not a general phase boundary

Under the repository's iid width scaling and fixed positive lag,
`M_s`, `M_w`, and `K` fluctuate at initialization scale `N^(-1/2)`, while `A` is
order one. Expanding the sigmoid gates around zero gives

```text
d_0 = rho (K/2 - A M_s/4) + O_P(N^(-1)).                      (P.3)
```

The leading term is centered and realization-dependent. Thus fixed
`(rho,ell,bulk_gain)` does not determine its sign. Replacing `K` or `M_s` by zero
would erase the leading finite-width effect rather than prove a population law.
This is the concrete obstruction to promoting P.1 into an analytic positive-lag
`rho_c`.

## 2. Conditional parameter-ball outcome theorem

Let `X=(theta_B,theta_W)` be the paired parameter state, let

```text
Delta(X)=M_w(theta_B)-M_w(theta_W),
d(X)=L_V Delta(X),
j(X)=L_V d(X),
```

for the joint unregularized full-batch logistic-loss field
`V=(-grad L_B,-grad L_W)`. Shared initialization gives `Delta(X_0)=0`.
Choose a product norm and its closed radius-`R` ball

```text
U_R={X: ||X-X_0||<=R}.
```

All constants below must be proved in this same norm and uniformly on `U_R`.

### Theorem P.2 (parameter-tube crossing, outcome, and learnability)

Assume

```text
||V(X)|| <= S_R,                         S_R>0,
-Lambda <= j(X) <= -lambda < 0,          0<lambda<=Lambda,
```

for every `X in U_R`, and define the certified horizon `h_R=R/S_R`. If `d_0>0`
and

```text
h_R > d_0/lambda,
```

then the paired trajectory has exactly one rate crossing `tau_d` before leaving
the ball, with

```text
d_0/Lambda <= tau_d <= d_0/lambda.                         (P.4)
```

If, more strongly,

```text
h_R > 2d_0/lambda,                                          (P.5)
```

then strict outcome suppression occurs within the certified horizon. Specifically,
for every `t` satisfying `2d_0/lambda<t<=h_R`,

```text
Delta(t) < 0.                                                (P.6)
```

Finally, if on the same ball

```text
dot M_w^W(X) >= nu > 0
```

and the target is nondegenerate,

```text
beta > M_w^W(0),
(beta-M_w^W(0))/nu < h_R,                                  (P.7)
```

then the weak-only member learns by reaching target `beta` at a strictly positive
time before leaving the ball. If `beta<=M_w^W(0)`, the target was already met at
initialization; that is degenerate attainment, not weak-only learnability, and it
cannot gate a causal-starvation certificate. Conditions (P.5) and the nondegenerate
(P.7) therefore certify causal starvation, rather than only a rate or outcome
ordering.

**Proof.** Until exit from `U_R`, path length is at most `S_R t`; hence exit cannot
occur before `h_R`. Along that interval, `d'=j`, so

```text
d_0-Lambda t <= d(t) <= d_0-lambda t.                       (P.8)
```

The lower bound is positive before `d_0/Lambda`, the upper bound is negative after
`d_0/lambda`, continuity gives a zero between them, and `d'<0` makes it unique.
Integrating the upper bound from `Delta(0)=0` gives

```text
Delta(t) <= d_0 t-lambda t^2/2,
```

which is strictly negative for `t>2d_0/lambda`, proving (P.6). For
`beta>M_w^W(0)`, integrating the weak-only drift lower bound proves target
attainment by the positive time in (P.7). A target at or below the initial response
requires no optimization and is excluded from the learnability and causal gates.
The causal conclusion is exactly outcome suppression plus independent,
nondegenerate weak-only learnability. □

### Corollary P.3 (certified safe horizon)

If instead `|j(X)|<=J` on `U_R` and

```text
d_0-J h_R > 0,                                               (P.9)
```

then throughout `[0,h_R]`

```text
d(t) >= d_0-Jt > 0,
Delta(t) >= d_0 t-Jt^2/2 > 0       for t>0.                  (P.10)
```

Thus relative transfer persists and outcome suppression is excluded on the
certified horizon. This is a finite-horizon safety statement, not a claim about all
future optimization time.

`theory.parameter_tube_certificate` evaluates P.2--P.3. Missing optional premises
fail closed. It records `target_met_at_initialization` explicitly and never lets a
true value set either weak-only learnability or causal starvation. The function
checks scalar consistency but deliberately does not estimate `S_R`, `lambda`,
`Lambda`, `nu`, or `J` from sampled trajectory points. No current repository
component constructs nonvacuous values for general trained positive-lag RNNs;
claiming otherwise would turn a conditional theorem into a trajectory diagnostic.

## 3. Zero-lag distribution-only corollary

At zero lag, zero background input, and zero initial hidden state, both cues occur
at the final step. Hence every state entering the final update is zero and

```text
h_T = W h_(T-1)+B x_(T-1) = B x_(T-1).
```

For every finite realized recurrent matrix `W`, the responses and loss are
independent of `W`, every recurrent gradient is zero, and `W` remains fixed. Thus
T9--T10's six-scalar closure holds at arbitrary realized recurrent bulk gain; only
positive lag, where earlier cue-induced states generally expose `W`, remains
blocked. The P.11--P.13 specialization below additionally requires the noiseless
positive rank-one cue law, explicitly `cue_noise=0`, as well as zero background
noise. At the deterministic symmetric reference

```text
(M_s,M_w,u_ss,u_sw,u_ww,w)=(0,0,1,0,1,1),
```

let `bar Delta` and `bar d` denote the deterministic both-minus-weak response gap
and rate gap.

### Corollary P.4 (zero-lag recurrence-invisible suppression and learnability)

Assume the noiseless positive rank-one cue law (`cue_noise=0`), zero background
noise, zero lag, and any finite realized recurrent matrix. For every `rho>0`,

```text
bar Delta(0)=0,       bar d(0)=0,       bar d'(0)=-rho^2/2<0. (P.11)
```

Therefore there exists `epsilon_rho>0` such that
`bar Delta(t)<0` for every `0<t<=epsilon_rho`: the deterministic reference is
suppressed from initialization, not transfer-then-suppression.

The deterministic weak-only response reaches every finite `beta>0` in finite time.
Indeed its invariant symmetric subsystem obeys

```text
u_ww=w=sqrt(1+M_w^2),
dot M_w=2 sigmoid(-M_w) sqrt(1+M_w^2)>0,                     (P.12)
```

and, while `0<=M_w<=beta`,

```text
dot M_w >= 2 sigmoid(-beta).
```

Hence a valid target-time upper bound is

```text
tau_beta <= beta/[2 sigmoid(-beta)].                         (P.13)
```

The deterministic zero-lag reference therefore exhibits causal suppression for
every `rho>0` and every finite weak target. It does not exhibit an initial transfer
phase, and this conclusion is independent of the arbitrary finite realized `W`.

**Proof.** The zero-pre-final-state argument above makes `W` loss-invisible and
reduces the flow to T9's six-scalar equations. Those equations give identical
initial weak drifts in the two conditions. Differentiating once at the symmetric
state gives `(dot M_w^B)'=-(rho^2+1)/2` and
`(dot M_w^W)'=-1/2`, proving (P.11). Taylor's theorem and continuity imply the
local strict sign. In the weak-only subsystem, symmetry preserves `u_ww=w`;
differentiating `u_ww^2-M_w^2` shows it remains one, which gives (P.12). The lower
bound and (P.13) follow by integration. No law or zero-gain premise for `W` enters
any step. □

For a finite random width, fix deterministic witness times `t_s,t_beta` with
`bar Delta(t_s)<0` and `bar M_w^W(t_beta)>beta`, and let `H` contain both. If

```text
0<delta<min(-bar Delta(t_s)/2, bar M_w^W(t_beta)-beta),
```

then T10's joint paired tube event implies finite-width outcome suppression at
`t_s` and weak-only target attainment by `t_beta`. With the common deterministic
tube radius `r` and the maximum paired Lipschitz constant `L_(H,r)`, its failure
probability is at most

```text
min(1, 9 exp(2 L_(H,r) H)/(N min(r,delta)^2)).               (P.14)
```

The witness margins in (P.14) must be established analytically or by validated
integration. Ordinary floating-point ODE output is not silently promoted to a
proof of them.

## 4. What is and is not predicted before training

The executable pre-trajectory decision chain is:

1. P.1 computes exact `d_0` from `(rho,ell,W,b_s,b_w,c)`.
2. If independently certified product-ball constants are available, P.2 yields a
   unique rate crossing, an outcome-suppression witness, and, with the explicit
   nondegenerate weak-target premises, a causal-starvation certificate; P.3 can
   instead exclude suppression on a finite horizon.
3. Without those constants, only the initialization sign is proved. `j_0` or a
   sampled frozen-kernel trajectory is a diagnostic/predictor, not a uniform tube
   certificate.
4. Corollary P.4 gives a genuine distribution-only statement at zero lag for
   arbitrary finite realized recurrent gain, under zero background input and its
   explicit noiseless positive rank-one cue law.

The unresolved main challenge is therefore sharply localized: derive computable,
nonvacuous `S_R`, `lambda`, `Lambda`, `nu`, or `J` from general positive-lag
recurrent parameters, or prove another closed positive-lag trajectory law. Until
that obligation is met, there is no defensible universal starvation/safe phase
diagram in `(rho,Delta ell)`.
