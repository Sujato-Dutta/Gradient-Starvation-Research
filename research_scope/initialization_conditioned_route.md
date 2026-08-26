# Initialization-conditioned route for recurrent Gradient Starvation

## Status

**Implemented finite-width predictor and completed held-out protocol; not a
starvation proof.** This route avoids the unavailable general
positive-disorder/positive-lag optimization-time DMFT. It combines exact
finite-width initialization derivatives, a full sample-logit empirical-NTK
surrogate, conditional kernel-movement bounds, and the already-proved deterministic
tail-area/learnability certificates.

The archived seeds below are retrospective design data. A sealed broad pilot was
then rejected, and a fresh sealed factorial passed only the subsequently restricted
drift/response-crossing endpoints. No experiment proved kernel stability, accurate
timing, broad phase or causal prediction, universal starvation, or DMFT.

## Exact initialization object

For the paired joint state `X=(theta_B,theta_W)` and the unregularized full-batch CE
field

```text
V(X)=(-grad L_B(theta_B), -grad L_W(theta_W)),
Delta(X)=M_w(theta_B)-M_w(theta_W),
d(X)=L_V Delta,
j(X)=L_V d,
```

shared initialization and the common probe response give `Delta(X_0)=0`. The helper
`theory.paired_initial_response_jet` now computes the exact finite-width tuple
`(Delta(0),d(0),j(0))` by higher-order autograd. Both the response Hessian and the
loss-Hessian contribution are retained. It also returns each condition's drift and
second Lie derivative so the paired subtraction is auditable.

The helper is valid for the implemented optimizer field only under full-batch
cross-entropy gradient flow: no weight decay, clipping, momentum, minibatch noise,
or alternate objective. It verifies bitwise-equal model state and an exact matched
strong-channel intervention before evaluating the jet.

## What can already be proved

**Local sign-persistence theorem.** Let the paired gradient-flow solution remain in
a tube `U` for `0 <= tau <= h`, and suppose `|L_V d(X)| <= J` throughout `U`. Then

```text
|d(tau)-d(0)| <= J tau,
Delta(tau) >= d(0) tau - J tau^2/2       when d(0)>0,
Delta(tau) <= d(0) tau + J tau^2/2       when d(0)<0.
```

Consequently, if `d(0)>0`, transfer is certified for
`0 < tau < min(h,d(0)/J)`; if `d(0)<0`, suppression from initialization is
certified for `0 < tau < min(h,|d(0)|/J)`.

**Proof.** Along the flow, `d'(tau)=L_V d(X(tau))`. Integrating the uniform bound
once controls `d`, and integrating the resulting one-sided inequality again from
`Delta(0)=0` controls `Delta`. This is the fundamental theorem of calculus. □

A useful stronger conditional statement is also immediate. If
`-Lambda <= j(tau) <= -lambda < 0` until the first drift zero and `d(0)>0`, then

```text
d(0)/Lambda <= tau_d <= d(0)/lambda,
A_+ = Delta(tau_d) <= d(0)^2/(2 lambda).
```

Combined with a certified post-crossover bound `d <= -kappa`, Theorem B.1/B.2 then
bounds the response crossing. These are conditional implications; evaluating only
`j(0)` does not establish the required tube bounds.

For explicit-Euler SGD, `d(0)` is the continuous-field derivative, not the exact
one-step gap increment. A theorem matching the experiment must either use the exact
one-step observable or add a validated Taylor remainder, e.g. a Hessian bound on
`Delta` and a velocity bound. Logged points every ten updates cannot supply that
remainder.

## Retrospective archived-seed reconstruction

Configuration: width 32, sequence length 20, `rho=4`, lag 2, positive regime,
`n=256`, cue noise 0.1. The implementation exactly reproduces every archived
`initial_d_w`; `Delta(0)=0` in all cases.

### tanh

| coupled seed | `d(0)` | `j(0)` | local `-d(0)/j(0)` | observed drift zero | observed response zero | archived outcome |
|---:|---:|---:|---:|---:|---:|---|
| 20 | -0.012709 | -0.035033 | — | — | — | suppression from initialization |
| 21 | 0.015542 | -0.264783 | 0.058697 | 0.070056 | 0.132684 | causal certificate |
| 22 | -0.007654 | -0.144671 | — | — | — | suppression from initialization |
| 23 | -0.005000 | -0.119963 | — | — | — | suppression from initialization |
| 24 | 0.028350 | -0.212527 | 0.133393 | 0.183421 | 0.392152 | causal certificate |
| 25 | 0.031098 | -0.167718 | 0.185419 | 0.329166 | 0.642035 | causal certificate |
| 26 | -0.024230 | -0.076679 | — | — | — | suppression from initialization |
| 27 | -0.006400 | -0.148911 | — | — | — | suppression from initialization |

The sign of exact `d(0)` partitions the observed `3/8` causal-certificate and `5/8`
suppression groups perfectly. All eight tanh `j(0)` values are negative. Among the
three positive-drift seeds, the linearized drift-zero ordering is exactly the
observed ordering, although the times are early by approximately 16%, 27%, and 44%.
The quadratic response-zero estimates `-2d(0)/j(0)` are also early and are not
certificates.

### GRU

| coupled seed | `d(0)` | `j(0)` | observed response crossing? |
|---:|---:|---:|---|
| 20 | -0.014605 | 0.004519 | no |
| 21 | 0.002104 | 0.001508 | yes |
| 22 | -0.009240 | 0.007351 | no |
| 23 | 0.007492 | -0.006254 | yes |
| 24 | 0.004899 | -0.004907 | yes |
| 25 | -0.009098 | 0.000393 | no |
| 26 | 0.006780 | -0.007713 | yes |
| 27 | 0.010397 | -0.001299 | yes |

The sign of `d(0)` again partitions the retrospective response-crossing groups, but
`j(0)>0` for seed 21 even though its drift eventually crosses at optimization time
3.43. Thus a second-order local predictor is not a credible full-horizon GRU
mechanism. GRU needs a larger certified tube, a higher-order jet, or the full frozen
sample-logit NTK route. No GRU run passes the weak-only learnability gate.

## Full empirical-NTK route and frozen protocol

For each causal condition, the implemented predictor materializes the complete
signed sample-logit Jacobian `J`, empirical NTK `Theta=J J^T`, and weak-response
cross-kernel `c=J grad M_w` at initialization. It then integrates the nonlinear
logistic tangent recursion with `Theta` and `c` frozen while the margin gate evolves.
This is the finite-width surrogate of Theorem A.4, not the two-probe Gram
approximation. At initialization its paired response drift agrees with the direct
jet; automated tanh/GRU tests also check the kernel identities, symmetry, absence of
state mutation, and unpopulated parameter gradients.

`run_enl_preflight` enforces full-batch unregularized cross-entropy, computes no
trained state, writes the kernel and prediction artifacts, and hashes the manifest,
predictions, resolved configuration, environment, executable-source fingerprint,
and every per-record kernel artifact. `run_enl_evaluate` verifies the externally
pinned manifest and exact seed/configuration/source match before constructing an
optimizer, loads rather than recomputes predictions, and verifies after training
that all frozen inputs are unchanged. Explicit `data_seed`/`model_seed` pairs are
supported; reused axes are analyzed with a two-way pigeonhole bootstrap rather than
row-wise independent intervals.

The available continuous and exact secant-SGD comparison bounds remain conditional
on caller-supplied uniform movement bounds for `Theta` and `c`. Neither preflight
nor held-out accuracy proves those bounds for tanh or GRU.

The pilot and factorial manifests record executable-source fingerprints, but the
corresponding dirty Python source snapshots were not archived and are no longer
reconstructible. The current tree therefore correctly fails their source guard. The
compact archives audit outcomes and seals, not bitwise executable replay; a newly
generated and repinned preflight is a new study.

## Completed held-out evidence

### Retrospective design set

Coupled seeds 20--27 remain design data only. Their exact initialization jets and
observed trajectories motivated the route, but their perfect sign partition is
confounded by seed reuse and model selection. They are excluded from confirmatory
evidence. The data-seed-100/model-seed-1000--1007 engineering set is also excluded:
an evaluator floating-grid failure generated unscored outcomes, so it contributes no
scientific evidence.

### Broad pilot: rejected

The initialization-only predictions for data seed 200 and model seeds 2000--2007,
for both tanh and GRU, were sealed under manifest
`f186d587e28f6408b68baf95566d2d1f1c5d5701e205e9a69ce86cbe4f0ffaf5` before the
16 trajectories were run. The preregistered primary outputs were phase, drift
crossing, response crossing, causal certificate, and weak-only learnability, with
complete coverage, overall accuracy at least `0.75`, and each-architecture accuracy
at least `0.625`.

The broad pilot failed. Tanh weak-only learnability was `4/8=0.5`, below the
architecture threshold; no post hoc success label can repair that rejection.
Descriptively, overall phase, drift-crossing, response-crossing, causal-certificate,
and learnability accuracies were respectively `12/16`, `14/16`, `14/16`, `14/16`,
and `12/16`. This pilot became design evidence for a narrower fresh test, not part
of that test.

### Fresh 4x8 factorial: restricted crossing endpoints passed

After the broad rejection, the primary claim was narrowed in advance to binary
drift-crossing and response-crossing events. Phase, causal certificate,
learnability, crossing times, and trajectory magnitudes were explicitly secondary.
The fresh design crossed data seeds 201--204 with model seeds 2010--2017 for each of
tanh and GRU: 64 architecture/seed records. Predictions were sealed under manifest
`bdcf02abbef5b9d8f7ce3979d3363ee8d39ab7817a1d554616bc4dd4a22a0db1` before
training. Acceptance required complete coverage, overall accuracy at least `0.75`,
each-architecture accuracy at least `0.625`, and an overall two-way-bootstrap 95%
lower endpoint at least `0.5` for each primary endpoint.

The restricted test passed: drift crossing was correct for `60/64` records and
response crossing for `64/64`. Their reuse-aware 95% bootstrap intervals were
`[0.8125,1.0]` and `[1.0,1.0]`. By architecture, drift crossing was `28/32` for GRU
and `32/32` for tanh; response crossing was `32/32` for each.

The result supports only an uncalibrated crossing-event classifier in this width-32,
`rho=4`, lag-2, positive-regime setting. The broader claim remains rejected: the
secondary phase, causal-certificate, and learnability accuracies were `52/64`,
`56/64`, and `49/64`, and tanh learnability was only `17/32` with bootstrap interval
`[0.1875,0.875]`. Crossing times were systematically early: overall drift and
response biases were `-0.8225` and `-1.1892`, with MAEs `0.8249` and `1.1892`.
Tanh trajectory magnitudes were poor despite strong event signs: response-gap,
weak-response, and drift RMSEs were `2.4552`, `2.4894`, and `0.4903`.

## Route conclusion and remaining obligations

The local jet theorem above remains a separate exact short-time route. It still
needs a certified flow tube and interval bound on `L_V d` to produce a nonzero
sign-persistence certificate. The frozen full empirical NTK supplies a stronger
finite-width empirical predictor and an exact conditional explicit-SGD error
formula, but no tanh/GRU kernel-movement bound has been established.

For an eventual starvation theorem one must still prove a unique drift crossing,
sufficient negative tail area, and weak-only target attainment, with first-entry
and discretization hypotheses where required. The completed factorial does not
supply those proofs and does not estimate a universal prevalence outside its
frozen design. The scientifically supported conclusion is therefore: exact
initialization diagnostics, conditional local and frozen-kernel theorems, rejection
of the broad predictor, and held-out support for only the restricted crossing-event
classification endpoints. It is not accurate timing, universal starvation, or
DMFT.
