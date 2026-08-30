# Paper theorem package — current authoritative summary

This document is the paper-facing map of what is proved, what is measured, and
what remains open. `claim_ledger.md` is the statement-level authority;
`e2_theorem.md` and `cdc_theorem.md` contain the proofs.

## 1. Headline theorem package

The paper no longer presents a general paired CE-RNN DMFT as an established
result. Its rigorous contribution is a finite-width causal package.

### Proposition A: exact finite-width CE response flow

For any differentiable response, full-batch logistic gradient flow induces the
exact response/logit cross-kernel dynamics. If signed logits are exactly linear in
parameter-independent feature coordinates, this reduces to `dot m=Gg`. The dense
linear recurrent geometry has a closed finite-width expression including readout,
input, and recurrent parameter blocks.

**Scope:** the closed two-mode form is exact for the linear synthetic model with
zero background noise. Tanh/GRU modes are common symmetric unit-probe responses,
not exact additive logit coordinates; their exact response
drift must be computed by the universal cross-kernel/direct-autograd form, with the
`Gg` residual reported.

### Exact signed-logit/response empirical-NTK flow

For one finite training set, let `r(theta)` be the vector of signed logits, let
`J_r(theta)` have rows `grad r_i`, define the full signed-logit empirical NTK
`K(theta)=J_r J_r^T`, and define the response cross-kernel
`C_i(theta)=<grad r_i,grad M(theta)>`. With `w(r)=sigma(-r)`, Proposition A.1 applied
also to each `r_i` gives the exact coupled finite-width flow

```text
dot r = K(theta) w(r)/n,
dot M = C(theta)^T w(r)/n.
```

This uses the complete sample-level kernels. The two-probe Gram matrix `G` cannot
replace `K` or `C` for nonlinear tanh/GRU responses.

### Initialization-frozen nonlinear logistic surrogate

Freeze `K_0=K(theta_0)` and `C_0=C(theta_0)`. The tangent surrogate

```text
dot r_bar = K_0 sigma(-r_bar)/n,
dot M_bar = C_0^T sigma(-r_bar)/n
```

is still nonlinear in its logits. It is an exact ODE for the initialization-frozen
surrogate. Under full-batch SGD with learning rate `eta`,
`integrate_frozen_logistic_sgd` exactly applies its explicit-Euler recursion

```text
r_bar_(k+1) = r_bar_k + eta K_0 sigma(-r_bar_k)/n,
M_bar_(k+1) = M_bar_k + eta C_0^T sigma(-r_bar_k)/n,
tau_k = eta k.
```

Neither exactness statement makes this the exact trained tanh/GRU trajectory.

### Conditional continuous/discrete secant-kernel comparison

Use Euclidean/operator norms, `b>=sup ||sigma(-r)||_2` (one may take `b=sqrt(n)`),
`a=||K_0||_2/(4n)`, and `c_0=||C_0||_2`. If a true continuous trajectory on
`[0,t]` satisfies uniform bounds

```text
||K(theta_s)-K_0||_2 <= delta_K,
||C(theta_s)-C_0||_2 <= delta_C,
```

then the true-versus-frozen errors obey

```text
||r(t)-r_bar(t)||_2 <= delta_K b Phi(t)/n,
|M(t)-M_bar(t)| <= t delta_C b/n
  + c_0 delta_K b Psi(t)/(4n^2),
Phi(t)=(exp(a t)-1)/a,
Psi(t)=(exp(a t)-1-a t)/a^2,
```

with continuous limits `Phi(t)=t` and `Psi(t)=t^2/2` when `a=0`.
For nonlinear SGD, apply the same argument to the exact segment-secant logit and
response kernels. If their distances from `K_0,C_0` are uniformly at most
`delta_K,delta_C` through step `k`, set

```text
Phi_k=((1+eta a)^k-1)/a,
Psi_k=(((1+eta a)^k-1)/(eta a)-k)/a.
```

Then

```text
||r_k-r_bar_k||_2 <= delta_K b Phi_k/n,
|M_k-M_bar_k| <= k eta delta_C b/n
  + eta c_0 delta_K b Psi_k/(4n^2),
```

with limits `Phi_k=k eta` and `Psi_k=eta k(k-1)/2` at `a=0`.
Subtracting the exact and frozen equations, using the `1/4` Lipschitz constant of
`sigma(-x)`, and applying Grönwall (continuous) or induction (discrete) proves the
bounds. `theory.frozen_kernel_discrete_error_bound` evaluates the discrete formula
from caller-supplied constants; it does not establish them. No uniform
instantaneous or secant-kernel movement bound has been proved for the tanh or GRU
experiments, so their frozen-surrogate agreement is empirical and conditional.

### Proposition B: cumulative drift balance and the single-crossing corollary

For `Delta=m_w^B-m_w^W` and exact equal-time derivative `d=Delta'`, split the
drift into positive and negative parts and let `P,N` be their cumulative areas.
Absolute continuity gives the general identity `Delta(t)=P(t)-N(t)` under arbitrary
sign changes. Strict outcome suppression therefore occurs exactly when cumulative
negative drift exceeds cumulative positive drift at some time. If there is one
`+ -> -` drift crossing at `tau_d`, the single-crossing corollary reduces this to
the familiar tail-area statement: the response later reaches equality iff the
post-crossover negative area reaches the positive area stored at the peak, and
strict suppression requires strict excess negative area. Quantitative drift bounds
yield explicit upper bounds on the response crossing time.

This is the exact correction to the retracted statement that a drift crossing
necessarily causes a response crossing. It also covers oscillatory drift and
suppression from initialization, which the single-crossing corollary does not.

### Theorem C: ordering-invariant noiseless rank-one criterion

For deterministic cues `z^B=(rho,1)` and `z^W=(0,1)`, positive weak drifts factor
as `F_w^B=s_B H_B` and `F_w^W=s_W H_W`. The log ratio

```
Psi = log(F_w^B/F_w^W)
```

has an exact derivative separating total geometry-ratio growth from margin-gate
contraction without choosing either non-canonical additive decomposition. A positive
`Psi` followed by a uniform negative derivative produces one rate crossover and an
explicit time bound. Outcome suppression still requires Proposition B's cumulative
balance; causal starvation additionally requires the independent weak-only first-
hit learnability gate.

**Prediction discipline:** bounds on `Psi'` proved before outcomes can prospectively
bound a crossover. Evaluating `Psi` on a completed trajectory certifies it but is
not a certified or quantitatively accurate held-out time prediction.

### Corollary D: hitting-time stability

For continuous trajectories, an absolutely continuous reference with an interior
first upward hit, an a.e. slope lower bound on a radius-`r` neighborhood, and a
positive quantitative prehistory margin has first-hit error at most
`epsilon/kappa` when `epsilon<eta` and `epsilon/kappa<=r`. A logged grid adds at
most its maximum mesh when the enlarged window remains inside `r`. Paired delays
require two finite hits satisfying their own assumptions; otherwise they are
censored. Descending response-gap crossings must be sign-reversed and localized
after a point where `Delta>0`, because `Delta(0)=0`. This licenses hitting-time
limits only after uniform trajectory convergence and the directional first-entry
hypotheses have independently been proved.

### Theorem E: exactly solvable recurrent anchor

At zero recurrent disorder, zero cue lag, and zero background noise, the dense
linear dynamics close at every width on six scalar inner products. Gaussian
initialization obeys the explicit Chebyshev bound `9/(N epsilon^2)`. For each fixed
cue law, the reference energy obeys `R<=3+4 W(e^-1)H`; a deterministic radius `r`
gives `L=4(A+(R_H+r)C)` and the bootstrapped trajectory failure bound
`9 exp(2LH)/(N min(r,delta)^2)`. Thus finite-horizon ODE stability yields uniform
convergence in probability. This is a genuine large-width theorem at a singular
solvable point, not a positive-disorder or positive-lag DMFT.

### CDC Results 1–3

All CDC gradients, projections, norms, and smoothness balls use the fixed
implemented tensor coordinates with the product Euclidean/Frobenius metric; there
is no reparameterization-invariance or natural-gradient claim.

1. CDC preserves the current strong-feature drift exactly at first order.
2. The uncapped feasible correction is the unique minimum Euclidean-norm,
   lower-bound-attaining correction in those coordinates. Exact positive-deficit
   feasibility is the nonvanishing tangent restriction represented by `q!=0`;
   `q`, its norm/tolerance, and the selected minimizer are metric/scale-dependent.
3. Under same-chart local smoothness on both straight step segments and a
   nondegenerate projected weak direction, one CDC step differs from the
   corresponding ERM strong response by an explicit `O(eta^2)` bound.

A nonbinding cap leaves Result 2 unchanged. A binding cap preserves Result 1 and
the actual-velocity Result 3 bound but generally misses the weak target, so no
target-attaining optimality claim applies. Result 3 is a local one-step theorem;
no trajectory-level or final strong-retention theorem follows.

## 2. Explicitly open theory

The following are not paper theorems:

- the positive-disorder/positive-lag joint CE-RNN optimization-time DMFT;
- certified or quantitatively accurate tanh/GRU drift- or response-crossing times;
  the initialization-frozen surrogate is an empirical event classifier whose time
  predictions remain uncalibrated;
- uniform tanh/GRU instantaneous or secant-kernel stability on the training horizon;
- an analytic learnability/starvation boundary `rho_c`;
- a general noisy-cue rank-one factorization;
- a trajectory-level CDC guarantee.

All general solver paths remain blocked. The former five DMFT obligations remain:
derive the effective process, enumerate visible and loss-invisible closure variables,
prove existence/uniqueness, prove joint quenched concentration, and combine the
limit with hitting-time/discretization control.

## 3. Empirical package that can be quoted

Use only final-scale artifacts named in `claim_ledger.md`.

### Controlled causal phase grid

`results/e1_dense_rerun-20260823-085142`

- positive weak AUC gap in all 32 `rho`–seed cells per lag (four strengths
  by eight seed IDs) at lags 0, 2, and 4;
- negative-control transfer in 8/8 seeds;
- lag 8 is unlearnable under the preregistered target and therefore not labelled
  starved;
- AUC gap is monotone in `rho` at every measured lag.

### Initialization-frozen empirical-NTK prediction

The two-stage commands and checked-in configs are:

```bash
python run_experiment.py enl-preflight --config configs/enl_ntk_pilot_preflight.yaml
python run_experiment.py enl-evaluate --config configs/enl_ntk_pilot_evaluate.yaml
python run_experiment.py enl-preflight --config configs/enl_ntk_crossing_factorial_preflight.yaml
python run_experiment.py enl-evaluate --config configs/enl_ntk_crossing_factorial_evaluate.yaml
```

The archived evaluate configs point to the named final preflight directories and
pin their manifest hashes. For a new preflight, its output directory and manifest
digest must be pinned in the matching evaluate config before evaluation.

Preflight writes `config.resolved.yaml`, `environment.json`, `predictions.csv`,
`prediction_summary.csv`, `prediction_aggregate.csv`,
`preflight_acceptance.json`, `manifest.json`, `manifest.sha256`, and one
`kernels/*.pt` file per record. Evaluation writes `config.resolved.yaml`,
`environment.json`, `trajectories.csv`, `summary.csv`, `aggregate.csv`,
`crossover.csv`, `scores.csv`, `metrics.json`, `evaluation_acceptance.json`, and
`provenance.json`.

Before creating an evaluation output directory or optimizer, `enl-evaluate`
requires the sidecar, actual manifest digest, and externally pinned digest to
agree; checks the accepted schema and no-training preflight contract; requires an
exact executable-source fingerprint/count and exact equality of the frozen task,
model, training, and held-out protocol; validates safe unique paths, required-file
and per-kernel hashes/sizes, prediction digest, complete configured seed factorial,
and exact step/`tau` grids. It consumes the hashed `predictions.csv`, not a
reconstruction from kernel files, and re-hashes every sealed input after training.
This is a fail-closed integrity/provenance contract, not a kernel-stability proof.

The broad 16-record pilot is a negative result. Classifier correctness was phase
`12/16`, drift crossing `14/16`, response crossing `14/16`, causal certificate
`14/16`, and weak-only learnability `12/16`; it **failed** because tanh learnability
was only `4/8`, below the frozen per-model threshold. The final pilot provenance is:

- evaluation `results/enl_ntk_pilot_evaluate-20260825-160757`;
- compact evidence `paper/artifacts/enl_ntk_pilot-20260825`;
- manifest `f186d587e28f6408b68baf95566d2d1f1c5d5701e205e9a69ce86cbe4f0ffaf5`;
- predictions `9787d99953f1046b4934aa6e7b3f969e30a4efafb3d02a67cc95926232f9dfaa`;
- recorded source fingerprint `8d8162558e7d25fff7a5059a4cf49139a354ca211fd6f0759417e75dd87f435c`.

After that failure, the fresh restricted factorial froze only drift- and
response-crossing events as primary. It crossed four unseen data seeds with eight
unseen model seeds for each architecture (`64` records total). Response-crossing
classification was `64/64`; drift-crossing classification was `60/64` (`32/32`
tanh, `28/32` GRU). The reuse-aware two-way-bootstrap 95% lower bounds were `1.0`
and `0.8125`, respectively. The final factorial provenance is:

- evaluation `results/enl_ntk_crossing_factorial_evaluate-20260825-162949`;
- compact evidence `paper/artifacts/enl_ntk_crossing_factorial-20260825`;
- manifest `bdcf02abbef5b9d8f7ce3979d3363ee8d39ab7817a1d554616bc4dd4a22a0db1`;
- predictions `c05b31810cb3869e7d051f539741732a8758f6cac542deb448a98d2ada7b6542`;
- recorded source fingerprint `ca878df376b01c0ea45ac352e353530bf20c567de59bb400617e5dd74d5b6b6b`.

The compact archives retain resolved configs, seals, acceptance/metrics/provenance,
and per-record scores with per-file hashes. They omit large kernel tensors and full
trajectory/prediction tables. The two exact dirty Python source snapshots are also
unavailable: the listed source digests are historical fingerprints, not archived
source bundles, and the later tree fails their source guard. Consequently, a clean
checkout can audit these outcomes and seals but cannot replay either archived
study; a regenerated and repinned preflight is a new study.

Phase `52/64`, causal-certificate `56/64`, and learnability `49/64` are
classifier-correctness counts, not prevalence; these endpoints were explicitly
secondary/rejected and cannot rescue the broad hypothesis. Crossing times were
systematically early (bias: tanh drift `-0.0593`, tanh response `-0.1380`, GRU
drift `-1.4330`, GRU response `-1.8199`), while tanh trajectory magnitudes were
poor (`2.455` response-gap RMSE and `2.489` weak-response RMSE). Thus the restricted
result is held-out event-classification evidence at one width/task cell, not causal
prediction, universal prevalence, kernel stability, or quantitatively accurate
time prediction.

### Nonlinear crossover

The final theorem-aligned artifact is
`paper/artifacts/enl_tanh_crossover-20260824-141207`. It uses common symmetric unit-probe
responses and the universal direct-autograd response drift at every logged point.

- tanh: exact drift crossings, response crossings, and causal tail-area
  certificates in **3/8** seeds; the remaining **5/8** are suppressed from
  initialization. Conditional on crossing, mean exact drift `tau*=0.1942` and mean
  response crossing `0.3890`.
- GRU: exact drift and response crossings in **5/8** seeds, but weak-only reaches
  neither the preregistered `beta=0.5` target nor causal learnability in any seed;
  therefore causal starvation is certified in **0/8**.
- A post-hoc terminal-response profile derived from the tracked per-seed summary,
  without training, gives tanh `B_W(H)` min/mean/max
  `4.41313266754 / 4.49664855003 / 4.67605876923` (8/8 above `0.5`) and GRU
  `0.0828229486942 / 0.129690139554 / 0.187142759562` (0/8). This shows that the
  GRU result is not marginal at the preregistered operating point; terminal reach
  is not generally a first-hit replacement and the audit supports no architecture
  ranking.
- maximum direct-versus-projected drift residual: `0.02197` for tanh and
  `0.002986` for GRU. Projected and matched-state quantities remain diagnostics.

The profile table, figure, and portable provenance record are stored under the
same artifact directory and can be rebuilt with
`python scripts/build_enl_learnability_profile.py`.

The old tanh 8/8 result and `tau*=1.6112` are invalid under the final response
definition. Crossing-time means above are conditional descriptive statistics and
must be reported with their denominators.

The post-hoc exploratory grid
`results/enl_exact_regime_search-20260824-105003` found no condition with 4/4
causal certificates. Its best cells reached 3/4, so it is a negative search audit,
not confirmatory evidence and not a basis for selecting a paper result.

### Width evidence

`results/e2_corrected_validation-*` is convergence toward a held-out finite-network
`closure_reference`, not DMFT validation. `results/e2r_solver_checks-20260823-084911`
validates only the exact zero-disorder and frozen-geometry special cases; its general
acceptance remains false by construction.

### CDC ablation

`results/e3_cdc_dense_ablation-20260823-090702`

- all shadow-based methods are numerically similar on weak rescue, but not
  statistically equivalent;
- CDC uniquely preserves instantaneous strong drift (rounding-scale residual);
- CDC does not have the best final strong response;
- the paper must disclose the weak-only shadow model and roughly doubled cost.

### Waterbirds

The adapter's CDC-style method is a minibatch head-coordinate surrogate outside
the full-batch matched-shadow theorem, and it has **not been run**. The weak
coordinate is an intercept, not an identified bird-shape feature. No Waterbirds
result belongs in the main result table until the pilot is run.

### Empirical scope

All citable outcomes are controlled synthetic experiments. E1 has eight seed IDs
reused over four strengths at each lag; E-NL has eight seeds per architecture at
one task cell. These counts are descriptive rather than event-prevalence estimates,
and no real-data experiment has been run.

## 4. Novelty boundary after reading the 2026 literature

Ger and Barak (2026) derive exact/asymptotic low-rank overlap learning equations and
show that loss-invisible overlaps can govern learning. This substantially narrows
any claim that projected geometry or hidden closure variables are new.

Clark et al. (2026) derive a task-trained RNN DMFT for a long-time
Langevin/Gibbs equilibrium and explicitly state that it does not track noiseless
optimization-time gradient flow. Their theory covers structure/disorder and learned
representations, but not this paired CE causal weak-feature trajectory.

Accordingly, the defensible novelty is the causal paired intervention, the general
cumulative drift balance and its single-crossing tail-area corollary, the ordering-
invariant rank-one drift criterion, and
the theorem-aligned recurrent experiments—not “the first theory of learning in
RNNs” and not a completed general recurrent DMFT. The nonlinear evidence is
seed-dependent and cannot support a claim of a universal recurrent crossover.

Sources: [Ger and Barak (2026)](https://arxiv.org/html/2605.04115v1) and
[Clark et al. (2026)](https://www.biorxiv.org/content/10.64898/2026.03.02.708943v1.full-text).
Content was rephrased for compliance with licensing restrictions.

## 5. Required paper language

Allowed:

> We prove exact finite-width signed-logit/response empirical-NTK flow. Freezing
> those full sample-level kernels gives an exact nonlinear logistic tangent
> surrogate, while comparison with trained tanh/GRU networks remains conditional on
> unproved kernel-movement bounds. After a broad held-out predictor failed on tanh
> learnability, a fresh restricted four-data-seed by eight-model-seed factorial at
> one controlled cell classified response-crossing events in 64/64 records and
> drift-crossing events in 60/64; the two-way-bootstrap lower bounds were 1.0 and
> 0.8125. Phase, causal certification, learnability, crossing times, and trajectory
> magnitudes were not validated by that restricted result.

> Separately, we prove the cumulative drift balance for arbitrary sign changes; in
> the single-crossing case, a rate crossover becomes outcome suppression exactly
> when its negative tail area exhausts the earlier transfer advantage. In a
> noiseless rank-one cue model, an ordering-invariant log drift ratio gives
> sufficient conditions for a unique rate crossover. Under the theorem-aligned
> nonlinear protocol, tanh causal crossover is seed-dependent (3/8), while GRU
> outcome crossings fail the causal learnability gate. General optimization-time
> DMFT and certified or quantitatively accurate nonlinear crossing-time prediction
> remain open.

Not allowed:

- “We prove that recurrent networks generally undergo transfer-to-starvation.”
- “The DMFT predicts `tau*` or `rho_c`.”
- “The broad empirical-NTK pilot passed.”
- “The frozen empirical-NTK surrogate is exact for trained tanh/GRU networks,” or
  “tanh/GRU kernels are stable.”
- “The restricted factorial predicts phase, causal starvation, learnability,
  calibrated crossing times, or universal event prevalence.”
- “Manifest verification proves the surrogate approximation.”
- “A drift crossing necessarily implies starvation.”
- “Late tanh suppression is geometry-dominated.”
- “CDC preserves the final strong feature or beats its ablations.”
- “Tests prove the theorem.”
