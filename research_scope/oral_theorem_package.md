# Paper theorem package — current authoritative summary

This document is the paper-facing map of what is proved, what is measured, and
what remains open. `claim_ledger.md` is the statement-level authority;
`e2_theorem.md` and `cdc_theorem.md` contain the proofs.

## 1. Headline theorem package

The paper no longer presents a general paired CE-RNN DMFT as an established
result. Its rigorous contribution is a finite-width causal package.

### Theorem A: exact finite-width CE response flow

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

### Theorem B: when rate suppression becomes outcome starvation

For `Delta=m_w^B-m_w^W` and exact equal-time derivative `d=Delta'`, assume one
`+ -> -` drift crossing at `tau_d`. Then the response later reaches equality iff
the accumulated post-crossover negative area reaches the positive area stored at
the peak. Strict starvation requires strict excess negative area. This condition is
necessary and sufficient, and quantitative drift bounds yield explicit upper bounds
on the response crossing time.

This is the exact correction to the retracted statement that a drift crossing
necessarily causes a response crossing.

### Theorem C: ordering-invariant noiseless rank-one criterion

For deterministic cues `z^B=(rho,1)` and `z^W=(0,1)`, positive weak drifts factor
as `F_w^B=s_B H_B` and `F_w^W=s_W H_W`. The log ratio

```
Psi = log(F_w^B/F_w^W)
```

has an exact derivative separating total geometry-ratio growth from margin-gate
contraction without choosing either non-canonical additive decomposition. A positive
`Psi` followed by a uniform negative derivative produces one rate crossover and an
explicit time bound. Outcome starvation still requires Theorem B.

**Prediction discipline:** independently proved bounds on `Psi'` predict a
crossover. Evaluating `Psi` on a completed trajectory certifies it but is not an
independent prediction.

### Corollary D: hitting-time stability

Uniform trajectory convergence plus an isolated transverse target crossing implies
hitting-time convergence, with error at most trajectory error divided by crossing
slope, plus grid resolution. This licenses hitting-time limits only after a
trajectory limit has independently been proved.

### Theorem E: exactly solvable recurrent anchor

At zero recurrent disorder and zero cue lag, the dense linear dynamics close at
every width on six scalar inner products. Gaussian initialization obeys the explicit
Chebyshev bound `9/(N epsilon^2)`; finite-horizon ODE stability propagates this into
uniform convergence in probability to deterministic initial data. This is a genuine
large-width theorem at a singular solvable point, not a general DMFT.

### CDC Results 1–3

1. CDC preserves the current strong-feature drift exactly at first order.
2. The uncapped feasible correction is the unique minimum-norm correction attaining
   the weak-only drift target under that constraint.
3. Under local smoothness and a nondegenerate projected weak direction, one CDC step
   differs from the corresponding ERM strong response by an explicit `O(eta^2)`
   bound.

Result 3 is now proved as a local one-step theorem. No trajectory-level or final
strong-retention theorem follows.

## 2. Explicitly open theory

The following are not paper theorems:

- the positive-disorder/positive-lag joint CE-RNN optimization-time DMFT;
- an independently predicted tanh or GRU crossover time;
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

- positive weak AUC gap in 32/32 seeds at lags 0, 2, and 4;
- negative-control transfer in 8/8 seeds;
- lag 8 is unlearnable under the preregistered target and therefore not labelled
  starved;
- AUC gap is monotone in `rho` at every measured lag.

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
- maximum direct-versus-projected drift residual: `0.02197` for tanh and
  `0.002986` for GRU. Projected and matched-state quantities remain diagnostics.

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

The adapter is an unrun surrogate probe. The weak coordinate is an intercept, not
an identified bird-shape feature. No Waterbirds result belongs in the main result
table until the pilot is run.

## 4. Novelty boundary after reading the 2026 literature

Ger and Barak (2026) derive exact/asymptotic low-rank overlap learning equations and
show that loss-invisible overlaps can govern learning. This substantially narrows
any claim that projected geometry or hidden closure variables are new.

Clark et al. (2026) derive a task-trained RNN DMFT for a long-time
Langevin/Gibbs equilibrium and explicitly state that it does not track noiseless
optimization-time gradient flow. Their theory covers structure/disorder and learned
representations, but not this paired CE causal weak-feature trajectory.

Accordingly, the defensible novelty is the causal paired intervention, the exact
rate-versus-outcome theorem, the ordering-invariant rank-one drift criterion, and
the theorem-aligned recurrent experiments—not “the first theory of learning in
RNNs” and not a completed general recurrent DMFT. The nonlinear evidence is
seed-dependent and cannot support a claim of a universal recurrent crossover.

Sources: [Ger and Barak (2026)](https://arxiv.org/html/2605.04115v1) and
[Clark et al. (2026)](https://www.biorxiv.org/content/10.64898/2026.03.02.708943v1.full-text).
Content was rephrased for compliance with licensing restrictions.

## 5. Required paper language

Allowed:

> We prove exact finite-width response dynamics and show that a rate crossover
> becomes outcome starvation exactly when its negative tail area exhausts the
> earlier transfer advantage. In a noiseless rank-one cue model, an
> ordering-invariant log drift ratio gives sufficient conditions for a unique rate
> crossover. Under a theorem-aligned nonlinear protocol, tanh crossover is
> seed-dependent (3/8 causal certificates), while GRU outcome crossings fail the
> causal learnability gate; general optimization-time DMFT prediction remains open.

Not allowed:

- “We prove that recurrent networks generally undergo transfer-to-starvation.”
- “The DMFT predicts `tau*` or `rho_c`.”
- “A drift crossing necessarily implies starvation.”
- “Late tanh suppression is geometry-dominated.”
- “CDC preserves the final strong feature or beats its ablations.”
- “Tests prove the theorem.”

## 6. Readiness status

The theorem package and theorem-aligned nonlinear rerun are complete, including
figures. The rerun is tracked with its resolved configuration and a content hash
over all executable Python source. That makes the project scientifically
defensible, not automatically oral-ready. Historical E1, E2, E2-R, exploratory
E-NL, and E3 runs remain dirty and cannot be reconstructed exactly because they
predate content fingerprinting. Paper readiness also requires verification of the
external baseline implementations and external review. The corrected nonlinear
crossover is seed-dependent, which materially weakens an oral-level empirical
headline. Waterbirds is optional only if the paper is explicitly framed as a
controlled synthetic/theory paper; if it is included, the mandatory pilot must
precede any full run.
