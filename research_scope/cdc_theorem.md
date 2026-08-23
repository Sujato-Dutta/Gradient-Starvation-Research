# Counterfactual Drift Correction: what is proved, and what is not

Status labels are used strictly. **Proved** means the argument is complete and
elementary. **Target** means the statement is believed and partially measured but
the proof is not written. Nothing here is a claim about the joint mean-field
theory, which remains blocked; see `e2_theorem.md` § "Proof obligations".

Implementation: `theory.counterfactual_drift_correction`, driven by
`training._train_paired_counterfactual_drift`.

---

## Setting

Let `theta` be the trainable parameters of the both-feature model and let
`L(theta)` be the full-batch cross-entropy. Write the two feature responses as
`m_s(theta)` and `m_w(theta)`, with gradients

```
a = grad m_s(theta),        p = grad m_w(theta).
```

The ERM parameter velocity is `v = -grad L(theta)`. Instantaneous mode drifts are
the directional derivatives along the velocity,

```
F_s = a . v,        F_w = p . v.
```

A matched weak-only shadow model, trained in lockstep from the same
initialization on the strong-ablated batch, supplies the *counterfactual* weak
drift `F_w^W`. The correction is restricted to the component of `p` orthogonal to
`a`:

```
q = p - (p . a / ||a||^2) a,        so   a . q = 0,   q != 0 unless p || a.
```

The applied update is `v' = v + alpha q` with

```
alpha = relu(F_w^W - F_w) / ||q||^2,
```

optionally capped at `max_alpha`. Assumptions used throughout: full-batch
gradients, zero weight decay, no gradient clipping. The trainer rejects
configurations violating these, because each would perturb the update after the
correction is computed and so invalidate the algebra.

---

## Result 1 — first-order strong-drift preservation. **Proved.**

**Claim.** `a . v' = a . v`, exactly, for any `alpha`.

**Proof.** `a . v' = a . (v + alpha q) = a . v + alpha (a . q)`. By construction
`q` is the residual of `p` after projection onto `a`, so

```
a . q = a . p - (p . a / ||a||^2)(a . a) = a . p - p . a = 0,
```

hence `a . v' = a . v`. The identity is independent of `alpha`, so capping the
coefficient does not weaken it. □

**Scope.** This concerns the *instantaneous first-order* strong-response drift at
the current parameter state. It is not a statement about the strong response at
any later time. See Result 3 and the tradeoff note below.

**Measured.** Maximum absolute instantaneous strong-drift change across logged
steps: `7.63e-6` in the linear validation and `7.45e-8` over 808 logged checks in
the frozen tanh validation. These are consistent with float32 rounding on the
inner products rather than with a systematic effect.

---

## Result 2 — optimality of the correction. **Proved.**

**Claim.** Among all corrections `d` satisfying `a . d = 0` and
`p . (v + d) >= F_w^W`, the choice `d* = alpha q` with `alpha` as above is the
unique minimiser of `||d||`.

**Proof.** Feasibility requires `p . d >= F_w^W - F_w =: delta`. If `delta <= 0`
then `d = 0` is feasible and trivially minimal, and `alpha = 0` by the `relu`.
Assume `delta > 0`. Decompose `p = q + (p . a / ||a||^2) a`. For any `d` with
`a . d = 0`,

```
p . d = q . d,
```

because the `a`-component of `p` is annihilated by `d`. So the problem is

```
minimise ||d||   subject to   a . d = 0,   q . d >= delta.
```

By Cauchy-Schwarz on the subspace `{d : a . d = 0}`, which contains `q`,

```
delta <= q . d <= ||q|| ||d||   =>   ||d|| >= delta / ||q||,
```

with equality iff `d` is a non-negative multiple of `q`. Taking
`d = (delta / ||q||^2) q` attains the bound and satisfies both constraints, and
Cauchy-Schwarz equality is strict unless `d || q`, giving uniqueness. □

Equivalently: `d*` maximises the strictly concave objective
`q . d - ||d||^2 / (2 lambda)` subject to `a . d = 0`, whose stationarity condition
`q - d / lambda = mu a` combined with `a . d = 0` yields `d = lambda q` after
eliminating the multiplier `mu`. The two formulations agree with
`lambda = delta / ||q||^2`.

**Feasibility condition.** The construction requires `q != 0`, i.e. `grad m_w` not
parallel to `grad m_s`. The trainer logs `protected_norm_sq = ||q||^2` and a
`cdc_feasible` flag at every logged step; both validations report 100%
feasibility and 100% target attainment.

---

## Result 3 — finite-step deviation. **Target, not proved.**

**Intended statement.** Let `m_s` have `L`-Lipschitz gradient in a neighbourhood
containing the segment traversed by one step. Then a single gradient-descent step
of size `eta` satisfies

```
| m_s(theta + eta v') - m_s(theta + eta v) | <= (L / 2) eta^2 ( ||v'||^2 + ||v||^2 ) ,
```

so the one-step strong-response deviation between CDC and ERM is `O(eta^2)` with a
constant governed by `L` and the update norms.

**Why it is not yet proved.** The Taylor argument is routine, but a usable
statement needs (i) an explicit neighbourhood on which the Lipschitz constant is
valid, (ii) a bound on `||v'||` in terms of `||v||` and `alpha ||q||`, which
requires controlling `1/||q||^2` away from the feasibility boundary, and (iii) a
statement about accumulation over many steps, which is what would actually license
a trajectory-level claim. None of these is written.

**Measured.** `run_e3r` sweeps `eta` and fits the log-log slope of the one-step
strong-response deviation. A slope near 2 is consistent with the target; the fitted
value is reported in the run's `r3_slope.json`. This is an empirical scaling
measurement, not the bound.

---

## The tradeoff that must always be reported

CDC preserves the instantaneous first-order strong drift (Result 1, proved). It
does **not** preserve the final strong response, because the corrected update
follows a different trajectory. Measured in the frozen tanh validation: final
strong response `1.10748` under ERM against `0.846715` under CDC, a reduction of
about `0.26`. The linear validation showed a reduction of about `0.258`.

Any table reporting CDC must therefore carry the final strong response in its own
column. Reporting only the instantaneous guarantee would misrepresent the method.

---

## Information setting

CDC requires a matched weak-only shadow model trained in lockstep on
strong-ablated data, using full-batch gradients. That is a real supervision and
compute requirement:

- it needs the ability to construct the strong-ablated counterfactual, which on
  synthetic data is exact and on real data generally is not;
- it doubles the model memory and roughly doubles the per-step cost;
- it assumes full-batch gradients.

Every comparison involving CDC must state this. A baseline that does not use the
counterfactual is not operating in the same information setting, and the
difference must be disclosed rather than absorbed into a win.

---

## Distinctness from generic gradient surgery

PCGrad and Bloop already project gradients to reduce interference or to preserve a
primary objective at first order. The claim here is *not* that projection is new.
What differs is the object being constrained:

- generic surgery constrains a **loss** gradient, typically `grad L_train`;
- CDC constrains `grad m_s`, a **theory-defined feature response** identified by
  the starvation decomposition, and rescues along a **causal** direction derived
  from the matched weak-only counterfactual.

This distinction is empirical, not rhetorical, and is what the Phase 5 ablations
exist to test.

### Measured outcome of that test — CDC does not win on weak rescue

`results/e3_cdc_dense_ablation-20260823-090702`, dense_linear, width 96, `ρ = 4`,
lag 2, 8 unseen seeds (40–47), 500 steps.

| method | shadow | causal AUC gap | paired vs ERM, 95% CI | max abs instantaneous strong-drift change | final `m_s` |
|---|---|---:|---|---:|---:|
| ERM | no | 10.5155 | — | — | 1.364 |
| Spectral Decoupling | no | 9.0229 | −1.493 [−1.755, −1.230] | — | 0.829 |
| interaction | no | 10.5145 | −0.001 [−0.002, −0.000] | — | 1.365 |
| **counterfactual_drift (CDC)** | yes | −0.0005 | −10.516 [−11.373, −9.659] | **4.77e-07** | 1.042 |
| loss_gradient_projection | yes | −0.0002 | −10.516 [−11.372, −9.659] | 3.75e-01 | 0.371 |
| unconstrained_rescue | yes | −0.0007 | −10.516 [−11.373, −9.660] | 1.48e-01 | **1.295** |
| bloop | yes | 0.0002 | −10.515 [−11.372, −9.659] | 3.75e-01 | 0.370 |
| pcgrad | yes | −0.0007 | −10.516 [−11.373, −9.660] | 1.48e-01 | 1.295 |

All five shadow-based methods attain the shared target at 100% of logged steps and
are **numerically similar on the causal weak gap, agreeing to about four decimal
places**. Once every family reaches the same weak-drift target at every step the weak
trajectory is essentially pinned, so the constraint cannot change the weak outcome by
much. The constraint governs *collateral damage*, not the rescue.

**They are not statistically indistinguishable, and that phrase must not be used.**
Overlapping confidence intervals against a common baseline do not establish
equivalence. Direct pairwise tests on the same eight seeds find 9 of 10 comparisons
significant at 0.05, with differences that are tiny but real:

| pair | mean difference | 95% CI | p |
|---|---:|---|---:|
| CDC − loss_gradient_projection | −0.000350 | `[−0.000383, −0.000317]` | 4.3e-08 |
| CDC − unconstrained_rescue | +0.000176 | `[+0.000158, +0.000194]` | 8.5e-08 |
| CDC − bloop | −0.000720 | `[−0.000811, −0.000630]` | 2.9e-07 |
| CDC − pcgrad | +0.000176 | `[+0.000158, +0.000194]` | 8.5e-08 |
| loss_gradient_projection − bloop | −0.000370 | `[−0.000443, −0.000297]` | 6.4e-06 |
| unconstrained_rescue − bloop | −0.000896 | `[−0.000982, −0.000811]` | 4.5e-08 |
| unconstrained_rescue − pcgrad | 0.000000 | `[0, 0]` | n/a |

`unconstrained_rescue` and `pcgrad` are *bitwise identical* on this run, because
PCGrad only projects when the rescue direction conflicts with the ERM velocity and no
conflict arose. That is a property of this regime, not of the methods.

The honest summary is therefore: the differences are statistically detectable and
practically negligible, roughly four orders of magnitude smaller than the
`≈ −10.5` effect each has against ERM. Claiming equivalence would require a
**preregistered equivalence margin** and a test against it, which has not been done.
Until then, say "numerically similar" and quote the table.

**What CDC uniquely delivers.** Exact instantaneous strong-drift preservation:
`4.77e-07`, consistent with float32 rounding, against `1.5e-01` to `3.8e-01` for
every alternative. Result 1 is real, unique to this constraint, and clearly
measurable. Projecting against `grad L_train` — the generic-surgery object —
perturbs the protected feature response by five to six orders of magnitude more.

**What CDC does not deliver, and this must be stated.** The instantaneous guarantee
does not translate into better *final* strong-feature retention. `unconstrained_rescue`
and `pcgrad` end at `m_s = 1.295`, **higher** than CDC's `1.042`, against ERM's
`1.364`. So in this regime the method with the exact first-order guarantee retains
*less* of the strong feature by the end of training than the variant with no
constraint at all. That is a direct empirical instance of the trajectory caveat
above, and it weakens the practical case for CDC as currently formulated.

**Consequences for positioning.** The novelty claim cannot rest on better weak
rescue, because there is none relative to Bloop or the ablations. It cannot rest on
better final strong-feature retention, because the unconstrained variant is better
on that axis here. What remains defensible is narrower: CDC is the unique member of
this family that preserves the instantaneous first-order response of a
theory-identified feature, exactly and by construction. Whether that property has
practical value is **not** demonstrated by this run, and a paper should not imply
that it is.

GSI-5 also moves the opposite way from the earlier tanh result: `0.247` under ERM
against `0.305` under CDC here, whereas the frozen tanh validation reported a
reduction from `0.124` to `0.032`. The two regimes differ; neither should be quoted
as the general behaviour.

Positioning against Ger & Barak (arXiv:2605.04115) and Clark et al.
(bioRxiv 2026.03.02.708943) is **not** written here: both are recorded as unread in
`claim_ledger.md`, and comparative-novelty prose based on titles alone would be
unfounded.
