# Implementation plan: remaining theory, proofs, and experiments

Derived from `Gradient Starvation ICLR'27.docx` (the "note"), cross-checked
against `a.md`, `research_scope/e2_theorem.md`,
`research_scope/oral_theorem_package.md`, and the actual source tree on
2026-08-22.

Baseline at time of writing: `pytest -q` → `46 passed`, Python 3.12.4,
`.venv-test`. Working tree dirty as inventoried in `a.md` section 18.

> **AMENDED 2026-08-22 after author scoping.** Sections marked `(AMENDED)` reflect
> the approved execution scope: Phase 3 is scaffolding plus exact special cases
> only, with DMFT checks B / D-solver / E / F **blocked** behind tested
> `NotImplementedError` contracts; Phase 0 git work stops at a local commit;
> Phase 1 lockstep/sequential divergence is a stop condition; Phase 4 is not
> started; Phase 7 is skipped. See §0.4 for the full scoping record and the
> blocked-surface policy.
>
> **Further author decisions, same date:** the lockstep trainer stays opt-in with
> `sequential` as the default, and E-NL is its only consumer; no derivation
> skeleton is created (§3a); the parameter-snapshot aliasing correction found
> during Phase 1 is recorded as its own commit rather than folded into the
> baseline commit.

---

## 0. Three corrections to the note itself

These must be settled before the plan is executed, because two of them change
the note's stated motivation.

### 0.1 The note's motivating E2 diagnostic uses superseded numbers

Note section 1.1 says: "weak-only particle RMSE improves with width while
BOTH-feature RMSE worsens." That is true of
`results/e2_width_extension-20260821-134952`, which is a superseded
`low_rank_linear` run with the old input scaling:

| condition | N=128 | N=192 | N=256 |
|---|---:|---:|---:|
| both | 19.862 | 21.779 | 25.975 |
| weak_only | 0.132 | 0.079 | 0.075 |

It is **not** true of the corrected dense run
`results/e2_corrected_validation-20260821-224630`:

| condition | N=32 | N=64 | N=128 | N=256 |
|---|---:|---:|---:|---:|
| both (NRMSE) | 0.09084 | 0.04761 | 0.05149 | 0.02958 |
| weak_only (NRMSE) | 1.56931 | 0.38792 | 0.40201 | 0.20024 |

**Action.** Keep the requirement for an independent solver, but restate the
reason. The correct reason is *epistemic, not empirical*: the current reference
is a mean over trained finite networks fed into `integrate_projected_flow`, so
it cannot falsify the theory it is being compared to. Do not argue "the closure
is wrong because BOTH RMSE grows" — that evidence no longer exists after the
scaling and model-family fixes.

Secondary honesty point: corrected both-condition NRMSE decreases end-to-end
but is **non-monotone** (0.0476 at N=64 rises to 0.0515 at N=128). Report the
non-monotonicity; the acceptance JSON only checks end-to-end decrease plus a
negative log-log slope.

### 0.2 Two citations resolve, but remain unread (AMENDED)

Both references exist and are recorded with full identifiers:

- Ger & Barak, "Learning reveals invisible structure in low-rank RNNs",
  arXiv:2605.04115 — <https://arxiv.org/abs/2605.04115>
- Clark et al., "Structure, disorder, and dynamics in task-trained recurrent
  neural circuits", bioRxiv 2026.03.02.708943 —
  <https://www.biorxiv.org/content/10.64898/2026.03.02.708943v1>

Status: **unread — not usable for novelty or theorem-positioning claims until
read by the author.** They are load-bearing for the "loss-invisible overlaps"
argument (Ger & Barak) and for scoping the DMFT contribution (Clark et al.).
Titles alone do not license comparative-novelty prose. Both are logged with this
flag in `research_scope/claim_ledger.md`.

### 0.3 The note's 2-term decomposition already exists as a 3-term identity

The note wants `D_w = T_geom − S_CE`. `theory.py::matched_weak_drift_decomposition`
already proves and tests `F_w^W − F_w^B = ce_gating + geometry_shift + cross_transport`.
The exact mapping is:

```
D_w     = F_w^B − F_w^W = −causal_deficit
S_CE    =  ce_gating       =  G_ww^B [ g_w(0,m_w) − g_w(m_s,m_w) ]
T_geom  = −geometry_shift − cross_transport
        =  (G_ww^B − G_ww^W) g_w(0,m_w)  +  G_ws^B g_s(m_s,m_w)
```

So Corollary A does not need a new identity. It needs (a) this reparameterization,
(b) per-step logging, which does not exist today, and (c) sufficient conditions
for the sign flip, which is the actual open work.

**Matching convention — decided.** `matched_weak_drift_decomposition` evaluates
the weak-only *CE field* at the both-condition `m_w`, while taking the weak-only
*geometry* at the weak-only parameter state. Lockstep training matches the two
models on `τ`, not on `m_w`. These are different matchings and yield different
numbers. The project adopts the **`m_w`-matched-field convention**, because it is
the one already implemented and covered by `tests/test_theory.py`. Corollary A
must state this convention explicitly. A dedicated test documents the numerical
difference between the two matchings so they are never conflated.

### 0.4 Approved execution scoping (AMENDED)

The user scoped this implementation pass as follows, and these constraints
override any broader ambition stated later in this document:

1. **Phase 3 is scaffolding plus exact special cases only.** Build the
   `DMFTSpec`/`DMFTSolution`/`solve_dmft` interface, the independence guard,
   determinism, check A (zero disorder) and check C (frozen geometry), and the
   MSE *objective* infrastructure. Checks **B**, **D-solver**, **E**, and **F**
   are **blocked**, not deferred — see §3c as amended.
2. **Phase 0 git scope is branch plus local commit only.** No `git push`, no pull
   request. Report the commit SHA and leave pushing to the author.
3. **Phase 1 divergence is a stop condition.** If lockstep ERM does not reproduce
   sequential ERM trajectories, investigate deterministic causes (initialization
   order, optimizer construction, RNG consumption order, diagnostic side
   effects), then stop and report. Do not hide it, do not silently change the
   default path, do not loosen the tolerance to force a pass.
4. **No placeholder DMFT solver may be implemented or accepted as scientific
   validation.**

#### Blocked-surface policy

Any function whose correctness depends on the undelivered closure derivation
must:

1. raise `NotImplementedError` naming the specific unmet proof obligation by
   number from `research_scope/e2_theorem.md` § "Proof obligations" — that file is
   the single home of the obligation list, since §3a declines to create a
   placeholder derivation document;
2. have a test asserting the raise (`pytest.raises(NotImplementedError)`) — not
   `skip`, not `xfail`, so that a later placeholder insertion fails loudly;
3. never return a plausible-looking numeric array, zeros, or a stub trajectory;
4. never be wired into an acceptance JSON as a passing field. Where an acceptance
   record mentions a blocked check, the value is the literal string `"blocked"`
   and the record's top-level `passed` is `false`.

---

## 1. Gap table: note requirement → repository state

| Note item | Repo state | Gap |
|---|---|---|
| P1–P5 proved lemmas | implemented, tested (`theory.py`, `tests/test_theory.py`) | write-up only |
| **Theorem A** joint CE-RNN DMFT | `integrate_projected_flow` integrates `dm=Gg` with `G(τ)` **measured from trained finite nets** (`experiments.py::run_e2`, `_mean_closure`) | no independent solver; `C`, `R`, cross-condition overlaps absent |
| **Corollary A** crossover | identity exists but `matched_weak_drift_decomposition` is called **only from a unit test** — never from `training.py` or `experiments.py` | no `T_geom`/`S_CE`/`D_w` trajectory columns, no `τ*` |
| **Theorem B** learnability + boundary | `_paired_summary` emits `{delayed_weak, jointly_learned, strongly_starved, indeterminate}` | does not match the note's 4 semantic regions; no `L` gate, no `δ` band, no `ρ_c` |
| **Theorem C** CDC R1, R2 | implemented and logged 100% (`counterfactual_drift_correction`, `_train_paired_counterfactual_drift`) | R3 (`O(η²)`) never measured |
| E-CDC fair baselines | `losses.py` supports `erm`, `spectral_decoupling`, `interaction`, `counterfactual_drift` | **no Bloop, no PCGrad**, no ablation arms |
| E2-R limits A–F | none | new solver + harness; **MSE objective not implemented** (limit D blocked) |
| E1-H held-out test | `run_e1` already accepts `task.theory_boundary_file` and forwards it to `plot_e1` | reuse this hook; need frozen boundary CSV, held-out config, 4-region figure |
| E-NL crossover | tanh long-horizon runs exist | no mechanistic columns; `τ*_pred` vs `τ*_obs` impossible today |
| E-REAL Waterbirds | adapter + ERM/SD/interaction | no CDC oracle variant, no label-free variant, no data setup |
| K-feature appendix | `synthetic.py`, `gradient_gram`, `projected_statistics` hardcode 2 modes | needs K-generalization; appendix scope |

Engineering debt carried from `a.md`: untracked `.DS_Store`, no commit/branch,
no clean Python 3.10 run, CI never observed remotely, no `dense_linear + CDC`
integration test.

---

## 2. Phase plan

Ordering rule: cheap instrumentation that unblocks a *measurement* comes before
expensive derivation, but no experiment that depends on a frozen theoretical
prediction runs before that prediction is derived and frozen.

### Phase 0 — Freeze the baseline (hygiene)

1. `git rm --cached`-free approach: add `.DS_Store` to `.gitignore`; delete the
   single file. No broad `git clean`.
2. Review the full dirty diff, then branch `codex/corrected-dense-baseline`,
   stage specific paths (never `git add .`), and commit **locally**.
   **AMENDED: no `git push`, no pull request.** Report the SHA; pushing and
   authorizing remote CI are the author's decisions.
3. Verify a clean Python 3.10 environment locally with a throwaway venv,
   `requirements.txt`, and `pytest -q`. `a.md` §13 flags 3.10 as never
   independently verified while `.github/workflows/tests.yml` claims to test it.
   Fix real incompatibilities. If Python 3.10 is unavailable on the machine, say
   so explicitly rather than assuming it works.
4. Add the missing `dense_linear + CDC` integration test.

Acceptance: local commit exists; `pytest -q` green on 3.12 and on 3.10 (or 3.10
explicitly reported unavailable); `git status --short` shows only intended files.

Gate: everything after this assumes a committed, reproducible baseline.

---

### Phase 1 — Crossover instrumentation (unblocks Corollary A and E-NL)

This is the highest value-per-hour item in the plan. The strongest empirical
result in the project (tanh transfer→starvation) currently has no mechanistic
data behind it.

**Design decision that must be made explicitly.** The existing non-CDC
`train_paired` trains both-feature **then** weak-only sequentially, so there is
no weak-only model available at each logged `τ` of the both run. Two options:

- (a) add a lockstep paired trainer (mirror the structure already used by
  `_train_paired_counterfactual_drift`, which does step both models together);
- (b) snapshot parameters and post-process.

Choose (a). It reuses a proven code path and keeps memory bounded.

**Matching convention caveat.** `matched_weak_drift_decomposition` evaluates the
weak-only *CE field* at the both-condition `m_w`, while using weak-only
*geometry* at the weak-only parameter state. Lockstep training matches on `τ`,
not on `m_w`. These are different matchings and give different numbers.
Corollary A must state which one it uses, and the code must not mix them
silently.

Work items:

1. `theory.py`: add `CrossoverDecomposition` + `crossover_decomposition(both, weak_only, z_w)`
   returning `d_w`, `t_geom`, `s_ce`, `reconstruction_error`, implemented as the
   sign mapping in §0.3 on top of the existing tested identity.
2. `training.py`: `train_paired_lockstep(...)` behind `training.paired_mode: lockstep`.
   New logged columns: `d_w`, `t_geom`, `s_ce`, `decomposition_reconstruction_error`,
   and the both-condition `G_ws` (currently only `G_sw` is logged; they are equal
   by symmetry of `gradient_gram`, so either alias it or log once and document).
3. `metrics.py`: `sign_crossing_time(times, values)` — first `+ → −` crossing by
   linear interpolation — plus `n_sign_changes`. Return `inf`/`nan` distinctly for
   "never crosses" vs "no data".
4. `experiments.py`: new `run_enl(config)` + CLI `enl`, writing
   `tau_star_observed`, `tau_star_predicted` (from `T_geom = S_CE`),
   `abs_tau_star_error`, per seed and per point.
5. `configs/enl_smoke.yaml`, `configs/enl_tanh_crossover.yaml` (tanh, the known
   `rho=4, lag=2` regime, seeds disjoint from the frozen validation seeds 6–17,
   e.g. 20–27).
6. GRU arm in the same config as the note's mechanistic contrast — record whether
   `S_CE` ever overtakes `T_geom`.

Acceptance:
- `reconstruction_error` below the tolerance already used in `tests/test_theory.py`
  (`rtol=2e-5, atol=2e-6`) at every logged step;
- `τ*_obs` recovered on the known tanh regime near the `a.md` figure of ≈2.15;
- `|τ*_pred − τ*_obs|` reported with per-seed spread — **no target value is
  promised in advance**;
- GRU result reported whatever it shows, including "never crosses".

Honest label: this is empirical support for Corollary A. It is not a proof of
the sufficient conditions, which remain open.

---

### Phase 2 — Learnability-gated causal classification (unblocks Theorem B figure)

1. `metrics.py`: extend `causal_metrics` (or add `classify_causal_regime`) with
   `beta`, `tau_max`, `delta`, returning `weak_only_learnable`
   (`T_w^0 ≤ tau_max`) and `regime ∈ {unlearnable, transfer, neutral, starvation}`.
   `delta_tw` stays `nan` outside `L`.
2. `experiments.py::_paired_summary`: emit the new `regime` column; keep the old
   value as `legacy_phase` so existing CSVs remain interpretable.
3. `plotting.py`: `plot_e1_regions` — categorical 4-region map, boundary overlay,
   held-out points with uncertainty bars. Keep the existing censoring dagger
   convention; do not plot missing cells as zero.
4. New config keys: `task.beta`, `task.tau_max`, `task.delta_tw_tolerance`.

Acceptance (verifiable against numbers already on disk): reprocess
`results/e1_corrected_validation-20260821-144112/summary.csv` and confirm
- lag 8 → `unlearnable` (weak-only final response was `0.001643`),
- positive cells at lag 0/2/4 → `starvation` for every seed,
- the matched negative control (mean gap `−0.750387`) → `transfer` or `neutral`,
  never `starvation`.

If any of those three fail, the gate is mis-specified — fix before Phase 4.

---

### Phase 3 — Theorem A: scaffolding and exact special cases only (E2-R)

The note is right that this is the central open item. Split it cleanly: the
derivation is mathematics and produces a document; the solver is code and must
be *derived from* that document. This pass writes neither the derivation nor a
placeholder standing in for it — only the scaffolding, the two special cases that
validate against machinery already exact in this repository, and tested blocks
everywhere else.

#### 3a. No derivation document this pass (AMENDED TWICE)

**`research_scope/dmft_derivation.md` is deliberately not created.** An empty
skeleton of numbered headings adds a file that looks like progress while
containing none, and it invites future edits that fill in plausible mathematics
without proof. The document should be created by the author at the moment real
derivation work begins.

The five proof obligations therefore continue to live in exactly one place,
`research_scope/e2_theorem.md` § "Proof obligations":

1. effective single-site process and its covariance/response kernels;
2. closure set, including loss-invisible order parameters;
3. finite-horizon existence and uniqueness of the closed Volterra/ODE system;
4. joint quenched concentration for both causal conditions under shared disorder;
5. hitting-time corollary under transversality, plus discretization error.

**Consequence for the blocked-surface policy:** every `NotImplementedError`
message cites `research_scope/e2_theorem.md` and its obligation number, not a
derivation file that does not exist. Sequence time `k` and optimization time `τ`
must remain distinct in every kernel the derivation eventually introduces.

**No mathematical content is written by the implementation pass.** The
generating functional, closure enumeration, and concentration proof are author
work. An assistant may later help draft them, but drafted mathematics is not a
proof until the author checks it.

#### 3b. Solver → `src/gradient_starvation/dmft.py`

- `DMFTSpec`: `sequence_length`, `tau_max`, `dtau`, `bulk_gain`, `rho`,
  `lag_separation`, `cue_noise`, `quadrature_size`, `tolerance`,
  `max_iterations`.
- `DMFTSolution`: `tau`, `m_s`, `m_w`, `G`, plus `C`/`R`/`residual_history`
  fields that are present but **unpopulated** in the special-case paths.
- `solve_dmft(spec)` dispatches:
  - `bulk_gain == 0` → check A path, implemented;
  - `frozen_geometry` supplied → check C path, implemented;
  - otherwise → `NotImplementedError` citing `e2_theorem.md` obligations 1 and 2.
- **Independence guard, enforced by test:** the module imports nothing from
  `training.py`, `experiments.py`, or `width_validation.py`, and reads no file
  under `results/`. Tested by AST-inspecting the import graph, not by convention.
  This is the property that would let the solver falsify the theory.
- Determinism: identical `DMFTSpec` → bitwise-identical output. Tested.

#### 3c. Limit checks: only A and C are implementable (AMENDED)

| Check | Status | Reference / rationale |
|---|---|---|
| A zero disorder `g=0` | **implemented** | validates against `exact_dense_linear_geometry` + autograd, which `tests/test_theory.py` already trusts; target relative error ≤1e-6 |
| C frozen geometry | **implemented** | validates against `integrate_projected_flow` with constant `G`; isolates CE margin gating from representation adaptation |
| D MSE *objective* | **implemented** | `losses.py` gains a real, testable MSE objective with its own tests |
| B weak-only reduction | **BLOCKED** | reducing the joint system to a weak-only DMFT itself requires the `C`/`R` closure kernels. There is nothing to reduce *to*. Imposing `g=0` as well collapses it into check A, so it would not be an independent test either way. |
| D MSE *solver* branch | **BLOCKED** | needs the closure equations before an MSE variant of them can exist |
| E internal convergence | **BLOCKED** | self-consistency residuals are only defined relative to *derived* self-consistency equations. A residual that "converges" for a non-derived system is a false pass. |
| F finite-width | **BLOCKED** | requires a working general solver to freeze a prediction from |

Each blocked check raises `NotImplementedError` naming its unmet obligation, and
a test asserts that raise. This is the amended scope: blocked, not deferred, and
never silently stubbed.

#### 3d. Wiring

- `run_e2r(config)` in `experiments.py`, CLI choice `e2r` in `run_experiment.py`.
- `configs/e2r_solver_checks.yaml` (A and C only, CPU-cheap).
- **AMENDED: do not create `configs/e2r_finite_width.yaml`** — check F is blocked.
- `e2r_acceptance.json`: `check_a` and `check_c` carry real values;
  `check_b`, `check_d_solver`, `check_e`, `check_f` carry the literal string
  `"blocked"`; top-level `"passed": false` with a `"blocked_on"` field listing the
  unmet obligations. **An acceptance record that cannot pass is the correct output
  for this pass.**
- Rename the particle path to `closure_reference` in all **new** outputs and label
  it a numerical approximation per note §3.1. Do not rewrite existing run CSVs.
  Keep it as a diagnostic; stop presenting it as theory.

**Forbidden in Phase 3:** any code path returning numbers for the general
closure; any acceptance field reporting a blocked check as passing; any use of the
phrase "DMFT validation" in code, docstrings, output, or the README for work done
in this phase.

---

### Phase 4 — BLOCKED, not started this pass (AMENDED)

Boundary derivation and the E1-H held-out test depend on a working general solver
from §3b, which does not exist. Therefore this pass does **not** create
`src/gradient_starvation/boundary.py`, does **not** create
`configs/e1_heldout_boundary.yaml`, and does **not** generate a frozen boundary
CSV.

Ready for reuse when Phase 4 unblocks: `run_e1` already reads
`task.theory_boundary_file` and forwards it to `plot_e1`, so the overlay hook
needs no new plumbing. `plot_e1_regions` from Phase 2 accepts the same boundary
frame.

When unblocked, the work is:

1. `boundary.py` with `learnability_region(...)` and
   `rho_critical(lag, bulk_gain, delta, tau_max)` by bisection on the solver's
   `ΔT_w`, evaluated **only inside** `L`.
2. Freeze as `research_scope/frozen_predictions/boundary_<hash>.csv`, record hash.
3. `configs/e1_heldout_boundary.yaml` — a *small* set of unrun points below / near
   / above the boundary, unseen seeds (e.g. 200–207). Not another broad grid.

**The one Phase-4-adjacent item in scope now:** rerun the corrected E1 grid under
the final width-scaled dense parameterization, since `a.md` §10 flags the existing
grid as predating that fix. New timestamped directory. Report whether the
conclusions hold.

---

### Phase 5 — Theorem C completion and fair CDC evaluation

**Theory.** `research_scope/cdc_theorem.md`: R1 and R2 are already algebraic —
write them up with the projector and the strictly concave QP / Lagrange
argument. R3 needs an explicit `L`-smoothness Taylor bound with named constants
and an update-norm assumption.

**Empirical R3.** Sweep `η`, measure one-step strong-response deviation between
CDC and ERM, fit log-log slope; expect ≈2. Report the fitted slope, not a claim.

**Baselines (currently absent).** Add to `losses.py` / `training.py`:
- `bloop` — EMA-smoothed auxiliary-gradient projection against `∇L_train`;
- `pcgrad` — secondary;
- `unconstrained_rescue` — same rescue direction, no `∇m_s` constraint (ablation ii);
- `loss_gradient_projection` — generic projection against `∇L_train` (ablation iii).

Bloop is the critical comparison. The note's distinctness argument stands or
falls on CDC constraining `∇m_s` (a theory-defined feature response) rather than
`∇L_train`. The repo already does the former — that is the defensible claim, and
ablation (iii) is what demonstrates it.

**Metrics.** Causal gap (hitting time and AUC), weak response, instantaneous
strong-drift error, **final strong response reported separately**, GSI-5,
wall time (already logged), peak memory, and sensitivity to `max_alpha` /
shadow-update frequency.

**Information-setting disclosure.** CDC uses a full-batch weak-only shadow model.
State that cost and supervision requirement in every table where CDC appears.

**Rerun.** `configs/e3_cdc_validation_dense.yaml` — the linear CDC result in
`a.md` §11.1 predates the input-scaling fix. The nonlinear tanh results are
unaffected and stand.

Acceptance: CDC beats Bloop and both ablations on the causal weak gap at equal
information setting, with paired CIs; or the result is reported honestly as not
beating them.

---

### Phase 6 — E-REAL: Waterbirds, code only (AMENDED)

**Implement, do not run.** No dataset download, no pretrained-asset download.
Report what setup the author must perform.

1. Setup requirements documented, kept outside mandatory CI (the existing
   actionable preflight test stays and is extended to the new methods).
2. `waterbirds.py`: two CDC variants, named to prevent confusion —
   - `counterfactual_drift_oracle`: builds the weak-only shadow using group
     labels. Label it an oracle mechanistic transfer test everywhere.
   - `counterfactual_drift_modal`: rescue direction estimated from model-internal
     feature modes, no group labels. This is the group-agnostic-comparable arm.
3. Report average accuracy, worst-group accuracy, per-group CIs, GSI-5, compute
   cost, sensitivity.

Acceptance: both variants reported side by side with ERM / SD / interaction. The
oracle variant is never placed in the same comparison column as group-agnostic
robustness methods.

---

### Phase 7 — SKIPPED this pass (AMENDED)

Cut by the approved scoping. Retained here as a future item.

Blocked on real generalization work — `SyntheticTaskSpec` and `synthetic.py`
hardcode 2 channels, `gradient_gram` hardcodes `range(2)` and a `2×2` output,
`projected_statistics` stacks exactly two modes, and `_linear_probe_responses`
builds exactly two probes.

Scope: one dominant feature plus 3–4 weaker ones with controlled strengths and
lags; test whether pairwise/low-rank interaction statistics predict which modes
starve, and whether CDC rescues several without collapsing the dominant response.
Mechanical but non-trivial. Appendix only, unless it shows qualitatively new
behavior. Cut this first if time is short.

---

### Phase 8 — Claim discipline

1. `research_scope/claim_ledger.md`: every paper-bound statement tagged
   **proved**, **target/conjectured**, or **empirical**, with the file or
   `results/` run directory that backs it. This is the single artifact that
   prevents the failure mode `a.md` §20 warns about. It records the two resolved
   citations with the `unread` flag from §0.2.
2. Frozen definitions per note §3.1: causal starvation requires weak-only
   learnability; `k` vs `τ` distinct; GSI-5 is a diagnostic, not the definition of
   starvation; no starvation label outside `L`.
3. Update `README.md`, `research_scope/e2_theorem.md`, and
   `research_scope/oral_theorem_package.md` to reflect what this pass delivered,
   including the §0.1 correction.

**No comparative-novelty prose is written this pass**, because the two 2026
references are unread (§0.2).

---

## 3. Dependency graph (as amended)

```
Phase 0 ──┬─> Phase 1 (crossover instrumentation) ──> E-NL results
          ├─> Phase 2 (learnability gate) ──> plot_e1_regions
          ├─> Phase 3a (skeleton) ──> 3b (scaffolding + checks A, C)
          │                              └─> [BLOCKED: B, D-solver, E, F]
          │                                     └─> [BLOCKED: Phase 4]
          ├─> Phase 5 (CDC theory + baselines)  ── solver-independent
          └─> Phase 4-adjacent: corrected-E1 rerun under dense scaling

Phase 5 ──> Phase 6 (Waterbirds, code only)
Phase 7 SKIPPED
Phase 8 closes last
```

Phase 5 is deliberately placed off the solver critical path: R1/R2 are algebraic
and the baselines are ordinary optimizer work, so none of it waits on Theorem A.
Phase 4 cannot start before a general solver exists.

---

## 4. What must not be claimed at any point

Carried forward from `a.md` §20, with additions from this plan:

1. A passing test suite proves software invariants, not the theorem.
2. `e2_acceptance.json` passing is convergence to a **held-out empirical
   closure**. Only a *future* `e2r_acceptance.json` carrying a frozen independent
   solution would support the phrase "DMFT validation". This pass deliberately
   produces a **non-passing** `e2r_acceptance.json`.
3. A positive AUC gap outside the learnable region `L` is indeterminate.
4. CDC preserves *instantaneous first-order strong-response drift*. It does not
   preserve the final strong response — the tanh validation shows ERM `1.10748`
   vs CDC `0.846715`.
5. Superseded runs (`e2_medium_validation`, `e2_high_confidence`,
   `e2_width_extension`, the 13:xx E1/E3 batch) stay out of quantitative claims.
6. The GRU negative result is reported, not hidden.
7. Corollary A's crossover conditions are open until §3a step 4 and the
   sufficient-condition derivation are written. Phase 1 produces evidence only.
8. No statement about acceptance outcome.

---

## 5. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Closure fails with `C`, `R`, `G` alone | author derivation stalls | expected per Ger & Barak (unread); add loss-invisible overlaps, do not force minimal closure |
| Pressure to stub the blocked solver to make an acceptance record pass | fabricated validation | blocked-surface policy §0.4: tested `NotImplementedError`, `"blocked"` literals, `passed: false` |
| `N=2048` infeasible locally | claimed scale drops | cluster, or reduce the claim in writing — never silently |
| Lockstep trainer changes measured trajectories vs sequential | Phase 1 numbers not comparable to `a.md` | tested equivalence assertion; divergence is a **stop condition**, investigated for deterministic causes (init order, optimizer construction, RNG consumption, diagnostic side effects) and reported, never papered over |
| `τ*` matching convention ambiguity | Corollary A numbers not reproducible | resolved in §0.3: `m_w`-matched-field convention, documented and tested against the alternative |
| Bloop beats CDC | novelty claim weakens | report it; fall back to the `∇m_s`-vs-`∇L_train` distinction, which ablation (iii) isolates |
| Waterbirds oracle mistaken for group-agnostic | reviewer credibility | naming plus explicit column separation |
| Python 3.10 clean env breaks | CI never green | Phase 0 gate, before any science |

---

## 6. Execution order for this pass (as amended)

1. **Phase 0** — `.DS_Store`, informed diff review, `dense_linear + CDC` test,
   branch, **local commit only**, Python 3.10 verification.
2. **Phase 1** — `crossover_decomposition`, lockstep trainer + equivalence test
   (stop condition), logged columns, `sign_crossing_time`, `run_enl`, configs,
   then execute the tanh crossover run. Largest scientific return per hour: it
   turns the strongest existing observation into a mechanistic measurement.
3. **Phase 2** — learnability gate, then the replay acceptance against
   `results/e1_corrected_validation-20260821-144112/summary.csv`. Three
   known-answer checks; any failure is a stop condition.
4. **Phase 3** — solver scaffolding, checks A and C, MSE objective, tested blocks
   for B / D-solver / E / F, non-passing `e2r_acceptance.json`. No derivation
   document; obligations stay in `e2_theorem.md`.
5. **Phase 4-adjacent only** — corrected-E1 rerun under dense scaling.
6. **Phase 5** — CDC write-up (R1/R2 proved, R3 as target), empirical R3 slope,
   Bloop / PCGrad / two ablations, dense-scaling CDC rerun.
7. **Phase 6** — Waterbirds CDC variants, code only.
8. **Phase 8** — claim ledger, frozen definitions, doc updates.

Stop conditions: Phase 1 divergence surviving investigation; Phase 2 replay
failure; any temptation to implement B / E / F or the general closure to make an
acceptance record pass; a Python 3.10 incompatibility needing a dependency
change; any request requiring a push, PR, or dataset download.
