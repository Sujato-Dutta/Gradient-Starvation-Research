# Causal Gradient Starvation in Recurrent Networks

This repository studies whether a dominant temporal feature causally slows the learning of a weaker predictive feature in recurrent models. The key comparison is paired: every both-feature run is matched to a weak-only run with the same weak signal, lag, noise, initialization, architecture, optimizer, and seed.

The codebase implements five experiment blocks:

- **E1 - causal phase diagram:** sweeps feature-strength ratio and temporal separation, measures weak-feature hitting-time delay, and includes a negative-control data regime. Classification is gated on weak-only learnability, so a point where the counterfactual never learns is reported as indeterminate rather than starved.
- **E2 - width convergence against an empirical reference:** checks the exact finite-width projected-flow identity, compares network trajectories with a held-out numerical closure reference, and measures error versus width. The reference is a mean over trained finite networks, not a solved theory.
- **E2-R - solver checks:** exercises the two mean-field special cases that are exact without the undelivered closure derivation, and records the remaining checks as blocked. Its acceptance record reports `passed: false` by construction.
- **E-NL - theorem-aligned crossover and frozen-kernel falsification:** uses common symmetric unit-probe responses for tanh/GRU, computes the universal direct-autograd response drift at every logged point, reports the projected `Gg` residual separately, and evaluates the exact finite-step tail-area certificate distinguishing rate suppression from outcome suppression; causal starvation additionally requires weak-only first-hit learnability. A separate `enl-preflight`/`enl-evaluate` path freezes the full signed-logit empirical NTK and response cross-kernel before training, then tests its crossing-event predictions against held-out nonlinear trajectories.
- **E3 - mitigation and transfer:** compares ERM, Spectral Decoupling, an interaction penalty, and a five-way ablation of constrained weak-rescue methods including Counterfactual Drift Correction plus Bloop-style and PCGrad-style baselines. The Waterbirds adapter contains a CDC-style minibatch head-coordinate surrogate outside the full-batch matched-shadow theorem, but it **has never been run**; see `research_scope/waterbirds_setup.md`.

### Read this before quoting any number

`research_scope/claim_ledger.md` tags every paper-bound statement as proved, target,
empirical, blocked or retracted, and names the artifact behind it. Several figures in
older notes are superseded. The established headline is now a finite-width causal
theorem package; the more ambitious joint mean-field extension remains a blocked
conjecture. The ledger exists so that checking which is which is faster than
rediscovering it. Five things worth knowing before reading anything else:

- The repository **proves** the exact finite-width CE cross-kernel flow, the
  general cumulative drift balance and its single-crossing tail-area corollary, a
  conditional noiseless rank-one rate-
  crossover theorem, and stability of quantitatively isolated transverse first
  hits under uniform-error, radius, and mesh hypotheses,
  the singular zero-disorder/zero-lag six-scalar limit, and three fixed-coordinate
  local CDC results. See `research_scope/e2_theorem.md` and
  `research_scope/cdc_theorem.md`.
- At full sample level, the exact finite-width signed-logit/response flow is
  `dot r=K(theta)sigma(-r)/n`, `dot M=C(theta)^T sigma(-r)/n`. Freezing `K,C` at
  initialization gives an exact nonlinear logistic tangent surrogate and exact
  explicit-Euler recursion for that surrogate. Its agreement with trained
  tanh/GRU networks is conditional: no required instantaneous/secant kernel-
  movement bound has been proved for either architecture.
- The **general positive-disorder/positive-lag joint CE-RNN optimization-time DMFT
  remains unproved**. `research_scope/e2_theorem.md` lists the five outstanding
  obligations; every solver surface that depends on them raises
  `NotImplementedError` naming the obligation, with a test asserting the raise.
- For tanh/GRU, `Gg` is a projected diagnostic rather than the exact derivative of
  the common probe response. E-NL therefore uses direct autograd drift as primary
  and reports the projection residual.
- **Counterfactual Drift Correction does not outperform its ablations** on the causal
  weak gap. What it uniquely provides is exact preservation of the instantaneous
  first-order strong-feature drift. See `research_scope/cdc_theorem.md`.

The older `gsi_pipeline/` and `legacy_docs/` directories are retained for provenance. New experiments live in `src/gradient_starvation/` and use explicit YAML configurations.

## Manuscript

The complete scientific source is `paper/main.tex`, with bibliography in
`paper/references.bib` and the generic compiled artifact at `paper/main.pdf`. The
official anonymous ICLR rendering is `paper/iclr2027/submission.pdf`; its wrapper
defines `\ICLRSubmission` and reuses `paper/main.tex`. In that mode, the compact
reviewer-facing synthesis is an additional claim-bearing surface before the shared
detailed theorem/result/proof package, so both surfaces must stay synchronized.
See `paper/iclr2027/review_resolution.md` for their cross-check and final build
record. Together they contain the finite-width proofs, executable-certificate map,
full synthetic protocol, corrected results, novelty boundary, and scientific
limitations.

## Setup

Python 3.12.4 is the canonical validated environment. Create a dedicated virtual environment and install from the content-addressed exact-version manifest:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.0
python -m pip install -r requirements-lock-py312.txt
python scripts/check_environment_lock.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. The unpinned
`requirements.txt` remains a Python 3.10 compatibility input, not the reproducible
claim-bearing environment.

For the optional, currently unrun Waterbirds experiment:

```bash
python -m pip install -r requirements-waterbirds.txt
```

The Waterbirds extras are not part of the validated lock and back no current
empirical claim. No project-package installation step is needed: the root entry
points add `src/` to the Python path.

## Quick validation

Run the unit and integration tests inside the dedicated environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run a small end-to-end experiment that is suitable for CPU validation:

```powershell
.\.venv\Scripts\python.exe run_experiment.py e1 --config configs/smoke.yaml
```

The test suite also runs compact E2 and E3 integrations from
`configs/e2_smoke.yaml` and `configs/e3_smoke.yaml`. They validate the complete
output contract (resolved configuration, environment record, trajectories,
summaries, aggregates, geometry check, and figures) without using the
compute-heavy research configurations. GitHub Actions runs this synthetic suite
on an unpinned Python 3.10 compatibility environment and the exact locked Python
3.12.4 environment for every push and pull request.

## Running the main experiments

Each command accepts repeatable `--set section.key=value` overrides. Results are written to a timestamped directory containing the resolved configuration, environment metadata, per-step trajectories, summaries, and figures.

### E1: causal phase diagram

```powershell
.\.venv\Scripts\python.exe run_experiment.py e1 --config configs/e1_phase.yaml
```

For the corrected, machine-conscious phase-grid validation used after the
pilot, run:

```powershell
python run_experiment.py e1 --config configs/e1_dense_rerun.yaml
```

It evaluates four strength values, four cue lags, eight independent seeds,
and a matched negative control. The E1 figure reports the finite AUC gap for
the complete grid and marks any cells with right-censored hitting times; a
censored cell is never displayed as zero delay.

The primary statistic is `delta_tw`: the difference between the both-feature and weak-only times required to reach the preregistered weak response `beta`. `weak_auc_gap` is the time-integrated difference between paired weak trajectories. Right-censored hitting times are recorded explicitly rather than silently converted to finite values.

### E2: projected dynamics and width convergence

```powershell
.\.venv\Scripts\python.exe run_experiment.py e2 --config configs/e2_field.yaml
```

E2 first verifies `dm/dtau = Gg` against direct autograd on the exact linear
scope. It then builds a held-out `closure_reference` from independent quenched
finite networks and evaluates held-out widths. This is empirical concentration
evidence, not an independent solver or a substitute for a concentration proof.

### E-NL: exact nonlinear crossover certificate

```bash
python run_experiment.py enl --config configs/enl_tanh_crossover.yaml
```

E-NL uses lockstep paired training solely so both causal conditions are available
at each logged optimization time. The common tanh/GRU feature responses are odd
symmetric unit-probe contrasts, hence shared initialization gives an exactly zero
initial weak-response gap. The primary drift is computed by direct autograd; the
projected `Gg` drift, matched-state decomposition, projection residual, response
crossing, and finite-step tail-area certificate are all retained under distinct
column names. A drift crossing alone is reported as rate suppression. Causal
starvation additionally requires an outcome crossing and weak-only learnability.

The authoritative completed run is
`paper/artifacts/enl_tanh_crossover-20260824-141207`: tanh has 3/8 exact drift crossings,
3/8 response crossings, and 3/8 causal certificates; its other 5/8 seeds are
suppressed from initialization. GRU has 5/8 exact drift and response crossings, but
weak-only never reaches `beta=0.5`, hence 0/8 causal certificates. The historical
tanh 8/8 result and `tau*=1.6112` used a superseded response convention and must
not be quoted.

A post-hoc threshold-sensitivity audit uses the tracked terminal weak-only response
`B_W(H)=M_w^W(H)` for each of those seeds. Tanh terminal min/mean/max are
`4.41313266754 / 4.49664855003 / 4.67605876923` (8/8 above `beta=0.5`), whereas
GRU min/mean/max are `0.0828229486942 / 0.129690139554 / 0.187142759562` (0/8),
so the GRU failure is not marginal at that operating point. Rebuild the audit
without training:

```bash
MPLBACKEND=Agg python scripts/build_enl_learnability_profile.py
```

The command reads only the tracked `summary.csv` and resolved config and writes
`learnability_profile.csv`, `learnability_profile_summary.csv`,
`enl_learnability_profile.pdf`, `enl_learnability_profile.png`, and
`learnability_profile_provenance.json` in the authoritative artifact directory.
This terminal-response profile is not generally equivalent to the preregistered
first-hit gate for a nonmonotone trajectory, is not a confirmatory endpoint, and
does not support an architecture ranking.

The post-hoc grid
`results/enl_exact_regime_search-20260824-105003` found no 4/4 condition and is
exploratory negative evidence only.

#### Frozen empirical-NTK preflight and held-out evaluation

The exact finite-width sample-level flow uses the signed-logit Jacobian `J_r`,
`K(theta)=J_r J_r^T`, and response cross-kernel
`C_i(theta)=<grad r_i,grad M_w>`:

```text
dot r = K(theta) sigma(-r)/n,
dot M_w = C(theta)^T sigma(-r)/n.
```

Freezing `K_0,C_0` gives an exact nonlinear logistic tangent-surrogate ODE; the
implemented recursion is its exact explicit-Euler discretization under
`tau=learning_rate*step`. This exactness is internal to the surrogate. Conditional
continuous/discrete comparison bounds depend on caller-supplied uniform movement
bounds for instantaneous or segment-secant `K,C`; the discrete evaluator is
`theory.frozen_kernel_discrete_error_bound`. No such tanh/GRU stability bound has
been proved, so the nonlinear comparison is empirical.

To regenerate either protocol under the current source tree, first run its
preflight:

```bash
python run_experiment.py enl-preflight --config configs/enl_ntk_pilot_preflight.yaml
python run_experiment.py enl-preflight --config configs/enl_ntk_crossing_factorial_preflight.yaml
```

The checked-in evaluate configs are historical evidence records: they pin the final
archived preflight directories and manifest hashes, whose dirty source fingerprints
do not match the later tree. They therefore fail closed if executed now. For a new
study, copy the matching evaluate config, repin `evaluation.preflight_dir` and
`evaluation.manifest_sha256` to the new preflight, then run `enl-evaluate` with that
local config. This is a new protocol run, not a bitwise reproduction of the archived
study.

`enl-preflight` never constructs an optimizer or observes a trained trajectory. It
writes `config.resolved.yaml`, `environment.json`, `predictions.csv`,
`prediction_summary.csv`, `prediction_aggregate.csv`,
`preflight_acceptance.json`, `manifest.json`, `manifest.sha256`, and
`kernels/*.pt`. Before training, `enl-evaluate` verifies the sidecar, actual and
externally pinned manifest hashes; accepted preflight contract; executable-source
fingerprint/count; exact task/model/training/protocol equality; safe unique paths;
all required-file and kernel hashes/sizes; prediction digest; complete configured
seed factorial; and exact step/`tau` grids. It consumes the hashed predictions CSV
rather than recomputing predictions and re-hashes every sealed input after training.
Evaluation writes `trajectories.csv`, `summary.csv`, `aggregate.csv`,
`crossover.csv`, `scores.csv`, `metrics.json`, `evaluation_acceptance.json`, and
`provenance.json`. These checks establish integrity and provenance, not the
scientific frozen-kernel approximation.

The broad 16-record pilot **failed**: classifier correctness was phase `12/16`,
drift `14/16`, response `14/16`, causal certificate `14/16`, and learnability
`12/16`, with tanh learnability only `4/8`. Its manifest, prediction, and source
hashes are respectively
`f186d587e28f6408b68baf95566d2d1f1c5d5701e205e9a69ce86cbe4f0ffaf5`,
`9787d99953f1046b4934aa6e7b3f969e30a4efafb3d02a67cc95926232f9dfaa`, and
`8d8162558e7d25fff7a5059a4cf49139a354ca211fd6f0759417e75dd87f435c`.

The revised fresh factorial crossed four unseen data seeds with eight unseen model
seeds for each architecture (`64` records) and froze only drift/response crossing
as primary. Response classification was `64/64`; drift was `60/64` (`32/32` tanh,
`28/32` GRU), with reuse-aware two-way-bootstrap 95% lower bounds `1.0` and
`0.8125`. Its manifest, prediction, and source hashes are respectively
`bdcf02abbef5b9d8f7ce3979d3363ee8d39ab7817a1d554616bc4dd4a22a0db1`,
`c05b31810cb3869e7d051f539741732a8758f6cac542deb448a98d2ada7b6542`, and
`ca878df376b01c0ea45ac352e353530bf20c567de59bb400617e5dd74d5b6b6b`.

Compact, checked-in copies of both studies are at
`paper/artifacts/enl_ntk_pilot-20260825` and
`paper/artifacts/enl_ntk_crossing_factorial-20260825`. Their archive manifests and
`paper/artifacts/provenance_manifest.json` record a SHA-256 digest and byte count
for each resolved config, sealed preflight manifest/acceptance record, evaluation
metrics/acceptance/provenance record, and `scores.csv`. The compact archives omit
`kernels/*.pt`, full trajectory/summary/crossover/aggregate/prediction tables, and
the exact dirty Python source snapshots. Those run-time files remain only in ignored
local `results/`; the source snapshots are unavailable. A clean checkout can audit
the compact outcomes and recorded seals, but it cannot replay either archived study.
Regenerating and repinning a preflight under current source creates a new study.

Phase `52/64`, causal certificate `56/64`, and learnability `49/64` were explicitly
secondary/rejected classifier endpoints, not event prevalence. Predicted crossing
times were systematically early, and tanh trajectory magnitudes were poor
(`2.455` response-gap RMSE; `2.489` weak-response RMSE). The allowed conclusion is
therefore narrow: the sealed frozen surrogate classified crossing events well on
this fresh restricted single-cell factorial after the broad predictor failed.
It does **not** establish causal prediction, universal prevalence, tanh/GRU kernel
stability, calibrated crossing times, or general recurrent behavior. Certified or
quantitatively accurate nonlinear crossing-time prediction remains open.

### E3: mitigation

```powershell
python run_experiment.py e3 --config configs/e3_cdc_dense_ablation.yaml
```

The interaction method penalizes the positive fixed-geometry starvation susceptibility estimated from `A_sw`, `G_sw`, and `G_ww`. `proxy` mode detaches the geometry term and is the practical default; `second_order` differentiates through the complete empirical geometry and costs substantially more memory.

The frozen nonlinear confirmation is `configs/e3_tanh_cdc_validation.yaml`.
For a machine-conscious width check, run
`configs/e3_tanh_cdc_width_validation.yaml` at widths 32, 64, and 96, then
combine matched-seed runs with:

```powershell
.\.venv\Scripts\python.exe summarize_width_validation.py --run 32=PATH_TO_WIDTH32 --run 64=PATH_TO_WIDTH64 --run 96=PATH_TO_WIDTH96 --output results/e3_tanh_cdc_width_validation
```

The summarizer rejects unmatched ERM/CDC seeds and duplicate run keys, then
writes paired effects, CDC diagnostics, and PNG/PDF width-robustness figures.

`counterfactual_drift` simultaneously trains a matched weak-only shadow model.
At each full-batch step it attempts the unique minimum Euclidean-norm correction,
in the fixed implemented tensor coordinates with the product Euclidean/Frobenius
metric, needed to match the shadow model's weak drift while projecting orthogonally
to the strong-mode gradient. Exact target attainment and fixed-coordinate
minimum-norm optimality require a feasible, uncapped direction; exact positive-
deficit feasibility is `q != 0`, while `feasibility_epsilon` is only a numerical
tolerance. The represented projection, norm, tolerance, and selected minimizer are
metric- and scale-dependent; no reparameterization-invariance or natural-gradient
claim is made. A nonbinding cap leaves the result unchanged, whereas a binding cap
generally loses the target and therefore has no target-attaining optimality claim.
Infeasible or capped steps remain explicitly logged. The instantaneous strong drift
is unchanged for every coefficient, including a cap. This method requires
cross-entropy diagnostics, zero weight decay, and no gradient clipping; the trainer
rejects configurations that would invalidate the contract. Its diagnostics
(`cdc_target_met`, `cdc_feasible`, `cdc_uncapped_alpha`, `cdc_alpha`,
`cdc_cap_binding`, `cdc_target_residual`, correction size, and strong-drift change)
are written to `trajectories.csv`.

A compact corrected-scaling comparison can be run with:

```powershell
.\.venv\Scripts\python.exe run_experiment.py e3 --config configs/e3_cdc_pilot.yaml
```

Use `configs/e3_cdc_dense_ablation.yaml` for the final eight-seed, width-96
controlled ablation. The older `e3_cdc_validation.yaml` predates the final dense
scaling and must not be quoted. The nonlinear search and validation configurations
remain secondary robustness studies:
`configs/e3_nonlinear_regime_search.yaml`,
`configs/e3_tanh_long_horizon_search.yaml`, and
`configs/e3_tanh_cdc_validation.yaml`. The selected tanh condition deliberately
uses a long horizon because it exhibits early transfer followed by late
starvation; short runs do not test the intended nonlinear phenomenon.

To run the optional real-data surrogate, first complete the mandatory sensitivity
pilot after installing the optional requirements:

```bash
python run_experiment.py waterbirds --config configs/waterbirds_pilot.yaml
```

Only after its learning-rate, weak-target, backbone-update, and modal-agreement
criteria pass should `configs/waterbirds.yaml` be run. Waterbirds remains **not
run** and supplies only a CDC-style minibatch head-coordinate surrogate outside the
full-batch matched-shadow theorem. Its constant weak coordinate is an intercept,
not an identified bird-shape feature.

Waterbirds is intentionally excluded from the mandatory synthetic test suite:
it requires the optional dependencies, dataset terms acceptance, local storage
through `data.root`, and (when enabled) pretrained model assets. The code raises
an actionable error when its optional dependencies are absent.

## Outputs and reproducibility

Every output directory contains:

- `config.resolved.yaml`: the exact configuration after command-line overrides;
- `environment.json`: Python, PyTorch, device, platform, and source-state metadata;
  if it records `git_dirty: true` without `source_sha256`, the commit hash alone
  does not reconstruct the executed source. New runs record a fingerprint over
  `run_experiment.py` plus `src/**/*.py`, but a digest identifies supplied candidate
  bytes; it does not archive omitted source by itself. The final E-NL source is
  reconstructible at commit `2d0ee83` / tag `theorem-aligned-v1`. The pilot and
  factorial retain recorded dirty-source fingerprints but not the corresponding
  source snapshots, so their exact execution cannot be reconstructed. These and the
  older dirty-run limitations are recorded in
  `paper/artifacts/provenance_manifest.json`;
- `finite_n_geometry_check.json` (E2): analytic-versus-autograd validation of the exact linear-RNN geometry;
- `trajectories.csv`: optimization-time measurements including mode responses, drifts, margins, `A`, `G`, susceptibility, and GSI-5;
- `summary.csv`: paired causal metrics and confidence-interval inputs;
- `aggregate.csv`: seed means and 95% confidence intervals for preregistered outcomes;
- experiment-specific PDF and PNG figures.

Seeds are paired across conditions. Synthetic data is generated once per `(rho, lag, regime, seed)` and the weak-only condition masks only the strong channel. The initial model state is also reused across the pair.

## Important configuration choices

- `task.beta` is the preregistered hitting-time target.
- `task.phase_delay` is the threshold used to label delayed versus jointly learned runs, and the default for `task.delta_tw_tolerance`.
- `task.tau_max` and `task.delta_tw_tolerance` configure the learnability gate. `tau_max` defaults to the trajectory horizon.
- `task.theory_boundary_file` optionally overlays a CSV with `rho` and `lag_separation` columns. No analytic boundary exists yet; the hook is ready for when the solver unblocks.
- `training.full_batch: true` matches the controlled gradient-flow setting.
- `training.paired_mode` is `sequential` (default) or `lockstep`. Lockstep advances both conditions together and is used by E-NL. The two update schedules are equivalent under deterministic full-batch training—asserted at `rtol=0` on final parameters across three model families.
- `training.exact_response_drift: true` computes the universal direct-autograd mode derivative at every logged point. E-NL forces this on and refuses non-finite exact drift rather than falling back to projected `Gg`.
- `model.kind` supports `low_rank_linear`, `dense_linear`, `tanh`, and `gru`. `dense_linear` is the family with the closed finite-width geometry. Nonlinear feature responses are common symmetric unit-probe contrasts; they are not assumed to form exact additive logit coordinates.
- `model.bulk_gain` controls recurrent disorder strength.
- `mitigation.objective` is `cross_entropy` (default) or `mse`.
- `mitigation.method` is one of `erm`, `spectral_decoupling`, `interaction`, or one of five shadow-based rescue methods: `counterfactual_drift`, `loss_gradient_projection`, `unconstrained_rescue`, `bloop`, `pcgrad`. All five share one causal target and differ only in what they protect, so comparing them isolates the constraint. **They all train a second weak-only model in lockstep**, roughly doubling memory and per-step cost, which the other methods do not require; disclose that in any comparison.

### Regime labels emitted by E1

`regime_class` takes one of five values. `unlearnable` means the weak-only
counterfactual never reached the target, so the point is indeterminate rather than
starved. `degenerate` means the target was already met at initialization, before any
optimization step — an artifact of threshold choice, not a phase. The remaining three
are `transfer`, `neutral` and `starvation`. `legacy_phase` preserves the pre-gate
label so run directories already on disk stay interpretable.

The default main configurations are intentionally compute-heavy. Start with `configs/smoke.yaml`, confirm the output contract, and then scale widths, samples, steps, and seeds on the intended compute system.
