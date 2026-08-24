# Causal Gradient Starvation in Recurrent Networks

This repository studies whether a dominant temporal feature causally slows the learning of a weaker predictive feature in recurrent models. The key comparison is paired: every both-feature run is matched to a weak-only run with the same weak signal, lag, noise, initialization, architecture, optimizer, and seed.

The codebase implements five experiment blocks:

- **E1 - causal phase diagram:** sweeps feature-strength ratio and temporal separation, measures weak-feature hitting-time delay, and includes a negative-control data regime. Classification is gated on weak-only learnability, so a point where the counterfactual never learns is reported as indeterminate rather than starved.
- **E2 - width convergence against an empirical reference:** checks the exact finite-width projected-flow identity, compares network trajectories with a held-out numerical closure reference, and measures error versus width. The reference is a mean over trained finite networks, not a solved theory.
- **E2-R - solver checks:** exercises the two mean-field special cases that are exact without the undelivered closure derivation, and records the remaining checks as blocked. Its acceptance record reports `passed: false` by construction.
- **E-NL - theorem-aligned crossover:** uses common symmetric unit-probe responses for tanh/GRU, computes the universal direct-autograd response drift at every logged point, reports the projected `Gg` residual separately, and evaluates the exact finite-step tail-area certificate distinguishing rate suppression from outcome starvation.
- **E3 - mitigation and transfer:** compares ERM, Spectral Decoupling, an interaction penalty, and a five-way ablation of constrained weak-rescue methods including Counterfactual Drift Correction plus Bloop-style and PCGrad-style baselines. A Waterbirds adapter exists but **has never been run**; see `research_scope/waterbirds_setup.md`.

### Read this before quoting any number

`research_scope/claim_ledger.md` tags every paper-bound statement as proved, target,
empirical, blocked or retracted, and names the artifact behind it. Several figures in
older notes are superseded. The established headline is now a finite-width causal
theorem package; the more ambitious joint mean-field extension remains a blocked
conjecture. The ledger exists so that checking which is which is faster than
rediscovering it. Four things worth knowing before reading anything else:

- The repository **proves** the exact finite-width CE cross-kernel flow, the
  transfer/suppression tail-area criterion, a conditional noiseless rank-one rate-
  crossover theorem, hitting-time stability, the zero-disorder six-scalar limit,
  and three local CDC results. See `research_scope/e2_theorem.md` and
  `research_scope/cdc_theorem.md`.
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

The complete scientific draft is `paper/main.tex`, with bibliography in
`paper/references.bib` and a compiled artifact at `paper/main.pdf`. It contains the
finite-width proofs, executable-certificate map, full synthetic protocol, corrected
results, novelty boundary, and limitations. It is a scientifically defensible
submission draft, not a guarantee of ICLR acceptance or oral selection.

## Setup

Python 3.10 or newer is recommended. Create a dedicated virtual environment and install the unpinned requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

For the optional Waterbirds experiment:

```powershell
python -m pip install -r requirements-waterbirds.txt
```

No package installation step is needed. The root entry points add `src/` to the Python path.

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
on Python 3.10 and 3.12 for every push and pull request.

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
not be quoted. The post-hoc grid
`results/enl_exact_regime_search-20260824-105003` found no 4/4 condition and is
exploratory negative evidence only.

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
At each full-batch step it attempts the unique minimum-norm correction needed to
match the shadow model's weak drift while projecting orthogonally to the strong-mode
gradient. Exact target attainment and minimum-norm optimality require a feasible,
uncapped direction; infeasible or capped steps remain explicitly logged. The
instantaneous strong drift is unchanged for every coefficient, including a cap.
This method requires cross-entropy diagnostics, zero weight decay, and no gradient
clipping; the trainer rejects configurations that would invalidate the contract.
Its diagnostics (`cdc_target_met`, `cdc_feasible`, correction size, and strong-drift
change) are written to `trajectories.csv`.

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
criteria pass should `configs/waterbirds.yaml` be run. Waterbirds is a surrogate
probe: its constant weak coordinate is an intercept, not an identified bird-shape
feature.

Waterbirds is intentionally excluded from the mandatory synthetic test suite:
it requires the optional dependencies, dataset terms acceptance, local storage
through `data.root`, and (when enabled) pretrained model assets. The code raises
an actionable error when its optional dependencies are absent.

## Outputs and reproducibility

Every output directory contains:

- `config.resolved.yaml`: the exact configuration after command-line overrides;
- `environment.json`: Python, PyTorch, device, platform, and source-state metadata;
  if it records `git_dirty: true` without `source_sha256`, the commit hash alone
  does not reconstruct the executed source. New runs content-address
  `run_experiment.py` plus `src/**/*.py`; the final E-NL compact archive and the
  explicit limitations of older dirty runs are recorded in
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
