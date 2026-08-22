# Causal Gradient Starvation in Recurrent Networks

This repository studies whether a dominant temporal feature causally slows the learning of a weaker predictive feature in recurrent models. The key comparison is paired: every both-feature run is matched to a weak-only run with the same weak signal, lag, noise, initialization, architecture, optimizer, and seed.

The codebase implements three experiment blocks:

- **E1 - causal phase diagram:** sweeps feature-strength ratio and temporal separation, measures weak-feature hitting-time delay, and includes a negative-control data regime.
- **E2 - field-theory validation:** checks the exact finite-width projected-flow identity, compares network trajectories with a joint numerical particle closure, and measures error versus width.
- **E3 - mitigation and transfer:** compares ERM, Spectral Decoupling, a susceptibility-targeted interaction penalty, and Counterfactual Drift Correction on linear and nonlinear recurrent models. An optional Waterbirds adapter measures average and worst-group accuracy.

The older `gsi_pipeline/` and `legacy_docs/` directories are retained for provenance. New experiments live in `src/gradient_starvation/` and use explicit YAML configurations.

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
.\.venv\Scripts\python.exe run_experiment.py e1 --config configs/e1_corrected_validation.yaml
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

E2 first verifies `dm/dtau = Gg` against direct autograd. It then builds a joint strong-weak particle closure from independent quenched recurrent systems and evaluates held-out finite networks at several widths. The particle closure is a numerical approximation and is deliberately reported as such; it is not a substitute for a concentration proof.

### E3: mitigation

```powershell
.\.venv\Scripts\python.exe run_experiment.py e3 --config configs/e3_mitigation.yaml
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
At each full-batch step it adds the minimum-norm correction needed to match the
shadow model's weak drift, after projecting the correction orthogonally to the
strong-mode gradient. The instantaneous strong drift is therefore unchanged.
This method requires zero weight decay and no gradient clipping; the trainer
rejects configurations that would invalidate the guarantee. Its diagnostics
(`cdc_target_met`, `cdc_feasible`, correction size, and strong-drift change) are
written to `trajectories.csv`.

A compact corrected-scaling comparison can be run with:

```powershell
.\.venv\Scripts\python.exe run_experiment.py e3 --config configs/e3_cdc_pilot.yaml
```

Use `configs/e3_cdc_validation.yaml` for the eight-seed, width-96 CPU
validation. These are validation-scale configurations, not substitutes for the
larger nonlinear and width-convergence experiments required for publication.
The nonlinear search and validation configurations are
`configs/e3_nonlinear_regime_search.yaml`,
`configs/e3_tanh_long_horizon_search.yaml`, and
`configs/e3_tanh_cdc_validation.yaml`. The selected tanh condition deliberately
uses a long horizon because it exhibits early transfer followed by late
starvation; short runs do not test the intended nonlinear phenomenon.

To run the optional real-data transfer after installing its requirements:

```powershell
.\.venv\Scripts\python.exe run_experiment.py waterbirds --config configs/waterbirds.yaml
```

Waterbirds is intentionally excluded from the mandatory synthetic test suite:
it requires the optional dependencies, dataset terms acceptance, local storage
through `data.root`, and (when enabled) pretrained model assets. The code raises
an actionable error when its optional dependencies are absent.

## Outputs and reproducibility

Every output directory contains:

- `config.resolved.yaml`: the exact configuration after command-line overrides;
- `environment.json`: Python, PyTorch, device, platform, and source-state metadata;
- `finite_n_geometry_check.json` (E2): analytic-versus-autograd validation of the exact linear-RNN geometry;
- `trajectories.csv`: optimization-time measurements including mode responses, drifts, margins, `A`, `G`, susceptibility, and GSI-5;
- `summary.csv`: paired causal metrics and confidence-interval inputs;
- `aggregate.csv`: seed means and 95% confidence intervals for preregistered outcomes;
- experiment-specific PDF and PNG figures.

Seeds are paired across conditions. Synthetic data is generated once per `(rho, lag, regime, seed)` and the weak-only condition masks only the strong channel. The initial model state is also reused across the pair.

## Important configuration choices

- `task.beta` is the preregistered hitting-time target.
- `task.phase_delay` is the threshold used to label delayed versus jointly learned runs.
- `task.theory_boundary_file` optionally overlays a CSV with `rho` and `lag_separation` columns from an external field-theory solve.
- `training.full_batch: true` matches the controlled gradient-flow setting.
- `model.kind` supports `low_rank_linear`, `dense_linear`, `tanh`, and `gru`.
- `model.bulk_gain` controls recurrent disorder strength.
- `mitigation.method` is one of `erm`, `spectral_decoupling`, `interaction`, or `counterfactual_drift`.

The default main configurations are intentionally compute-heavy. Start with `configs/smoke.yaml`, confirm the output contract, and then scale widths, samples, steps, and seeds on the intended compute system.
