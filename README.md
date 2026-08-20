# Causal Gradient Starvation in Recurrent Networks

This repository studies whether a dominant temporal feature causally slows the learning of a weaker predictive feature in recurrent models. The key comparison is paired: every both-feature run is matched to a weak-only run with the same weak signal, lag, noise, initialization, architecture, optimizer, and seed.

The codebase implements three experiment blocks:

- **E1 - causal phase diagram:** sweeps feature-strength ratio and temporal separation, measures weak-feature hitting-time delay, and includes a negative-control data regime.
- **E2 - field-theory validation:** checks the exact finite-width projected-flow identity, compares network trajectories with a joint numerical particle closure, and measures error versus width.
- **E3 - mitigation and transfer:** compares ERM, Spectral Decoupling, and a susceptibility-targeted interaction penalty on linear and nonlinear recurrent models. An optional Waterbirds adapter measures average and worst-group accuracy.

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

## Running the main experiments

Each command accepts repeatable `--set section.key=value` overrides. Results are written to a timestamped directory containing the resolved configuration, environment metadata, per-step trajectories, summaries, and figures.

### E1: causal phase diagram

```powershell
.\.venv\Scripts\python.exe run_experiment.py e1 --config configs/e1_phase.yaml
```

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

To run the optional real-data transfer after installing its requirements:

```powershell
.\.venv\Scripts\python.exe run_experiment.py waterbirds --config configs/waterbirds.yaml
```

The WILDS downloader may require accepting the dataset's terms and providing local storage through `data.root`.

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
- `mitigation.method` is one of `erm`, `spectral_decoupling`, or `interaction`.

The default main configurations are intentionally compute-heavy. Start with `configs/smoke.yaml`, confirm the output contract, and then scale widths, samples, steps, and seeds on the intended compute system.
