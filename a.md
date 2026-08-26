# Gradient Starvation Research: Complete AI Handoff

> **SUPERSEDED (2026-08-24).** This historical handoff predates the final
> theorem-aligned nonlinear protocol and must not be used as a claim authority.
> Use `README.md`, `research_scope/claim_ledger.md`, and
> `research_scope/oral_theorem_package.md`; in particular, the final E-NL result is
> tanh 3/8 causal certificates and GRU 0/8 after the learnability gate.

> **Read this file before changing code or interpreting results.** It records the current scientific goal, implementation state, experimental evidence, known-invalid runs, unresolved theorem work, and recommended continuation order as of **2026-08-22**.

## 1. Project identity and working state

- Repository: `https://github.com/Sujato-Dutta/Gradient-Starvation-Research`
- Local repository: `/Users/prithvirajsangramsinhpatil/Documents/ChatGPT/Gradient-Starvation/Gradient-Starvation-Research`
- Current branch: `main`
- Remote: `origin = https://github.com/Sujato-Dutta/Gradient-Starvation-Research.git`
- Last committed revision: `3607991 first commit`
- Blueprint PDF: `/Users/prithvirajsangramsinhpatil/Downloads/gradient_starvation_iclr27_blueprint-1.pdf`
- Local test environment: `.venv-test`, Python 3.12.4

The working tree contains a large amount of valuable **uncommitted work**. Do not reset, clean, overwrite, or discard it. Results under `results/` are ignored by Git but remain locally available. Before beginning a major rewrite, inspect `git status`, preserve unrelated changes, and create a branch with the `codex/` prefix if a commit workflow is requested.

The user wants publication-quality work aimed at an ICLR 2027 oral-level paper, but also wants simple explanations and honest conclusions. Never describe simulation evidence as a mathematical proof and never promise oral acceptance.

## 2. Scientific question and causal definition

The project studies **gradient starvation** in sequential neural networks. A task contains a strong feature and a weak feature. The central causal comparison uses a paired intervention:

1. **Both-feature condition:** the model receives strong and weak cues.
2. **Weak-only condition:** the exact same samples, labels, noise, weak cue, and model initialization are used, but the strong input channel is set exactly to zero.

Starvation means that the strong feature causally suppresses the learning of the weak feature. The primary quantity is the paired weak-mode trajectory difference, commonly summarized as

`AUC gap = AUC(weak-only weak response) - AUC(both-feature weak response)`.

A positive gap indicates causal starvation only when the weak-only counterfactual meaningfully learns the weak feature. If weak-only itself does not learn, the point is **unlearnable/indeterminate**, not evidence of starvation. This counterfactual-learnability condition is an important correction to the original framing.

Optimization time and sequence time are distinct. Use `tau` for optimization time and `t` for positions within the input sequence.

## 3. Blueprint theorem inventory

The supplied blueprint proposes or motivates:

1. an exact signed function-flow identity;
2. an exact two-mode projected-flow identity;
3. cross-entropy sensitivity/gating results;
4. a feature-strength equilibrium theorem;
5. a common-metric weak-only comparison theorem;
6. exact finite-width geometry for a dense linear RNN;
7. a joint cross-entropy dynamical mean-field theory (CE-DMFT) and starvation phase boundary.

The finite-dimensional identities are implemented and numerically tested. The
general positive-disorder/positive-lag joint paired CE-DMFT theorem, its quenched
concentration proof, and analytic phase boundary are **not proved**. A separate
singular dense-linear zero-disorder/zero-lag six-scalar limit is proved under its
explicit deterministic tube assumptions. The authoritative current theorem-scoping
files are:

- `research_scope/e2_theorem.md`
- `research_scope/oral_theorem_package.md`

These files intentionally distinguish proved algebraic identities, implemented mitigation guarantees, empirical evidence, and future proof targets.

## 4. Repository structure and entry points

Important paths:

- `run_experiment.py`: main experiment CLI.
- `src/gradient_starvation/data/synthetic.py`: deterministic paired synthetic data.
- `src/gradient_starvation/models/recurrent.py`: low-rank linear, dense linear, tanh, and GRU model families.
- `src/gradient_starvation/training.py`: training, projected diagnostics, mitigation logic, and CDC logging.
- `src/gradient_starvation/theory.py`: projected-flow identities, finite-width geometry, matched drift decomposition, and related theory utilities.
- `src/gradient_starvation/experiments.py`: E1/E2/E3 orchestration, artifacts, E2 closure evaluation, and acceptance records.
- `src/gradient_starvation/plotting.py`: experiment figures.
- `src/gradient_starvation/width_validation.py`: width-robustness summary utilities.
- `summarize_width_validation.py`: reproducible combiner for nonlinear width runs.
- `configs/`: smoke, pilot, validation, and larger experiment configurations.
- `tests/`: unit, model/loss, theorem, training, smoke-integration, and width-validation tests.
- `.github/workflows/tests.yml`: local GitHub Actions workflow definition.
- `results/`: generated CSV, JSON, PNG, PDF, and environment records; ignored by Git.

## 5. Implemented synthetic data repair

The previously missing `src/gradient_starvation/data/` package has been restored.

Implemented objects:

- `SyntheticTaskSpec`: sequence length, cue strengths, temporal positions/separation, noise, sample count, and regime.
- `SyntheticBatch`: model inputs, binary labels, signed labels, strong/weak latent coordinates, condition, originating spec, and `.to(device)` support.
- `make_paired_task(spec, seed)`: deterministically constructs both-feature and weak-only batches.

Critical invariant: paired samples are identical outside the strong input channel, and the weak-only strong channel is exactly zero. Tests fail if samples diverge elsewhere.

The positive regime uses a strong latent coordinate centered around `rho` and a weak coordinate centered around one, plus configured noise. The negative-control regime introduces an anti-correlated latent factor so that the expected causal sign is reversed. `.gitignore` keeps downloaded top-level datasets ignored while explicitly tracking `src/gradient_starvation/data/**`.

## 6. Exact theory and mitigation implemented

### 6.1 Exact projected finite-width flow

For the dense linear RNN

`h_(t+1) = W h_t + B x_t`, and `f(x) = c^T h_T`,

the controlled initialization is

- `W_ij ~ N(0, g^2/N)`;
- `B_ia ~ N(0, 1/N)`;
- `c_i ~ N(0, 1/N)`.

For channel-localized modes with responses `m_a`, impulse quantities `p_a`, `q_a`, and recurrent derivatives `S_a`, the exact finite-width identity is

`d m_a/d tau = sum_b G_ab g_b`,

with

`G_ab = p_a^T p_b + 1[channel_a = channel_b] q_a^T q_b + <S_a,S_b>_F`.

The channel indicator is essential because distinct input columns are distinct parameter blocks. This is checked against autograd and direct loss-gradient flow. Latest validation has exact-identity absolute error at most `2.86102294921875e-06` and finite-width geometry relative error `0.0`.

### 6.2 Exact matched weak-drift decomposition

`matched_weak_drift_decomposition` verifies

`weak-only drift - both-feature drift = CE gating + geometry shift + cross transport`.

This separates suppression caused by cross-entropy saturation from geometry changes and cross-mode transport. It is an exact algebraic decomposition at matched state, not a large-width theorem.

### 6.3 Counterfactual Drift Correction (CDC)

CDC uses a matched weak-only shadow model to estimate the counterfactual weak drift.
It works in the fixed implemented tensor coordinates with the product
Euclidean/Frobenius metric and adds a correction along the component of
`grad(m_w)` orthogonal to `grad(m_s)` in that metric. There is no
reparameterization-invariance or natural-gradient claim.

When the positive deficit is exactly feasible (`q != 0`) and the correction is
uncapped:

1. corrected instantaneous weak drift reaches at least the weak-only target;
2. instantaneous strong drift is unchanged;
3. the correction is the unique minimum Euclidean-norm, lower-bound-attaining
   feasible correction in those coordinates.

The represented `q`, its norm, the numerical `feasibility_epsilon`, and the
selected minimizer are metric- and scale-dependent. A nonbinding cap leaves the
result unchanged; a binding cap generally misses the target, so no target-attaining
optimality claim applies. The trainer logs feasibility, target attainment,
correction norm, strong-drift preservation error, uncapped and applied
coefficients, whether the cap binds, and the target residual. CDC requires
full-batch training, no weight decay, and no gradient clipping. The guarantee is
**instantaneous** and local; same-chart finite-step smoothness gives only the proved
one-step bound. The final strong response can be lower because CDC follows a
different trajectory, and this tradeoff must always be reported.

## 7. Critical E2 corrections

Two major problems were found in the earlier E2 design.

### 7.1 Incorrect input scaling

Linear input weights were initialized as `randn(width, input_size) / sqrt(input_size)`. Their norm therefore grew as approximately `sqrt(N)`, and projected geometry such as `G_ss` grew linearly with width. Observed mean `G_ss` values at widths 16, 32, 64, 128, and 256 were approximately `9.5, 17.4, 32.1, 64.7, 127.6`. There was no common optimization-time large-width limit.

Both `DenseLinearRNN.input` and `LowRankLinearRNN.input` now use width scaling, `1/sqrt(width)`.

### 7.2 Wrong model family

Earlier E2 experiments used `low_rank_linear` while claiming the dense-`W` finite-width geometry theorem. The low-rank parameter metric is different. Corrected E2 uses `dense_linear`.

As a consequence, all earlier E2 medium/high/width-extension runs are scientifically superseded and must not be cited as DMFT validation.

## 8. Corrected E2 theorem target

The intended oral-level theorem is a **joint paired large-width convergence theorem**, not merely convergence of two separate conditions.

For fixed sequence length, finite optimization horizon, bounded feature moments, and stable bulk gain, define joint order parameters for both-feature and weak-only systems under shared quenched initialization. They must include mode responses, projected geometry, margin law, CE sensitivity, and all recurrent correlation/response observables required for closure.

The target is uniform-in-time convergence in probability of the paired finite-width process to the unique solution of one coupled recurrent generating-functional/DMFT system. The joint law must retain `G_sw` and corresponding cross-response terms. A hitting-time result additionally requires counterfactual learnability and transverse target crossing.

This target is **not yet proved**. The remaining proof obligations are:

1. derive the effective single-site process and covariance/response kernels from the finite-width probability space;
2. enumerate every visible and loss-invisible order parameter required for closure;
3. prove finite-horizon existence and uniqueness of the Volterra/ODE system;
4. prove joint quenched concentration for both causal conditions under shared disorder;
5. prove the hitting-time corollary under transversality and bound discrete-step error.

The current numerical reference is correctly called a **held-out quenched ensemble closure**, not a proved or independent DMFT solver.

## 9. Corrected E2 experiments and results

### 9.1 Pilot

Directory: `results/e2_corrected_pilot-20260821-224411`

- widths: 24, 48, 96, 192;
- four evaluation seeds;
- eight disjoint closure particles at width 256;
- points `(rho, lag) = (2, 2), (4, 4)`;
- 200 steps;
- acceptance passed.

Mode NRMSE from smallest to largest width:

| Point | Condition | Width 24 | Width 192 |
|---|---:|---:|---:|
| `(2,2)` | both | 0.20595 | 0.04283 |
| `(2,2)` | weak-only | 0.26102 | 0.10935 |
| `(4,4)` | both | 0.09660 | 0.02280 |
| `(4,4)` | weak-only | 0.55941 | 0.20755 |

### 9.2 High-confidence corrected validation

Directory: `results/e2_corrected_validation-20260821-224630`

Design:

- points: `(1,0), (2,2), (4,4), (8,8)`;
- widths: 32, 64, 128, 256;
- evaluation seeds: 0 through 7;
- held-out closure width: 384;
- closure seeds: 100 through 115;
- samples: 1024;
- steps: 400;
- 256 complete, finite, non-duplicate summary rows.

Automated acceptance:

```json
{
  "closure_and_evaluation_seeds_disjoint": true,
  "finite_n_geometry_relative_error": 0.0,
  "max_projected_identity_absolute_error": 2.86102294921875e-06,
  "all_summary_diagnostics_finite": true,
  "mode_error_decreases_for_every_point_and_condition": true,
  "at_least_three_widths": true,
  "passed": true
}
```

Mode trajectory NRMSE, width 32 to width 256, with fitted log-log slope:

| Point | Condition | NRMSE at 32 | NRMSE at 256 | Slope |
|---|---|---:|---:|---:|
| `(1,0)` | both | 0.168733 | 0.036794 | -0.659925 |
| `(1,0)` | weak-only | 0.103293 | 0.022774 | -0.760243 |
| `(2,2)` | both | 0.089176 | 0.036419 | -0.354439 |
| `(2,2)` | weak-only | 0.112559 | 0.057971 | -0.316892 |
| `(4,4)` | both | 0.047785 | 0.013352 | -0.514434 |
| `(4,4)` | weak-only | 4.563753 | 0.315860 | -1.139775 |
| `(8,8)` | both | 0.057647 | 0.031766 | -0.276435 |
| `(8,8)` | weak-only | 1.497647 | 0.404340 | -0.560949 |

All geometry, margin, GSI, and trajectory convergence slopes are negative for every point and condition. Large weak-only normalized errors at `(4,4)` and `(8,8)` are inflated by a near-zero normalization denominator; absolute RMSE also decreases.

Overall, both-condition mode NRMSE falls from about `0.09084` at width 32 to `0.02958` at width 256. Weak-only mode NRMSE falls from about `1.5693` to `0.20024`, while absolute mode RMSE falls from `0.22695` to `0.08780`.

Interpretation: the corrected dense parameterization shows strong, broad finite-width convergence to a held-out empirical wide-network closure. This is substantially better evidence than the previous E2 results, but it is not a proof and not an independent DMFT validation because the closure is still calibrated from finite trained networks.

## 10. E1 phase-grid result

Primary directory: `results/e1_corrected_validation-20260821-144112`

Design:

- width 96;
- 256 samples;
- 500 steps;
- eight independent seeds, 4 through 11;
- full grid `rho in {1,2,4,8}` by `lag in {0,2,4,8}` plus negative controls;
- 136 paired summaries.

Results:

- Every positive-regime cell through lag 4 had a positive AUC gap for every seed.
- Lag 0 mean gaps ranged from `11.937663` to `23.326406`.
- Lag 2 mean gaps ranged from `8.170996` to `9.363293`.
- Lag 4 mean gaps ranged from `1.451338` to `1.517044`.
- Matched negative control mean gap: `-0.750387`, 95% CI `[-0.876793, -0.623980]`.
- Final accuracy was 1.0.
- Lag 8 is indeterminate because weak-only final response was only `0.001643`; neither condition meaningfully learned the weak feature.

The phase plot was fixed so that missing or censored cells are not silently plotted as zero.

**Important consistency warning:** this E1 run predates the latest width-scaled linear input correction. It was valid under its then-current parameterization but is not fully aligned with the final dense E2 theorem parameterization. Before using E1 in a paper, rerun it with the corrected scaling and regenerate the phase diagram.

Earlier E1 medium/high runs around the initial 13:xx experiment batch predate a duplicate-`rho` scaling fix and are superseded.

## 11. E3 mitigation results

### 11.1 Corrected linear CDC validation

Directory: `results/e3_cdc_validation-20260821-142502`

- width 96;
- eight seeds;
- ERM AUC gap: `3.115968`;
- interaction proxy: `3.115618`;
- Spectral Decoupling: `3.038149`;
- CDC: `0.000001`;
- final accuracy: 1.0 for every method;
- CDC feasibility and target attainment: 100%;
- maximum instantaneous strong-drift change: `7.63e-6`;
- paired CDC-minus-ERM change: `-3.115966`, 95% CI `[-3.707183, -2.524750]`;
- final strong response under CDC was about `0.258` lower than ERM.

This is strong evidence that CDC eliminates the chosen linear weak-trajectory deficit while preserving instantaneous strong drift. However, it predates the newest linear input-scaling correction, so rerun the linear E3 benchmark under the final dense parameterization before publication.

### 11.2 Nonlinear regime search

Short scan: `results/e3_nonlinear_regime_search-20260821-143300`

- tanh and GRU;
- width 24, 128 samples, 80 steps, seeds 0 and 1;
- `rho in {2,4,8}`, `lag in {0,2,4,8}`.

No robust positive starvation appeared over the short horizon; the strong cue initially transferred useful gradient and accelerated weak learning.

Long tanh scan: `results/e3_tanh_long_horizon_search-20260821-143400`

- width 24, 128 samples, 1000 steps, seeds 0 and 1;
- at `rho=4`, lag 2, the mean AUC gap was `12.956939`, minimum seed gap `9.160787`;
- final weak response: both `1.208475`, weak-only `4.340887`;
- approximate sign crossover around optimization time `tau = 2.15`.

Interpretation: tanh exhibits **early positive transfer followed by late starvation**. A valid trajectory theorem must permit the drift-deficit sign to change over time; a static positive-susceptibility condition is insufficient.

Long GRU scan: `results/e3_gru_long_horizon_search-20260821-143732`

- eight summaries;
- mean gap `-8.109928`.

This is a negative result. The current GRU regime should not be presented as a successful starvation or mitigation benchmark.

### 11.3 Tanh CDC pilot and frozen validation

Pilot: `results/e3_tanh_cdc_pilot-20260821-143459`

- width 32, 256 samples, 750 steps, seeds 2 through 5;
- ERM `7.560667`;
- interaction `7.220363`;
- Spectral Decoupling `7.669840`;
- CDC `-2.182139`;
- final accuracy 1.0; CDC feasibility/target attainment 100%.

Frozen validation: `results/e3_tanh_cdc_validation_combined-20260821-161600`

- width 64, 512 samples, 1000 steps, unseen seeds 6 through 13;
- ERM gap `19.728794`, 95% CI `[18.811261, 20.646327]`;
- interaction gap `19.431932`;
- Spectral Decoupling gap `16.942663`;
- CDC gap `-1.933250`, 95% CI `[-2.204207, -1.662293]`;
- paired CDC-minus-ERM `-21.662044`, 95% CI `[-22.337681, -20.986407]`, approximate p-value `1.83e-11`;
- final accuracy 1.0 for all runs;
- CDC target and feasibility 100% over 808 logged checks;
- maximum instantaneous strong-drift change `7.45e-8`;
- final strong response: ERM `1.10748`, CDC `0.846715`;
- final weak response: ERM `0.96642`, CDC `5.10491`;
- GSI: ERM `0.12405`, CDC `0.031735`.

### 11.4 Tanh CDC width robustness

Directory: `results/e3_tanh_cdc_width_validation_combined-20260821-162700`

Matched seeds 14 through 17 at widths 32, 64, and 96:

| Width | Paired CDC-minus-ERM AUC change | 95% CI |
|---:|---:|---:|
| 32 | -17.846010 | `[-20.237667, -15.454352]` |
| 64 | -21.852428 | `[-23.035754, -20.669101]` |
| 96 | -22.091320 | `[-23.780840, -20.401799]` |

Every seed improved at every width; accuracy was 1.0 and target attainment remained 100%. Figure: `results/e3_tanh_cdc_width_validation_combined-20260821-162700/e3_tanh_cdc_width_robustness.png`.

These nonlinear results are unaffected by the later linear input-scaling change and currently provide the strongest empirical contribution in the project.

## 12. Superseded or non-citable results

Do not use the following as final scientific evidence without clearly labeling them obsolete:

- `results/e2_medium_validation-20260821-133012`
- `results/e2_high_confidence-20260821-133710`
- `results/e2_width_extension-20260821-134952`
- early E1/E3 medium/high runs from the initial 13:xx batch

Reasons include duplicate feature-strength scaling, the wrong `low_rank_linear` model for a dense-geometry claim, and input weights whose norm grew with width. In the obsolete E2 width extension, both-condition RMSE worsened from roughly `19.86` to `25.97`, which was an important signal that the setup was incorrect.

The exact finite-width geometry tests themselves remain valid; the problem was the mismatch between the research claim and the model/parameterization used in those earlier experiments.

## 13. Test suite and CI status

The latest full local run was:

```text
46 passed
```

There was one harmless Seaborn `PendingDeprecationWarning`. `git diff --check` passed.

Coverage includes:

- deterministic synthetic data, shapes, device transfer, temporal cue placement, exact ablation, and regime moments;
- model families `low_rank_linear`, `dense_linear`, `tanh`, and `gru`;
- ERM, Spectral Decoupling, interaction proxy, and second-order behavior where practical;
- CDC in linear and tanh smoke/integration paths;
- invalid configuration/input handling and stable paired initialization;
- exact finite-width geometry and projected-flow identities;
- E1, E2, and E3 tiny CPU integrations with temporary outputs;
- required CSV/JSON/PNG/PDF contracts, schemas, finite diagnostics, and paired summaries;
- E2 disjoint closure/evaluation seeds and convergence acceptance;
- Waterbirds missing-data/dependency errors with actionable messages;
- width-validation aggregation.

The GitHub Actions file installs `requirements.txt`, sets a headless Matplotlib backend, and runs tests on Ubuntu with Python 3.10 and 3.12 for pushes and pull requests. It exists only in the dirty local tree and has not been observed running remotely. A clean Python 3.10 environment has not independently been verified yet.

Potential minor coverage gap: add an explicit `dense_linear + CDC` integration test if complete Cartesian model/mitigation coverage is required; current CDC coverage centers on low-rank linear and tanh experiment paths.

## 14. Waterbirds status

Waterbirds support is intentionally excluded from mandatory CI because it requires external dataset access, pretrained assets, and optional dependencies.

Implemented:

- adapter: `src/gradient_starvation/waterbirds.py`;
- config: `configs/waterbirds.yaml`;
- actionable missing-data/dependency preflight test;
- ERM, Spectral Decoupling, interaction proxy, and CDC-style minibatch
  head-coordinate surrogate variants.

The CDC-style Waterbirds variants are outside the full-batch matched-shadow theorem:
they use minibatches and head-coordinate surrogates rather than its exact causal
response setting. They are implementation-tested but **have not been run**.

Not implemented or run:

- any Waterbirds experiment or result;
- dataset download/preparation;
- pretrained asset setup;
- real-data experiments;
- average-group/worst-group results.

This is a major oral-readiness blocker because all current positive evidence is synthetic.

## 15. ICLR oral-readiness assessment

Current verdict: **not yet ICLR oral ready**.

Strengths:

- exact causal pairing and strong tests;
- exact finite-width projected identities;
- corrected dense scaling and clean E2 convergence across four widths, eight evaluation seeds, and multiple regimes;
- a novel-looking causal drift decomposition and constrained CDC rule;
- strong nonlinear tanh mitigation with held-out seeds and width robustness;
- honest negative controls and GRU negative results;
- automated test and artifact contracts.

Main blockers:

1. the decisive joint paired CE-DMFT theorem is a target, not a proof;
2. the E2 closure is empirical finite-network calibration, not an independently derived solver;
3. there is no proved analytic phase boundary;
4. the trajectory-level CDC guarantee and nonlinear sign crossover lack a theorem;
5. the linear E1/E3 results should be rerun under the final scaling;
6. no real-data experiment has been completed;
7. novelty has not been exhaustively compared against gradient starvation/Spectral Decoupling, PCGrad, Bloop/gradient surgery, bilevel methods, or the latest recurrent-DMFT literature;
8. CI has not yet run remotely from a committed branch.

No model should claim that the work is guaranteed to receive an oral. The strongest defensible statement is that the corrected E2 convergence and nonlinear CDC results are promising, while theory completion and real-data validation remain necessary for a serious oral-level submission.

## 16. Recommended continuation order

### Priority 1: complete the mathematical E2 theorem

- Write the exact probability space and coupled both/weak system under shared quenched disorder.
- Derive the generating functional or cavity equations.
- Identify the full covariance and response kernels, including loss-invisible observables.
- Prove finite-horizon existence/uniqueness and joint concentration.
- Derive the counterfactually learnable hitting-time corollary.
- Only after this derivation, implement an independent deterministic DMFT solver.

Do not reverse the logic by calling the current ensemble average “DMFT.”

### Priority 2: rerun final-scale E2 against the independent solver

- Use the publication configuration in `configs/e2_field.yaml` as a starting point: widths 128, 512, and 2048 with a wide closure/reference.
- Use 3–5 or more independent seeds, respecting machine memory and runtime.
- Keep closure/reference seeds disjoint from evaluation seeds.
- Report trajectories, `G_ss/G_sw/G_ww`, margin distribution, GSI, absolute and normalized error, slopes, and exact-identity residual.
- Use a cluster if width 2048 is too expensive locally; do not silently reduce the claimed scale.

### Priority 3: derive and validate the phase boundary

- Derive `rho_c(lag, bulk_gain)` only inside the counterfactually learnable region.
- Separately characterize the learnability boundary.
- Rerun E1 with the corrected dense scaling.
- Overlay the analytic boundary and finite-width uncertainty on the corrected E1 phase diagram.

### Priority 4: strengthen CDC theory and evaluation

- Prove a trajectory/hitting-time comparison under explicit smoothness, monotonicity, and persistent-feasibility assumptions.
- Allow time-varying drift sign to capture early transfer and late starvation.
- Quantify finite-step optimization error.
- Run sensitivity analyses for target definition, coefficient/correction caps, batch approximation, learning rate, runtime, and memory.
- Report the longer-horizon strong-feature tradeoff, not just instantaneous preservation.
- Rerun linear CDC under the final dense scaling.

### Priority 5: real-data experiment

- Treat the existing Waterbirds CDC-style minibatch head-coordinate implementation
  only as a surrogate outside the full-batch matched-shadow theorem.
- Run the mandatory pilot before any full experiment; it remains **not run**.
- Establish data and pretrained-backbone setup separately from mandatory CI.
- Compare ERM, Spectral Decoupling, interaction proxy, and the proposed method.
- Report average accuracy, worst-group accuracy, per-group confidence intervals, compute cost, and sensitivity.

### Priority 6: novelty and adversarial paper review

- Search current primary literature, including work published after the blueprint was written.
- Compare formally against Gradient Starvation/Spectral Decoupling, PCGrad, Bloop and related gradient surgery, bilevel counterfactual training, causal representation learning, and recurrent DMFT.
- Position CDC specifically as targeting a **causal weak-only drift** while protecting strong instantaneous drift, not merely as generic gradient projection.
- In the paper, label each statement as proved, conjectured/target, or empirical.

## 17. Commands for the next model

From the repository root:

```bash
source .venv-test/bin/activate
python --version
pytest -q
git diff --check
git status --short
```

Useful smoke or validation commands:

```bash
python run_experiment.py e1 --config configs/e1_corrected_pilot.yaml
python run_experiment.py e2 --config configs/e2_smoke.yaml
python run_experiment.py e2 --config configs/e2_corrected_pilot.yaml
python run_experiment.py e2 --config configs/e2_corrected_validation.yaml
python run_experiment.py e3 --config configs/e3_smoke.yaml
python run_experiment.py e3 --config configs/e3_tanh_cdc_validation.yaml
python summarize_width_validation.py
```

The high-confidence corrected E2 run is already complete. Do not rerun it merely to rediscover the same outputs unless checking reproducibility after a code change.

Before a publication-scale run, inspect the exact CLI and configuration parser in `run_experiment.py` and `src/gradient_starvation/experiments.py`; do not assume configurations are interchangeable across experiment families.

## 18. Current uncommitted file inventory

At handoff, modified tracked files include:

```text
.gitignore
README.md
configs/e2_field.yaml
configs/e3_mitigation.yaml
src/gradient_starvation/experiments.py
src/gradient_starvation/losses.py
src/gradient_starvation/models/recurrent.py
src/gradient_starvation/plotting.py
src/gradient_starvation/theory.py
src/gradient_starvation/training.py
tests/test_synthetic.py
tests/test_theory.py
tests/test_training.py
```

Important untracked additions include:

```text
.github/workflows/tests.yml
configs/e1_corrected_pilot.yaml
configs/e1_corrected_validation.yaml
configs/e2_corrected_pilot.yaml
configs/e2_corrected_validation.yaml
configs/e2_smoke.yaml
configs/e3_cdc_pilot.yaml
configs/e3_cdc_validation.yaml
configs/e3_nonlinear_regime_search.yaml
configs/e3_smoke.yaml
configs/e3_tanh_cdc_pilot.yaml
configs/e3_tanh_cdc_validation.yaml
configs/e3_tanh_cdc_width_validation.yaml
configs/e3_tanh_long_horizon_search.yaml
research_scope/e2_theorem.md
research_scope/oral_theorem_package.md
src/gradient_starvation/data/
src/gradient_starvation/width_validation.py
summarize_width_validation.py
tests/test_experiment_smoke.py
tests/test_models_and_losses.py
tests/test_width_validation.py
```

There is also an untracked `.DS_Store`; remove it or ignore it before committing, but do not use a broad destructive clean command.

## 19. Original implementation-plan completion status

Completed locally:

- restored synthetic data package;
- deterministic paired data and exact-ablation tests;
- shape/device/timing/regime-moment tests;
- model and mitigation tests;
- input/config validation and paired initialization checks;
- E1/E2/E3 tiny CPU integrations and artifact/schema validation;
- dedicated smoke configurations;
- README test documentation and Waterbirds prerequisites;
- GitHub Actions workflow for Python 3.10 and 3.12;
- actionable Waterbirds preflight;
- corrected E2 dense parameterization, seed separation, convergence tables, and acceptance JSON;
- nonlinear CDC validation and width robustness.

Operationally remaining:

- review the entire dirty diff;
- remove or ignore `.DS_Store`;
- run tests in a genuinely clean environment, especially Python 3.10;
- create a branch, commit, push, and observe remote CI if authorized by the user;
- optionally add explicit dense-linear CDC integration coverage.

Scientifically remaining:

- complete joint paired CE-DMFT proof and independent solver;
- derive the phase and learnability boundaries;
- prove the trajectory-level mitigation result;
- rerun final linear E1/E3 under corrected scaling;
- execute publication-scale width experiments;
- run a real-data benchmark;
- complete primary-literature novelty analysis and adversarial review.

## 20. Non-negotiable interpretation rules

1. A passed test suite proves software invariants, not the research theorem.
2. A passed E2 acceptance JSON is evidence of numerical convergence to a held-out empirical closure, not an independent DMFT proof.
3. Positive AUC gap without weak-only learnability is indeterminate.
4. CDC preserves instantaneous strong drift under its assumptions; it does not guarantee the same final strong response.
5. Negative results such as the current GRU regime should be retained and reported honestly.
6. Use corrected E2 outputs, not superseded low-rank/wrong-scaling runs.
7. Preserve paired seeds, samples, and initializations in every causal comparison.
8. Report absolute errors when normalized errors have near-zero denominators.
9. Keep closure/calibration and evaluation seeds disjoint.
10. Do not claim ICLR oral readiness until the proof, independent solver, real-data study, and final reruns are complete.

## 21. Best immediate next action

The next model should first run `pytest -q`, inspect `git diff`, and read both
`research_scope` theorem files. Then it should focus on the actual coupled
generating-functional derivation and specify every order parameter needed for an
independent E2 solver. Running more finite networks before that derivation has
diminishing scientific value. In parallel, if engineering time is available,
rerun E1 and linear E3 with the final width-scaled dense model; only then run the
mandatory pilot for the existing, still-unrun Waterbirds CDC-style minibatch
head-coordinate surrogate, which remains outside the full-batch matched-shadow
theorem.

This handoff is deliberately conservative: the repository contains strong synthetic evidence and a much cleaner experimental foundation than it began with, but its headline oral-level theorem remains unfinished.
