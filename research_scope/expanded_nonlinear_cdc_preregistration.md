# Expanded nonlinear, beta, and CDC tradeoff study

## Status

**Scientific design frozen; outcome generation blocked.** The machine-readable
contract is `configs/expanded_nonlinear_cdc_preregistration.yaml`, checked by
`scripts/check_expanded_studies_preregistration.py`. It reports
`execution_authorized: false`.

No model was trained and no preflight prediction or scientific outcome was created
in this pass. Historical E-NL/E19/E20 and E3 results were audited only to avoid
promoting post-hoc or mismatched evidence.

## Why a new study is required

The final E-NL result contains eight coupled seeds at one task cell per architecture.
Its terminal beta profile is post-hoc and does not recompute first-hitting times. E19
failed its broad rule. E20 passed only drift/response crossing classification at one
cell; phase, causal certificate, learnability, timing, and magnitudes were not
accepted primary endpoints. The exact dirty source bytes for E19/E20 were not
archived, so a current-source run would be a new study rather than a replay.

The only completed CDC/Bloop/PCGrad comparison is a dense-linear constraint
ablation. Weak rescue was nearly equal in magnitude, CDC alone preserved the
instantaneous strong drift, but PCGrad-style and unconstrained rescue had higher
final strong response. No preregistered equivalence margin or nonlinear style-arm
comparison exists. Those facts do not show that CDC wins the requested final
tradeoff.

## A. Prospective nonlinear beta breadth

The new nonlinear design fixes three cells before outcomes:

```text
(rho, lag) = (3,1), (4,2), (5,3),
```

for tanh and GRU at width 32. Four candidate data seeds are crossed with eight
candidate model seeds, giving 192 paired architecture/cell/seed records. Candidate
IDs still require an untouched-outcome attestation against ignored local results;
their presence in this file alone does not certify freshness.

Each trajectory is trained once. The fixed beta grid

```text
[0.25, 0.5, 1.0, 2.0]
```

is then evaluated in long form using the original, linearly interpolated **first
hit**. Terminal response is not a substitute, nonmonotone trajectories retain their
earliest crossing, a target met at initialization is degenerate, and a weak-only
nonhit is unlearnable. Outcome suppression is beta-independent; the causal label at
a beta is the conjunction of that independent outcome certificate and a
nondegenerate weak-only first hit.

`study_metrics.first_hitting_profile` implements this pure analysis contract. It
rejects empty, duplicate, unsorted, or nonfinite beta grids and malformed trajectory
grids. Its tests cover interpolation, nonmonotone first hits, degeneracy, censoring,
and the outcome/learnability conjunction.

The primary output is the complete model/cell/seed/beta profile, including
weak-only learnability and causal certification. An all-beta robust certificate is
a fixed summary, not a replacement for the profile. No beta, cell, or architecture
may be selected after outcomes to create a favorable claim.

## B. CDC weak-rescue versus strong-preservation tradeoff

The nonlinear tradeoff study uses tanh width 64 at `(rho,lag)=(4,2)` and `(5,3)`,
with the same crossed 4-by-8 seed axes. The four frozen arms are:

- ERM;
- CDC;
- **Bloop-style shadow-target rescue**;
- **PCGrad-style shadow-target rescue**.

The style labels are load-bearing. The Bloop-style arm applies an EMA/projection
motif to `grad(M_w)` and chooses its multiplier to hit a privileged weak-only-shadow
target. The PCGrad-style arm conditionally projects only the rescue direction
against the ERM velocity and then chooses the same target-hitting multiplier. They
are not canonical multi-objective Bloop or randomized multi-task-loss PCGrad.
There is no external parity test or canonical reference implementation. Method IDs
remain unchanged for historical compatibility, but publication labels cannot omit
“style.”

All shadow arms receive the same information and target, use full-batch SGD with no
weight decay or clipping, and report feasibility and cap status. ERM does not train
a weak-only shadow; that cost/information difference must remain visible.

### Frozen endpoint orientation

Weak rescue is

```text
R_w = -abs(weak-response AUC gap),
```

so larger is better and both under- and over-rescue are penalized. First-hit delay,
final weak response, target attainment, feasibility, and cap binding are also
reported.

Strong preservation is measured relative to the paired ERM trajectory at the exact
same task/data/model record:

```text
D_traj  = abs(strong-response AUC_method - strong-response AUC_ERM),
D_final = abs(M_s,method(H) - M_s,ERM(H)).
```

Smaller is better. Maximum absolute instantaneous strong-drift change is a separate
mechanistic endpoint. The instantaneous CDC theorem is not allowed to stand in for
`D_traj` or `D_final`.

`study_metrics.pareto_frontier` reports the empirical maximize-rescue/minimize-
deviation frontier, including ties. It does not itself establish sampling
uncertainty.

### Frozen “CDC beats” rule

For each style comparator, let the weak-rescue paired difference be CDC minus the
comparator. Its reuse-aware 95% interval must lie entirely inside `[-0.05,0.05]`.
For strong preservation, the paired advantage is comparator absolute deviation
minus CDC absolute deviation. Both the trajectory and final-response advantage
intervals must have strictly positive lower endpoints. Feasibility, cap, runtime,
and memory records must be complete.

Only the intersection of all those requirements permits “CDC beats this comparator
on the weak-rescue/strong-preservation tradeoff.” Failing any component yields no
win claim, even if CDC is on the empirical Pareto frontier. The frontier and all
component intervals are still reported.

`study_metrics.tradeoff_decision` implements the equivalence-plus-superiority
intersection for one strong endpoint; the frozen evaluator must apply it separately
to trajectory and final deviations and intersect both decisions.

## Inference and cost

All method contrasts are paired within task/data/model record. Because data and
model seeds are reused across the crossed factorial, uncertainty uses a two-way
pigeonhole bootstrap with 5,000 fixed-seed replicates. Cells are reported
separately; a cross-cell result is an equal-weight fixed-cell macro-average, not
independent replication. Holm correction applies within each primary family. No
architecture ranking is authorized.

Cumulative wall time already exists in trajectory logs. The future study must also
run each method/record in an isolated process and collect peak RSS; CUDA runs must
reset and record peak allocated memory. Static parameter bytes are secondary only.
Peak-memory telemetry is not currently implemented, so it remains an execution
blocker rather than an estimated result.

## Authorization blockers

Before any outcome, the study still requires:

1. a reviewed clean source state;
2. a committed machine-validated preregistration (no commit was authorized here);
3. an exact environment digest;
4. an initialization-only preflight manifest with an externally pinned digest;
5. an untouched-seed attestation;
6. implemented and tested isolated-process memory telemetry;
7. a reviewed claim-set transition allowing new outcomes.

The existing submission claim set explicitly says new confirmatory outcomes are not
allowed. Therefore the added work is limited to pure analysis utilities, semantic
baseline corrections, a frozen blocked design, and tests. It supplies no stronger
nonlinear or CDC empirical claim yet.
