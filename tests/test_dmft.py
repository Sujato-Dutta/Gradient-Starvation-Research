"""Solver scaffolding: the two exact special cases, and the blocked surfaces.

The blocked-surface tests assert that a ``NotImplementedError`` is *raised*.  They
are deliberately not ``skip`` or ``xfail``: the block is a tested contract, so if
somebody later inserts a placeholder that returns numbers, these tests fail loudly
rather than quietly starting to pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from gradient_starvation.data.synthetic import SyntheticTaskSpec, make_paired_task
from gradient_starvation.dmft import (
    DMFTSpec,
    cross_entropy_field,
    cue_quadrature,
    refine_until_converged,
    self_consistency_residual,
    solve_dmft,
    solve_frozen_geometry,
    solve_weak_only_reduction,
    solve_zero_disorder,
)
from gradient_starvation.losses import base_objective, training_objective
from gradient_starvation.models.recurrent import DenseLinearRNN
from gradient_starvation.theory import (
    exact_dense_linear_geometry,
    gradient_gram,
    integrate_projected_flow,
)

DMFT_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "gradient_starvation" / "dmft.py"
)
FORBIDDEN_MODULES = ("training", "experiments", "width_validation", "waterbirds")


# ---------------------------------------------------------------------------
# Independence and determinism
# ---------------------------------------------------------------------------


def test_solver_module_imports_nothing_from_the_finite_network_stack():
    """Enforced by AST inspection, not by convention.

    A solver that imported the training stack could be calibrated from trained
    networks, and would then be unable to falsify the theory it is compared with.
    """
    tree = ast.parse(DMFT_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            if node.level:  # relative import, e.g. `from .training import ...`
                imported.add((node.module or "").split(".")[-1])
    for forbidden in FORBIDDEN_MODULES:
        offenders = {name for name in imported if forbidden in name}
        assert not offenders, f"dmft.py must not import {forbidden}: {offenders}"


def test_solver_module_performs_no_file_or_network_reads():
    """Inspect executable AST nodes, not prose.

    Checking raw source text would trip over this module's own docstrings, which
    legitimately discuss ``results/`` when explaining why the solver must not read
    it.  Only calls and non-docstring literals are inspected.
    """
    tree = ast.parse(DMFT_SOURCE.read_text(encoding="utf-8"))
    forbidden_calls = {
        "open", "read_csv", "read_json", "loadtxt", "genfromtxt", "load",
        "read_text", "read_bytes", "urlopen", "get", "glob", "iterdir",
    }
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                called.add(function.id)
            elif isinstance(function, ast.Attribute):
                called.add(function.attr)
    assert not (called & forbidden_calls), called & forbidden_calls

    # Docstrings are the first statement of a module, class or function body.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            candidate = node.body[0] if node.body else None
            if isinstance(candidate, ast.Expr) and isinstance(candidate.value, ast.Constant):
                if isinstance(candidate.value.value, str):
                    docstrings.add(id(candidate.value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            assert "results/" not in node.value, node.value


@pytest.mark.parametrize(
    "spec",
    [
        DMFTSpec(bulk_gain=0.0, lag_separation=0, tau_max=0.5, dtau=0.05, cue_noise=0.1),
        DMFTSpec(frozen_geometry=((2.0, 0.1), (0.1, 1.5)), tau_max=0.5, dtau=0.05),
    ],
    ids=["zero_disorder", "frozen_geometry"],
)
def test_solutions_are_bitwise_reproducible(spec):
    first, second = solve_dmft(spec), solve_dmft(spec)
    assert np.array_equal(first.m_s, second.m_s)
    assert np.array_equal(first.m_w, second.m_w)
    assert np.array_equal(first.geometry, second.geometry)


def test_unpopulated_kernels_are_empty_not_fabricated():
    """The special-case paths must not invent covariance or response kernels."""
    solution = solve_dmft(DMFTSpec(bulk_gain=0.0, lag_separation=0, tau_max=0.2, dtau=0.05))
    assert solution.covariance.size == 0
    assert solution.response.size == 0
    assert solution.residual_history.size == 0
    assert solution.path == "zero_disorder"


# ---------------------------------------------------------------------------
# Check A: zero disorder
# ---------------------------------------------------------------------------


def _zero_disorder_discrepancy(learning_rate: float, tau_max: float = 2.0) -> float:
    """Max relative gap between the solver and a directly simulated network."""
    steps = int(round(tau_max / learning_rate))
    task = SyntheticTaskSpec(
        sequence_length=5, n_samples=4096, rho=2.0, lag_separation=0, cue_noise=0.0
    )
    both, _ = make_paired_task(task, seed=0)
    torch.manual_seed(4)
    model = DenseLinearRNN(width=64, bulk_gain=0.0)
    with torch.no_grad():
        b_s = model.input[:, 0].clone()
        b_w = model.input[:, 1].clone()
        c = model.readout.clone()
    solution = solve_zero_disorder(
        DMFTSpec(
            sequence_length=5, bulk_gain=0.0, lag_separation=0, rho=2.0, cue_noise=0.0,
            tau_max=tau_max, dtau=min(learning_rate / 50, 1e-3),
            initial_mode=(float(c @ b_s), float(c @ b_w)),
            initial_input_gram=(
                (float(b_s @ b_s), float(b_s @ b_w)),
                (float(b_s @ b_w), float(b_w @ b_w)),
            ),
            initial_readout_norm_sq=float(c @ c),
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    observed = []
    for _ in range(steps + 1):
        observed.append(model.mode_responses(task).detach().numpy().copy())
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_objective(model, both, {"method": "erm"})
        loss.backward()
        optimizer.step()
    # The premise of the reduction: the recurrent block never moves.
    assert float(model.recurrent.detach().abs().max()) == 0.0
    observed = np.array(observed)
    grid = np.arange(steps + 1) * learning_rate
    predicted = np.stack(
        [np.interp(grid, solution.tau, solution.m_s),
         np.interp(grid, solution.tau, solution.m_w)], axis=1
    )
    return float(np.abs(observed - predicted).max() / max(np.abs(observed).max(), 1e-8))


def test_check_a_is_the_correct_continuous_limit_of_the_finite_network():
    """The residual gap must be the O(eta) discretization error, and nothing else.

    The solver integrates gradient *flow*; the network takes discrete SGD steps.
    Those differ at first order in the learning rate, so an absolute tolerance would
    either be vacuous or fail for the wrong reason.  The meaningful check is the
    scaling: if the solver is the correct continuous limit, the discrepancy vanishes
    linearly in eta.  Measured slope is ~1.01 with error/eta ~ 0.24 across a 16x
    range of eta.

    This is an empirical measurement of the discretization gap, not the proved bound
    that obligation 5 of research_scope/e2_theorem.md calls for.
    """
    learning_rates = [0.08, 0.04, 0.02, 0.01]
    errors = [_zero_disorder_discrepancy(rate) for rate in learning_rates]

    assert all(np.isfinite(errors))
    # Monotone in eta, and first-order.
    assert errors == sorted(errors, reverse=True)
    slope = float(np.polyfit(np.log(learning_rates), np.log(errors), 1)[0])
    assert 0.9 <= slope <= 1.1, f"expected first-order scaling, fitted slope {slope}"
    # The ratio error/eta must be stable, confirming a single constant governs it.
    ratios = np.array(errors) / np.array(learning_rates)
    assert ratios.max() / ratios.min() < 1.1
    # And the finest step must actually be accurate.
    assert errors[-1] < 5e-3


def test_check_a_matches_the_network_exactly_at_initialization():
    """Seeding from the network's own inner products must be exact at tau = 0."""
    task = SyntheticTaskSpec(
        sequence_length=5, n_samples=256, rho=2.0, lag_separation=0, cue_noise=0.0
    )
    torch.manual_seed(4)
    model = DenseLinearRNN(width=64, bulk_gain=0.0)
    with torch.no_grad():
        b_s = model.input[:, 0].clone()
        b_w = model.input[:, 1].clone()
        c = model.readout.clone()
    solution = solve_zero_disorder(
        DMFTSpec(
            sequence_length=5, bulk_gain=0.0, lag_separation=0, rho=2.0, cue_noise=0.0,
            tau_max=0.1, dtau=0.05,
            initial_mode=(float(c @ b_s), float(c @ b_w)),
            initial_input_gram=(
                (float(b_s @ b_s), float(b_s @ b_w)),
                (float(b_s @ b_w), float(b_w @ b_w)),
            ),
            initial_readout_norm_sq=float(c @ c),
        )
    )
    observed = model.mode_responses(task).detach().numpy()
    assert abs(solution.m_s[0] - observed[0]) < 1e-6
    assert abs(solution.m_w[0] - observed[1]) < 1e-6


def test_check_a_geometry_matches_the_exact_analytic_gram():
    """``G_ab = u_ab + delta_ab w`` must equal theory.exact_dense_linear_geometry."""
    torch.manual_seed(7)
    task = SyntheticTaskSpec(
        sequence_length=4, n_samples=64, rho=2.0, lag_separation=0, cue_noise=0.0
    )
    model = DenseLinearRNN(width=48, bulk_gain=0.0)
    autograd, _ = gradient_gram(model.mode_responses(task), model)
    analytic = exact_dense_linear_geometry(model, task)

    with torch.no_grad():
        b_s = model.input[:, 0].clone()
        b_w = model.input[:, 1].clone()
        c = model.readout.clone()
    spec = DMFTSpec(
        sequence_length=4, bulk_gain=0.0, lag_separation=0, tau_max=0.1, dtau=0.05,
        initial_mode=(float(c @ b_s), float(c @ b_w)),
        initial_input_gram=(
            (float(b_s @ b_s), float(b_s @ b_w)),
            (float(b_s @ b_w), float(b_w @ b_w)),
        ),
        initial_readout_norm_sq=float(c @ c),
    )
    solver_geometry = solve_zero_disorder(spec).geometry[0]

    reference = analytic.detach().numpy()
    assert np.abs(solver_geometry - reference).max() / np.abs(reference).max() < 1e-6
    torch.testing.assert_close(autograd, analytic, rtol=2e-5, atol=2e-6)


def test_check_a_refuses_positive_lag_because_the_recurrent_block_is_not_inert():
    with pytest.raises(NotImplementedError, match="lag_separation == 0"):
        solve_zero_disorder(DMFTSpec(bulk_gain=0.0, lag_separation=2))


def test_check_a_requires_zero_bulk_gain():
    with pytest.raises(ValueError, match="bulk_gain == 0"):
        solve_zero_disorder(DMFTSpec(bulk_gain=0.5, lag_separation=0))


# ---------------------------------------------------------------------------
# Check C: frozen geometry
# ---------------------------------------------------------------------------


def test_check_c_matches_integrate_projected_flow_with_constant_geometry():
    """Both integrate dm/dtau = G g by RK4, so a constant G must agree."""
    geometry = ((1.8, 0.2), (0.2, 1.1))
    spec = DMFTSpec(
        frozen_geometry=geometry, rho=2.0, cue_noise=0.1, tau_max=2.0, dtau=0.005,
        quadrature_size=61,
    )
    solution = solve_frozen_geometry(spec)

    coordinates, weights = cue_quadrature(spec)
    # `integrate_projected_flow` averages uniformly, so replicate the quadrature
    # measure by weighting the coordinate list into a matched empirical sample.
    counts = np.maximum((weights / weights.max() * 4000).astype(int), 1)
    empirical = np.repeat(coordinates, counts, axis=0)
    matrix = np.asarray(geometry, dtype=float)
    reference = integrate_projected_flow(
        np.zeros(2), empirical, solution.tau,
        np.repeat(matrix[None, :, :], len(solution.tau), axis=0),
    )
    scale = max(float(np.abs(reference).max()), 1e-8)
    assert np.abs(solution.modes - reference).max() / scale < 5e-3
    assert np.allclose(solution.geometry, matrix)


def test_check_c_isolates_margin_gating_by_holding_geometry_fixed():
    """With G fixed, growth must decelerate purely through the margin gate."""
    spec = DMFTSpec(
        frozen_geometry=((1.0, 0.0), (0.0, 1.0)), rho=4.0, cue_noise=0.0,
        tau_max=6.0, dtau=0.01,
    )
    solution = solve_frozen_geometry(spec)
    increments = np.diff(solution.m_s)
    assert (increments > 0).all()          # monotone rise
    assert increments[-1] < increments[0]  # gated, so decelerating
    assert np.allclose(solution.geometry[0], solution.geometry[-1])


def test_check_c_validates_its_geometry_argument():
    with pytest.raises(ValueError, match="frozen_geometry"):
        solve_frozen_geometry(DMFTSpec())
    with pytest.raises(ValueError, match="symmetric"):
        solve_frozen_geometry(DMFTSpec(frozen_geometry=((1.0, 0.3), (0.9, 1.0))))


def test_quadrature_reduces_to_a_point_mass_without_cue_noise():
    coordinates, weights = cue_quadrature(DMFTSpec(rho=3.0, cue_noise=0.0))
    assert coordinates.shape == (1, 2)
    assert coordinates[0].tolist() == [3.0, 1.0]
    assert weights.tolist() == [1.0]


def test_cross_entropy_field_is_positive_and_decreasing_in_the_margin():
    coordinates, weights = cue_quadrature(DMFTSpec(rho=2.0, cue_noise=0.1))
    early = cross_entropy_field(np.zeros(2), coordinates, weights)
    late = cross_entropy_field(np.array([3.0, 3.0]), coordinates, weights)
    assert (early > 0).all()
    assert (late < early).all()


# ---------------------------------------------------------------------------
# Blocked surfaces: the raise is the contract
# ---------------------------------------------------------------------------


def test_general_solve_is_blocked_naming_obligations_one_and_two():
    with pytest.raises(NotImplementedError) as excinfo:
        solve_dmft(DMFTSpec(bulk_gain=0.8, lag_separation=2))
    message = str(excinfo.value)
    assert "research_scope/e2_theorem.md" in message
    assert "obligations 1, 2" in message
    assert "Proof obligations" in message


def test_check_b_weak_only_reduction_is_blocked():
    with pytest.raises(NotImplementedError) as excinfo:
        solve_weak_only_reduction(DMFTSpec())
    assert "obligations 1, 2" in str(excinfo.value)
    assert "nothing to reduce" in str(excinfo.value) or "has no target" in str(excinfo.value)


def test_check_d_mse_solver_branch_is_blocked_but_the_objective_is_not():
    with pytest.raises(NotImplementedError) as excinfo:
        solve_dmft(DMFTSpec(bulk_gain=0.0, lag_separation=0), objective="mse")
    assert "obligations 1, 2" in str(excinfo.value)

    # The finite-network MSE objective is real, and must actually work.
    task = SyntheticTaskSpec(sequence_length=4, n_samples=32, rho=2.0, lag_separation=1)
    both, _ = make_paired_task(task, seed=1)
    torch.manual_seed(0)
    model = DenseLinearRNN(width=8, bulk_gain=0.3)
    _, mse = base_objective(model, both, "mse")
    _, ce = base_objective(model, both, "cross_entropy")
    assert torch.isfinite(mse) and mse.item() > 0
    assert not torch.isclose(mse, ce)


def test_check_e_residual_and_refinement_are_blocked():
    solution = solve_dmft(DMFTSpec(bulk_gain=0.0, lag_separation=0, tau_max=0.2, dtau=0.1))
    with pytest.raises(NotImplementedError) as excinfo:
        self_consistency_residual(solution)
    assert "obligations 2, 3" in str(excinfo.value)
    with pytest.raises(NotImplementedError):
        refine_until_converged(DMFTSpec())


def test_unknown_objective_is_a_value_error_not_a_block():
    with pytest.raises(ValueError, match="Unknown objective"):
        solve_dmft(DMFTSpec(bulk_gain=0.0, lag_separation=0), objective="hinge")


# ---------------------------------------------------------------------------
# MSE objective
# ---------------------------------------------------------------------------


def test_mse_objective_trains_and_is_selectable_through_the_mitigation_mapping():
    task = SyntheticTaskSpec(sequence_length=4, n_samples=64, rho=2.0, lag_separation=1)
    both, _ = make_paired_task(task, seed=2)
    torch.manual_seed(3)
    model = DenseLinearRNN(width=12, bulk_gain=0.3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    first = None
    for step in range(15):
        optimizer.zero_grad(set_to_none=True)
        loss, info = training_objective(model, both, {"method": "erm", "objective": "mse"})
        assert info["objective"] == "mse"
        if step == 0:
            first = loss.item()
        loss.backward()
        optimizer.step()
    assert loss.item() < first


def test_unknown_base_objective_is_rejected():
    task = SyntheticTaskSpec(sequence_length=4, n_samples=16, rho=2.0, lag_separation=1)
    both, _ = make_paired_task(task, seed=3)
    model = DenseLinearRNN(width=6, bulk_gain=0.3)
    with pytest.raises(ValueError, match="Unknown objective"):
        base_objective(model, both, "huber")
