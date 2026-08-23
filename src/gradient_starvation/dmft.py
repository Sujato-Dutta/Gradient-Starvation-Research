"""Deterministic solver scaffolding for the joint CE-RNN mean-field target.

Scope of this module
--------------------
This is **not** a DMFT solver.  The joint cross-entropy recurrent mean-field
theory of ``research_scope/e2_theorem.md`` is unproved, so the general solve is a
blocked surface that raises :class:`NotImplementedError` naming the specific unmet
proof obligation.  What is implemented here are two *exact special cases* that
require no part of that derivation and that can be validated against machinery the
repository already trusts:

``check A`` -- zero disorder
    At ``bulk_gain == 0`` and ``lag_separation == 0`` the projected dynamics close
    exactly on six scalars at every width.  See :func:`solve_zero_disorder`.

``check C`` -- frozen geometry
    With the projected metric held fixed, the mode dynamics reduce to a plain ODE
    driven by the cross-entropy field.  See :func:`solve_frozen_geometry`.

Independence
------------
The module imports only NumPy and the standard library.  It must never import
``training``, ``experiments`` or ``width_validation``, and must never read a file
under ``results/``.  A solver calibrated from trained networks cannot falsify the
theory it is compared against, so this independence is the property that gives the
special-case checks their meaning.  It is enforced by an AST test rather than by
convention: see ``tests/test_dmft.py``.

Determinism
-----------
Every routine is pure NumPy with no random draws, so identical inputs produce
bitwise-identical outputs.  Also asserted by test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

_OBLIGATIONS = "research_scope/e2_theorem.md"


def _blocked(summary: str, *obligations: int) -> NotImplementedError:
    """Build the standard blocked-surface error naming its unmet obligations."""
    numbers = ", ".join(str(number) for number in obligations)
    plural = "obligations" if len(obligations) > 1 else "obligation"
    return NotImplementedError(
        f"{summary} This is blocked on {plural} {numbers} of "
        f'{_OBLIGATIONS} § "Proof obligations". No placeholder result is returned, '
        "because a plausible-looking number here would be indistinguishable from a "
        "validated one."
    )


@dataclass(frozen=True)
class DMFTSpec:
    """Problem specification for a deterministic solve.

    ``frozen_geometry`` selects the check-C path when supplied.  ``bulk_gain == 0``
    with ``lag_separation == 0`` selects the check-A path.  Anything else is
    blocked.
    """

    sequence_length: int = 20
    tau_max: float = 5.0
    dtau: float = 0.01
    bulk_gain: float = 0.8
    rho: float = 2.0
    lag_separation: int = 0
    cue_noise: float = 0.0
    quadrature_size: int = 41
    tolerance: float = 1e-10
    max_iterations: int = 100
    frozen_geometry: tuple[tuple[float, float], tuple[float, float]] | None = None
    initial_mode: tuple[float, float] = (0.0, 0.0)
    # Wide-limit initial projected metric: B_ia ~ N(0,1/N) and c_i ~ N(0,1/N) give
    # unit input and readout norms with vanishing cross terms.
    initial_input_gram: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
    )
    initial_readout_norm_sq: float = 1.0

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
        if self.tau_max <= 0:
            raise ValueError("tau_max must be positive.")
        if self.dtau <= 0 or self.dtau > self.tau_max:
            raise ValueError("dtau must be positive and no larger than tau_max.")
        if self.bulk_gain < 0:
            raise ValueError("bulk_gain must be non-negative.")
        if self.rho <= 0:
            raise ValueError("rho must be positive.")
        if not 0 <= self.lag_separation < self.sequence_length:
            raise ValueError("lag_separation must be in [0, sequence_length).")
        if self.cue_noise < 0:
            raise ValueError("cue_noise must be non-negative.")
        if self.quadrature_size < 1:
            raise ValueError("quadrature_size must be at least 1.")


@dataclass
class DMFTSolution:
    """Result of a deterministic solve.

    ``covariance``, ``response`` and ``residual_history`` are declared because the
    joint theory requires them, and are deliberately left empty on the special-case
    paths.  An empty array is honest here: these kernels are only defined relative
    to the underdelivered closure, so fabricating values would misrepresent what
    was computed.  :attr:`path` records which branch produced the solution.
    """

    tau: np.ndarray
    m_s: np.ndarray
    m_w: np.ndarray
    geometry: np.ndarray
    path: str
    spec: DMFTSpec
    covariance: np.ndarray = field(default_factory=lambda: np.empty(0))
    response: np.ndarray = field(default_factory=lambda: np.empty(0))
    residual_history: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def modes(self) -> np.ndarray:
        """Return the mode trajectory as an ``(n_tau, 2)`` array."""
        return np.stack((self.m_s, self.m_w), axis=1)


# ---------------------------------------------------------------------------
# Cross-entropy field, evaluated from the specified cue law
# ---------------------------------------------------------------------------


def cue_quadrature(spec: DMFTSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Hermite nodes and weights for the positive-regime cue law.

    The synthetic task draws ``z_s ~ N(rho, (rho * cue_noise)^2)`` and
    ``z_w ~ N(1, cue_noise^2)`` independently.  Integrating against that analytic
    law -- rather than averaging a sampled batch -- is what keeps the solver
    independent of any finite network or dataset realisation.
    """
    if spec.cue_noise == 0:
        return np.array([[spec.rho, 1.0]]), np.array([1.0])
    nodes, weights = np.polynomial.hermite_e.hermegauss(spec.quadrature_size)
    weights = weights / weights.sum()
    z_s = spec.rho * (1.0 + spec.cue_noise * nodes)
    z_w = 1.0 + spec.cue_noise * nodes
    grid_s, grid_w = np.meshgrid(z_s, z_w, indexing="ij")
    weight_s, weight_w = np.meshgrid(weights, weights, indexing="ij")
    coordinates = np.stack((grid_s.ravel(), grid_w.ravel()), axis=1)
    return coordinates, (weight_s * weight_w).ravel()


def cross_entropy_field(
    mode: np.ndarray, coordinates: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return ``g_a = E[z_a sigma(-z . m)]`` under the quadrature measure."""
    margin = coordinates @ mode
    # 1/(1+exp(x)) with a clipped exponent; equals sigma(-x) without overflow.
    gate = 1.0 / (1.0 + np.exp(np.clip(margin, -60.0, 60.0)))
    return (coordinates * (weights * gate)[:, None]).sum(axis=0)


# ---------------------------------------------------------------------------
# Check A: zero disorder
# ---------------------------------------------------------------------------


def solve_zero_disorder(spec: DMFTSpec) -> DMFTSolution:
    """Exact projected dynamics at ``bulk_gain == 0`` and ``lag_separation == 0``.

    Why this closes without any mean-field argument.  With a zero recurrent
    initialization and both cues at the final step, every mode response is
    ``m_a = c . b_a`` with no dependence on the recurrent block, so
    ``dm_a/dW = 0``.  Since the synthetic logits depend on the parameters only
    through the two mode responses, the recurrent gradient vanishes identically and
    ``W`` stays exactly zero for the whole trajectory.  Gradient flow then reduces
    to

        db_a/dtau = g_a c,      dc/dtau = sum_b g_b b_b,

    which closes on six scalars -- the two modes ``m_a = c . b_a``, the input Gram
    ``u_ab = b_a . b_b`` and the readout norm ``w = c . c``:

        dm_a/dtau = g_a w + sum_b g_b u_ab
        du_ab/dtau = g_a m_b + g_b m_a
        dw/dtau    = 2 sum_b g_b m_b

    and the projected metric is exactly ``G_ab = u_ab + delta_ab w``, matching
    ``theory.exact_dense_linear_geometry`` term by term (``p_a . p_b = u_ab``,
    the same-channel ``q_a . q_b = w``, and ``S_a = 0``).

    This system is exact at *every* width, not only in the limit; width enters only
    through the initial conditions, which concentrate on ``u = I`` and ``w = 1``.

    Restricted to ``lag_separation == 0`` on purpose.  At any positive lag the
    recurrent block is no longer inert -- verified numerically -- so the projected
    system is not known to close on finitely many scalars without the derivation.
    """
    if spec.bulk_gain != 0:
        raise ValueError("solve_zero_disorder requires bulk_gain == 0.")
    if spec.lag_separation != 0:
        raise _blocked(
            "The zero-disorder reduction is exact only at lag_separation == 0, "
            "where the recurrent block is inert. At positive lag the recurrent "
            "block evolves and the projected system is not known to close on "
            "finitely many order parameters.",
            1,
            2,
        )

    coordinates, weights = cue_quadrature(spec)
    n_steps = int(round(spec.tau_max / spec.dtau))
    tau = np.linspace(0.0, n_steps * spec.dtau, n_steps + 1)

    gram = np.asarray(spec.initial_input_gram, dtype=float)
    if not np.allclose(gram, gram.T):
        raise ValueError("initial_input_gram must be symmetric.")

    def unpack(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        mode = state[:2]
        u = np.array([[state[2], state[3]], [state[3], state[4]]])
        return mode, u, float(state[5])

    def derivative(state: np.ndarray) -> np.ndarray:
        mode, u, readout = unpack(state)
        g = cross_entropy_field(mode, coordinates, weights)
        d_mode = g * readout + u @ g
        d_u = np.outer(g, mode) + np.outer(mode, g)
        d_readout = 2.0 * float(g @ mode)
        return np.array([d_mode[0], d_mode[1], d_u[0, 0], d_u[0, 1], d_u[1, 1], d_readout])

    state = np.array(
        [
            spec.initial_mode[0], spec.initial_mode[1],
            gram[0, 0], gram[0, 1], gram[1, 1],
            float(spec.initial_readout_norm_sq),
        ],
        dtype=float,
    )
    modes = np.zeros((n_steps + 1, 2))
    geometry = np.zeros((n_steps + 1, 2, 2))
    for index in range(n_steps + 1):
        mode, u, readout = unpack(state)
        modes[index] = mode
        geometry[index] = u + readout * np.eye(2)
        if index == n_steps:
            break
        dt = spec.dtau
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return DMFTSolution(
        tau=tau, m_s=modes[:, 0], m_w=modes[:, 1], geometry=geometry,
        path="zero_disorder", spec=spec,
    )


# ---------------------------------------------------------------------------
# Check C: frozen geometry
# ---------------------------------------------------------------------------


def solve_frozen_geometry(spec: DMFTSpec) -> DMFTSolution:
    """Mode dynamics with the projected metric held fixed.

    Isolates cross-entropy margin gating from representation adaptation: with ``G``
    constant, every change in the drift comes from the field ``g(m)``.  Integrates

        dm_a/dtau = sum_b G_ab g_b(m)

    by classical RK4, which is the same scheme ``theory.integrate_projected_flow``
    applies to an externally supplied geometry trajectory, so the two agree when
    that trajectory is constant.
    """
    if spec.frozen_geometry is None:
        raise ValueError("solve_frozen_geometry requires spec.frozen_geometry.")
    geometry_matrix = np.asarray(spec.frozen_geometry, dtype=float)
    if geometry_matrix.shape != (2, 2):
        raise ValueError("frozen_geometry must be a 2x2 matrix.")
    if not np.allclose(geometry_matrix, geometry_matrix.T):
        raise ValueError("frozen_geometry must be symmetric.")

    coordinates, weights = cue_quadrature(spec)
    n_steps = int(round(spec.tau_max / spec.dtau))
    tau = np.linspace(0.0, n_steps * spec.dtau, n_steps + 1)

    def derivative(mode: np.ndarray) -> np.ndarray:
        return geometry_matrix @ cross_entropy_field(mode, coordinates, weights)

    modes = np.zeros((n_steps + 1, 2))
    modes[0] = np.asarray(spec.initial_mode, dtype=float)
    for index in range(1, n_steps + 1):
        dt = spec.dtau
        mode = modes[index - 1]
        k1 = derivative(mode)
        k2 = derivative(mode + 0.5 * dt * k1)
        k3 = derivative(mode + 0.5 * dt * k2)
        k4 = derivative(mode + dt * k3)
        modes[index] = mode + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return DMFTSolution(
        tau=tau, m_s=modes[:, 0], m_w=modes[:, 1],
        geometry=np.repeat(geometry_matrix[None, :, :], n_steps + 1, axis=0),
        path="frozen_geometry", spec=spec,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def solve_dmft(
    spec: DMFTSpec, *, objective: Literal["cross_entropy", "mse"] = "cross_entropy"
) -> DMFTSolution:
    """Dispatch to an implemented special case, or refuse.

    ``frozen_geometry`` supplied
        Check C, :func:`solve_frozen_geometry`.
    ``bulk_gain == 0`` and ``lag_separation == 0``
        Check A, :func:`solve_zero_disorder`.
    anything else
        :class:`NotImplementedError`.  The general solve needs the effective
        single-site process and the enumerated closure set, neither of which has
        been derived.
    """
    if objective == "mse":
        raise _blocked(
            "An MSE variant of the mean-field equations cannot be written before "
            "the cross-entropy equations themselves exist. The MSE *objective* is "
            "implemented for finite networks in losses.py; only the solver branch "
            "is blocked.",
            1,
            2,
        )
    if objective != "cross_entropy":
        raise ValueError(f"Unknown objective: {objective!r}")
    if spec.frozen_geometry is not None:
        return solve_frozen_geometry(spec)
    if spec.bulk_gain == 0 and spec.lag_separation == 0:
        return solve_zero_disorder(spec)
    raise _blocked(
        "The general joint cross-entropy recurrent mean-field solve is not "
        "available: the effective single-site process and its covariance/response "
        "kernels have not been derived, and the closure set of visible and "
        "loss-invisible order parameters has not been enumerated. Implemented "
        "special cases are bulk_gain == 0 with lag_separation == 0 (zero disorder) "
        "and a supplied frozen_geometry.",
        1,
        2,
    )


# ---------------------------------------------------------------------------
# Blocked surfaces
# ---------------------------------------------------------------------------


def solve_weak_only_reduction(spec: DMFTSpec) -> DMFTSolution:
    """Check B. Blocked.

    Reducing the joint both/weak-only system to a weak-only mean-field theory
    presupposes the covariance and response kernels of that theory, so there is
    nothing to reduce *to*.  Imposing ``bulk_gain == 0`` as well would collapse the
    check into check A, so it would not be an independent test either way.
    """
    raise _blocked(
        "The weak-only reduction has no target: a weak-only mean-field theory is "
        "itself defined by the covariance and response kernels that have not been "
        "derived. Note also that imposing bulk_gain == 0 would collapse this into "
        "check A rather than providing an independent test.",
        1,
        2,
    )


def self_consistency_residual(solution: DMFTSolution) -> np.ndarray:
    """Check E. Blocked.

    A self-consistency residual is defined only relative to derived
    self-consistency equations.  Reporting a residual that "converges" for a system
    whose equations do not exist would be a false pass, which is precisely the
    failure this project's interpretation rules exist to prevent.
    """
    raise _blocked(
        "A self-consistency residual is only defined relative to derived "
        "self-consistency equations. Until the closed Volterra/ODE system exists "
        "and is shown to have a unique finite-horizon solution, a converging "
        "residual would be a false pass rather than evidence.",
        2,
        3,
    )


def refine_until_converged(spec: DMFTSpec) -> DMFTSolution:
    """Check E, quadrature and step refinement. Blocked for the same reason."""
    raise _blocked(
        "Internal convergence refinement reports self-consistency residuals, which "
        "presuppose derived self-consistency equations and a finite-horizon "
        "existence/uniqueness result.",
        2,
        3,
    )
