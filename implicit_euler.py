"""Implicit Euler integration for first-order ordinary differential equations.

The method advances ``y' = f(t, y)`` by solving the nonlinear equation
``y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}) = 0`` at every time step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple


VectorField = Callable[[float, float], float]
Jacobian = Callable[[float, float], float]


class IntegrationResult(NamedTuple):
    """Time grid and solution values returned by :func:`implicit_euler`."""

    times: list[float]
    values: list[float]


def implicit_euler(
    f: VectorField,
    t0: float,
    y0: float,
    step_size: float,
    steps: int,
    *,
    jacobian: Jacobian | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 20,
) -> IntegrationResult:
    """Solve a scalar ODE with the implicit (backward) Euler method.

    Args:
        f: Right-hand side of ``y' = f(t, y)``.
        t0: Initial time.
        y0: Initial value.
        step_size: Positive integration step ``h``.
        steps: Number of steps to take.
        jacobian: Optional derivative ``df/dy``.  If omitted, a centered
            finite-difference approximation is used.
        tolerance: Absolute Newton residual tolerance.
        max_iterations: Maximum Newton iterations per time step.

    Raises:
        ValueError: If integration or Newton parameters are invalid.
        RuntimeError: If Newton's method does not converge.
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    times = [t0]
    values = [y0]
    for step in range(steps):
        t_next = t0 + (step + 1) * step_size
        y_previous = values[-1]
        # The previous value is a robust initial guess for sufficiently small h.
        y_next = y_previous

        for _ in range(max_iterations):
            residual = y_next - y_previous - step_size * f(t_next, y_next)
            if abs(residual) <= tolerance:
                break

            if jacobian is None:
                delta = 1e-7 * max(1.0, abs(y_next))
                derivative_f = (
                    f(t_next, y_next + delta) - f(t_next, y_next - delta)
                ) / (2.0 * delta)
            else:
                derivative_f = jacobian(t_next, y_next)

            derivative = 1.0 - step_size * derivative_f
            if derivative == 0.0:
                raise RuntimeError(
                    f"Newton derivative vanished at step {step + 1} (t={t_next})"
                )
            y_next -= residual / derivative
        else:
            final_residual = y_next - y_previous - step_size * f(t_next, y_next)
            if abs(final_residual) > tolerance:
                raise RuntimeError(
                    f"Newton's method did not converge at step {step + 1} (t={t_next})"
                )

        times.append(t_next)
        values.append(y_next)

    return IntegrationResult(times, values)


if __name__ == "__main__":
    # Stiff test problem: y' = -15y, y(0) = 1.
    result = implicit_euler(
        lambda _t, y: -15.0 * y,
        t0=0.0,
        y0=1.0,
        step_size=0.1,
        steps=10,
        jacobian=lambda _t, _y: -15.0,
    )
    for t, y in zip(result.times, result.values):
        print(f"t={t:.1f}, y={y:.8f}")
