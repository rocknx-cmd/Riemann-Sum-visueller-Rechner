"""
Convergence analysis: sequence of approximations and Richardson error estimate.

Computes Q_n for n = n0, 2*n0, 4*n0, 8*n0, ... and builds a table of
n, Q_n, and |Q_n - Q_previous|. Also computes the Richardson estimate
E_est = (Q_2n - Q_n) / (2^p - 1) for methods of order p.
"""

from typing import Callable

import numpy as np

from src.quadrature import compute_quadrature
from src.types import (
    ConvergenceRow,
    ConvergenceTable,
    QuadratureMethod,
    RichardsonEstimate,
)


def build_convergence_table(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    initial_n: int,
    method: QuadratureMethod,
    num_doublings: int = 5,
) -> ConvergenceTable:
    """
    Compute approximations for n = n0, 2*n0, 4*n0, ... and record changes.

    For each n, computes Q_n and appends a row with n, Q_n, and
    |Q_n - Q_previous| (None for the first row).
    """
    table = ConvergenceTable(method=method, initial_n=initial_n)
    prev_q: float | None = None
    n = initial_n
    for _ in range(num_doublings):
        result = compute_quadrature(f, a, b, n, method)
        q_n = result.approximation
        change = abs(q_n - prev_q) if prev_q is not None else None
        table.add_row(n=n, q_n=q_n, change=change)
        prev_q = q_n
        n *= 2
    return table


def richardson_estimate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    method: QuadratureMethod,
) -> RichardsonEstimate:
    """
    Richardson error estimate for a method of order p:

        E_est = (Q_2n - Q_n) / (2^p - 1).

    For Trapezoidal and Midpoint p = 2; for Left/Right Riemann p = 1.
    """
    res_n = compute_quadrature(f, a, b, n, method)
    res_2n = compute_quadrature(f, a, b, 2 * n, method)
    factor = method.richardson_factor()
    estimated_error = (res_2n.approximation - res_n.approximation) / factor
    return RichardsonEstimate(
        method=method,
        n=n,
        q_n=res_n.approximation,
        q_2n=res_2n.approximation,
        estimated_error=estimated_error,
        order=method.order,
    )
