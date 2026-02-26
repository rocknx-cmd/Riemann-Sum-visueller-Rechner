"""
Quadrature module: definite integral approximation via Q_n = Σ w_i f(x_i).

Implements five methods with explicit node and weight structure.
All computations use NumPy vectorization over the subintervals/nodes.
"""

from typing import Callable

import numpy as np

from src.types import QuadratureMethod, QuadratureResult


def _uniform_nodes(a: float, b: float, n: int) -> np.ndarray:
    """Subinterval boundaries: x_0 = a, x_1, ..., x_n = b (n+1 points)."""
    return np.linspace(a, b, n + 1, dtype=float)


def _left_riemann_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Left Riemann sum: nodes x_i = x_{i-1} (left endpoints), weight = h per node.
    Q_n = h * Σ_{i=0}^{n-1} f(x_i), with x_i = a + i*h, h = (b-a)/n.
    """
    h = (b - a) / n
    nodes = a + np.arange(n, dtype=float) * h
    weights = np.full(n, h)
    return nodes, weights


def _right_riemann_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Right Riemann sum: nodes x_i = x_i (right endpoints), weight = h per node.
    Q_n = h * Σ_{i=1}^{n} f(x_i), with x_i = a + i*h.
    """
    h = (b - a) / n
    nodes = a + np.arange(1, n + 1, dtype=float) * h
    weights = np.full(n, h)
    return nodes, weights


def _midpoint_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Midpoint rule: nodes at subinterval midpoints, weight = h each.
    x_i = a + (i + 1/2)*h, Q_n = h * Σ_{i=0}^{n-1} f(x_i). Order p = 2.
    """
    h = (b - a) / n
    nodes = a + (np.arange(n, dtype=float) + 0.5) * h
    weights = np.full(n, h)
    return nodes, weights


def _trapezoidal_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Trapezoidal rule: nodes at subinterval endpoints (n+1 points), with
    weights h/2 at endpoints and h at interior nodes (composite: h/2, h, ..., h, h/2).
    Q_n = (h/2)*(f(x_0) + 2*Σ_{i=1}^{n-1} f(x_i) + f(x_n)). Order p = 2.
    """
    nodes = _uniform_nodes(a, b, n)
    h = (b - a) / n
    weights = np.full(n + 1, h)
    weights[0] = weights[-1] = h / 2
    return nodes, weights


def _simpson_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simpson's rule: requires n even. Nodes at x_0,...,x_n (n+1 points).
    Weights: h/3 * (1, 4, 2, 4, ..., 2, 4, 1). Order p = 4.
    """
    if n % 2 != 0:
        n = n + 1  # ensure even for Simpson
    nodes = _uniform_nodes(a, b, n)
    h = (b - a) / n
    weights = np.empty(n + 1)
    weights[0] = weights[-1] = h / 3
    weights[1:-1:2] = 4 * h / 3
    weights[2:-1:2] = 2 * h / 3
    return nodes, weights


def _compute_quadrature(
    f: Callable[[np.ndarray], np.ndarray],
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Evaluate Q_n = Σ w_i f(x_i) in a vectorized way."""
    f_vals = np.asarray(f(nodes), dtype=float)
    return float(np.dot(weights, f_vals))


def compute_quadrature(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    method: QuadratureMethod,
    *,
    return_nodes_weights: bool = False,
) -> QuadratureResult:
    """
    Compute the definite integral approximation ∫_a^b f(x) dx using the chosen method.

    Implements the general quadrature formula Q_n = Σ_i w_i f(x_i) with
    method-specific nodes and weights. All evaluations of f are vectorized.

    Args:
        f: Vectorized function (accepts array, returns array).
        a, b: Integration interval [a, b].
        n: Number of subintervals (for Simpson, n is forced to be even).
        method: Quadrature method enum.
        return_nodes_weights: If True, attach nodes and weights to the result.

    Returns:
        QuadratureResult with approximation and optional nodes/weights.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if a >= b:
        raise ValueError("Interval must satisfy a < b.")

    dispatcher = {
        QuadratureMethod.LEFT_RIEMANN: _left_riemann_nodes_weights,
        QuadratureMethod.RIGHT_RIEMANN: _right_riemann_nodes_weights,
        QuadratureMethod.MIDPOINT: _midpoint_nodes_weights,
        QuadratureMethod.TRAPEZOIDAL: _trapezoidal_nodes_weights,
        QuadratureMethod.SIMPSON: _simpson_nodes_weights,
    }
    # Simpson requires even n; others use n as-is
    actual_n = (n if n % 2 == 0 else n + 1) if method == QuadratureMethod.SIMPSON else n
    get_nw = dispatcher[method]
    nodes, weights = get_nw(a, b, actual_n)
    approximation = _compute_quadrature(f, nodes, weights)

    out_nodes = tuple(float(x) for x in nodes) if return_nodes_weights else None
    out_weights = tuple(float(w) for w in weights) if return_nodes_weights else None

    return QuadratureResult(
        method=method,
        n=actual_n,
        a=a,
        b=b,
        approximation=approximation,
        nodes=out_nodes,
        weights=out_weights,
    )


def get_nodes_weights_for_plot(
    a: float, b: float, n: int, method: QuadratureMethod
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (nodes, weights) for the given method and n (for visualization).
    Simpson: n is adjusted to be even.
    """
    if method == QuadratureMethod.SIMPSON and n % 2 != 0:
        n = n + 1
    dispatcher = {
        QuadratureMethod.LEFT_RIEMANN: _left_riemann_nodes_weights,
        QuadratureMethod.RIGHT_RIEMANN: _right_riemann_nodes_weights,
        QuadratureMethod.MIDPOINT: _midpoint_nodes_weights,
        QuadratureMethod.TRAPEZOIDAL: _trapezoidal_nodes_weights,
        QuadratureMethod.SIMPSON: _simpson_nodes_weights,
    }
    return dispatcher[method](a, b, n)
