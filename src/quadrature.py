# quadrature: links/rechts/mittelpunkt/trapez, alle vectorized

from typing import Callable

import numpy as np

from src.types import QuadratureMethod, QuadratureResult


def _uniform_nodes(a: float, b: float, n: int) -> np.ndarray:
    return np.linspace(a, b, n + 1, dtype=float)


def _left_riemann_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    # links: x_i links, gewicht h
    h = (b - a) / n
    nodes = a + np.arange(n, dtype=float) * h
    weights = np.full(n, h)
    return nodes, weights


def _right_riemann_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    # rechts: x_i rechts, gewicht h
    h = (b - a) / n
    nodes = a + np.arange(1, n + 1, dtype=float) * h
    weights = np.full(n, h)
    return nodes, weights


def _midpoint_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    # mittelpunkt pro intervall
    h = (b - a) / n
    nodes = a + (np.arange(n, dtype=float) + 0.5) * h
    weights = np.full(n, h)
    return nodes, weights


def _trapezoidal_nodes_weights(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    # trapez: randpunkte, gewichte h/2 am rand
    nodes = _uniform_nodes(a, b, n)
    h = (b - a) / n
    weights = np.full(n + 1, h)
    weights[0] = weights[-1] = h / 2
    return nodes, weights


def _compute_quadrature(
    f: Callable[[np.ndarray], np.ndarray],
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    # Q_n = sum w_i * f(x_i)
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
    # näherung für int_a^b f(x) dx
    if n < 1:
        raise ValueError("n must be at least 1.")
    if a >= b:
        raise ValueError("Interval must satisfy a < b.")

    dispatcher = {
        QuadratureMethod.LEFT_RIEMANN: _left_riemann_nodes_weights,
        QuadratureMethod.RIGHT_RIEMANN: _right_riemann_nodes_weights,
        QuadratureMethod.MIDPOINT: _midpoint_nodes_weights,
        QuadratureMethod.TRAPEZOIDAL: _trapezoidal_nodes_weights,
    }
    get_nw = dispatcher[method]
    nodes, weights = get_nw(a, b, n)
    approximation = _compute_quadrature(f, nodes, weights)

    out_nodes = tuple(float(x) for x in nodes) if return_nodes_weights else None
    out_weights = tuple(float(w) for w in weights) if return_nodes_weights else None

    return QuadratureResult(
        method=method,
        n=n,
        a=a,
        b=b,
        approximation=approximation,
        nodes=out_nodes,
        weights=out_weights,
    )


def get_nodes_weights_for_plot(
    a: float, b: float, n: int, method: QuadratureMethod
) -> tuple[np.ndarray, np.ndarray]:
    # für plot
    dispatcher = {
        QuadratureMethod.LEFT_RIEMANN: _left_riemann_nodes_weights,
        QuadratureMethod.RIGHT_RIEMANN: _right_riemann_nodes_weights,
        QuadratureMethod.MIDPOINT: _midpoint_nodes_weights,
        QuadratureMethod.TRAPEZOIDAL: _trapezoidal_nodes_weights,
    }
    return dispatcher[method](a, b, n)
