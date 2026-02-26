"""
Visualization: presentation-ready plots of the integrand and quadrature structure.

Plots the continuous function f(x) and the sampling structure (rectangles,
trapezoids, or nodes) according to the selected method.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.quadrature import get_nodes_weights_for_plot
from src.types import QuadratureMethod


def _fine_curve_x(a: float, b: float, num_points: int = 500) -> np.ndarray:
    """Dense x grid for smooth function curve."""
    return np.linspace(a, b, num_points)


def _plot_riemann_rectangles(
    ax: plt.Axes,
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    use_left: bool,
) -> None:
    """Draw rectangles for Left (use_left=True) or Right (use_left=False) Riemann sum."""
    h = (b - a) / n
    for i in range(n):
        x_left = a + i * h
        x_right = a + (i + 1) * h
        x_sample = x_left if use_left else x_right
        y_val = float(f(np.array([x_sample]))[0])
        # Rectangle: base [x_left, x_right], height from 0 to y_val
        verts = [
            [x_left, 0],
            [x_right, 0],
            [x_right, y_val],
            [x_left, y_val],
            [x_left, 0],
        ]
        ax.plot([v[0] for v in verts], [v[1] for v in verts], color="steelblue", alpha=0.6, linewidth=1)
        ax.fill([v[0] for v in verts], [v[1] for v in verts], color="steelblue", alpha=0.25)


def _plot_midpoint_rectangles(
    ax: plt.Axes,
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
) -> None:
    """Draw rectangles with height at midpoint of each subinterval."""
    h = (b - a) / n
    for i in range(n):
        x_left = a + i * h
        x_right = a + (i + 1) * h
        x_mid = (x_left + x_right) / 2
        y_val = float(f(np.array([x_mid]))[0])
        verts = [
            [x_left, 0],
            [x_right, 0],
            [x_right, y_val],
            [x_left, y_val],
            [x_left, 0],
        ]
        ax.plot([v[0] for v in verts], [v[1] for v in verts], color="steelblue", alpha=0.6, linewidth=1)
        ax.fill([v[0] for v in verts], [v[1] for v in verts], color="steelblue", alpha=0.25)


def _plot_trapezoids(
    ax: plt.Axes,
    f: Callable[[np.ndarray], np.ndarray],
    nodes: np.ndarray,
) -> None:
    """Draw trapezoids between consecutive (x_i, f(x_i)) and (x_{i+1}, f(x_{i+1}))."""
    f_vals = f(nodes)
    for i in range(len(nodes) - 1):
        x0, x1 = nodes[i], nodes[i + 1]
        y0, y1 = float(f_vals[i]), float(f_vals[i + 1])
        verts = [[x0, 0], [x1, 0], [x1, y1], [x0, y0], [x0, 0]]
        ax.plot([v[0] for v in verts], [v[1] for v in verts], color="steelblue", alpha=0.6, linewidth=1)
        ax.fill([v[0] for v in verts], [v[1] for v in verts], color="steelblue", alpha=0.25)


def _plot_nodes_only(
    ax: plt.Axes,
    f: Callable[[np.ndarray], np.ndarray],
    nodes: np.ndarray,
) -> None:
    """Draw nodes (x_i, f(x_i)) for Simpson (parabolic arcs omitted for clarity)."""
    f_vals = f(nodes)
    ax.scatter(nodes, f_vals, color="steelblue", s=30, zorder=5, label="Stützstellen")


def plot_integration(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    method: QuadratureMethod,
    title: str | None = None,
) -> plt.Figure:
    """
    Create a presentation-ready figure: function curve + quadrature structure.

    The plot shows the continuous curve and the sampling structure (rectangles,
    trapezoids, or nodes) according to the method. Includes labels, legend, and title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x_fine = _fine_curve_x(a, b)
    y_fine = f(x_fine)
    ax.plot(x_fine, y_fine, "k-", linewidth=2, label="f(x)")

    nodes, _ = get_nodes_weights_for_plot(a, b, n, method)

    if method == QuadratureMethod.LEFT_RIEMANN:
        _plot_riemann_rectangles(ax, f, a, b, n, use_left=True)
    elif method == QuadratureMethod.RIGHT_RIEMANN:
        _plot_riemann_rectangles(ax, f, a, b, n, use_left=False)
    elif method == QuadratureMethod.MIDPOINT:
        _plot_midpoint_rectangles(ax, f, a, b, n)
    elif method == QuadratureMethod.TRAPEZOIDAL:
        _plot_trapezoids(ax, f, nodes)
    else:  # Simpson
        _plot_nodes_only(ax, f, nodes)

    ax.set_xlim(a, b)
    y_min, y_max = np.min(y_fine), np.max(y_fine)
    margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.5
    ax.set_ylim(min(y_min - margin, 0), y_max + margin)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        title or f"{method.display_name()}, n = {n}",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def show_plot(fig: plt.Figure) -> None:
    """Display the figure (for interactive use)."""
    plt.figure(fig.number)
    plt.show()


def save_plot(fig: plt.Figure, path: str) -> None:
    """Save the figure to a file."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
