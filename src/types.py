"""
Numerical Integration Framework — Type definitions and enumerations.

Provides QuadratureMethod enum for method selection and dataclasses
for structured integration results, convergence data, and error estimates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class QuadratureMethod(Enum):
    """Numerical quadrature method. Each corresponds to a formula Q_n = Σ w_i f(x_i)."""

    LEFT_RIEMANN = "left_riemann"
    RIGHT_RIEMANN = "right_riemann"
    MIDPOINT = "midpoint"
    TRAPEZOIDAL = "trapezoidal"
    SIMPSON = "simpson"

    @property
    def order(self) -> int:
        """Asymptotic convergence order p: error ∝ n^{-p}."""
        if self in (QuadratureMethod.LEFT_RIEMANN, QuadratureMethod.RIGHT_RIEMANN):
            return 1
        if self in (QuadratureMethod.MIDPOINT, QuadratureMethod.TRAPEZOIDAL):
            return 2
        if self == QuadratureMethod.SIMPSON:
            return 4
        return 1

    def richardson_factor(self) -> float:
        """Factor 2^p - 1 used in Richardson extrapolation error estimate."""
        p = self.order
        return (2**p) - 1

    def display_name(self) -> str:
        """Human-readable name for output (German)."""
        names = {
            QuadratureMethod.LEFT_RIEMANN: "Linksseitige Rechtecksumme",
            QuadratureMethod.RIGHT_RIEMANN: "Rechtsseitige Rechtecksumme",
            QuadratureMethod.MIDPOINT: "Mittelpunktregel",
            QuadratureMethod.TRAPEZOIDAL: "Trapezregel",
            QuadratureMethod.SIMPSON: "Simpson-Regel",
        }
        return names[self]


@dataclass(frozen=True)
class QuadratureResult:
    """Result of a single quadrature computation Q_n ≈ ∫_a^b f(x) dx."""

    method: QuadratureMethod
    n: int
    a: float
    b: float
    approximation: float
    nodes: Optional[tuple[float, ...]] = None  # sampling nodes x_i
    weights: Optional[tuple[float, ...]] = None  # weights w_i (if desired for inspection)


@dataclass
class ConvergenceRow:
    """One row in the convergence table: n, Q_n, and change from previous."""

    n: int
    q_n: float
    change: Optional[float] = None  # |Q_n - Q_previous|


@dataclass
class ConvergenceTable:
    """Convergence analysis: sequence n0, 2*n0, 4*n0, ... with Q_n and changes."""

    method: QuadratureMethod
    initial_n: int
    rows: list[ConvergenceRow] = field(default_factory=list)

    def add_row(self, n: int, q_n: float, change: Optional[float] = None) -> None:
        self.rows.append(ConvergenceRow(n=n, q_n=q_n, change=change))


@dataclass
class RichardsonEstimate:
    """Richardson error estimate: E_est = (Q_2n - Q_n) / (2^p - 1)."""

    method: QuadratureMethod
    n: int
    q_n: float
    q_2n: float
    estimated_error: float
    order: int


@dataclass
class ExactIntegralResult:
    """Exact integral from SymPy (when available)."""

    exact_value: float
    absolute_error: float
    relative_error: float
    success: bool
