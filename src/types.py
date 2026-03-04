# types + enums für quadratur

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class QuadratureMethod(Enum):
    # links/rechts/mittelpunkt/trapez

    LEFT_RIEMANN = "left_riemann"
    RIGHT_RIEMANN = "right_riemann"
    MIDPOINT = "midpoint"
    TRAPEZOIDAL = "trapezoidal"

    @property
    def order(self) -> int:
        # für richardson: riemann p=1, trapez/mittel p=2
        if self in (QuadratureMethod.LEFT_RIEMANN, QuadratureMethod.RIGHT_RIEMANN):
            return 1
        if self in (QuadratureMethod.MIDPOINT, QuadratureMethod.TRAPEZOIDAL):
            return 2
        return 1

    def richardson_factor(self) -> float:
        return (2**self.order) - 1

    def display_name(self) -> str:
        names = {
            QuadratureMethod.LEFT_RIEMANN: "Linksseitige Rechtecksumme (Untersumme)",
            QuadratureMethod.RIGHT_RIEMANN: "Rechtsseitige Rechtecksumme (Obersumme)",
            QuadratureMethod.MIDPOINT: "Mittelpunktregel",
            QuadratureMethod.TRAPEZOIDAL: "Trapezregel",
        }
        return names[self]


@dataclass(frozen=True)
class QuadratureResult:
    method: QuadratureMethod
    n: int
    a: float
    b: float
    approximation: float
    nodes: Optional[tuple[float, ...]] = None
    weights: Optional[tuple[float, ...]] = None


@dataclass
class ConvergenceRow:

    n: int
    q_n: float
    change: Optional[float] = None


@dataclass
class ConvergenceTable:
    method: QuadratureMethod
    initial_n: int
    rows: list[ConvergenceRow] = field(default_factory=list)

    def add_row(self, n: int, q_n: float, change: Optional[float] = None) -> None:
        self.rows.append(ConvergenceRow(n=n, q_n=q_n, change=change))


@dataclass
class RichardsonEstimate:
    method: QuadratureMethod
    n: int
    q_n: float
    q_2n: float
    estimated_error: float
    order: int


@dataclass
class ExactIntegralResult:
    exact_value: float
    absolute_error: float
    relative_error: float
    success: bool
