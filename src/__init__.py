"""
Numerical Integration Framework.

A modular, university-level framework for definite integration with
LaTeX input, multiple quadrature methods, convergence analysis,
Richardson error estimation, and visualization.
"""

from src.types import (
    ConvergenceRow,
    ConvergenceTable,
    ExactIntegralResult,
    QuadratureMethod,
    QuadratureResult,
    RichardsonEstimate,
)

__all__ = [
    "QuadratureMethod",
    "QuadratureResult",
    "ConvergenceRow",
    "ConvergenceTable",
    "RichardsonEstimate",
    "ExactIntegralResult",
]
