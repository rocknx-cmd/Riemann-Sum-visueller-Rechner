# exaktes integral wenn sympy es schafft

from typing import Optional

import numpy as np
import sympy as sp

from src.function_parser import get_symbol
from src.types import ExactIntegralResult


def compute_exact_integral(
    expr: sp.Expr,
    a: float,
    b: float,
    numerical_approximation: float,
) -> ExactIntegralResult:
    # F(b)-F(a), sonst success=False
    symbol = get_symbol(expr)
    try:
        antiderivative = sp.integrate(expr, symbol)
        if hasattr(antiderivative, "evalf") and not antiderivative.has(sp.Integral):
            F_b = float(antiderivative.subs(symbol, b).evalf())
            F_a = float(antiderivative.subs(symbol, a).evalf())
            exact_value = F_b - F_a
        else:
            return ExactIntegralResult(
                exact_value=0.0,
                absolute_error=0.0,
                relative_error=0.0,
                success=False,
            )
    except Exception:
        return ExactIntegralResult(
            exact_value=0.0,
            absolute_error=0.0,
            relative_error=0.0,
            success=False,
        )

    absolute_error = abs(numerical_approximation - exact_value)
    relative_error = (
        absolute_error / abs(exact_value) if exact_value != 0 else float("inf")
    )
    return ExactIntegralResult(
        exact_value=exact_value,
        absolute_error=absolute_error,
        relative_error=relative_error,
        success=True,
    )
