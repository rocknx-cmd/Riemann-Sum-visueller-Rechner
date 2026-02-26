"""
Function parser: LaTeX → SymPy expression → numerical callable.

Parses a user-supplied LaTeX string (e.g. "\\sin(x)", "\\exp(-x^2)")
using SymPy's parse_latex, simplifies the expression symbolically,
and produces a vectorized NumPy function via lambdify for use in quadrature.
"""

from typing import Optional, Tuple

import numpy as np
import sympy as sp

# Default symbol used for the variable (e.g. in f(x))
DEFAULT_SYMBOL = "x"


def _try_parse_latex(latex_str: str) -> Optional[sp.Expr]:
    """Try parsing with parse_latex; return None if unavailable or failing."""
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex_str)
        return expr if expr is not None else None
    except Exception:
        return None


def _latex_to_sympify_fallback(s: str) -> str:
    """Replace common LaTeX with SymPy-friendly form for sympify fallback."""
    import re
    s = s.replace("\\sin", "sin").replace("\\cos", "cos").replace("\\tan", "tan")
    s = s.replace("\\exp", "exp").replace("\\log", "log").replace("\\sqrt", "sqrt")
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\left\s*\(|\\right\s*\)", "", s)
    return s


def parse_latex_expression(latex_str: str) -> sp.Expr:
    """
    Parse a LaTeX or expression string into a SymPy expression.

    Tries SymPy's parse_latex first (requires antlr4). On failure or missing
    dependency, falls back to sympify with LaTeX-like substitutions.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    latex_str = latex_str.strip()
    if not latex_str:
        raise ValueError("Empty expression.")
    expr = _try_parse_latex(latex_str)
    if expr is None:
        try:
            fallback = _latex_to_sympify_fallback(latex_str)
            expr = sp.sympify(fallback)
        except Exception as e:
            raise ValueError(f"Failed to parse expression: {e}") from e
    if not isinstance(expr, sp.Expr):
        raise ValueError("Parsing did not produce a symbolic expression.")
    return expr


def simplify_expression(expr: sp.Expr) -> sp.Expr:
    """Simplify the symbolic expression (expand, cancel, etc.)."""
    return sp.simplify(expr)


def get_symbol(expr: sp.Expr, default: str = DEFAULT_SYMBOL) -> sp.Symbol:
    """Extract the single free symbol from the expression, or use default."""
    free = list(expr.free_symbols)
    if len(free) == 0:
        return sp.Symbol(default)
    if len(free) == 1:
        return free[0]
    # Prefer 'x' if present
    for s in free:
        if str(s) == default:
            return s
    return free[0]


def expression_to_numpy(
    expr: sp.Expr, symbol: sp.Symbol
) -> Tuple[sp.Expr, "np.ufunc"]:
    """
    Convert a SymPy expression to a vectorized NumPy function.

    Returns the symbolic expression (for display) and a callable
    that accepts a NumPy array and returns an array of the same shape.
    """
    func = sp.lambdify(symbol, expr, modules="numpy")
    # Ensure we use a vectorized version (lambdify with numpy is already vectorized)
    def vectorized_func(x: np.ndarray) -> np.ndarray:
        return np.asarray(func(x), dtype=float)

    return expr, vectorized_func


def latex_to_function(latex_str: str) -> Tuple[sp.Expr, "np.ufunc", sp.Symbol]:
    """
    Full pipeline: LaTeX string → simplified SymPy expression → NumPy function.

    Returns:
        expr: Simplified SymPy expression (for pretty printing).
        f: Vectorized callable f(x) for x a NumPy array.
        symbol: The integration variable (e.g. x).
    """
    expr = parse_latex_expression(latex_str)
    expr = simplify_expression(expr)
    symbol = get_symbol(expr)
    _, f = expression_to_numpy(expr, symbol)
    return expr, f, symbol


def format_expression(expr: sp.Expr) -> str:
    """Return a nicely formatted string of the expression (e.g. for console)."""
    return sp.pretty(expr, use_unicode=True)
