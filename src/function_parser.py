# latex/expression -> sympy -> numpy fkt

from typing import Optional, Tuple

import numpy as np
import sympy as sp

DEFAULT_SYMBOL = "x"


def _try_parse_latex(latex_str: str) -> Optional[sp.Expr]:
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex_str)
        return expr if expr is not None else None
    except Exception:
        return None


def _latex_to_sympify_fallback(s: str) -> str:
    import re
    s = s.replace("\\sin", "sin").replace("\\cos", "cos").replace("\\tan", "tan")
    s = s.replace("\\exp", "exp").replace("\\log", "log").replace("\\sqrt", "sqrt")
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\left\s*\(|\\right\s*\)", "", s)
    return s


def parse_latex_expression(latex_str: str) -> sp.Expr:
    # latex oder expression -> sympy. fallback sympify wenn latex fehlt
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
    return sp.simplify(expr)


def get_symbol(expr: sp.Expr, default: str = DEFAULT_SYMBOL) -> sp.Symbol:
    free = list(expr.free_symbols)
    if len(free) == 0:
        return sp.Symbol(default)
    if len(free) == 1:
        return free[0]
    for s in free:
        if str(s) == default:
            return s
    return free[0]


def expression_to_numpy(
    expr: sp.Expr, symbol: sp.Symbol
) -> Tuple[sp.Expr, "np.ufunc"]:
    func = sp.lambdify(symbol, expr, modules="numpy")

    def vectorized_func(x: np.ndarray) -> np.ndarray:
        return np.asarray(func(x), dtype=float)

    return expr, vectorized_func


def latex_to_function(latex_str: str) -> Tuple[sp.Expr, "np.ufunc", sp.Symbol]:
    expr = parse_latex_expression(latex_str)
    expr = simplify_expression(expr)
    symbol = get_symbol(expr)
    _, f = expression_to_numpy(expr, symbol)
    return expr, f, symbol


def format_expression(expr: sp.Expr) -> str:
    return sp.pretty(expr, use_unicode=True)
