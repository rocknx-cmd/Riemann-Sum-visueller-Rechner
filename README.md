# Numerical Integration Framework

A university-level numerical integration framework in Python with LaTeX input, multiple quadrature methods, convergence analysis, Richardson error estimation, and presentation-ready visualization.

## Features

- **LaTeX input**: Enter integrands in LaTeX (e.g. `\sin(x)`, `\exp(-x^2)`, `0.9*(x-3)^3 + 2*(x-3)^2 + 2`). Parsed with SymPy (and optional antlr4), simplified, and converted to a vectorized NumPy function.
- **Quadrature methods**: Left/Right Riemann sum, Midpoint rule, Trapezoidal rule, Simpson's rule. Each implemented as \( Q_n = \sum_i w_i f(x_i) \) with explicit nodes and weights.
- **Convergence table**: Approximations for \( n, 2n, 4n, \ldots \) with \( |Q_n - Q_{\text{previous}}| \).
- **Richardson error estimate**: \( E_{\text{est}} = (Q_{2n} - Q_n) / (2^p - 1) \) with \( p = 2 \) (Trapezoidal/Midpoint) or \( p = 4 \) (Simpson).
- **Exact integral**: When SymPy can compute the antiderivative, reports exact value, absolute error, and relative error.
- **Visualization**: Matplotlib plot of the function and the quadrature structure (rectangles, trapezoids, or nodes).

## Requirements

- Python 3.10+
- NumPy, SymPy, Matplotlib  
- For full LaTeX parsing: `antlr4-python3-runtime>=4.11` (optional; expression-style input works without it)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the interactive CLI:

```bash
python main.py
```

Flow: enter LaTeX function → view symbolic form → enter interval \([a, b]\) → choose method → enter \( n \) → see approximation, convergence table, Richardson estimate → optionally exact integral and plot.

## Project structure

```
MathePL/
├── main.py                 # Entry point
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── types.py            # QuadratureMethod enum, dataclasses
    ├── function_parser.py  # LaTeX/expression → SymPy → lambdify
    ├── quadrature.py       # All five quadrature methods (vectorized)
    ├── convergence.py     # Convergence table, Richardson estimate
    ├── exact_integral.py  # Symbolic exact integral when possible
    ├── visualization.py   # Matplotlib figures
    └── cli.py             # Interactive workflow and output
```

## Example (programmatic)

```python
from src.function_parser import latex_to_function, format_expression
from src.quadrature import compute_quadrature
from src.types import QuadratureMethod

expr, f, _ = latex_to_function(r"\exp(-x**2)")
print(format_expression(expr))
result = compute_quadrature(f, 0, 1, 20, QuadratureMethod.SIMPSON)
print("Q_n =", result.approximation)
```

## License

Use for educational and presentation purposes.
