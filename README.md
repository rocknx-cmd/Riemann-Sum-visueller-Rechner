# Matematik PL -> integrale Numerisch rechnen

Ein Python-Framework, welches numerische Integration mit einem LaTeX-Input, und verschiedenen Methoden der Summenformel, das Integral einer Funktion ausrechnet. Ganz ohne analysis (nur wenn man will) und mit visualisierung der angewandten Rieman-Summe.

## Features

- **LaTeX Input**: Formel eingeben in LaTeX (z.B.: $`\sin(x)`$, $`\exp(-x^2)`$, $`0.9*(x-3)^3 + 2*(x-3)^2 + 2`$). Mit SymPy wird das dann geparsed, simplifiziert, and konvertiert zu einer NumPy Funktion (diesmal mit Vektoren).
- **Rechnungsmethode**: Es gibt folgende Arten: Links/Rechts- Summe  (Untersumme/Obersumme), Mittelsumme und Trapezsumme. Jede Methode wird als $\sum_i \Delta(x)_i f(x_i)$ (mit $\Delta(x) = \frac{b-a}{n}$) implementiert (wobei manche komplizierter werden als andere).
- **Konvergenz Tabelle**: Approximierung für $\( n, 2n, 4n, \ldots \)$ mit $\( |Q_n - Q_{\text{prev}}| \)$.
- **Richardson-Error**: $\( E_{\text{est}} = (Q_{2n} - Q_n) / (2^p - 1) \)$ mit $\( p = 1 \)$ (Riemann) oder $\( p = 2 \)$ (Mittelpunkt/Trapez).
- **Exaktes integral**: Falls SymPy die Aufleitung rechnen kann, wird der genaue Wert des bestimmten Integrals widergegeben. Dazu kommt auch der relative Error zu der Summen-Methode.
- **Visualisierung**: Matplotlib kann die Funktion inklusive der eingezeichneten Methode Plotten um zu zeigen wie akkurat die verschiednenen Methoden sind (abgesehen von den Werten).

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

Flow: enter LaTeX function → view symbolic form → enter interval $\([a, b]\)$ → choose method → enter $\( n \)$ → see approximation, convergence table, Richardson estimate → optionally exact integral and plot.

## Project structure

```
Riemann-Sum-visueller-Rechner/
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

## License

Use for educational and presentation purposes.
