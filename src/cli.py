# cli für numerische integration

from typing import Optional

import matplotlib.pyplot as plt

from src.convergence import build_convergence_table, richardson_estimate
from src.exact_integral import compute_exact_integral
from src.function_parser import format_expression, latex_to_function
from src.quadrature import compute_quadrature
from src.types import QuadratureMethod
from src.visualization import plot_integration, show_plot


def _prompt(prompt_text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    line = input(f"{prompt_text}{suffix}: ").strip()
    if not line and default is not None:
        return default
    return line


def _parse_float(s: str, name: str) -> float:
    try:
        return float(s.replace(",", "."))
    except ValueError:
        raise ValueError(f"Ungültige Zahl für {name}: {s}")


def _select_method() -> QuadratureMethod:
    methods = list(QuadratureMethod)
    print("\n  Welche Methode soll verwendet werden?")
    for i, m in enumerate(methods, start=1):
        print(f"    {i}. {m.display_name()}")
    while True:
        choice = _prompt("Nummer (1–4)", "3")
        try:
            idx = int(choice)
            if 1 <= idx <= len(methods):
                return methods[idx - 1]
        except ValueError:
            pass
        print("  Bitte eine Zahl zwischen 1 und 4 eingeben.")


def _run_workflow() -> None:
    print("\n" + "=" * 60)
    print("  Numerische Integration")
    print("  Fläche unter einer Kurve f(x) von a bis b")
    print("=" * 60)

    # funktion
    print("\n  Gib die Funktion f(x) ein.")
    print("  Beispiele:  sin(x)   oder   x**2   oder   exp(-x**2)")
    latex_input = _prompt("f(x)", "x**2")
    try:
        expr, f, symbol = latex_to_function(latex_input)
    except ValueError as e:
        print(f"\n  Fehler: {e}")
        return

    print("\n  Deine Funktion (vereinfacht):")
    print("  " + "-" * 40)
    for line in format_expression(expr).splitlines():
        print("  " + line)
    print("  " + "-" * 40)

    # intervall
    print("\n  Von wo bis wo soll integriert werden?")
    a_str = _prompt("Untere Grenze a", "0")
    b_str = _prompt("Obere Grenze b", "1")
    try:
        a = _parse_float(a_str, "a")
        b = _parse_float(b_str, "b")
    except ValueError as e:
        print(f"\n  Fehler: {e}")
        return
    if a >= b:
        print("\n  Fehler: a muss kleiner als b sein.")
        return

    # methode + n
    method = _select_method()
    print("\n  Wie fein soll unterteilt werden? (Mehr = genauer, aber langsamer)")
    n_str = _prompt("Anzahl Unterteilungen n", "10")
    try:
        n = int(n_str)
        if n < 1:
            raise ValueError("n muss mindestens 1 sein")
    except ValueError as e:
        print(f"\n  Fehler: {e}")
        return

    # näherung
    result = compute_quadrature(f, a, b, n, method)
    print("\n" + "=" * 60)
    print("  ERGEBNIS")
    print("=" * 60)
    print(f"  Methode:       {method.display_name()}")
    print(f"  Intervall:     von {a} bis {b}")
    print(f"  Unterteilungen: {result.n}")
    print(f"  Näherungswert: {result.approximation:.10g}")
    print()

    # konvergenz
    print("  Was passiert, wenn man feiner unterteilt?")
    print("  (Je mehr Unterteilungen, desto genauer wird das Ergebnis.)")
    print("  " + "-" * 52)
    table = build_convergence_table(f, a, b, n, method, num_doublings=5)
    print(f"  {'n':>8}  {'Näherung':>16}  {'Unterschied zum vorherigen':>24}")
    print("  " + "-" * 52)
    for row in table.rows:
        change_str = f"{row.change:.4e}" if row.change is not None else "—"
        print(f"  {row.n:>8}  {row.q_n:>16.10g}  {change_str:>24}")
    print()

    # richardson
    rich = richardson_estimate(f, a, b, n, method)
    print("  Geschätzter Fehler (Richardson):")
    print("  Aus der Verbesserung von n auf 2*n lässt sich der Fehler schätzen.")
    print(f"  Geschätzter Fehler ≈ {abs(rich.estimated_error):.4e}")
    print()

    # exakt
    exact_choice = _prompt("Exakten Wert berechnen (wenn möglich)? (j/n)", "j")
    if exact_choice.lower().startswith("j") or exact_choice.lower() == "y":
        exact_result = compute_exact_integral(expr, a, b, result.approximation)
        if exact_result.success:
            print("  Exakter Wert (per Formel):")
            print(f"    Exakt:          {exact_result.exact_value:.10g}")
            print(f"    Absoluter Fehler:  {exact_result.absolute_error:.4e}")
            print(f"    Relativer Fehler:   {exact_result.relative_error:.4e}")
        else:
            print("  Exakter Wert konnte nicht berechnet werden.")
        print()

    # plot
    plot_choice = _prompt("Grafik anzeigen? (j/n)", "j")
    if plot_choice.lower().startswith("j") or plot_choice.lower() == "y":
        fig = plot_integration(
            f,
            a,
            b,
            result.n,
            method,
            title=f"{method.display_name()}, n = {result.n}",
        )
        show_plot(fig)
        plt_close = _prompt("Grafik schließen und beenden? (j/n)", "j")
        if plt_close.lower().startswith("j") or plt_close.lower() == "y":
            plt.close(fig)

    print("\n  Fertig.")


def run_cli() -> None:
    try:
        _run_workflow()
    except KeyboardInterrupt:
        print("\n\n  Abgebrochen.")
    except EOFError:
        print("\n\n  Eingabe beendet.")
