from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution


@dataclass(frozen=True)
class TLSResult:
    params: np.ndarray        # best-fit parameters, in the order the model takes them
    uncertainties: np.ndarray # standard uncertainties, √diag(covariance)
    covariance: np.ndarray    # parameter covariance matrix
    x_fitted: np.ndarray      # best estimate of each point's true x (x + δ)
    chi2: float               # χ² at the minimum
    dof: int                  # ν = n − m (2n measurements, n + m unknowns)

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    @property
    def p_value(self) -> float:
        """Chance of a χ² at least this large if the model and the σ are right."""
        return float(chi2_distribution.sf(self.chi2, self.dof)) if self.dof > 0 else float("nan")


def tls_fit(
    model: Callable[..., np.ndarray],
    data: dict,
    p0: list[float],
    *,
    x: str = "x",
    y: str = "y",
    sigma_x: str = "sigma_x",
    sigma_y: str = "sigma_y",
) -> TLSResult:
    """Fit model(x, *params) to the named columns of data, with errors on both x and y.

    p0 is a starting guess for the parameters. The keyword arguments are the
    column names, so data with other names works too, e.g. x="L", y="T".

    Usage example: 
    line_fit = tls_fit(line, line_data, p0=[0.0, 1.0])
    where line is 
    def line(x, a, b):
            return a + b * x
    """
    xs, ys = np.asarray(data[x], dtype=float), np.asarray(data[y], dtype=float)
    sx, sy = np.asarray(data[sigma_x], dtype=float), np.asarray(data[sigma_y], dtype=float)
    if not (len(xs) == len(ys) == len(sx) == len(sy)):
        raise ValueError("x, y, sigma_x and sigma_y must all have the same length")
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("every sigma must be greater than 0")
    n, m = len(xs), len(p0)

    def residuals(unknowns: np.ndarray) -> np.ndarray:
        p, delta = unknowns[:m], unknowns[m:]
        return np.concatenate([(ys - model(xs + delta, *p)) / sy, delta / sx])

    start = np.concatenate([np.asarray(p0, dtype=float), np.zeros(n)])
    solution = least_squares(residuals, start, x_scale="jac")
    if not solution.success:
        raise RuntimeError(f"fit didn't converge: {solution.message}")
    J = solution.jac
    covariance = np.linalg.inv(J.T @ J)[:m, :m]
    return TLSResult(
        params=solution.x[:m],
        uncertainties=np.sqrt(np.diag(covariance)),
        covariance=covariance,
        x_fitted=xs + solution.x[m:],
        chi2=float(np.sum(solution.fun**2)),
        dof=n - m,
    )


def report(title: str, result: TLSResult, names: list[str], true_values: list[float] | None = None) -> None:
    print(title)
    for i, name in enumerate(names):
        line = f"  {name:>6} = {result.params[i]:.5g} ± {result.uncertainties[i]:.2g}"
        if true_values is not None:
            pull = (result.params[i] - true_values[i]) / result.uncertainties[i]
            line += f"   (true {true_values[i]:g}, {pull:+.1f}σ away)"
        print(line)
    print(f"  χ² = {result.chi2:.2f}, ν = {result.dof}, χ²/ν = {result.reduced_chi2:.2f}, p = {result.p_value:.2f}\n")


# ---------------------------------------------------------------- examples
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    def line(x, a, b):
        return a + b * x

    x_true = np.linspace(0, 10, 15)
    line_data = {
        "x": x_true + rng.normal(0, 0.3, x_true.size),
        "y": line(x_true, 1.0, 2.0) + rng.normal(0, 0.5, x_true.size),
        "sigma_x": np.full(x_true.size, 0.3),
        "sigma_y": np.full(x_true.size, 0.5),
    }
    line_fit = tls_fit(line, line_data, p0=[0.0, 1.0])
    report("Straight line, y = a + b·x", line_fit, ["a", "b"], true_values=[1.0, 2.0])

