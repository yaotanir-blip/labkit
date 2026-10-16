import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import chi2 as chi2_distribution

def plot_fit(data,x='x',y='y',dx='dx',dy = 'dy', *model, params, covariance, x_fitted, chi2):
    xs, ys = np.asarray(data[x], dtype=float), np.asarray(data[y], dtype=float)
    sx, sy = np.asarray(data[dx], dtype=float), np.asarray(data[dy], dtype=float)
    p = np.asarray(params, dtype=float)
    sigma_p = np.asarray(uncertainties, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    x_fit = np.asarray(x_fitted, dtype=float)
    n, m = len(xs), len(p)
    dof = n - m

    # Constrained layout makes room for the results box placed outside the axes.
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, height_ratios=(3, 1), layout="constrained")

    # --- Top panel: data, fit, 1σ band, and each point's shift onto the curve
    lo, hi = min(xs.min(), x_fit.min()), max(xs.max(), x_fit.max())
    curve_x = np.linspace(lo, hi, 400)
    curve_y = model(curve_x, *p)
    band = _curve_uncertainty(model, curve_x, p, sigma_p, cov)

    # fmt="o" draws the data as points; without it errorbar joins them with a line.
    top.errorbar(xs, ys, xerr=sx, yerr=sy, fmt="o", label="Data")
    (fit_line,) = top.plot(curve_x, curve_y, label="Fit")
    # The band takes the fit line's colour and is see-through: a default fill is opaque
    # and would hide the curve it surrounds.
    top.fill_between(curve_x, curve_y - band, curve_y + band, color=fit_line.get_color(),
                     alpha=0.3, label="1σ band")
    # All segments as one line (NaN breaks it between points), so they share one colour.
    gaps = np.full(n, np.nan)
    top.plot(np.column_stack([xs, x_fit, gaps]).ravel(),
             np.column_stack([ys, model(x_fit, *p), gaps]).ravel(), label="Shift to fitted x")
    top.set_ylabel(y_label or y)
    top.legend(loc="best")

    # --- Bottom panel: residuals with the effective (x and y) error bar
    slope = _derivative(model, xs, p, step=1e-3 * sx)
    effective_sigma = np.sqrt(sy**2 + (slope * sx) ** 2)
    bottom.axhline(0)
    bottom.errorbar(xs, ys - model(xs, *p), yerr=effective_sigma, fmt="o")
    bottom.set_xlabel(x_label or x)
    bottom.set_ylabel("y − f(x)")

    # --- Results box, beside the plot so it never covers data
    names = _param_names(model, m)
    lines = [f"{name} = {_format(value, sigma)}" for name, value, sigma in zip(names, p, sigma_p)]
    for i in range(m):
        for j in range(i + 1, m):
            lines.append(f"ρ({names[i]}, {names[j]}) = {cov[i, j] / (sigma_p[i] * sigma_p[j]):+.2f}")
    lines.append("")
    lines.append(f"χ² = {chi2:.2f}")
    lines.append(f"ν = {dof}")
    if dof > 0:
        lines.append(f"χ²/ν = {chi2 / dof:.2f}")
        lines.append(f"p = {chi2_distribution.sf(chi2, dof):.2f}")
    else:
        lines.append("χ²/ν undefined (ν = 0)")
    blank = Line2D([], [], linestyle="none")
    # A legend with empty handles, used as a text box: handlelength=0 removes the gap
    # where a line sample would normally go.
    fig.legend([blank] * len(lines), lines, loc="outside right upper", title="Fit results",
               handlelength=0, handletextpad=0, alignment="left")
    return fig


def _param_names(model: Callable[..., np.ndarray], m: int) -> list[str]:
    """Names from the model's signature, e.g. line(x, a, b) → [a, b]; p0, p1, … otherwise."""
    try:
        names = list(inspect.signature(model).parameters)[1:]
    except (TypeError, ValueError):
        names = []
    return names if len(names) == m else [f"p{j}" for j in range(m)]


def _derivative(model, xs, p, step):
    """f′(x) by a central difference. The step is a small fraction of σ_x, the scale that matters."""
    return (model(xs + step, *p) - model(xs - step, *p)) / (2 * step)


def _curve_uncertainty(model, xs, p, sigma_p, cov):
    """1σ uncertainty on the fitted curve, σ_f(x)² = g(x)ᵀ C g(x) with g = ∂f/∂p.

    This is error propagation through the covariance matrix C, so it includes the
    correlations between parameters. Each derivative uses a step of 10⁻⁴ σ_p.
    """
    gradient = np.empty((len(p), len(xs)))
    for j in range(len(p)):
        h = 1e-4 * sigma_p[j] if sigma_p[j] > 0 else 1e-8
        up, down = p.copy(), p.copy()
        up[j] += h
        down[j] -= h
        gradient[j] = (model(xs, *up) - model(xs, *down)) / (2 * h)
    return np.sqrt(np.einsum("ix,ij,jx->x", gradient, cov, gradient))


def _format(value: float, sigma: float) -> str:
    """value ± σ, with σ to 2 significant figures and the value to the same decimal place."""
    if not (np.isfinite(value) and np.isfinite(sigma)) or sigma <= 0:
        return f"{value:.4g} ± {sigma:.2g}"
    exponent = int(np.floor(np.log10(max(abs(value), sigma))))
    scientific = exponent >= 4 or exponent <= -3
    if scientific:
        value, sigma = value / 10.0**exponent, sigma / 10.0**exponent
    decimals = 1 - int(np.floor(np.log10(sigma)))
    if np.floor(np.log10(round(sigma, decimals))) > np.floor(np.log10(sigma)):  # 0.0996 → 0.10
        decimals -= 1
    shown = max(decimals, 0)
    text = f"{round(value, decimals) + 0.0:.{shown}f} ± {round(sigma, decimals):.{shown}f}"
    return f"({text}) × 10{str(exponent).translate(_SUPERSCRIPT)}" if scientific else text