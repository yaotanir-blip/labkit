# labkit

A small toolkit for fitting lab data when **both x and y have uncertainties**, with a Streamlit app so you can fit a CSV without writing any code.

Ordinary least squares assumes x is known exactly. In many lab measurements it isn't. labkit uses **total least squares**: it fits the model parameters together with a small shift δ for each x value, weighting every residual by its uncertainty. You get:

- best-fit parameters with standard uncertainties and the full covariance matrix
- the parameter correlations
- χ², degrees of freedom ν, χ²/ν and the p-value
- a plot of the data and fitted curve with a 1σ band and a residuals panel

## Installation

Requires Python 3.14 or newer.

```bash
git clone https://github.com/yaotanir-blip/labkit.git
cd labkit
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

`requirements.txt` installs labkit in editable mode (`-e .`) along with Streamlit and pytest, so changes you make under `src/labkit/` take effect straight away.

## Running the app

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501). Stop it with **Ctrl+C** in the terminal.

In the app:

1. **Choose your data.** Pick *Example data* (randomly generated straight-line data) or *Upload CSV* in the sidebar.
2. **Pick the columns.** Choose which columns hold x, y, dx and dy.
3. **Choose a model.** The options are linear, exponential, polynomial (with a degree selector) or custom.

The plot and fit results then appear on the page.

## Data format

A comma-separated file with one header row, then one row per measurement:

```csv
x,dx,y,dy
10,0.05,0.43,0.2
20,0.05,0.86,0.2
30,0.05,1.29,0.2
```

- **Column names:** labkit tidies the headers when it reads the file. Units in brackets are dropped, letters become lowercase and spaces become underscores, so `Voltage (V)` becomes `voltage`. Use the tidied names when you refer to columns in code.
- **Uncertainties:** every uncertainty must be greater than zero.
- **Example file:** [example data/ex1.csv](<example%20data/ex1.csv>).

## Using labkit in Python

```python
from labkit.data import fileread
from labkit.fitting import tls_fit
from labkit.results import plot_fit

def line(x, a, b):
    return a + b * x

data = fileread("example data/ex1.csv")
fit = tls_fit(line, data, p0=[0, 1], x="x", y="y", sigma_x="dx", sigma_y="dy")

print(fit.params, fit.uncertainties)
print(f"χ²/ν = {fit.reduced_chi2:.2f}, p = {fit.p_value:.2f}")

fig = plot_fit(line, data, x="x", y="y", dx="dx", dy="dy",
               params=fit.params, uncertainties=fit.uncertainties,
               covariance=fit.covariance, x_fitted=fit.x_fitted, chi2=fit.chi2)
fig.savefig("fit.png")
```

### Writing a model

**Signature.** A model is any function `f(x, p1, p2, ...)` that works on NumPy arrays: x comes first, then one argument per fit parameter. `p0` gives a starting value for each parameter, in the same order.

**Parameter names.** The plot labels each parameter with its argument name. A model written as `f(x, *c)` takes any number of parameters, and the plot labels them `p0`, `p1`, ….

**Starting values.** A good starting guess matters for nonlinear models. If a fit fails to converge or gives a nonsensical result, try `p0` values closer to what you expect.

**Example.** The capacitor voltage in a driven RLC circuit:

```python
import numpy as np

def rlc(f, E0, omega0, R=500, L=22e-3, C=1.8e-9):
    w = 2 * np.pi * f                     # f in Hz
    return E0 / (C * np.sqrt(w**2 * R**2 + L**2 * (w**2 - omega0**2)**2))

fit = tls_fit(rlc, data, p0=[1, 1.6e5], sigma_x="dx", sigma_y="dy")
```

The fit only works if the data covers the resonance. Far from resonance the curve is almost flat, so both parameters are poorly determined.

## Project layout

```
app.py               Streamlit app
src/labkit/
  data.py            CSV reading and example-data generation
  fitting.py         tls_fit and the TLSResult it returns
  results.py         plot_fit: data, fit, 1σ band and residuals
example data/        sample CSV files
```

## Licence

MIT. See [LICENCE](LICENCE).
