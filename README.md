# Vekta

Small NumPy-based toolkit for numerical simulation work: grids, finite differences,
a 2D Laplace (steady-state heat) solver, Fourier series, and ODE integrators.
Vekta only computes: it returns arrays and contains no plotting or animation code, so
results can go to Matplotlib, Manim, PyQt6, or anything else.

> **Status:** early development (0.x). The API may change between releases.

## Installation

Not yet published on PyPI. From a local clone:

```bash
python -m pip install -e .
```

Requires Python 3.11 or later. The only dependency is NumPy.

## Functions

| Function | Module | Purpose |
|---|---|---|
| `remap` | `core` | Linearly map values from one range to another |
| `make_grid` | `grid` | Uniform 2D grid (`[i, j]` indexing: i along x, j along y) |
| `laplacian` | `fd` | 5-point finite-difference Laplacian of a 2D field |
| `solve_plate_laplace` | `laplace` | Steady-state 2D heat (Laplace) solution, red-black SOR |
| `fourier_series` | `fourier` | Coefficients and evaluator for a periodic function |
| `euler`, `rk4` | `ode` | Fixed-step ODE integrators (1st and 4th order) |
| `rk45` | `ode` | Adaptive Dormand-Prince integrator |

## Quick start

Steady-state temperature on a plate. The edge values of the initial field are the
fixed boundary temperatures:

```python
import numpy as np
import vekta

x_grid, y_grid, dx, dy = vekta.make_grid((0, 1), (0, 1), (101, 101))

field = np.zeros(x_grid.shape)
field[-1, :] = 90.0              # right edge held at 90

temperature, iterations, residual = vekta.solve_plate_laplace(field, dx, dy)
```

ODE integration, using `derivative(t, state)`:

```python
import numpy as np
import vekta

def decay(t, state):
    return -state

times, states = vekta.rk4(decay, np.array([1.0]), 0.0, 5.0, step_size=0.01)
times, states = vekta.rk45(decay, np.array([1.0]), 0.0, 5.0)  # adaptive, uneven times
```

Fourier series of a periodic function:

```python
import numpy as np
import vekta

square_wave = lambda x: np.sign(np.sin(x))
mean_value, cosine_coeffs, sine_coeffs, evaluate = vekta.fourier_series(
    square_wave, period=2 * np.pi, num_terms=15)
approximation = evaluate(np.linspace(0, 2 * np.pi, 400))
```

## Notes

- `fourier_series` needs a vectorised function and a known period.
- `rk45` returns unevenly spaced times; interpolate them if you need animation frames.
- `laplacian` returns zero on the boundary points.

## Development

```bash
python -m pip install -e .
```

```text
src/vekta/   library source
tests/       automated tests
examples/    usage scripts
```

## License

MIT. See [LICENSE](LICENSE).