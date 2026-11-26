import numpy as np

__all__ = ["make_grid"]


def make_grid(x_limits, y_limits, shape):
    """Uniform 2D grid.

    x_limits, y_limits : (min, max) of each axis.
    shape              : (num_points_x, num_points_y), each >= 3.

    Returns x_grid, y_grid, dx, dy. Arrays are indexed [i, j] with i along x
    and j along y (indexing="ij"), so field[i, j] sits at (x_grid[i, j], y_grid[i, j]).
    """
    num_x, num_y = shape
    if num_x < 3 or num_y < 3:
        raise ValueError("shape must be at least (3, 3)")
    x_values = np.linspace(*x_limits, num_x)
    y_values = np.linspace(*y_limits, num_y)
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="ij")
    return x_grid, y_grid, x_values[1] - x_values[0], y_values[1] - y_values[0]