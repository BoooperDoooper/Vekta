import numpy as np

__all__ = ["laplacian"]


def laplacian(field, dx, dy):
    """5-point finite-difference Laplacian: d2u/dx2 + d2u/dy2.

    Measures how far each point sits from the average of its 4 neighbours
    (positive: colder than neighbours, so heat flows in). Heat equation:
    du/dt = alpha * laplacian(u).
    Interior points only; boundary points are returned as 0. Second-order accurate.
    """
    result = np.zeros_like(field, dtype=float)
    centre = field[1:-1, 1:-1]
    result[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2 * centre + field[:-2, 1:-1]) / dx**2   # x direction
        + (field[1:-1, 2:] - 2 * centre + field[1:-1, :-2]) / dy**2  # y direction
    )
    return result