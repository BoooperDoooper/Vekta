import numpy as np

from .fd import laplacian

__all__ = ["solve_plate_laplace"]


def solve_plate_laplace(initial_field, dx, dy, tolerance=1e-8, max_iterations=50000,
                        relaxation_factor=None, check_every=10):
    """Steady-state 2D heat (Laplace) solution by red-black SOR.

    initial_field : 2D array. Its EDGE values are the fixed (Dirichlet)
                    boundary temperatures; interior values are the initial guess.
    tolerance     : stop when max |laplacian| over the interior is below this.
    relaxation_factor : SOR omega in (1, 2). None -> estimated optimum.
    check_every   : iterations between residual checks (the check costs time).

    Returns temperature, iterations_used, final_residual.
    """
    temperature = np.array(initial_field, dtype=float)  # copy: input is untouched
    num_x, num_y = temperature.shape
    if relaxation_factor is None:
        spectral_radius = (np.cos(np.pi / (num_x - 1)) + np.cos(np.pi / (num_y - 1))) / 2
        relaxation_factor = 2 / (1 + np.sqrt(1 - spectral_radius**2))

    weight_x, weight_y = 1 / dx**2, 1 / dy**2
    weight_total = 2 * (weight_x + weight_y)

    # Checkerboard index sets, built once. Same-colour points do not neighbour
    # each other, so each colour can be updated in one vectorised step.
    i_index, j_index = np.meshgrid(np.arange(1, num_x - 1), np.arange(1, num_y - 1), indexing="ij")
    colour_masks = [(i_index + j_index) % 2 == parity for parity in (0, 1)]
    colour_points = [(i_index[mask], j_index[mask]) for mask in colour_masks]

    residual = np.inf
    for iteration in range(1, max_iterations + 1):
        for i, j in colour_points:
            neighbour_average = (
                weight_x * (temperature[i + 1, j] + temperature[i - 1, j])
                + weight_y * (temperature[i, j + 1] + temperature[i, j - 1])
            ) / weight_total
            temperature[i, j] += relaxation_factor * (neighbour_average - temperature[i, j])
        if iteration % check_every == 0:
            residual = np.max(np.abs(laplacian(temperature, dx, dy)[1:-1, 1:-1]))
            if residual < tolerance:
                return temperature, iteration, residual
    return temperature, max_iterations, residual