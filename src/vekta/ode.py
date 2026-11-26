import numpy as np

__all__ = ["euler", "rk4", "rk45"]


def _fixed_step_setup(state0, t_start, t_end, step_size):
    # Shared by euler/rk4: step count and exact step so the last point lands on t_end.
    num_steps = int(np.ceil((t_end - t_start) / step_size - 1e-12))
    step = (t_end - t_start) / num_steps
    times = t_start + step * np.arange(num_steps + 1)
    states = np.empty((num_steps + 1,) + np.shape(state0))
    states[0] = state0
    return num_steps, step, times, states


def euler(derivative, state0, t_start, t_end, step_size):
    """Explicit Euler, 1st order. derivative(t, state) -> d(state)/dt.
    Returns times, states (states[k] is the state at times[k])."""
    num_steps, step, times, states = _fixed_step_setup(state0, t_start, t_end, step_size)
    for k in range(num_steps):
        states[k + 1] = states[k] + step * derivative(times[k], states[k])
    return times, states


def rk4(derivative, state0, t_start, t_end, step_size):
    """Classic Runge-Kutta, 4th order, fixed step. Same interface as euler."""
    num_steps, step, times, states = _fixed_step_setup(state0, t_start, t_end, step_size)
    for k in range(num_steps):
        t, state = times[k], states[k]
        slope_1 = derivative(t, state)
        slope_2 = derivative(t + step / 2, state + step / 2 * slope_1)
        slope_3 = derivative(t + step / 2, state + step / 2 * slope_2)
        slope_4 = derivative(t + step, state + step * slope_3)
        states[k + 1] = state + step / 6 * (slope_1 + 2 * slope_2 + 2 * slope_3 + slope_4)
    return times, states


# Dormand-Prince coefficients (module level: built once).
_DP_NODES = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_DP_MATRIX = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_DP_WEIGHTS_5TH = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_DP_WEIGHTS_4TH = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
_DP_ERROR_WEIGHTS = _DP_WEIGHTS_5TH - _DP_WEIGHTS_4TH


def rk45(derivative, state0, t_start, t_end, rel_tol=1e-6, abs_tol=1e-9,
         initial_step=None, max_step=np.inf):
    """Adaptive Dormand-Prince RK45. Step size is chosen from the 4th/5th-order
    difference so the local error stays near the tolerances.
    Returns times, states at the accepted steps (times are NOT evenly spaced)."""
    state = np.asarray(state0, dtype=float)
    t = t_start
    step = initial_step if initial_step else (t_end - t_start) / 100
    times, states = [t], [state]
    slope_first = derivative(t, state)  # reused as slope_7 of the previous step (FSAL)

    while t < t_end:
        step = min(step, max_step, t_end - t)
        slopes = [slope_first]
        for stage in range(1, 7):
            increment = sum(coeff * slope for coeff, slope in zip(_DP_MATRIX[stage], slopes))
            slopes.append(derivative(t + _DP_NODES[stage] * step, state + step * increment))
        new_state = state + step * np.tensordot(_DP_WEIGHTS_5TH, np.array(slopes), axes=1)
        error = step * np.tensordot(_DP_ERROR_WEIGHTS, np.array(slopes), axes=1)

        scale = abs_tol + rel_tol * np.maximum(np.abs(state), np.abs(new_state))
        error_ratio = np.max(np.abs(error) / scale)  # <= 1 means step accepted

        if error_ratio <= 1:
            t += step
            state = new_state
            slope_first = slopes[6]
            times.append(t)
            states.append(state)
        # Grow or shrink the step (safety 0.9, limited to a factor 0.2..5).
        step *= min(5.0, max(0.2, 0.9 * error_ratio ** -0.2)) if error_ratio > 0 else 5.0

    return np.array(times), np.array(states)