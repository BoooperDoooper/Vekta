import numpy as np

__all__ = ["fourier_series"]


def fourier_series(function, period, num_terms, num_samples=4096, start=0.0):
    """Fourier series of a periodic function on [start, start + period).

    function : vectorised callable, function(x_array) -> array.
    Series   : f(x) ~ mean_value + sum_n a_n cos(n w x) + b_n sin(n w x), w = 2 pi / period.

    Coefficients come from the periodic trapezoid rule on num_samples points
    (num_samples should be much larger than num_terms).

    Returns mean_value, cosine_coeffs, sine_coeffs, evaluate(x).
    """
    angular_freq = 2 * np.pi / period
    sample_x = start + period * np.arange(num_samples) / num_samples  # endpoint excluded
    sample_f = function(sample_x)
    harmonics = np.arange(1, num_terms + 1)
    phase = angular_freq * np.outer(harmonics, sample_x)  # shape (num_terms, num_samples)

    mean_value = sample_f.mean()
    cosine_coeffs = 2 * (np.cos(phase) @ sample_f) / num_samples
    sine_coeffs = 2 * (np.sin(phase) @ sample_f) / num_samples

    def evaluate(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        phase_x = angular_freq * np.outer(harmonics, x)
        return mean_value + cosine_coeffs @ np.cos(phase_x) + sine_coeffs @ np.sin(phase_x)

    return mean_value, cosine_coeffs, sine_coeffs, evaluate