import numpy as np

__all__ = ["remap"]


def remap(values, in_min, in_max, out_min, out_max):
    """Linearly map values from [in_min, in_max] to [out_min, out_max].

    Works on scalars and arrays. Values outside the input range are
    extrapolated, not clipped.
    """
    if in_min == in_max:
        raise ValueError("in_min and in_max must differ")
    fraction = (np.asarray(values, dtype=float) - in_min) / (in_max - in_min)
    return out_min + fraction * (out_max - out_min)