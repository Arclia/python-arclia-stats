
from typing import overload
from numbers import Number

import numpy as np
import numpy.typing as npt


@overload
def calculate_weighted_quantiles(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    q: Number,
)-> Number:
    ...


@overload
def calculate_weighted_quantiles(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    q: npt.ArrayLike,
)-> np.ndarray:
    ...


def calculate_weighted_quantiles(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    q: npt.ArrayLike,
):
    qq, needs_unpacking = (
        (np.array([q]), True) if isinstance(q, Number)
        else (np.asarray(q), False)
    )

    result = _calculate_weighted_quantiles(
        values = np.asarray(values),
        weights = np.asarray(weights),
        q = qq,
    )

    return (
        result[0] if needs_unpacking
        else result
    )




def _calculate_weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
):
    if np.any(q < 0) or np.any(q > 1):
        raise ValueError("All `q` must be within the interval `[0, 1]`")

    if values.shape != weights.shape:
        raise ValueError("`values` and `weights` must have the same shape")

    # TODO: Perhaps we should allow for N > 0 dimensions
    if len(values.shape) != 1:
        raise ValueError("`values` and `weights` must be 1-Dimensional Arrays")

    # TODO: This will need to be revisited if we allow N-Dimensional Arrays
    if len(values) == 0:
        raise ValueError("`values` and `weights` cannot be empty")

    if np.any(weights < 0):
        raise ValueError("All `weights` must be non-negative")


    sort_order = np.argsort(values)
    sorted_values = values[sort_order]
    sorted_weights = weights[sort_order]

    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]

    target_cumulative_weights = q * total_weight

    target_indices = np.searchsorted(
        a = cumulative_weights[:-1],
        v = target_cumulative_weights,
        side = "right",
    )

    return sorted_values[target_indices]
