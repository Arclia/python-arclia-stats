
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
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    
    if len(values.shape) != 1:
        raise ValueError("values and weights must be 1-Dimensional Arrays")

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
