
import pytest

from numbers import Number
from random import randint

import numpy as np

from arclia.stats._quantiles import (
    calculate_weighted_quantiles,
)


def create_random_weighted_values():
    size = randint(10, 20)
    return (
        [randint(1, 100) for _ in range(size)],
        [randint(1, 100) for _ in range(size)],
    )


class Test_calculate_weighted_quantiles:
    class Test_when_q_is_a_Number:
        def test_0_always_returns_the_min_value(self):
            for _ in range(3):
                weights, values = create_random_weighted_values()
                min_value = min(values)
                assert calculate_weighted_quantiles(
                    values = values,
                    weights = weights,
                    q = 0,
                ) == min_value

        def test_1_always_returns_the_max_value(self):
            for _ in range(3):
                weights, values = create_random_weighted_values()
                max_value = max(values)
                assert calculate_weighted_quantiles(
                    values = values,
                    weights = weights,
                    q = 1,
                ) == max_value

        def test_returns_first_value_that_crosses_specified_threshold(self):
            def check(q: Number, v: Number):
                assert calculate_weighted_quantiles(
                    values  = [2, 0, 1, 4, 3],
                    weights = [2, 3, 2, 2, 1],
                    q = q,
                ) == v

            with pytest.raises(AssertionError):
                # Confirm that the `check(..)` function is working
                check(q = 0, v = 1)

            # [0.0, 0.3) -> 0
            check(q = 0, v = 0)
            check(q = 0.29, v = 0)

            # [0.3, 0.5) -> 1
            check(q = 0.30, v = 1)
            check(q = 0.45, v = 1)

            # [0.5, 0.7) -> 2
            check(q = 0.5, v = 2)
            check(q = 0.69, v = 2)

            # [0.7, 0.8) -> 3
            check(q = 0.7, v = 3)
            check(q = 0.78, v = 3)

            # [0.8, 1.0] -> 4
            check(q = 0.8, v = 4)
            check(q = 0.91, v = 4)
            check(q = 1.0, v = 4)


    class Test_when_q_is_a_1_dimensional_Array:
        def test_it_returns_an_Array_of_quantiles(self):
            assert np.array_equal(
                calculate_weighted_quantiles(
                    values  = [4, 2, 6, 1, 9],
                    weights = [2, 2, 1, 3, 2],
                    q = [0.0, 0.29, 0.3, 0.72, 0.66, 0.90],
                ),
                [1, 1, 2, 6, 4, 9]
            )
