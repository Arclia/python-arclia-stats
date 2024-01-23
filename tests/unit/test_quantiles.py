
from random import randint, shuffle

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

        def test_q_between_0_and_1_returns_the_first_value_that_crosses_the_threshold(self):
            assert calculate_weighted_quantiles(
                values = [0, 1, 2, 3, 4],
                weights = [1, 1, 1, 1, 1],
                q = 0.2,
            ) == 1
