# `arclia-stats` #

Basic statistical functions.

## Installation ##

```shell
pip install arclia-stats
```

## Usage ##

### `calculate_weighted_quantile` ###

#### When `q` is a scalar ####

```python
from arclia.stats import calculate_weighted_quantile

result = calculate_weighted_quantile(
    values  = [1, 3, 0, 2, 4],
    weights = [1, 3, 1, 2, 3],
    q = 0.5,
)

# Prints: 3
print(repr(result))
```

#### When `q` is an `ArrayLike` ####

```python
from arclia.stats import calculate_weighted_quantile

result = calculate_weighted_quantile(
    values  = [1, 3, 0, 2, 4],
    weights = [1, 3, 1, 2, 3],
    q = [0.0, 0.5, 1.0],
)

# Prints: array([0, 3, 4])
print(repr(result))
```

