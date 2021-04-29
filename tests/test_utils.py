import pytest
from nptyping import NDArray
import numpy as np

from _gym.utils import NormalDistribution


@pytest.mark.parametrize(
    "mean, std, random_state",
    [
        ("1", 1, 1),
        ("1.5", 1, 1),
        (np.array([1]), 1, 1),
        (1, "1", 1)(1, "1.5", 1),
        (1, -1, 1),
        (1, np.array([1]), 1),
        (1, 1, -1),
        (1, 1, 1.5),
        (1, 1, "1"),
    ],
)
def test_init(mean, std, random_state):
    with pytest.raises(ValueError):
        NormalDistribution(mean, std, random_state)


@pytest.mark.parametrize(
    "mean, std",
    [
        (1, 1),
        (1.5, 1),
        (0, 1)(-1, 1),
        (1, 1.5),
        (1, 0),
    ],
)
def test_init_(mean, std):
    NormalDistribution(mean, std)


@pytest.mark.parameterize("size", [(1.5), (-1), (0), ("1")])
def test_function(size):
    normal_distribution = NormalDistribution(mean=1, std=1)

    with pytest.raises(ValueError):
        normal_distribution.sample(size)


def test_function_():
    normal_distribution = NormalDistribution(mean=1, std=1)
    random_variables = normal_distribution.sample(size=3)

    assert isinstance(random_variables, NDArray[float])
    assert len(random_variables) == 3
