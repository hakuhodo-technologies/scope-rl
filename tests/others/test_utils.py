import pytest

import numpy as np

from _gym.utils import NormalDistribution


@pytest.mark.parametrize(
    "mean, std, random_state",
    [
        ("1", 1, 1),
        ("1.5", 1, 1),
        (np.array([1]), 1, 1),
        (1, "1", 1),
        (1, "1.5", 1),
        (1, -1, 1),
        (1, np.array([1]), 1),
        (np.array([1, 2]), np.array([1]), 1),
        (np.array([[1], [2]]), np.array([[1], [2]]), 1),
        (np.array([1, 2]), np.array([-1, 1]), 1),
        (1, 1, -1),
        (1, 1, 1.5),
        (1, 1, "1"),
    ],
)
def test_init_failure_case(mean, std, random_state):
    with pytest.raises(ValueError):
        NormalDistribution(mean, std, random_state)


@pytest.mark.parametrize(
    "mean, std",
    [
        (1, 1),
        (1.5, 1),
        (0, 1),
        (-1, 1),
        (1, 1.5),
        (1, 0),
        (np.array([1]), np.array([1])),
        (np.array([1.5, 2]), np.array([1, 2])),
        (np.array([0, 2]), np.array([1, 2])),
        (np.array([-1, 2]), np.array([1, 2])),
        (np.array([1, 2]), np.array([1.5, 2])),
        (np.array([1, 2]), np.array([0, 2])),
    ],
)
def test_init_success_case(mean, std):
    NormalDistribution(mean, std)


@pytest.mark.parametrize("size", [(1.5), (-1), (0), ("1")])
def test_function_failure_case(size):
    normal_distribution = NormalDistribution(mean=1, std=1)

    with pytest.raises(ValueError):
        normal_distribution.sample(size)


def test_function_success_case():
    normal_distribution = NormalDistribution(mean=1, std=1)
    random_variables = normal_distribution.sample(size=3)

    assert random_variables.shape == (3,)

    normal_distribution = NormalDistribution(
        mean=np.array([1, 2]), std=np.array([1, 2])
    )
    random_variables = normal_distribution.sample(size=3)

    assert random_variables.shape == (3, 2)
