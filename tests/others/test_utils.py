import pytest

import numpy as np

from _gym.utils import NormalDistribution


# mean, std, random_state, err, description
invalid_input_of_init = [
    (
        "1",
        1,
        1,
        ValueError,
        "mean must be a float number or",
    ),
    (
        np.array([[1], [2]]),
        np.array([[1], [2]]),
        1,
        ValueError,
        "mean must be a float number or an 1-dimensional",
    ),
    (
        1,
        "1",
        1,
        ValueError,
        "std must be a non-negative float number or",
    ),
    (
        1,
        -1,
        1,
        ValueError,
        "std must be a non-negative float number or",
    ),
    (
        np.array([1, 2]),
        np.array([-1, 1]),
        1,
        ValueError,
        "std must be a non-negative float number or",
    ),
    (
        np.array([1]),
        1,
        1,
        ValueError,
        "mean and std must have the same length",
    ),
    (
        1,
        np.array([1]),
        1,
        ValueError,
        "mean and std must have the same length",
    ),
    (
        np.array([1, 2]),
        np.array([1]),
        1,
        ValueError,
        "mean and std must have the same length",
    ),
    (1, 1, -1, ValueError, ""),
    (1, 1, 1.5, ValueError, ""),
    (1, 1, "1", ValueError, ""),
]


@pytest.mark.parametrize(
    "mean, std, random_state, err, description",
    invalid_input_of_init,
)
def test_init_using_invalid_input(mean, std, random_state, err, description):
    with pytest.raises(err, match=f"{description}*"):
        NormalDistribution(mean, std, random_state)


# mean, std
valid_input_of_init = [
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
]


@pytest.mark.parametrize(
    "mean, std",
    valid_input_of_init,
)
def test_init_using_valid_input(mean, std):
    NormalDistribution(mean, std)


@pytest.mark.parametrize("size", [(1.5), (-1), (0), ("1")])
def test_function_using_invalid_input(size):
    normal_distribution = NormalDistribution(mean=1, std=1)

    with pytest.raises(ValueError):
        normal_distribution.sample(size)


def test_function_using_valid_input():
    normal_distribution = NormalDistribution(mean=1, std=1)
    random_variables = normal_distribution.sample(size=3)
    assert random_variables.shape == (3,)

    normal_distribution = NormalDistribution(
        mean=np.array([1, 2]), std=np.array([1, 2])
    )
    random_variables = normal_distribution.sample(size=4)
    assert random_variables.shape == (4, 2)
