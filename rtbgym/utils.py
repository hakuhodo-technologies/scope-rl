# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Useful tools."""
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .types import Numeric


@dataclass
class NormalDistribution:
    """Class to sample from normal distribution.

    Parameters
    -------
    mean: {int, float, array-like}
        Mean parameter of the normal distribution.

    std: {int, float, array-like}
        Standard deviation of the normal distribution.

    random_state: int, default=None (>= 0)
        Random state.

    """

    mean: Union[int, float, np.ndarray]
    std: Union[int, float, np.ndarray]
    random_state: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.mean, Numeric) and not (
            isinstance(self.mean, np.ndarray) and self.mean.ndim == 1
        ):
            raise ValueError(
                "mean must be a float number or an 1-dimensional NDArray of float values"
            )
        if not (isinstance(self.std, Numeric) and self.std >= 0) and not (
            isinstance(self.std, np.ndarray)
            and self.std.ndim == 1
            and self.std.min() >= 0
        ):
            raise ValueError(
                "std must be a non-negative float number or an 1-dimensional NDArray of non-negative float values"
            )
        if not (
            isinstance(self.mean, Numeric) and isinstance(self.std, Numeric)
        ) and not (
            isinstance(self.mean, np.ndarray)
            and isinstance(self.std, np.ndarray)
            and len(self.mean) == len(self.std)
        ):
            raise ValueError("mean and std must have the same length")
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        self.is_single_parameter = False
        if isinstance(self.mean, Numeric):
            self.is_single_parameter = True

    def sample(self, size: int = 1) -> np.ndarray:
        """Sample random variables from the pre-determined normal distribution.

        Parameters
        -------
        size: int, default=1 (> 0)
            Total number of the random variable to sample.

        Returns
        -------
        random_variables: ndarray of shape (size, )
            Random variables sampled from the normal distribution.

        """
        check_scalar(size, name="size", target_type=int, min_val=1)
        if self.is_single_parameter:
            random_variables = self.random_.normal(
                loc=self.mean, scale=self.std, size=size
            )
        else:
            random_variables = self.random_.normal(
                loc=self.mean, scale=self.std, size=(size, len(self.mean))
            )
        return random_variables


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def check_array(
    array: np.ndarray,
    name: str,
    expected_dim: int = 1,
    expected_dtype: Optional[type] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> ValueError:
    """Input validation on array.

    Parameters
    -------
    array: object
        Input array to check.

    name: str
        Name of the input array.

    expected_dim: int, default=1
        Expected dimension of the input array.

    expected_dtype: {type, tuple of type}, default=None
        Expected dtype of the input array.

    min_val: float, default=None
        Minimum value allowed in the input array.

    max_val: float, default=None
        Maximum value allowed in the input array.

    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be {expected_dim}D array, but got {type(array)}")
    if array.ndim != expected_dim:
        raise ValueError(
            f"{name} must be {expected_dim}D array, but got {expected_dim}D array"
        )
    if expected_dtype is not None:
        if not np.issubsctype(array, expected_dtype):
            raise ValueError(
                f"The elements of {name} must be {expected_dtype}, but got {array.dtype}"
            )
    if min_val is not None:
        if array.min() < min_val:
            raise ValueError(
                f"The elements of {name} must be larger than {min_val}, but got minimum value {array.min()}"
            )
    if max_val is not None:
        if array.max() > max_val:
            raise ValueError(
                f"The elements of {name} must be smaller than {max_val}, but got maximum value {array.max()}"
            )
