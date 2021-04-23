"""Useful tools."""
from dataclasses import dataclass
from typing import Union
from nptyping import NDArray

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class NormalDistribution:
    """Class to sample from normal distribution.

    Parameters
    -------
    mean: float
        Mean value of the normal distribution.

    std: float
        Standard deviation of the normal distribution.

    random_state: int, default=12345
        Random state.

    """

    mean: Union[int, float]
    std: Union[int, float]
    random_state: int = 12345

    def __post_init__(self):
        if not isinstance(self.mean, (int, float)):
            raise ValueError(f"mean must be a float number, but {self.mean} is given")
        if not isinstance(self.std, (int, float)):
            raise ValueError("std must be a float number, but {self.std} is given")
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def sample(self, size: int = 1) -> NDArray[float]:
        """Sample random variables from the pre-determined normal distribution.

        Parameters
        -------
        size: int, default=1
            Total numbers of the random variable to sample.

        Returns
        -------
        random_variables: NDArray[float], shape (size, )
            Random variables sampled from the normal distribution.

        """
        if not (isinstance(size, int) and size > 0):
            raise ValueError(f"size must be a positive integer, but {size} is given")
        return self.random_.normal(loc=self.mean, scale=self.std, size=size)
