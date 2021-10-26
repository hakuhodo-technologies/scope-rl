"""Useful tools."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Union, Any, Optional

import gym
import numpy as np
from sklearn.utils import check_random_state

from _gym.types import LoggedDataset, OPEInputDict


@dataclass
class NormalDistribution:
    """Class to sample from normal distribution.

    Parameters
    -------
    mean: Union[int, float, NDArray[int], NDArray[float]]
        Mean value of the normal distribution.

    std: Union[int, float, NDArray[int], NDArray[float]]
        Standard deviation of the normal distribution.

    random_state: int, default=12345
        Random state.

    """

    mean: Union[int, float, np.ndarray]
    std: Union[int, float, np.ndarray]
    random_state: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.mean, (int, float)) and not (
            isinstance(self.mean, np.ndarray) and self.mean.ndim == 1
        ):
            raise ValueError(
                "mean must be a float number or an 1-dimensional NDArray of float values"
            )
        if not (isinstance(self.std, (int, float)) and self.std >= 0) and not (
            isinstance(self.std, np.ndarray)
            and self.std.ndim == 1
            and self.std.min() >= 0
        ):
            raise ValueError(
                "std must be a non-negative float number or an 1-dimensional NDArray of non-negative float values"
            )
        if not (
            isinstance(self.mean, (int, float)) and isinstance(self.std, (int, float))
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
        if isinstance(self.mean, (int, float)):
            self.is_single_parameter = True

    def sample(self, size: int = 1) -> np.ndarray:
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


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate confidence interval by nonparametric bootstrap-like procedure."""
    random_ = check_random_state(random_state)
    boot_samples = [
        np.mean(random_.choice(samples, size=samples.shape[0]))
        for i in range(n_bootstrap_samples)
    ]
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
    return {
        "mean": np.mean(boot_samples),
        f"{100 * (1. - alpha)}% CI (lower)": lower_bound,
        f"{100 * (1. - alpha)}% CI (upper)": upper_bound,
    }


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def check_logged_dataset(logged_dataset: LoggedDataset):
    pass
    # raise NotImplementedError()


def check_scaler(x: Union[int, float]):
    pass
    # raise NotImplementedError


def check_if_valid_env_and_logged_dataset(
    env: gym.Env,
    logged_dataset: LoggedDataset,
):
    pass
    # raise NotImplementedError()


def check_input_dict(
    input_dict: OPEInputDict,
):
    pass
    # raise NotImplementedError()


def check_synthetic_dataset_configurations(
    configurations: Dict[str, Any],
):
    pass
    # raise NotImplementedError()
