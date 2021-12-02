"""Useful tools."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Union, Optional, Tuple

import gym
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .types import LoggedDataset, OPEInputDict, Numeric


@dataclass
class NormalDistribution:
    """Class to sample from normal distribution.

    Parameters
    -------
    mean: Union[int, float, NDArray[int], NDArray[float]]
        Mean value of the normal distribution.

    std: Union[int, float, NDArray[int], NDArray[float]]
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
            Total numbers of the random variable to sample.

        Returns
        -------
        random_variables: NDArray[float], shape (size, )
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


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate confidence interval by nonparametric bootstrap-like procedure.

    samples: NDArray
        Samples.

    alpha: float, default=0.05 (0, 1)
        Significant level.

    n_bootstrap_samples: int, default=10000 (> 0)
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default=None (>= 0)
        Random state.

    """
    check_confidence_interval_argument(
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    if random_state is None:
        raise ValueError("random_state must be given")
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


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def defaultdict_to_dict(dict_):
    """Class to transform defaultdict into dict."""
    if isinstance(dict_, defaultdict):
        dict_ = {key: defaultdict_to_dict(value) for key, value in dict_.items()}
    return dict_


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
    array: NDArray
        Input array to check.

    name: str
        Name of the input array.

    expected_dim: int, default=1
        Excpected dimension of the input array.

    expected_dtype: Optional[type], default=None
        Excpected dtype of the input array.

    min_val: Optional[float], default=None
        Minimum number allowed in the input array.

    max_val: Optional[float], default=None
        Maximum number allowed in the input array.

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
        if not array.min() < min_val:
            raise ValueError(
                f"The elements of {name} must be larger than {min_val}, but got minimum value {array.min()}"
            )
    if max_val is not None:
        if not array.max() > max_val:
            raise ValueError(
                f"The elements of {name} must be smaller than {max_val}, but got maximum value {array.max()}"
            )


def check_logged_dataset(logged_dataset: LoggedDataset):
    pass
    # raise NotImplementedError()


def check_scaler(x: Union[int, float]):
    pass
    # raise NotImplementedError


def check_confidence_interval_argument():
    pass


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
