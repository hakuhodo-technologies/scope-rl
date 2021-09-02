"""Useful tools."""
from numpy.lib.index_tricks import diag_indices
from dataclasses import dataclass
from typing import Dict, Union, Any

import numpy as np
from sklearn.utils import check_random_state

import gym
from d3rlpy.dataset import MDPDataset
from d3rlpy.ope import DiscreteFQE
from d3rlpy.ope import FQE as ContinuousFQE
from d3rlpy.algos import DiscreteRandomPolicy
from d3rlpy.algos import RandomPolicy as ContinuousRandomPolicy


from _gym.types import LoggedDataset, OPEInputDict
from _gym.ope import (
    BaseOffPolicyEstimator,
    DiscreteDirectMethod,
    DiscreteDoublyRobust,
    ContinuousDirectMethod,
    ContinuousDoublyRobust,
)


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
    random_state: int = 12345

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


def action_scaler():
    raise NotImplementedError()


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: int = 12345,
):
    raise NotImplementedError()


def check_logged_dataset(logged_dataset: LoggedDataset):
    raise NotImplementedError()


def check_scaler(x: Union[int, float]):
    raise NotImplementedError


def convert_logged_dataset_into_MDPDataset(logged_dataset: LoggedDataset):
    check_logged_dataset(logged_dataset)
    if logged_dataset["action_type"] == "discrete":
        return MDPDataset(
            observations=logged_dataset["state"],
            actions=logged_dataset["action"],
            rewards=logged_dataset["reward"],
            terminals=logged_dataset["done"],
            episode_terminals=logged_dataset["done"],
            discrete_action=True,
        )
    else:
        return MDPDataset(
            observations=logged_dataset["state"],
            actions=logged_dataset["action"],
            rewards=logged_dataset["reward"],
            terminals=logged_dataset["done"],
            episode_terminals=logged_dataset["done"],
        )


def check_base_model_args(
    dataset: MDPDataset,
    args: Dict[str, Any],
    action_type: str,
):
    args_ = copy.deepcopy(args)
    algo = (
        DiscreteRandomPolicy()
        if action_type == "discrete"
        else ContinuousRandomPolicy()
    )
    algo.build_with_dataset(dataset)
    args_["algo"] = algo

    try:
        fqe = (
            DiscreteFQE(**args_) if action_type == "discrete" else ContinuousFQE(**args)
        )
    except:
        raise ValueError(
            "base_model_args are invalid, please use default or refer to d3rlpy docs https://d3rlpy.readthedocs.io/en/v0.91/references/off_policy_evaluation.html"
        )


def check_if_valid_env_and_logged_dataset(
    env: gym.Env,
    logged_dataset: LoggedDataset,
):
    raise NotImplementedError()


def check_input_dict(
    input_dict: OPEInputDict,
):
    raise NotImplementedError()
