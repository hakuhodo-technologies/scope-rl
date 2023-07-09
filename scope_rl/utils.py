# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Useful tools."""
from dataclasses import dataclass
from collections import defaultdict
from typing import DefaultDict, Dict, Union, Optional, Any, Tuple
from pathlib import Path
import pickle

import gym
import scipy
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .types import LoggedDataset, OPEInputDict

# for scalers
from typing import ClassVar, List
import gym
import torch
from d3rlpy.dataset import MDPDataset, Transition
from d3rlpy.preprocessing import Scaler, ActionScaler


@dataclass
class MultipleLoggedDataset:
    """This class contains paths to multiple logged datasets and returns logged_dataset.

    Parameters
    -------
    action_type: {"discrete", "continuous"}
        Type of the action space.

    path: str
        Path to the directory. Either absolute or relative path is acceptable.

    save_relative_path: bool, default=False.
        Whether to save a relative path.
        If `True`, a path relative to the scope-rl directory will be saved.
        If `False`, the absolute path will be saved.

        Note that this option was added in order to run examples in the documentation properly.
        Otherwise, the default setting (`False`) is recommended.

    """

    action_type: str
    path: str
    save_relative_path: bool = False

    def __post_init__(self):
        self.dataset_ids = defaultdict(int)
        self.abs_path = None
        self.relative_path = None

        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

        if self.save_relative_path:
            abs_path = str(self.path.resolve())
            relative_path = abs_path.split("scope-rlrl/scope_rl/")

            if len(relative_path) == 1:
                self.relative_path = abs_path.split("scope_rl/")
            else:
                self.relative_path = "scope_rl/" + relative_path
        else:
            self.abs_path = self.path.resolve()

    def add(self, logged_dataset: LoggedDataset, behavior_policy_name: str):
        """Save logged dataset.

        Parameters
        -------
        logged_dataset: LoggedDataset.
            Logged dataset to save.

        behavior_policy_name: str
            Name of the behavior policy that generated the logged dataset.

        """
        dataset_id = self.dataset_ids[behavior_policy_name]
        self.dataset_ids[behavior_policy_name] += 1
        logged_dataset["dataset_id"] = dataset_id

        with open(
            self.path
            / f"logged_dataset_{self.action_type}_{behavior_policy_name}_{dataset_id}.pickle",
            "wb",
        ) as f:
            pickle.dump(logged_dataset, f)

    def get(self, behavior_policy_name: str, dataset_id: int):
        """Load logged dataset.

        Parameters
        -------
        behavior_policy_name: str
            Name of the behavior policy that generated the logged dataset.

        dataset_id: int
            Id of the logged dataset.

        Returns
        -------
        logged_dataset: LoggedDataset.
            Logged dataset.

        """
        if self.save_relative_path:
            abs_path = str(Path.cwd())
            abs_path = abs_path.split("scope-rl/scope_rl/")

            if len(abs_path) == 1:
                abs_path = abs_path.split("scope_rl/")
                abs_path = Path(abs_path[0] + "scope_rl/" + self.relative_path)
            else:
                abs_path = Path(abs_path[0] + "scope-rl/scope_rl/" + self.relative_path)
        else:
            path = self.abs_path

        with open(
            path
            / f"logged_dataset_{self.action_type}_{behavior_policy_name}_{dataset_id}.pickle",
            "rb",
        ) as f:
            logged_dataset = pickle.load(f)

        return logged_dataset

    @property
    def behavior_policy_names(self):
        return list(self.dataset_ids.keys())

    @property
    def n_datasets(self):
        return defaultdict_to_dict(self.dataset_ids)


@dataclass
class MultipleInputDict:
    """This class contains paths to multiple input dictionaries for OPE and returns input_dict.

    Parameters
    -------
    action_type: {"discrete", "continuous"}
        Type of the action space.

    path: str
        Path to the directory. Either absolute or relative path is acceptable.

    save_relative_path: bool, default=False.
        Whether to save a relative path.
        If `True`, a path relative to the scope-rl directory will be saved.
        If `False`, the absolute path will be saved.

        Note that this option was added in order to run examples in the documentation properly.
        Otherwise, the default setting (`False`) is recommended.

    """

    action_type: str
    path: str
    save_relative_path: bool = False

    def __post_init__(self):
        self.dataset_ids = defaultdict(list)
        self.eval_policy_name_list = defaultdict(list)
        self.abs_path = None
        self.relative_path = None

        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

        if self.save_relative_path:
            abs_path = str(self.path.resolve())
            relative_path = abs_path.split("scope-rl/scope_rl/")

            if len(relative_path) == 1:
                self.relative_path = abs_path.split("scope_rl/")
            else:
                self.relative_path = "scope_rl/" + relative_path
        else:
            self.abs_path = self.path.resolve()

    def add(self, input_dict: OPEInputDict, behavior_policy_name: str, dataset_id: int):
        """Save input_dict.

        Parameters
        -------
        input_dict: OPEInputDict.
            Input dictionary for OPE to save.

        behavior_policy_name: str
            Name of the behavior policy that generated the logged dataset.

        dataset_id: int
            Id of the logged dataset.

        """
        self.dataset_ids[behavior_policy_name].append(dataset_id)
        self.eval_policy_name_list[behavior_policy_name].append(list(input_dict.keys()))

        with open(
            self.path
            / f"input_dict_{self.action_type}_{behavior_policy_name}_{dataset_id}.pickle",
            "wb",
        ) as f:
            pickle.dump(input_dict, f)

    def get(self, behavior_policy_name: str, dataset_id: int):
        """Load input_dict.

        Parameters
        -------
        behavior_policy_name: str
            Name of the behavior policy that generated the logged dataset.

        dataset_id: int
            Id of the logged dataset.

        Returns
        -------
        input_dict: OPEInputDict.
            Input dictionary for OPE.

        """
        if self.save_relative_path:
            abs_path = str(Path.cwd())
            abs_path = abs_path.split("scope-rl/scope_rl/")

            if len(abs_path) == 1:
                abs_path = abs_path.split("scope_rl/")
                abs_path = Path(abs_path[0] + "scope_rl/" + self.relative_path)
            else:
                abs_path = Path(abs_path[0] + "scope-rl/scope_rl/" + self.relative_path)
        else:
            path = self.abs_path

        with open(
            path
            / f"input_dict_{self.action_type}_{behavior_policy_name}_{dataset_id}.pickle",
            "rb",
        ) as f:
            input_dict = pickle.load(f)

        return input_dict

    @property
    def use_same_eval_policy_across_dataset(self):
        """Check if the contained logged datasets use the same evaluation policies."""
        use_same_eval_policy = defaultdict(lambda: True)

        for behavior_policy, dataset_ids in self.dataset_ids.items():
            base_eval_policy_set = set(
                self.eval_policy_name_list[behavior_policy][dataset_ids[0]]
            )

            for dataset_id in dataset_ids:
                eval_policy_set = set(
                    self.eval_policy_name_list[behavior_policy][dataset_id]
                )

                if len(base_eval_policy_set.symmetric_difference(eval_policy_set)):
                    use_same_eval_policy[behavior_policy] = False

        return defaultdict_to_dict(use_same_eval_policy)

    @property
    def n_eval_policies(self):
        """Check the number of evaluation policies of each input dict."""
        n_eval_policies = {}

        for behavior_policy, eval_policy_names in self.eval_policy_name_list.items():
            n_eval_policies[behavior_policy] = np.zeros(
                len(eval_policy_names), dtype=int
            )

            for i in range(len(eval_policy_names)):
                n_eval_policies[behavior_policy][i] = len(eval_policy_names[i])

        return n_eval_policies

    @property
    def behavior_policy_names(self):
        return list(self.dataset_ids.keys())

    @property
    def n_datasets(self):
        return {key: len(value) for key, value in self.dataset_ids.items()}


def l2_distance(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float = 1.0,
):
    """Calcilate L2 distance.

    Parameters
    -------
    x: array-like of shape (n_samples, n_dim)
        Input array 1.

    y: array-like of shape (n_samples, n_dim)
        Input array 2.

    Returns
    -------
    distance: ndarray of (n_samples, )
        distance between x and y.

    """
    x_2 = (x**2).sum(axis=1)
    y_2 = (y**2).sum(axis=1)
    x_y = (x[:, np.newaxis, :] @ y[:, :, np.newaxis]).flatten()
    return x_2 + y_2 - 2 * x_y


def gaussian_kernel(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float = 1.0,
):
    """Gaussian kernel.

    x: array-like of shape (n_samples, n_dim)
        Input array 1.

    y: array-like of shape (n_samples, n_dim)
        Input array 2.

    bandwidth: float, default=1.0
        Bandwidth hyperparameter of the Gaussian kernel.

    Returns
    -------
    kernel_density: ndarray of (n_samples, )
        kernel density of x given y.

    """
    distance = l2_distance(x, y)
    return np.exp(-distance / (2 * bandwidth**2)) / np.sqrt(
        2 * np.pi * bandwidth**2
    )


def triangular_kernel(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float = 1.0,
):
    """Triangular kernel.

    Parameters
    -------
    x: array-like of shape (n_samples, n_dim)
        Input array 1.

    y: array-like of shape (n_samples, n_dim)
        Input array 2.

    bandwidth: float, default=1.0
        Bandwidth hyperparameter of the Trianglar kernel.

    Returns
    -------
    kernel_density: ndarray of (n_samples, )
        kernel density of x given y.

    """
    distance = np.sqrt(l2_distance(x, y))
    norm_dist = np.clip(distance / bandwidth)
    return (norm_dist < 1) * (1 - norm_dist) / bandwidth


def epanechnikov_kernel(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float = 1.0,
):
    """Epanechnikov kernel.

    Parameters
    -------
    x: array-like of shape (n_samples, n_dim)
        Input array 1.

    y: array-like of shape (n_samples, n_dim)
        Input array 2.

    bandwidth: float, default=1.0
        Bandwidth hyperparameter of the Trianglar kernel.

    Returns
    -------
    kernel_density: ndarray of (n_samples, )
        kernel density of x given y.

    """
    distance = np.sqrt(l2_distance(x, y))
    clipped_norm_dist = np.clip(distance / bandwidth, None, 1.0)
    return 0.75 * (1 - clipped_norm_dist**2) / bandwidth


def cosine_kernel(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float = 1.0,
):
    """Cosine kernel.

    x: array-like of shape (n_samples, n_dim)
        Input array 1.

    y: array-like of shape (n_samples, n_dim)
        Input array 2.

    bandwidth: float, default=1.0
        Bandwidth hyperparameter of the Trianglar kernel.

    Returns
    -------
    kernel_density: ndarray of (n_samples, )
        kernel density of x given y.

    """
    distance = np.sqrt(l2_distance(x, y))
    norm_dist = np.clip(distance / bandwidth)
    return (norm_dist < 1) * (np.pi / 4) * np.cos(norm_dist * np.pi / 2) / bandwidth


def uniform_kernel(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float = 1.0,
):
    """Uniform kernel.

    Parameters
    -------
    x: array-like of shape (n_samples, n_dim)
        Input array 1.

    y: array-like of shape (n_samples, n_dim)
        Input array 2.

    bandwidth: float, default=1.0
        Bandwidth hyperparameter of the Trianglar kernel.

    Returns
    -------
    kernel_density: ndarray of (n_samples, )
        kernel density of x given y.

    """
    distance = np.sqrt(l2_distance(x, y))
    norm_dist = np.clip(distance / bandwidth)
    return (norm_dist < 1) / (2 * bandwidth)


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate the confidence interval by a nonparametric bootstrap-like procedure.

    Parameters
    -------
    samples: array-like
        Samples.

    alpha: float, default=0.05
        Significance level. The value should be within `[0, 1)`.

    n_bootstrap_samples: int, default=10000 (> 0)
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default=None (>= 0)
        Random state.

    Returns
    -------
    estimated_confidence_interval: dict
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)
    check_scalar(
        n_bootstrap_samples, name="n_bootstrap_samples", target_type=int, min_val=1
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


def estimate_confidence_interval_by_hoeffding(
    samples: np.ndarray,
    alpha: float = 0.05,
    **kwargs,
) -> Dict[str, float]:
    """Estimate the confidence interval by the Hoeffding's inequality.

    Note
    -------
    The Hoeffding's inequality provides high-probability bounds of the expectation :math:`\\mu := \\mathbb{E}[X], X \\sim p(X)` as follows.

    .. math::

        |\\hat{\\mu} - \\mu| \\leq X_{\\max} \\sqrt{\\frac{\\log(1 / \\alpha)}{2 n}},

    which holds with probability :math:`1 - \\alpha` where :math:`n` is the data size.

    Parameters
    -------
    samples: array-like
        Samples.

    alpha: float, default=0.05
        Significance level. The value should be within `[0, 1)`.

    Returns
    -------
    estimated_confidence_interval: dict
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)
    mean = samples.mean()
    ci = samples.max() * np.sqrt(np.log(2 / alpha) / 2 * len(samples))
    return {
        "mean": mean,
        f"{100 * (1. - alpha)}% CI (lower)": mean - ci,
        f"{100 * (1. - alpha)}% CI (upper)": mean + ci,
    }


def estimate_confidence_interval_by_empirical_bernstein(
    samples: np.ndarray,
    alpha: float = 0.05,
    **kwargs,
) -> Dict[str, float]:
    """Estimate the confidence interval by the empirical bernstein inequality.

    Note
    -------
    The empirical bernstein inequality provides high-probability bounds of the expectation :math:`\\mu := \\mathbb{E}[X], X \\sim p(X)` as follows.

    .. math::

        |\\hat{\\mu} - \\mu| \\leq \\frac{7 X_{\\max} \\log(2 / \\alpha)}{3 (n - 1)} + \\sqrt{\\frac{2 \\hat{\\mathbb{V}}(X) \\log(2 / \\alpha)}{n(n - 1)}},

    which holds with probability :math:`1 - \\alpha` where :math:`n` is the data size and :math:`\\hat{\\mathbb{V}}` is the sample variance.

    Parameters
    -------
    samples: array-like
        Samples.

    alpha: float, default=0.05
        Significance level. The value should be within `[0, 1)`.

    Returns
    -------
    estimated_confidence_interval: dict
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)
    n = len(samples)
    mean = samples.mean()
    ci = 7 * samples.max() * np.log(2 / alpha) / (3 * (n - 1)) + np.sqrt(
        2 * np.log(2 / alpha) * samples.var() / (n - 1)
    )
    return {
        "mean": mean,
        f"{100 * (1. - alpha)}% CI (lower)": mean - ci,
        f"{100 * (1. - alpha)}% CI (upper)": mean + ci,
    }


def estimate_confidence_interval_by_t_test(
    samples: np.ndarray,
    alpha: float = 0.05,
    **kwargs,
) -> Dict[str, float]:
    """Estimate the confidence interval by Student T-test.

    Note
    -------
    Student T-test assumes that :math:`X \\sim p(X)` follows a normal distribution.
    Based on this assumption, the :math:`1 - \\alpha` \% confidence interval of :math:`\\mu := \\mathbb{E}[X]` is derived as follows.

    .. math::

        |\\hat{\\mu} - \\mu| \\leq \\frac{T_{\\mathrm{test}}(1 - \\alpha, n-1)}{\\sqrt{n} / \\hat{\\sigma}},

    where :math:`n` is the data size, :math:`T_{\\mathrm{test}}(\\cdot,\\cdot)` is the T-value, and :math:`\\sigma` is the standard deviation, respectively.

    Parameters
    -------
    samples: NDArray
        Samples.

    alpha: float, default=0.05
        Significance level. The value should be within `[0, 1)`.

    Returns
    -------
    estimated_confidence_interval: dict
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)
    n = len(samples)
    t = scipy.stats.t.ppf(1 - alpha, n - 1)
    mean = samples.mean()
    ci = t * samples.std(ddof=1) / np.sqrt(n)
    return {
        "mean": mean,
        f"{100 * (1. - alpha)}% CI (lower)": mean - ci,
        f"{100 * (1. - alpha)}% CI (upper)": mean + ci,
    }


def defaultdict_to_dict(dict_: Union[Dict[Any, Any], DefaultDict[Any, Any]]):
    """Transform a defaultdict into a corresponding dict."""
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
            f"{name} must be {expected_dim}D array, but got {array.ndim}D array"
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


def check_logged_dataset(logged_dataset: LoggedDataset):
    """Check logged dataset keys.

    Parameters
    -------
    logged_dataset: LoggedDataset
        Logged dataset.

    """
    dataset_keys = logged_dataset.keys()
    for expected_key in [
        "n_trajectories",
        "action_type",
        "n_actions",
        "action_dim",
        "state_dim",
        "step_per_trajectory",
        "state",
        "action",
        "reward",
        "pscore",
        "done",
        "terminal",
    ]:
        if expected_key not in dataset_keys:
            raise RuntimeError(f"{expected_key} does not exist in logged_dataset")


def check_input_dict(input_dict: OPEInputDict):
    """Check input dict keys.

    Parameters
    -------
    input_dict: OPEInputDict
        Input Dict.

    """
    for eval_policy in input_dict.keys():
        input_dict_keys = input_dict[eval_policy].keys()
        for expected_key in [
            "evaluation_policy_action",
            "evaluation_policy_action_dist",
            "state_action_value_prediction",
            "initial_state_value_prediction",
            "on_policy_policy_value",
            "gamma",
        ]:
            if expected_key not in input_dict_keys:
                raise RuntimeError(
                    f"{expected_key} does not exist in input_dict['{eval_policy}']"
                )


class NewGymAPIWrapper:
    """This class converts old gym outputs (gym<0.26.0) to the new ones (gym>=0.26.0)."""

    def __init__(
        self,
        env: gym.Env,
    ):
        self.env = env

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self.env.seed(seed)
        state = self.env.reset()
        return state, {}

    def step(self, action: Any) -> Tuple[Any]:
        state, action, done, info = self.env.step(action)
        return state, action, False, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __getattr__(self, key) -> Any:
        return object.__getattribute__(self.env, key)


class OldGymAPIWrapper:
    """This class converts new gym outputs (gym>=0.26.0) to the old ones (gym<0.26.0)."""

    def __init__(
        self,
        env: gym.Env,
    ):
        self.env = env

    def reset(self) -> np.ndarray:
        state, info = self.env.reset()
        return state

    def step(self, action: Any) -> Tuple[Any]:
        state, action, done, truncated, info = self.env.step(action)
        return state, action, done or truncated, info

    def render(self, mode: str = "human"):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed: Optional[int] = None):
        self.env.reset(seed=seed)

    def __getattr__(self, key) -> Any:
        return object.__getattribute__(self.env, key)


class MinMaxActionScaler(ActionScaler):
    r"""Min-Max normalization action preprocessing (temporally supported).

    Actions will be normalized in range ``[-1.0, 1.0]``.

    .. math::

        a' = (a - \min{a}) / (\max{a} - \min{a}) * 2 - 1

    Args:
        dataset (d3rlpy.dataset.MDPDataset): dataset object.

        min (numpy.ndarray): minimum values at each entry.

        max (numpy.ndarray): maximum values at each entry.

    """

    TYPE: ClassVar[str] = "min_max"
    _minimum: Optional[np.ndarray]
    _maximum: Optional[np.ndarray]

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        maximum: Optional[np.ndarray] = None,
        minimum: Optional[np.ndarray] = None,
    ):
        self._minimum = None
        self._maximum = None
        if dataset:
            transitions = []
            for episode in dataset.episodes:
                transitions += episode.transitions
            self.fit(transitions)
        elif maximum is not None and minimum is not None:
            self._minimum = np.asarray(minimum)
            self._maximum = np.asarray(maximum)

    def fit(self, transitions: List[Transition]) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        for i, transition in enumerate(transitions):
            action = np.asarray(transition.action)
            if i == 0:
                minimum = action
                maximum = action
            else:
                minimum = np.minimum(minimum, action)
                maximum = np.maximum(maximum, action)

        self._minimum = minimum.reshape((1,) + minimum.shape)
        self._maximum = maximum.reshape((1,) + maximum.shape)

    def fit_with_env(self, env: gym.Env) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        assert isinstance(env.action_space, gym.spaces.Box)
        shape = env.action_space.shape
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        self._minimum = low.reshape((1,) + shape)
        self._maximum = high.reshape((1,) + shape)

    def transform(self, action: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(self._minimum, dtype=torch.float32, device=action.device)
        maximum = torch.tensor(self._maximum, dtype=torch.float32, device=action.device)
        # transform action into [-1.0, 1.0]
        return ((action - minimum) / (maximum - minimum)) * 2.0 - 1.0

    def reverse_transform(self, action: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(self._minimum, dtype=torch.float32, device=action.device)
        maximum = torch.tensor(self._maximum, dtype=torch.float32, device=action.device)
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((action + 1.0) / 2.0)) + minimum

    def transform_numpy(self, action: np.ndarray) -> np.ndarray:
        assert self._minimum is not None and self._maximum is not None
        minimum, maximum = self._minimum, self._maximum
        # transform action into [-1.0, 1.0]
        return ((action - minimum) / (maximum - minimum)) * 2.0 - 1.0

    def reverse_transform_numpy(self, action: np.ndarray) -> np.ndarray:
        assert self._minimum is not None and self._maximum is not None
        minimum, maximum = self._minimum, self._maximum
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((action + 1.0) / 2.0)) + minimum

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._minimum is not None:
            minimum = self._minimum.copy() if deep else self._minimum
        else:
            minimum = None

        if self._maximum is not None:
            maximum = self._maximum.copy() if deep else self._maximum
        else:
            maximum = None

        return {"minimum": minimum, "maximum": maximum}


class MinMaxScaler(Scaler):
    """Min-Max normalization preprocessing (temporally supported).

    .. math::

        x' = (x - \min{x}) / (\max{x} - \min{x})

    Args:
        dataset (d3rlpy.dataset.MDPDataset): dataset object.

        min (numpy.ndarray): minimum values at each entry.

        max (numpy.ndarray): maximum values at each entry.

    """

    TYPE: ClassVar[str] = "min_max"
    _minimum: Optional[np.ndarray]
    _maximum: Optional[np.ndarray]

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        maximum: Optional[np.ndarray] = None,
        minimum: Optional[np.ndarray] = None,
    ):
        self._minimum = None
        self._maximum = None
        if dataset:
            transitions = []
            for episode in dataset.episodes:
                transitions += episode.transitions
            self.fit(transitions)
        elif maximum is not None and minimum is not None:
            self._minimum = np.asarray(minimum)
            self._maximum = np.asarray(maximum)

    def fit(self, transitions: List[Transition]) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        for i, transition in enumerate(transitions):
            observation = np.asarray(transition.observation)
            if i == 0:
                minimum = observation
                maximum = observation
            else:
                minimum = np.minimum(minimum, observation)
                maximum = np.maximum(maximum, observation)

        self._minimum = minimum.reshape((1,) + minimum.shape)
        self._maximum = maximum.reshape((1,) + maximum.shape)

    def fit_with_env(self, env: gym.Env) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        assert isinstance(env.observation_space, gym.spaces.Box)
        shape = env.observation_space.shape
        low = np.asarray(env.observation_space.low)
        high = np.asarray(env.observation_space.high)
        self._minimum = low.reshape((1,) + shape)
        self._maximum = high.reshape((1,) + shape)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(self._minimum, dtype=torch.float32, device=x.device)
        maximum = torch.tensor(self._maximum, dtype=torch.float32, device=x.device)
        return (x - minimum) / (maximum - minimum)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(self._minimum, dtype=torch.float32, device=x.device)
        maximum = torch.tensor(self._maximum, dtype=torch.float32, device=x.device)
        return ((maximum - minimum) * x) + minimum

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self._minimum is not None and self._maximum is not None
        minimum, maximum = self._minimum, self._maximum
        return (x - minimum) / (maximum - minimum)

    def reverse_transform_numpy(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum, maximum = self._minimum, self._maximum
        return ((maximum - minimum) * x) + minimum

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._maximum is not None:
            maximum = self._maximum.copy() if deep else self._maximum
        else:
            maximum = None

        if self._minimum is not None:
            minimum = self._minimum.copy() if deep else self._minimum
        else:
            minimum = None

        return {"maximum": maximum, "minimum": minimum}
