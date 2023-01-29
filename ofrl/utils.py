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


@dataclass
class MultipleLoggedDataset:
    """This class contains paths to multiple logged datasets and returns logged_dataset.

    Parameters
    -------
    path: str
        Path to the directory. Either absolute and relative path is acceptable.

    save_relative_path: bool, default=False.
        Whether to save a relative path.
        If `True`, a path relative to the ofrl directory will be saved.
        If `False`, the absolute path will be saved.

        Note that, this option was added in order to run examples in the documentation properly.
        Otherwise, the default setting (`False`) is recommended.

    """

    path: str
    save_relative_path: str

    def __post_init__(self):
        self.n_datasets = 0
        self.name_to_id_mapping = {}
        self.id_to_name_mapping = []

        self.abs_path = None
        self.relative_path = None

        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

        if self.save_relative_path:
            abs_path = str(self.path.resolve())
            relative_path = abs_path.split("ofrl/ofrl/")

            if len(relative_path) == 1:
                self.relative_path = abs_path.split("ofrl/")
            else:
                self.relative_path = "ofrl/" + relative_path
        else:
            self.abs_path = self.path.resolve()

    def __len__(self):
        return self.n_datasets

    def add(self, logged_dataset: LoggedDataset, name: Optional[str] = None):
        """Save logged dataset.

        Parameters
        -------
        logged_dataset: LoggedDataset.
            Logged dataset to save.

        name: str, default=None
            Name of the logged dataset.

        """
        id = self.n_datasets
        self.n_datasets += 1

        self.id_to_name_mapping.append(name)

        if name is not None:
            self.name_to_id_mapping[name] = id

        with open(self.path / f"logged_dataset_{id}.pickle", "wb") as f:
            pickle.dump(logged_dataset, f)

    def get(self, id: Union[int, str]):
        """Load logged dataset.

        Parameters
        -------
        id: {int, str}
            Id (or name) of the logged dataset.

        Returns
        -------
        logged_dataset: LoggedDataset.
            Logged dataset.

        """
        id = id if isinstance(id, int) else self.id_to_name_mapping[id]

        if self.save_relative_path:
            abs_path = str(Path.cwd())
            abs_path = abs_path.split("ofrl/ofrl/")

            if len(abs_path) == 1:
                abs_path = abs_path.split("ofrl/")
                abs_path = Path(abs_path[0] + "ofrl/" + self.relative_path)
            else:
                abs_path = Path(abs_path[0] + "ofrl/ofrl/" + self.relative_path)
        else:
            path = self.abs_path

        with open(path / f"logged_dataset_{id}.pickle", "rb") as f:
            logged_dataset = pickle.load(f)

        return logged_dataset


@dataclass
class MultipleInputDict:
    """This class contains paths to multiple input dictionaries for OPE and returns input_dict.

    Parameters
    -------
    path: str
        Path to the directory. Either absolute and relative path is acceptable.

    save_relative_path: bool, default=False.
        Whether to save a relative path.
        If `True`, a path relative to the ofrl directory will be saved.
        If `False`, the absolute path will be saved.

        Note that, this option was added in order to run examples in the documentation properly.
        Otherwise, the default setting (`False`) is recommended.

    """

    path: str
    save_relative_path: str

    def __post_init__(self):
        self.n_datasets = 0
        self.name_to_id_mapping = {}
        self.id_to_name_mapping = []
        self.eval_policy_name_list = []

        self.abs_path = None
        self.relative_path = None

        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

        if self.save_relative_path:
            abs_path = str(self.path.resolve())
            relative_path = abs_path.split("ofrl/ofrl/")

            if len(relative_path) == 1:
                self.relative_path = abs_path.split("ofrl/")
            else:
                self.relative_path = "ofrl/" + relative_path
        else:
            self.abs_path = self.path.resolve()

    def __len__(self):
        return self.n_datasets

    def add(self, input_dict: OPEInputDict, name: Optional[str] = None):
        """Save input_dict.

        Parameters
        -------
        input_dict: OPEInputDict.
            Input dictionary for OPE to save.

        name: str, default=None
            Name of the input_dict.

        """
        id = self.n_datasets
        self.n_datasets += 1

        self.id_to_name_mapping.append(name)
        self.eval_policy_name_list.append(list(input_dict.keys()))

        if name is not None:
            self.name_to_id_mapping[name] = id

        with open(self.path / f"input_dict_{id}.pickle", "wb") as f:
            pickle.dump(input_dict, f)

    def get(self, id: Union[int, str]):
        """Load input_dict.

        Parameters
        -------
        id: {int, str}
            Id (or name) of the input dictionary.

        Returns
        -------
        input_dict: OPEInputDict.
            Input dictionary for OPE.

        """
        id = id if isinstance(id, int) else self.id_to_name_mapping[id]

        if self.save_relative_path:
            abs_path = str(Path.cwd())
            abs_path = abs_path.split("ofrl/ofrl/")

            if len(abs_path) == 1:
                abs_path = abs_path.split("ofrl/")
                abs_path = Path(abs_path[0] + "ofrl/" + self.relative_path)
            else:
                abs_path = Path(abs_path[0] + "ofrl/ofrl/" + self.relative_path)
        else:
            path = self.abs_path

        with open(path / f"input_dict_{id}.pickle", "rb") as f:
            input_dict = pickle.load(f)

        return input_dict

    @property
    def use_same_eval_policy_across_dataset(self):
        """Check if the contained logged datasets use the same evaluation policies."""
        use_same_eval_policy = True
        base_eval_policy_set = set(self.eval_policy_name_list[0])

        for i in range(1, self.n_datasets):
            eval_policy_set = set(self.eval_policy_name_list[i])

            if len(base_eval_policy_set.symmetric_difference(eval_policy_set)):
                use_same_eval_policy = False

        return use_same_eval_policy

    @property
    def n_eval_policies(self):
        """Check the number of evaluation policies of each input dict."""
        n_eval_policies = np.zeros(self.n_datasets, dtype=int)
        for i in range(self.n_datasets):
            n_eval_policies[i] = len(self.eval_policy_name_list[i])

        return n_eval_policies


def gaussian_kernel(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
):
    """Gaussian kernel.

    x: array-like of shape (n_samples, n_dim)
        Input array 1.

    y: array-like of shape (n_samples, n_dim)
        Input array 2.

    sigma: float, default=1.0
        Bandwidth hyperparameter of gaussian kernel.

    Returns
    -------
    kernel_density: ndarray of (n_samples, )
        kernel density of x given y.

    """
    x_2 = (x ** 2).sum(axis=1)
    y_2 = (y ** 2).sum(axis=1)
    x_y = (x[:, np.newaxis, :] @ y[:, :, np.newaxis]).flatten()
    distance = x_2 + y_2 - 2 * x_y
    return np.exp(-distance / (2 * sigma ** 2))


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate the confidence interval by nonparametric bootstrap-like procedure.

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
    The Hoeffding's inequality derives the confidence intervals of :math:`\\mu := \\mathbb{E}[X], X \\sim p(X)` with probability :math:`1 - \\alpha` as follows.

    .. math::

        |\\hat{\\mu} - \\mu]| \\leq X_{\\max} \\sqrt{\\frac{\\log(1 / \\alpha)}{2 n}}`

    where :math:`n` is the data size.

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
    The empirical bernstein inequality derives the confidence intervals of :math:`\\mu := \\mathbb{E}[X], X \\sim p(X)` with probability :math:`1 - \\alpha` as follows.

    .. math::

        |\\hat{\\mu} - \\mu]| \\leq \\frac{7 X_{\\max} \\log(2 / \\alpha)}{3 (n - 1)} + \\sqrt{\\frac{2 \\hat{\\mathbb{V}}(X) \\log(2 / \\alpha)}{n(n - 1)}}`

    where :math:`n` is the data size and :math:`\\hat{\\mathbb{V}}` is the sample variance.

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
    Based on this assumption, the confidence intervals of :math:`\\mu := \\mathbb{E}[X]` with probability :math:`1 - \\alpha` is derived as follows.

    .. math::

        |\\hat{\\mu} - \\mu]| \\leq \\frac{T_{\\mathrm{test}}(1 - \\alpha, n-1)}{\\sqrt{n} / \\hat{\\sigma}}``

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
    """Transform a defaultdict into the dict."""
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
        Minimum number allowed in the input array.

    max_val: float, default=None
        Maximum number allowed in the input array.

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
        return state, action, done, info

    def render(self, mode: str = "human"):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed: Optional[int] = None):
        self.env.reset(seed)

    def __getattr__(self, key) -> Any:
        return object.__getattribute__(self.env, key)
