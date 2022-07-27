"""Useful tools."""
from collections import defaultdict
from typing import DefaultDict, Dict, Union, Optional, Any

import scipy
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .types import LoggedDataset, OPEInputDict


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
        Significant level. The value should be within `[0, 1)`.

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
    """Estimate the confidence interval by the hoeffding' inequality.

    Note
    -------
    The hoeffding's inequality derives the confidence intervals of :math:`\\mu := \\mathbb{E}[X], X \\sim p(X)` with probability :math:`1 - \\alpha` as follows.

    .. math::

        |\\hat{\\mu} - \\mu]| \\leq X_{\\max} \\sqrt{\\frac{\\log(1 / \\alpha)}{2 n}}`

    where :math:`n` is the data size.

    Parameters
    -------
    samples: array-like
        Samples.

    alpha: float, default=0.05
        Significant level. The value should be within `[0, 1)`.

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

        |\\hat{\\mu} - \\mu]| \\leq \\frac{7 X_{\\max} \\log(2 / \\alpha)}{3 (n - 1)} + \\sqrt{\\frac{2 \hat{\mathbb{V}}(X) \log(2 / \\alpha)}{n(n - 1)}}`

    where :math:`n` is the data size and :math:`\\hat{\\mathbb{V}}` is the sample variance.

    Parameters
    -------
    samples: array-like
        Samples.

    alpha: float, default=0.05
        Significant level. The value should be within `[0, 1)`.

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

        |\\hat{\\mu} - \\mu]| \\leq \\frac{T_{\\mathrm{test}}(1 - \\alpha, n-1)}{\\sqrt{n} / \hat{\\sigma}}``

    where :math:`n` is the data size, :math:`T_{\\mathrm{test}}(\\cdot,\\cdot)` is the T-value, and :math:`\\sigma` is the standard deviation, respectively.

    Parameters
    -------
    samples: NDArray
        Samples.

    alpha: float, default=0.05
        Significant level. The value should be within `[0, 1)`.

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
        Excpected dimension of the input array.

    expected_dtype: {type, tuple of type}, default=None
        Excpected dtype of the input array.

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
        "n_episodes",
        "action_type",
        "n_actions",
        "action_dim",
        "state_dim",
        "step_per_episode",
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
            "evaluation_policy_step_wise_pscore",
            "evaluation_policy_trajectory_wise_pscore",
            "evaluation_policy_action",
            "evaluation_policy_action_dist",
            "state_action_value_prediction",
            "initial_state_value_prediction",
            "initial_state_action_distribution",
            "on_policy_policy_value",
            "gamma",
        ]:
            if expected_key not in input_dict_keys:
                raise RuntimeError(
                    f"{expected_key} does not exist in input_dict['{eval_policy}']"
                )