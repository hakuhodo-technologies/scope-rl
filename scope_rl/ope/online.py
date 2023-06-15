# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"On-Policy performance comparison."
from tqdm.auto import tqdm
from typing import List, Union, Optional
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.stats import norm

from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import gym
from gym.wrappers import TimeLimit
from d3rlpy.algos import AlgoBase
from sklearn.utils import check_scalar, check_random_state

from ..policy.head import BaseHead, OnlineHead
from ..utils import (
    estimate_confidence_interval_by_bootstrap,
    check_array,
)


def visualize_on_policy_policy_value(
    env: gym.Env,
    policies: List[Union[AlgoBase, BaseHead]],
    policy_names: List[str],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
    fig_dir: Optional[Path] = None,
    fig_name: str = "on_policy_policy_value.png",
):
    """Visualize on-policy policy value estimates of the given policies.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policies: list of {AlgoBase, BaseHead}
        List of policies to be evaluated.

    policy_names: list of str
        Name of policies.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    alpha: float, default=0.05
        Significance level. The value should be within `[0, 1)`.

    n_bootstrap_samples: int, default=10000 (> 0)
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default=None (>= 0)
        Random state.

    fig_dir: Path, default=None
        Path to store the bar figure.
        If `None` is given, the figure will not be saved.

    fig_name: str, default="on_policy_policy_value.png"
        Name of the bar figure.

    """
    check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)
    check_scalar(
        n_bootstrap_samples, name="n_bootstrap_samples", target_type=int, min_val=1
    )
    if random_state is None:
        raise ValueError("random_state must be given")
    check_random_state(random_state)

    if fig_dir is not None and not isinstance(fig_dir, Path):
        raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
    if fig_name is not None and not isinstance(fig_name, str):
        raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

    on_policy_Policy_value_dict = {}
    for policy, name in zip(policies, policy_names):
        on_policy_Policy_value_dict[name] = rollout_policy_online(
            env=env,
            policy=policy,
            n_trajectories=n_trajectories,
            step_per_trajectory=step_per_trajectory,
            evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
            gamma=gamma,
            random_state=random_state,
        )

    plt.style.use("ggplot")
    plt.figure(figsize=(2 * len(policies), 4))

    sns.barplot(
        data=DataFrame(on_policy_Policy_value_dict),
        ci=100 * (1 - alpha),
        n_boot=n_bootstrap_samples,
        seed=random_state,
    )

    plt.ylabel(
        f"On-Policy Policy Value (± {np.int64(100*(1 - alpha))}% CI)", fontsize=12
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if fig_dir:
        plt.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")


def visualize_on_policy_policy_value_with_variance(
    env: gym.Env,
    policies: List[Union[AlgoBase, BaseHead]],
    policy_names: List[str],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    fig_dir: Optional[Path] = None,
    fig_name: str = "estimated_policy_value.png",
) -> None:
    """Visualize the policy value estimated by OPE estimators.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policies: list of {AlgoBase, BaseHead}
        List of policies to be evaluated.

    policy_names: list of str
        Name of policies.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    alpha: float, default=0.05
        Significance level. The value should be within `[0, 1)`.

    random_state: int, default=None (>= 0)
        Random state.

    fig_dir: Path, default=None
        Path to store the bar figure.
        If `None` is given, the figure will not be saved.

    fig_name: str, default="estimated_policy_value.png"
        Name of the bar figure.

    """
    check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)
    if fig_dir is not None and not isinstance(fig_dir, Path):
        raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
    if fig_name is not None and not isinstance(fig_name, str):
        raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(2 * len(policies), 4))
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_colors = len(color)

    n_policies = len(policies)
    mean = np.zeros(n_policies)
    variance = np.zeros(n_policies)

    for i, policy in enumerate(policies):
        statistics_dict = calc_on_policy_statistics(
            env=env,
            policy=policy,
            n_trajectories=n_trajectories,
            step_per_trajectory=step_per_trajectory,
            evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
            gamma=gamma,
            quartile_alpha=alpha,
            random_state=random_state,
        )
        mean[i] = statistics_dict["mean"]
        variance[i] = statistics_dict["variance"]

    upper, lower = norm.interval(1 - alpha, loc=mean, scale=np.sqrt(variance))

    for i in range(n_policies):
        ax.errorbar(
            np.arange(i, i + 1),
            mean[i],
            xerr=[0.4],
            yerr=[
                np.array([mean[i] - lower[i]]),
                np.array([upper[i] - mean[i]]),
            ],
            color=color[i % n_colors],
            elinewidth=5.0,
        )

    elines = ax.get_children()
    for i in range(n_policies):
        elines[2 * i + 1].set_color("black")
        elines[2 * i + 1].set_linewidth(2.0)

    ax.set_xticks(np.arange(n_policies))
    ax.set_xticklabels(policy_names)
    ax.set_ylabel(
        f"On-Policy Policy Value (± {np.int64(100*(1 - alpha))}% CI)",
        fontsize=12,
    )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlim(-0.5, n_policies - 0.5)

    if fig_dir:
        fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")


def visualize_on_policy_cumulative_distribution_function(
    env: gym.Env,
    policies: List[Union[AlgoBase, BaseHead]],
    policy_names: List[str],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    use_custom_reward_scale: bool = False,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    n_partition: Optional[int] = None,
    random_state: Optional[int] = None,
    legend: bool = True,
    fig_dir: Optional[Path] = None,
    fig_name: str = "on_policy_cumulative_distribution_function.png",
) -> None:
    """Visualize the cumulative distribution function of the on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    use_custom_reward_scale: bool, default=False
        Whether to use a customized reward scale or the reward observed under the behavior policy.
        If True, the reward scale is uniform, following Huang et al. (2021).
        If False, the reward scale follows the one defined in Chundak et al. (2021).

    scale_min: float, default=None
        Minimum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    n_partition: int, default=None
        Number of partitions in the reward scale (x-axis of the CDF).
        When `use_custom_reward_scale == True`, a value must be given.

    random_state: int, default=None (>= 0)
        Random state.

    legend: bool, default=True
        Whether to include a legend in the figure.

    fig_dir: Path, default=None
        Path to store the bar figure.
        If `None` is given, the figure will not be saved.

    fig_name: str, default="on_policy_cumulative_distribution_function.png"
        Name of the bar figure.

    """
    if fig_dir is not None and not isinstance(fig_dir, Path):
        raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
    if fig_name is not None and not isinstance(fig_name, str):
        raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(4, 3))

    for policy, policy_name in zip(policies, policy_names):
        cdf, reward_scale = calc_on_policy_cumulative_distribution_function(
            env=env,
            policy=policy,
            n_trajectories=n_trajectories,
            step_per_trajectory=step_per_trajectory,
            evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
            gamma=gamma,
            scale_min=scale_min,
            scale_max=scale_max,
            n_partition=n_partition,
            use_custom_reward_scale=use_custom_reward_scale,
            random_state=random_state,
        )
        ax.plot(reward_scale, cdf, label=policy_name)

    ax.set_title("cumulative distribution function")
    ax.set_xlabel("trajectory-wise reward")
    ax.set_ylabel("cumulative probability")
    if legend:
        ax.legend()

    fig.tight_layout()
    plt.show()

    if fig_dir:
        fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")


def visualize_on_policy_conditional_value_at_risk(
    env: gym.Env,
    policies: List[Union[AlgoBase, BaseHead]],
    policy_names: List[str],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    alphas: Optional[np.ndarray] = None,
    use_custom_reward_scale: bool = False,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    n_partition: Optional[int] = None,
    random_state: Optional[int] = None,
    legend: bool = True,
    fig_dir: Optional[Path] = None,
    fig_name: str = "on_policy_conditional_value_at_risk.png",
) -> None:
    """Visualize the conditional value at risk of the on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    alphas: array-like of shape (n_alpha, ), default=None
        Set of proportions of the shaded region. The values should be within `[0, 1)`.
        If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

    use_custom_reward_scale: bool, default=False
        Whether to use a customized reward scale or the reward observed under the behavior policy.
        If True, the reward scale is uniform, following Huang et al. (2021).
        If False, the reward scale follows the one defined in Chundak et al. (2021).

    scale_min: float, default=None
        Minimum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    n_partition: int, default=None
        Number of partitions in the reward scale (x-axis of the CDF).
        When `use_custom_reward_scale == True`, a value must be given.

    random_state: int, default=None (>= 0)
        Random state.

    legend: bool, default=True
        Whether to include a legend in the figure.

    fig_dir: Path, default=None
        Path to store the bar figure.
        If `None` is given, the figure will not be saved.

    fig_name: str, default="on_policy_conditional_value_at_risk.png"
        Name of the bar figure.

    """
    if fig_dir is not None and not isinstance(fig_dir, Path):
        raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
    if fig_name is not None and not isinstance(fig_name, str):
        raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

    plt.style.use("ggplot")
    fig, ax = plt.subplots((2 * len(policies), 4))

    for policy, policy_name in zip(policies, policy_names):
        cvar = calc_on_policy_conditional_value_at_risk(
            env=env,
            policy=policy,
            n_trajectories=n_trajectories,
            step_per_trajectory=step_per_trajectory,
            evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
            gamma=gamma,
            alphas=alphas,
            scale_min=scale_min,
            scale_max=scale_max,
            n_partition=n_partition,
            use_custom_reward_scale=use_custom_reward_scale,
            random_state=random_state,
        )
        ax.plot(alphas, cvar, label=policy_name)

    ax.set_title("conditional value at risk (CVaR)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("CVaR")
    if legend:
        ax.legend()

    fig.tight_layout()
    plt.show()

    if fig_dir:
        fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")


def visualize_on_policy_interquartile_range(
    env: gym.Env,
    policies: List[Union[AlgoBase, BaseHead]],
    policy_names: List[str],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    alpha: float = 0.05,
    use_custom_reward_scale: bool = False,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    n_partition: Optional[int] = None,
    random_state: Optional[int] = None,
    fig_dir: Optional[Path] = None,
    fig_name: str = "on_policy_interquartile_range.png",
) -> None:
    """Visualize the interquartile range of the on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    alpha: float, default=0.05
        Significance level. The value should be within `[0, 1)`.

    use_custom_reward_scale: bool, default=False
        Whether to use a customized reward scale or the reward observed under the behavior policy.
        If True, the reward scale is uniform, following Huang et al. (2021).
        If False, the reward scale follows the one defined in Chundak et al. (2021).

    scale_min: float, default=None
        Minimum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    n_partition: int, default=None
        Number of partitions in the reward scale (x-axis of the CDF).
        When `use_custom_reward_scale == True`, a value must be given.

    random_state: int, default=None (>= 0)
        Random state.

    fig_dir: Path, default=None
        Path to store the bar figure.
        If `None` is given, the figure will not be saved.

    fig_name: str, default="on_policy_conditional_value_at_risk.png"
        Name of the bar figure.

    """
    if fig_dir is not None and not isinstance(fig_dir, Path):
        raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
    if fig_name is not None and not isinstance(fig_name, str):
        raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

    plt.style.use("ggplot")
    fig, ax = plt.subplots((2 * len(policies), 4))
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_colors = len(color)

    n_policies = len(policies)
    mean = np.zeros(n_policies)
    median = np.zeros(n_policies)
    upper = np.zeros(n_policies)
    lower = np.zeros(n_policies)

    for i, policy in enumerate(policies):
        statistics_dict = calc_on_policy_statistics(
            env=env,
            policy=policy,
            n_trajectories=n_trajectories,
            step_per_trajectory=step_per_trajectory,
            evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
            gamma=gamma,
            quartile_alpha=alpha,
            scale_min=scale_min,
            scale_max=scale_max,
            n_partition=n_partition,
            use_custom_reward_scale=use_custom_reward_scale,
            random_state=random_state,
        )
        mean[i] = statistics_dict["mean"]
        median[i] = statistics_dict["interquartile_range"]["median"]
        upper[i] = statistics_dict["interquartile_range"][
            f"{100 * (1. - alpha)}% quartile (upper)"
        ]
        lower[i] = statistics_dict["interquartile_range"][
            f"{100 * (1. - alpha)}% quartile (lower)"
        ]

    ax.bar(
        np.arange(n_policies),
        upper - lower,
        bottom=lower,
        color=color,
        edgecolor="black",
        linewidth=0.3,
        tick_label=policy_names,
        alpha=0.3,
    )

    for i in range(n_policies):
        ax.errorbar(
            np.arange(i, i + 1),
            median[i],
            xerr=[0.4],
            color=color[i % n_colors],
            elinewidth=5.0,
            fmt="o",
            markersize=0.1,
        )
        ax.errorbar(
            np.arange(i, i + 1),
            mean[i],
            color=color[i % n_colors],
            fmt="o",
            markersize=10.0,
        )

        ax.set_title("interquartile range")
        ax.set_ylabel(
            f"{np.int64(100*(1 - alpha))}% range",
            fontsize=12,
        )
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlim(-0.5, n_policies - 0.5)

    if fig_dir:
        fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")


def calc_on_policy_statistics(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    quartile_alpha: float = 0.05,
    cvar_alpha: float = 0.05,
    use_custom_reward_scale: bool = False,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    n_partition: Optional[int] = None,
    random_state: Optional[int] = None,
):
    """Calculate the mean, variance, conditional value at risk, interquartile range of the on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    quartile_alpha: float, default=0.05
        Proportion of the shaded region. The value should be within (0, 1].

    cvar_alpha: float, default=0.05
        Proportion of the shaded region. The value should be within (0, 1].

    use_custom_reward_scale: bool, default=False
        Whether to use a customized reward scale or the reward observed under the behavior policy.
        If True, the reward scale is uniform, following Huang et al. (2021).
        If False, the reward scale follows the one defined in Chundak et al. (2021).

    scale_min: float, default=None
        Minimum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    n_partition: int, default=None
        Number of partitions in the reward scale (x-axis of the CDF).
        When `use_custom_reward_scale == True`, a value must be given.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    statistics_dict: dict
        Dictionary containing the mean, variance, CVaR, and interquartile range of the on-policy policy value.

    """
    check_scalar(
        quartile_alpha,
        name="quartile_alpha",
        target_type=float,
        min_val=0.0,
        max_val=0.5,
    )
    check_scalar(
        cvar_alpha, name="cvar_alpha", target_type=float, min_val=0.0, max_val=1.0
    )
    if use_custom_reward_scale:
        if scale_min is None:
            raise ValueError(
                "scale_min must be given when `use_custom_reward_scale == True`"
            )
        if scale_max is None:
            raise ValueError(
                "scale_max must be given when `use_custom_reward_scale == True`"
            )
        if n_partition is None:
            raise ValueError(
                "n_partition must be given when `use_custom_reward_scale == True`"
            )
        check_scalar(
            scale_min,
            name="scale_min",
            target_type=float,
        )
        check_scalar(
            scale_max,
            name="scale_max",
            target_type=float,
        )
        check_scalar(
            n_partition,
            name="n_partition",
            target_type=int,
            min_val=1,
        )

    on_policy_policy_values = rollout_policy_online(
        env=env,
        policy=policy,
        n_trajectories=n_trajectories,
        step_per_trajectory=step_per_trajectory,
        evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
        gamma=gamma,
        random_state=random_state,
    )

    mean = on_policy_policy_values.mean()
    variance = on_policy_policy_values.var(ddof=1)

    if use_custom_reward_scale:
        reward_scale = np.linspace(scale_min, scale_max, num=n_partition)
    else:
        reward_scale = np.sort(np.unique(on_policy_policy_values))

    density = np.histogram(
        on_policy_policy_values,
        bins=reward_scale,
        density=True,
    )[0]
    probability_density_function = density * np.diff(reward_scale)
    cumulative_distribution_function = np.insert(
        probability_density_function, 0, 0
    ).cumsum()

    idx = np.nonzero(cumulative_distribution_function[1:] > cvar_alpha)[0]
    if len(idx) == 0:
        cvar = (
            np.diff(cumulative_distribution_function) * reward_scale[1:]
        ).sum() / cumulative_distribution_function[-1]
    elif idx[0] == 0:
        cvar = reward_scale[1]
    else:
        lower_idx = idx[0]
        relative_probability_density = (
            np.diff(cumulative_distribution_function)[: lower_idx + 1]
            / cumulative_distribution_function[lower_idx + 1]
        )
        cvar = (relative_probability_density * reward_scale[1 : lower_idx + 2]).sum()

    def target_value_given_idx(idx):
        if len(idx):
            target_idx = idx[0]
            target_value = (reward_scale[target_idx] + reward_scale[target_idx + 1]) / 2
        else:
            target_value = reward_scale[-1]
        return target_value

    lower_idx = np.nonzero(density.cumsum() > quartile_alpha)[0]
    median_idx = np.nonzero(density.cumsum() > 0.5)[0]
    upper_idx = np.nonzero(density.cumsum() > 1 - quartile_alpha)[0]

    interquartile_range_dict = {
        "median": target_value_given_idx(median_idx),
        f"{100 * (1. - quartile_alpha)}% quartile (lower)": target_value_given_idx(
            lower_idx
        ),
        f"{100 * (1. - quartile_alpha)}% quartile (upper)": target_value_given_idx(
            upper_idx
        ),
    }

    statistics_dict = {
        "mean": mean,
        "variance": variance,
        "conditional_value_at_risk": cvar,
        "interquartile_range": interquartile_range_dict,
    }

    return statistics_dict


def calc_on_policy_policy_value(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    alpha: float = 0.05,
    use_bootstrap: bool = False,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
):
    """Calculate an on-policy policy value of a given policy.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    alpha: float, default=0.05
        Significance level. The value should be within (0, 1].

    use_bootstrap: bool, default=False
        Whether to use bootstrap sampling or not.

    n_bootstrap_samples: int, default=10000 (> 0)
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    on_policy_policy_value: float
        Average on-policy policy value.

    """
    if use_bootstrap:
        on_policy_policy_value = calc_on_policy_policy_value_interval(
            env=env,
            policy=policy,
            n_trajectories=n_trajectories,
            step_per_trajectory=step_per_trajectory,
            evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
            gamma=gamma,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    else:
        on_policy_policy_value = rollout_policy_online(
            env=env,
            policy=policy,
            n_trajectories=n_trajectories,
            step_per_trajectory=step_per_trajectory,
            evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
            gamma=gamma,
            random_state=random_state,
        ).mean()
    return on_policy_policy_value


def calc_on_policy_policy_value_interval(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
):
    """Estimate the confidence interval of on-policy policy value by nonparametric bootstrap.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    alpha: float, default=0.05
        Significance level. The value should be within `[0, 1)`.

    n_bootstrap_samples: int, default=10000 (> 0)
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    on_policy_confidence_interval: dict
        Dictionary storing the calculated mean and upper-lower confidence bounds.

    """
    on_policy_policy_values = rollout_policy_online(
        env=env,
        policy=policy,
        n_trajectories=n_trajectories,
        step_per_trajectory=step_per_trajectory,
        evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
        gamma=gamma,
        random_state=random_state,
    )
    return estimate_confidence_interval_by_bootstrap(
        samples=on_policy_policy_values,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )


def calc_on_policy_variance(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    random_state: Optional[int] = None,
):
    """Calculate the variance of the on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    on_policy_variance: float
        Variance of the on-policy policy value.

    """
    on_policy_policy_values = rollout_policy_online(
        env=env,
        policy=policy,
        n_trajectories=n_trajectories,
        step_per_trajectory=step_per_trajectory,
        evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
        gamma=gamma,
        random_state=random_state,
    )
    return on_policy_policy_values.var(ddof=1)


def calc_on_policy_conditional_value_at_risk(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    alphas: Optional[Union[np.ndarray, float]] = None,
    use_custom_reward_scale: bool = False,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    n_partition: Optional[int] = None,
    random_state: Optional[int] = None,
):
    """Calculate the conditional value at risk (CVaR) of the on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    alphas: array-like of shape (n_alpha, ) or float, default=None
        Set of proportions of the shaded region. The values should be within `[0, 1)`.
        If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

    use_custom_reward_scale: bool, default=False
        Whether to use a customized reward scale or the reward observed under the behavior policy.
        If True, the reward scale is uniform, following Huang et al. (2021).
        If False, the reward scale follows the one defined in Chundak et al. (2021).

    scale_min: float, default=None
        Minimum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    n_partition: int, default=None
        Number of partitions in the reward scale (x-axis of the CDF).
        When `use_custom_reward_scale == True`, a value must be given.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    on_policy_conditional_value_at_risk: np.ndarray
        CVaR of the on-policy policy value.

    """
    if alphas is None:
        alphas = np.linspace(0, 1, 21)
    elif isinstance(alphas, float):
        check_scalar(alphas, name="alphas", target_type=float, min_val=0.0, max_val=1.0)
        alphas = np.array([alphas], dtype=float)
    elif isinstance(alphas, np.ndarray):
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
    else:
        raise ValueError(
            f"alphas must be float or np.ndarray, but {type(alphas)} is given"
        )
    if use_custom_reward_scale:
        if scale_min is None:
            raise ValueError(
                "scale_min must be given when `use_custom_reward_scale == True`"
            )
        if scale_max is None:
            raise ValueError(
                "scale_max must be given when `use_custom_reward_scale == True`"
            )
        if n_partition is None:
            raise ValueError(
                "n_partition must be given when `use_custom_reward_scale == True`"
            )
        check_scalar(
            scale_min,
            name="scale_min",
            target_type=float,
        )
        check_scalar(
            scale_max,
            name="scale_max",
            target_type=float,
        )
        check_scalar(
            n_partition,
            name="n_partition",
            target_type=int,
            min_val=1,
        )

    on_policy_policy_values = rollout_policy_online(
        env=env,
        policy=policy,
        n_trajectories=n_trajectories,
        step_per_trajectory=step_per_trajectory,
        evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
        gamma=gamma,
        random_state=random_state,
    )

    if use_custom_reward_scale:
        reward_scale = np.linspace(scale_min, scale_max, num=n_partition)
    else:
        reward_scale = np.sort(np.unique(on_policy_policy_values))

    density = np.histogram(
        on_policy_policy_values,
        bins=reward_scale,
        density=True,
    )[0]
    probability_density_function = density * np.diff(reward_scale)
    cumulative_distribution_function = np.insert(
        probability_density_function, 0, 0
    ).cumsum()

    cvar = np.zeros_like(alphas)
    for i, alpha in enumerate(alphas):
        idx_ = np.nonzero(cumulative_distribution_function[1:] > alpha)[0]
        if len(idx_) == 0:
            cvar[i] = (
                np.diff(cumulative_distribution_function) * reward_scale[1:]
            ).sum() / cumulative_distribution_function[-1]
        elif idx_[0] == 0:
            cvar[i] = reward_scale[1]
        else:
            lower_idx_ = idx_[0]
            relative_probability_density = (
                np.diff(cumulative_distribution_function)[: lower_idx_ + 1]
                / cumulative_distribution_function[lower_idx_ + 1]
            )
            cvar[i] = (
                relative_probability_density * reward_scale[1 : lower_idx_ + 2]
            ).sum()

    return cvar


def calc_on_policy_interquartile_range(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    alpha: float = 0.05,
    use_custom_reward_scale: bool = False,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    n_partition: Optional[int] = None,
    random_state: Optional[int] = None,
):
    """Calculate the interquartile range of the on-policy policy value.

    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    alpha: float, default=0.05
        Proportion of the shaded region. The value should be within (0, 1].

    use_custom_reward_scale: bool, default=False
        Whether to use a customized reward scale or the reward observed under the behavior policy.
        If True, the reward scale is uniform, following Huang et al. (2021).
        If False, the reward scale follows the one defined in Chundak et al. (2021).

    scale_min: float, default=None
        Minimum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    n_partition: int, default=None
        Number of partitions in the reward scale (x-axis of the CDF).
        When `use_custom_reward_scale == True`, a value must be given.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    interquartile_range_dict: dict
        Dictionary containing the interquartile range of the on-policy policy value.

    """
    check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)
    if use_custom_reward_scale:
        if scale_min is None:
            raise ValueError(
                "scale_min must be given when `use_custom_reward_scale == True`"
            )
        if scale_max is None:
            raise ValueError(
                "scale_max must be given when `use_custom_reward_scale == True`"
            )
        if n_partition is None:
            raise ValueError(
                "n_partition must be given when `use_custom_reward_scale == True`"
            )
        check_scalar(
            scale_min,
            name="scale_min",
            target_type=float,
        )
        check_scalar(
            scale_max,
            name="scale_max",
            target_type=float,
        )
        check_scalar(
            n_partition,
            name="n_partition",
            target_type=int,
            min_val=1,
        )

    on_policy_policy_values = rollout_policy_online(
        env=env,
        policy=policy,
        n_trajectories=n_trajectories,
        step_per_trajectory=step_per_trajectory,
        evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
        gamma=gamma,
        random_state=random_state,
    )

    if use_custom_reward_scale:
        reward_scale = np.linspace(scale_min, scale_max, num=n_partition)
    else:
        reward_scale = np.sort(np.unique(on_policy_policy_values))

    def target_value_given_idx(idx):
        if len(idx):
            target_idx = idx[0]
            target_value = (reward_scale[target_idx] + reward_scale[target_idx + 1]) / 2
        else:
            target_value = reward_scale[-1]
        return target_value

    density = np.histogram(
        on_policy_policy_values,
        bins=reward_scale,
        density=True,
    )[0]
    density = density * np.diff(reward_scale)

    lower_idx = np.nonzero(density.cumsum() > alpha)[0]
    median_idx = np.nonzero(density.cumsum() > 0.5)[0]
    upper_idx = np.nonzero(density.cumsum() > 1 - alpha)[0]

    interquartile_range_dict = {
        "median": target_value_given_idx(median_idx),
        f"{100 * (1. - alpha)}% quartile (lower)": target_value_given_idx(lower_idx),
        f"{100 * (1. - alpha)}% quartile (upper)": target_value_given_idx(upper_idx),
    }

    return interquartile_range_dict


def calc_on_policy_cumulative_distribution_function(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    use_custom_reward_scale: bool = False,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    n_partition: Optional[int] = None,
    random_state: Optional[int] = None,
):
    """Calculate the cumulative distribution of the on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    use_custom_reward_scale: bool, default=False
        Whether to use a customized reward scale or the reward observed under the behavior policy.
        If True, the reward scale is uniform, following Huang et al. (2021).
        If False, the reward scale follows the one defined in Chundak et al. (2021).

    scale_min: float, default=None
        Minimum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in the CDF.
        When `use_custom_reward_scale == True`, a value must be given.

    n_partition: int, default=None
        Number of partitions in the reward scale (x-axis of the CDF).
        When `use_custom_reward_scale == True`, a value must be given.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    cumulative_distribution_function: np.ndarray
        Cumulative distribution function of the on-policy policy value.

    reward_scale: ndarray of shape (n_unique_reward, ) or (n_partition, )
        Reward Scale (x-axis of the cumulative distribution function).

    """
    if use_custom_reward_scale:
        if scale_min is None:
            raise ValueError(
                "scale_min must be given when `use_custom_reward_scale == True`"
            )
        if scale_max is None:
            raise ValueError(
                "scale_max must be given when `use_custom_reward_scale == True`"
            )
        if n_partition is None:
            raise ValueError(
                "n_partition must be given when `use_custom_reward_scale == True`"
            )
        check_scalar(
            scale_min,
            name="scale_min",
            target_type=float,
        )
        check_scalar(
            scale_max,
            name="scale_max",
            target_type=float,
        )
        check_scalar(
            n_partition,
            name="n_partition",
            target_type=int,
            min_val=1,
        )

    on_policy_policy_values = rollout_policy_online(
        env=env,
        policy=policy,
        n_trajectories=n_trajectories,
        step_per_trajectory=step_per_trajectory,
        evaluate_on_stationary_distribution=evaluate_on_stationary_distribution,
        gamma=gamma,
        random_state=random_state,
    )

    if use_custom_reward_scale:
        reward_scale = np.linspace(scale_min, scale_max, num=n_partition)
    else:
        reward_scale = np.sort(np.unique(on_policy_policy_values))

    density = np.histogram(
        on_policy_policy_values,
        bins=reward_scale,
        density=True,
    )[0]
    density = density * np.diff(reward_scale)

    cumulative_distribution_function = np.insert(density, 0, 0).cumsum()
    return cumulative_distribution_function, reward_scale


def rollout_policy_online(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_trajectories: int = 100,
    step_per_trajectory: Optional[int] = None,
    evaluate_on_stationary_distribution: bool = False,
    gamma: float = 1.0,
    random_state: Optional[int] = None,
):
    """Rollout a given policy on the environment and generate trajectory-wise rewards under the policy online.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_trajectories: int, default=100 (> 0)
        Number of trajectories to rollout.

    step_per_trajectory: int, default=None (> 0)
        Number of timesteps in an trajectory.

    evaluate_on_stationary_distribution: bool, default=False
        Whether to evaluate a policy based on the stationary state distribution induced by it.
        When True, the evaluation policy is evaluated by rollout without resetting environment at each trajectory.
        This argument is irrelevant when working on the finite horizon setting.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    on_policy_policy_values: ndarray of shape (n_trajectories, )
        Trajectory-wise on-policy policy values.

    """
    if not isinstance(env, gym.Env):
        raise ValueError(
            "env must be a child class of gym.Env",
        )
    if not isinstance(policy, (AlgoBase, BaseHead)):
        raise ValueError("policy must be a child class of either AlgoBase or BaseHead")
    check_scalar(
        n_trajectories,
        name="n_trajectories",
        target_type=int,
        min_val=1,
    )
    check_scalar(
        gamma,
        name="gamma",
        target_type=float,
        min_val=0.0,
        max_val=1.0,
    )
    if step_per_trajectory is not None:
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
    elif evaluate_on_stationary_distribution:
        raise ValueError(
            "step_per_trajectory must be given when `evaluate_on_stationary_distribution == True`."
        )
    elif isinstance(env, TimeLimit):
        step_per_trajectory = env.spec.max_episode_steps
    else:
        step_per_trajectory = np.infty
        warn(
            "step_per_trajectory is currently set to np.infty. The evaluation may not end if the environment will not terminate by themselves."
        )

    on_policy_policy_values = np.zeros(n_trajectories)
    env.reset(seed=random_state)

    if not isinstance(policy, BaseHead):
        policy = OnlineHead(policy, name="tmp")

        if evaluate_on_stationary_distribution:
            done = True

            for i in tqdm(
                range(n_trajectories),
                desc="[calculate on-policy policy value]",
                total=n_trajectories,
            ):
                cumulative_reward = 0

                if done:
                    state, _ = env.reset()
                    done = False

                for t in range(step_per_trajectory):
                    action = policy.predict_online(state)
                    state, reward, done, _, _ = env.step(action)
                    cumulative_reward += gamma**t * reward

                on_policy_policy_values[i] = cumulative_reward

        else:
            for i in tqdm(
                range(n_trajectories),
                desc="[calculate on-policy policy value]",
                total=n_trajectories,
            ):
                state, _ = env.reset()
                done = False
                episode_reward = 0
                t = 0

                while not done:
                    action = policy.predict_online(state)
                    state, reward, done, _, _ = env.step(action)
                    t += 1

                    episode_reward += gamma**t * reward
                    done = done or (t == step_per_trajectory)

                on_policy_policy_values[i] = episode_reward

    else:
        if evaluate_on_stationary_distribution:
            done = True

            for i in tqdm(
                range(n_trajectories),
                desc="[calculate on-policy policy value]",
                total=n_trajectories,
            ):
                cumulative_reward = 0

                if done:
                    state, _ = env.reset()
                    done = False

                for t in range(step_per_trajectory):
                    action = policy.sample_action_online(state)
                    state, reward, done, _, _ = env.step(action)
                    cumulative_reward += gamma**t * reward

                on_policy_policy_values[i] = cumulative_reward

        else:
            for i in tqdm(
                range(n_trajectories),
                desc="[calculate on-policy policy value]",
                total=n_trajectories,
            ):
                state, _ = env.reset()
                done = False
                episode_reward = 0
                t = 0

                while not done:
                    action = policy.sample_action_online(state)
                    state, reward, done, _, _ = env.step(action)
                    t += 1

                    episode_reward += gamma**t * reward
                    done = done or (t == step_per_trajectory)

                on_policy_policy_values[i] = episode_reward

    return on_policy_policy_values
