"On-Policy Performance Comparison."
from pandas.core.frame import DataFrame
from tqdm.autonotebook import tqdm
from typing import List, Union, Optional
from pathlib import Path

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import gym
from d3rlpy.algos import AlgoBase

from ..policy.head import BaseHead, OnlineHead
from ..utils import estimate_confidence_interval_by_bootstrap


def visualize_on_policy_policy_value(
    env: gym.Env,
    policies: List[Union[AlgoBase, BaseHead]],
    policy_names: List[str],
    n_episodes: int = 100,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
    fig_dir: Optional[Path] = None,
    fig_name: str = "on_policy_policy_value.png",
):
    """Visualize on-policy policy value of the given policies.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policies: List[Union[AlgoBase, BaseHead]]
        List of policies to be evaluated.

    policy_names: List[str]
        Name of policies.

    n_episodes: int, default=100 (> 0)
        Number of trajectories to rollout.

    alpha: float, default=0.05 (0, 1)
        Significant level.

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
    on_policy_Policy_value_dict = {}
    for policy, name in zip(policies, policy_names):
        on_policy_Policy_value_dict[name] = rollout_policy_online(
            env=env,
            policy=policy,
            n_episodes=n_episodes,
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
    plt.ylabel(f"On-Policy Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if fig_dir:
        fig.savefig(str(fig_dir / fig_name))


def calc_on_policy_policy_value(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_episodes: int = 100,
    use_bootstrap: bool = False,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
):
    """Calculate on-policy policy value of the given policy.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: Union[AlgoBase, BaseHead]
        A policy to be evaluated.

    use_bootstrap: bool, default=False
        Whether to use bootstrap sampling or not.

    n_episodes: int, default=100 (> 0)
        Number of trajectories to rollout.

    alpha: float, default=0.05 (0, 1)
        Significant level.

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
            n_episodes=n_episodes,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    else:
        on_policy_policy_value = rollout_policy_online(
            env=env,
            policy=policy,
            n_episodes=n_episodes,
            random_state=random_state,
        ).mean()
    return on_policy_policy_value


def calc_on_policy_policy_value_interval(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_episodes: int = 100,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
):
    """Calculate confidence interval of on-policy policy value by nonparametric bootstrap procedure.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: Union[AlgoBase, BaseHead]
        A policy to be evaluated.

    n_episodes: int, default=100 (> 0)
        Number of trajectories to rollout.

    alpha: float, default=0.05 (0, 1)
        Significant level.

    n_bootstrap_samples: int, default=10000 (> 0)
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    on_policy_confidence_interval: Dict[str, float]
        Dictionary storing the calculated mean and upper-lower confidence bounds.

    """
    on_policy_policy_values = rollout_policy_online(
        env=env,
        policy=policy,
        n_episodes=n_episodes,
        random_state=random_state,
    )
    return estimate_confidence_interval_by_bootstrap(
        samples=on_policy_policy_values,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )


def rollout_policy_online(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_episodes: int = 100,
    random_state: Optional[int] = None,
):
    """Rollout policy on the environment and collect trajectory-wise on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: Union[AlgoBase, BaseHead]
        A policy to be evaluated.

    n_episodes: int, default=100 (> 0)
        Number of trajectories to rollout.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    on_policy_policy_values: NDArray
        Trajectory-wise on-policy policy values.

    """
    on_policy_policy_values = np.zeros(n_episodes)
    env.seed(random_state)

    if isinstance(policy, AlgoBase):
        policy = OnlineHead(policy)

        for i in tqdm(
            range(n_episodes),
            desc="[calculate on-policy policy value]",
            total=n_episodes,
        ):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = policy.predict_online(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward

            on_policy_policy_values[i] = episode_reward

    else:
        for i in tqdm(
            range(n_episodes),
            desc="[calculate on-policy policy value]",
            total=n_episodes,
        ):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = policy.sample_action_online(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward

            on_policy_policy_values[i] = episode_reward

    return on_policy_policy_values
