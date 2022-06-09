"On-Policy Performance Comparison."
from tqdm.autonotebook import tqdm
from typing import List, Union, Optional
from pathlib import Path

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import gym
from d3rlpy.algos import AlgoBase
from sklearn.utils import check_scalar, check_random_state

from ..policy.head import BaseHead, OnlineHead
from ..utils import (
    estimate_confidence_interval_by_bootstrap,
)


def visualize_on_policy_policy_value(
    env: gym.Env,
    policies: List[Union[AlgoBase, BaseHead]],
    policy_names: List[str],
    n_episodes: int = 100,
    gamma: float = 1.0,
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

    policies: list of {AlgoBase, BaseHead}
        List of policies to be evaluated.

    policy_names: list of str
        Name of policies.

    n_episodes: int, default=100 (> 0)
        Number of trajectories to rollout.

    gamma: float, default=1.0
        Discount factor. The value should be within `(0, 1]`.

    alpha: float, default=0.05
        Significant level. The value should be within `[0, 1)`.

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
            n_episodes=n_episodes,
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
    plt.ylabel(f"On-Policy Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if fig_dir:
        plt.savefig(str(fig_dir / fig_name))


def calc_on_policy_policy_value(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_episodes: int = 100,
    gamma: float = 1.0,
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

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    use_bootstrap: bool, default=False
        Whether to use bootstrap sampling or not.

    gamma: float, default=1.0
        Discount factor. The value should be within `(0, 1]`.

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
            gamma=gamma,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    else:
        on_policy_policy_value = rollout_policy_online(
            env=env,
            policy=policy,
            n_episodes=n_episodes,
            gamma=gamma,
            random_state=random_state,
        ).mean()
    return on_policy_policy_value


def calc_on_policy_policy_value_interval(
    env: gym.Env,
    policy: Union[AlgoBase, BaseHead],
    n_episodes: int = 100,
    gamma: float = 1.0,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 100,
    random_state: Optional[int] = None,
):
    """Calculate confidence interval of on-policy policy value by nonparametric bootstrap procedure.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_episodes: int, default=100 (> 0)
        Number of trajectories to rollout.

    gamma: float, default=1.0
        Discount factor. The value should be within `(0, 1]`.

    alpha: float, default=0.05
        Significant level. The value should be within `[0, 1)`.

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
        n_episodes=n_episodes,
        gamma=gamma,
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
    gamma: float = 1.0,
    random_state: Optional[int] = None,
):
    """Rollout policy on the environment and collect trajectory-wise on-policy policy value.

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    policy: {AlgoBase, BaseHead}
        A policy to be evaluated.

    n_episodes: int, default=100 (> 0)
        Number of trajectories to rollout.

    gamma: float, default=1.0
        Discount factor. The value should be within `(0, 1]`.

    random_state: int, default=None (>= 0)
        Random state.

    Return
    -------
    on_policy_policy_values: ndarray of shape (n_episodes, )
        Trajectory-wise on-policy policy values.

    """
    if not isinstance(env, gym.Env):
        raise ValueError(
            "env must be a child class of gym.Env",
        )
    if not isinstance(policy, (AlgoBase, BaseHead)):
        raise ValueError("policy must be a child class of either AlgoBase or BaseHead")
    check_scalar(
        n_episodes,
        name="n_episodes",
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

    on_policy_policy_values = np.zeros(n_episodes)
    env.seed(random_state)

    if not isinstance(policy, BaseHead):
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

            t = 0
            while not done:
                action = policy.sample_action_online(state)
                state, reward, done, _ = env.step(action)
                episode_reward += gamma ** t * reward
                t += 1

            on_policy_policy_values[i] = episode_reward

    return on_policy_policy_values
