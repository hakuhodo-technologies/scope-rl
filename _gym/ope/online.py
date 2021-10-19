"On-Policy Performance Comparison."
from pandas.core.frame import DataFrame
from tqdm import tqdm
from typing import List, Union, Optional
from pathlib import Path

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import gym
from d3rlpy.algos import AlgoBase

from _gym.policy import BaseHead, OnlineHead
from _gym.utils import estimate_confidence_interval_by_bootstrap


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
    on_policy_Policy_value_dict = {}
    for policy, name in zip(policies, policy_names):
        on_policy_Policy_value_dict[name] = rollout_policy_online(
            env=env,
            policy=policy,
            n_episodes=n_episodes,
            random_state=random_state,
        )
    plt.style.use("ggplot")
    sns.barplot(
        data=DataFrame(on_policy_Policy_value_dict),
        ci=100 * (1 - alpha),
        n_boot=n_bootstrap_samples,
        seed=random_state,
    )
    plt.ylabel(f"On-Policy Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=8)
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
