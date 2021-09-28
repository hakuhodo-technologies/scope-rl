"""Synthetic Dataset Generation."""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from tqdm import tqdm

import gym
from gym.spaces import Discrete
import numpy as np
from sklearn.utils import check_random_state

from _gym.env import RTBEnv, CustomizedRTBEnv
from _gym.dataset import BaseDataset
from _gym.policy import BaseHead
from _gym.types import LoggedDataset
from _gym.utils import check_synthetic_dataset_configurations


@dataclass
class SyntheticDataset(BaseDataset):
    """Class for synthetic data generation.

    Note
    -------
    Generate dataset for Offline reinforcement learning (RL) and off-policy evaluation (OPE).


    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    behavior_policy: AlgoBase
        RL policy for data collection.

    random_state: int, default=12345
        Random state.

    Examples
    -------

    .. ::code-block:: python

        # import necessary module from _gym
        >>> from _gym.env import RTBEnv, CustomizedRTBEnv
        >>> from _gym.dataset import SyntheticDataset
        >>> from _gym.policy import DQN
        # import necessary module from other library
        >>> from sklearn.linear_model import LogisticRegression

        # initialize environment and define (RL) agent (i.e., policy)
        >>> env = RTBEnv()
        >>> dqn = DQN()

        # customize environment from the decision makers' perspective
        >>> env = CustomizedRTBEnv(
                original_env=env,
                reward_predictor=LogisticRegression(),
                action_type="discrete",
            )

        # initialize dataset class
        >>> dataset = SyntheticDataset(
                env=env,
                behavior_policy=dqn,
            )

        # pretrain behavior policy before data collection
        >>> dataset.pretrain_behavior_policy()

        # data collection
        >>> logged_dataset = dataset.obtain_trajectories(n_episodes=100)
        >>> logged_dataset
        {'size': 2400,
        'n_episodes': 100,
        'step_per_episode': 24,
        'action_type': 'discrete',
        'action_dim': 10,
        'action_meaning': array([ 0.1       ,  0.16681005,  0.27825594,  0.46415888,  0.77426368,
                1.29154967,  2.15443469,  3.59381366,  5.9948425 , 10.        ]),
        'state_keys': ['timestep',
        'remaining_budget',
        'budget_consumption_rate',
        'cost_per_mille_of_impression',
        'winning_rate',
        'reward',
        'adjust_rate'],
        'state': array([[0.00000000e+00, 1.00000000e+04, 2.15246222e-01, ...,
                1.88961951e-01, 2.00000000e+00, 2.98123434e-01],
                [1.00000000e+00, 9.00300000e+03, 9.97000000e-02, ...,
                9.07079646e-01, 0.00000000e+00, 1.29154967e+00],
                [2.00000000e+00, 9.00300000e+03, 0.00000000e+00, ...,
                0.00000000e+00, 0.00000000e+00, 1.00000000e-01],
                ...,
                [2.10000000e+01, 3.00000000e+00, 0.00000000e+00, ...,
                1.32158590e-02, 0.00000000e+00, 4.64158883e-01],
                [2.20000000e+01, 3.00000000e+00, 0.00000000e+00, ...,
                8.46560847e-02, 0.00000000e+00, 5.99484250e+00],
                [2.30000000e+01, 3.00000000e+00, 0.00000000e+00, ...,
                0.00000000e+00, 0.00000000e+00, 5.99484250e+00]]),
        'action': array([5., 0., 9., ..., 8., 8., 5.]),
        'reward': array([0., 0., 6., ..., 0., 0., 0.]),
        'done': array([0., 0., 0., ..., 0., 0., 1.]),
        'info': {'impression': array([205.,   0., 226., ...,  16.,   0.,  15.]),
        'click': array([20.,  0., 20., ...,  0.,  0.,  0.]),
        'conversion': array([0., 0., 6., ..., 0., 0., 0.]),
        'average_bid_price': array([ 68.64159292,   4.93777778, 487.30088496, ..., 348.17460317,
                302.79534884,  58.60294118])},
        'pscore': array([0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1])}

        # on-policy policy value of behavior policy
        >>> on_policy_policy_value = dataset.calc_on_policy_policy_value(
                n_episodes=10000
            )
        >>> on_policy_policy_value
        48.7

    """

    env: gym.Env
    behavior_policy: BaseHead
    configurations: Optional[Dict[str, Any]] = None
    random_state: int = 12345

    def __post_init__(self):
        if isinstance(self.env.action_space, Discrete):
            self.action_type = "discrete"
            self.n_actions = self.env.action_space.n
            self.action_dim = None
        else:
            self.action_type = "continuous"
            self.n_actions = None
            self.action_dim = len(self.env.action_space.high)

        if isinstance(self.env, (RTBEnv, CustomizedRTBEnv)):
            self.step_per_episode = self.env.step_per_episode
            self.action_meaning = (
                self.env.action_meaning if self.action_type == "discrete" else None
            )
            self.state_keys = self.env.obs_keys
        else:
            self.step_per_episode = None
            self.action_meaning = None
            self.state_keys = None

        if self.configurations is not None:
            self.configurations = check_synthetic_dataset_configurations(
                self.configurations
            )
            self.step_per_episode = self.configurations["step_per_episode"]
            self.action_meaning = self.configurations["action_meaning"]
            self.state_keys = self.configurations["state_keys"]

        if self.step_per_episode is None:
            self.max_episode_steps = self.env._max_episode_steps
        else:
            self.max_episode_steps = self.step_per_episode

        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def obtain_trajectories(self, n_episodes: int = 10000) -> LoggedDataset:
        """Rollout behavior policy and obtain trajectories.

        Parameters
        -------
        n_episodes: int, default=10000
            Number of trajectories to rollout behavior policy and collect data.

        Returns
        -------
        logged_dataset: Dict
            size: int
                Total steps the dataset records.

            n_episodes: int
                Total episodes the dataset records.

            step_per_episode: int
                Total timesteps in an episode.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            action_dim: int
                Action dimensions of discrete actions.
                If action_type is "continuous", None is recorded.

            state: NDArray[float], shape (size, 7)
                State of the RL environment.

            action: Union[NDArray[int], NDArray[float]], shape (size, )
                Action chosen by the behavior policy.

            reward: NDArray[int], shape (size, )
                Reward observed for each (state, action) pair.

            done: NDArray[int], shape (size, )
                Whether an episode ends or not.

            terminal: NDArray[int], shape (size, )
                Whether an episode reaches pre-defined maximum steps.

            info: Dict[str, NDArray[int]], shape (size, )
                Additional feedback information from RL environment.

            pscore: NDArray[float], shape (size, )
                Action choice probability of the behavior policy for the chosen action.

        """
        if self.step_per_episode is None:
            raise ValueError(
                f"when the total timestep in episode is flexible, use .obtain_steps() instead"
            )
        if not (isinstance(n_episodes, int) and n_episodes > 0):
            raise ValueError(
                f"n_episodes must be a positive integer, but {n_episodes} is given"
            )
        states = np.empty(
            (
                n_episodes * self.step_per_episode,
                self.env.observation_space.shape[0],
            )
        )
        if self.action_type == "discrete":
            actions = np.empty(n_episodes * self.step_per_episode)
            action_probs = np.empty(n_episodes * self.step_per_episode)
        else:
            actions = np.empty((n_episodes * self.step_per_episode, self.action_dim))
            action_probs = np.empty(
                (n_episodes * self.step_per_episode, self.action_dim)
            )

        rewards = np.empty(n_episodes * self.step_per_episode)
        dones = np.empty(n_episodes * self.step_per_episode)
        info = {}

        idx = 0
        for _ in tqdm(
            np.arange(n_episodes),
            desc="[obtain_trajectories]",
            total=n_episodes,
        ):
            state = self.env.reset()
            done = False

            while not done:
                (
                    action,
                    action_prob,
                ) = self.behavior_policy.stochastic_action_with_pscore_online(state)
                next_state, reward, done, info_ = self.env.step(action)

                if idx == 0:
                    info_keys = info_.keys()
                    for key in info_keys:
                        info[key] = np.empty(n_episodes * self.step_per_episode)

                states[idx] = state
                actions[idx] = action
                action_probs[idx] = action_prob
                rewards[idx] = reward
                dones[idx] = done

                for key, value in info_.items():
                    info[key][idx] = value

                idx += 1
                state = next_state

        logged_dataset = {
            "size": n_episodes * self.step_per_episode,
            "n_episodes": n_episodes,
            "step_per_episode": self.step_per_episode,
            "action_type": self.action_type,
            "n_actions": self.n_actions,
            "action_dim": self.action_dim,
            "action_meaning": self.action_meaning,
            "state_keys": self.state_keys,
            "state": states,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "terminal": dones,
            "info": info,
            "pscore": action_probs,
        }
        return logged_dataset

    def obtain_steps(self, n_steps: int = 100000) -> LoggedDataset:
        """Rollout behavior policy and obtain steps.

        Parameters
        -------
        n_steps: int, default=100000
            Number of steps to rollout behavior policy and collect data.

        Returns
        -------
        logged_dataset: Dict
            size: int
                Total steps the dataset records.

            n_episodes: int
                Total episodes the dataset records.

            step_per_episode: int
                Total timesteps in an episode.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            action_dim: int
                Action dimensions of discrete actions.
                If action_type is "continuous", None is recorded.

            state: NDArray[float], shape (size, 7)
                State of the RL environment.

            action: Union[NDArray[int], NDArray[float]], shape (size, )
                Action chosen by the behavior policy.

            reward: NDArray[int], shape (size, )
                Reward observed for each (state, action) pair.

            done: NDArray[int], shape (size, )
                Whether an episode ends or not.

            terminal: NDArray[int], shape (size, )
                Whether an episode reaches pre-defined maximum steps.

            info: Dict[str, NDArray[int]], shape (size, )
                Additional feedback information from RL environment.

            pscore: NDArray[float], shape (size, )
                Action choice probability of the behavior policy for the chosen action.

        """
        if not (isinstance(n_steps, int) and n_steps > 0):
            raise ValueError(
                f"n_steps must be a positive integer, but {n_steps} is given"
            )
        states = np.empty(
            (
                n_steps,
                self.env.observation_space.shape[0],
            )
        )
        if self.action_type == "discrete":
            actions = np.empty(n_steps)
            action_probs = np.empty(n_steps)
        else:
            actions = np.empty(n_steps, self.action_dim)
            action_probs = np.empty(n_steps, self.action_dim)

        rewards = np.empty(n_steps)
        dones = np.empty(n_steps)
        terminals = np.empty(n_steps)
        info = {}

        n_episodes = 0
        done = True

        for idx in tqdm(
            np.arange(n_steps),
            desc="[obtain_steps]",
            total=n_steps,
        ):
            if done:
                n_episodes += 1
                state = self.env.reset()
                step = 0
                done = False
            (
                action,
                action_prob,
            ) = self.behavior_policy.stochastic_action_with_pscore_online(state)
            next_state, reward, done, info_ = self.env.step(action)
            terminal = step == self.max_episode_steps - 1

            if idx == 0:
                info_keys = info_.keys()
                for key in info_keys:
                    info[key] = np.empty(n_steps)

            states[idx] = state
            actions[idx] = action
            action_probs[idx] = action_prob
            rewards[idx] = reward
            dones[idx] = done or terminal
            terminals[idx] = terminal

            for key, value in info_.items():
                info[key][idx] = value

            state = next_state
            step += 1

        logged_dataset = {
            "size": n_steps,
            "n_episodes": n_episodes,
            "step_per_episode": self.step_per_episode,
            "action_type": self.action_type,
            "n_actions": self.n_actions,
            "action_dim": self.action_dim,
            "action_meaning": self.action_meaning,
            "state_keys": self.state_keys,
            "state": states,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "terminal": terminals,
            "info": info,
            "pscore": action_probs,
        }
        return logged_dataset

    def calc_on_policy_policy_value(self, n_episodes: int = 10000) -> float:
        """Calculate on-policy policy value of the behavior policy by rollout.

        Parameters
        -------
        n_episodes: int, default=10000
            Number of episodes to train behavior policy.

        Returns
        -------
        mean_reward: float
            Mean episode reward calculated through rollout.

        """
        total_reward = 0.0
        for _ in tqdm(
            np.arange(n_episodes),
            desc="[calc_on_policy_policy_value]",
            total=n_episodes,
        ):
            state = self.env.reset()
            done = False

            while not done:
                action = self.behavior_policy.sample_action_online(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

        return total_reward / n_episodes
