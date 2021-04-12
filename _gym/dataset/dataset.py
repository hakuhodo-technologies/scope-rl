from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import gym
import numpy as np
from sklearn.utils import check_random_state

from policy.policy import BasePolicy


@dataclass
class BaseDataset(metaclass=ABCMeta):
    """Base Class for dataset."""
    @abstractmethod
    def obtain_trajectories(self, n_episodes: int) -> Dict[str, Any]:
        """Roullout behavior policy and obtain trajectories."""
        raise NotImplementedError

    @abstractmethod
    def pretrain_behavior_policy(self, n_episodes: int) -> None:
        """Pre-train behavior policy by interacting with environment."""
        raise NotImplementedError

    @abstractmethod
    def calc_ground_truth_policy_value(self, n_episodes: int) -> float:
        """Calculate ground-truth policy value of beahavior policy by rollout."""
        raise NotImplementedError


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

    behavior_policy: BasePolicy
        RL policy for data collection.

    random_state: int, default=12345.
        Random state.

    Examples
    -------

    .. ::code-block:: python

        # import necessary module from _gym
        >>> from env.env import RTBEnv
        >>> from dataset.dataset import SyntheticDataset
        >>> from policy.dqn import DQN

        # initialize environment and define (RL) agent (i.e., policy)
        >>> env = RTBEnv(
                use_reward_predictor=True,
                reward_predictor=LogisticRegression(),
            )
        >>> dqn = DQN()

        # initialize dataset class
        >>> dataset = SyntheticDataset(
                env=env,
                behavior_policy=dqn
            )

        # pretrain behavior policy before data collection
        >>> dataset.pretrain_behavior_policy()
        >>> dataset
        {
            'size': 240000,
            'n_episodes': 10000,
            'step_per_episode': 24,
            'action_type': 'discrete',
            'action_dim': 10,
            'state': array([[...]]),
            'action': array([...]),
            'reward': array([...]),
            'done': array([...]),
            'info': array([...]),
            'terminal': array([...]),
            'pscore': array([[...]]),
        }

        # data collection
        >>> obtained_dataset = dataset.obtain_trajectories(n_episodes=10000)

        # ground-truth policy value of behavior policy
        >>> ground_truth_policy_value = dataset.calculate_ground_truth_policy_value(
                n_episodes=10000
            )
        >>> ground_truth_policy_value
        ...
    
    """
    env: gym.Env
    behavior_policy: BasePolicy
    random_state: int = 12345

    def __post_init__(self):
        if not isinstance(env, gym.Env):
            raise ValueError(
                "env must be the gym.Env or a child class of the gym.Env")
        if not isinstance(behavior_policy, BasePolicy):
            raise ValueError(
                "behavior_policy must be the BasePolicy or a child class of the BasePolicy"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def obtain_trajectories(self, n_episodes: int = 10000) -> Dict[str, Any]:
        """Roullout behavior policy and obtain trajectories.
        
        Parameters
        -------
        n_episodes: int, default=10000.
            Number of trajectories to rollout behavior policy and collect data.

        Returns
        -------
        logged_dataset: Dict.
            size: int.
                Total steps the dataset records.

            n_episodes: int.
                Total episode the dataset records.

            step_per_episode: int.
                Number of steps recorded in an episode.

            action_type: str.
                Action type of the RL agent.
                Either "discrete" or "continuous".

            action_dim: int.
                Action dimention of discrete actions.
                If action_type is "continuous", None is recorded.

            state: NDArray[float], shape (size, 7).
                State of the RL environment.

            action: NDArray[Union[int, float]], shape (size, ).
                Action chosen by the behavior policy.

            reward: NDArray[int], shape (size, ).
                Reward observed for the (state, action) pair.

            done: NDArray[int], shape (size, ).
                Whether an episode ends or not.

            info: NDArray[Dict[str, int]], shape (size, ).
                Additional feedback information from RL environment.

            pscore: NDArray[float], shape (size, ).
                Action choice probability of the behavior policy for the chosen action.
        
        """
        if not (isinstance(n_episodes, int) and n_episodes > 0):
            raise ValueError(
                f"n_episodes must be a positive integer, but {n_episodes} is given"
            )
        states = np.zeros(
            (
                n_episodes * self.env.step_per_episode,
                self.env.observation_space.shape[0],
            )
        )
        actions = np.zeros(n_episodes * self.env.step_per_episode)
        action_probs = np.zeros(n_episodes * self.env.step_per_episode)
        rewards = np.zeros(n_episodes * self.env.step_per_episode)
        dones = np.zeros(n_episodes * self.env.step_per_episode)
        infos = {}

        idx = 0
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action, action_prob = self.behavior_policy.act(state)  # fix later
                next_state, reward, done, info = self.env.step(action)

                if idx == 0:
                    info_keys = info.keys()
                    for key in info_keys:
                        infos[key] = np.zeros(n_episodes * self.env.step_per_episode)

                states[idx] = state
                actions[idx] = action
                action_probs[idx] = action_prob
                rewards[idx] = reward
                dones[idx] = done

                for key, value in info.items():
                    infos[key][idx] = value

                idx += 1
                state = next_state

        logged_dataset = {
            "size": n_episodes * self.env.step_per_episode,
            "n_episodes": n_episodes,
            "step_per_episode": self.env.step_per_episode,
            "action_type": self.env.action_type,
            "action_dim": self.env.action_dim
            if self.env.action_type == "discrete"
            else None,
            "state": states,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "info": infos,
            "pscore": action_probs,
        }

        return logged_dataset

    def pretrain_behavior_policy(self, n_episodes: int = 10000) -> None:
        """Pre-train behavior policy by interacting with environment.
        
        Parameters
        -------
        n_episodes: int, default=10000.
            Number of episodes to train behavior policy.
        
        """
        if not (isinstance(n_episodes, int) and n_episodes > 0):
            raise ValueError(
                f"n_episodes must be a positive integer, but {n_episodes} is given"
            )
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action, action_prob = self.behavior_policy.act(state)  # fix later
                next_state, reward, done, _ = self.env.step(action)

                self.behavior_policy.update(
                    state, action, next_state, reward, done
                )  # fix later
                state = next_state

    def calc_ground_truth_policy_value(self, n_episodes: int = 10000) -> float:
        """Calculate ground-truth policy value of beahavior policy by rollout.
        
        Parameters
        -------
        n_episodes: int, default=10000.
            Number of episodes to train behavior policy.

        Returns
        -------
        mean_reward: float.
            Mean episode reward calculated through rollout.
        
        """
        return self.env.calc_ground_truth_policy_value(self.behavior_policy, n_episodes)
