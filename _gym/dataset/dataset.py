from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import gym
import numpy as np
from sklearn.utils import check_random_state

from policy.policy import BasePolicy


@dataclass
class BaseDataset(metaclass=ABCMeta):
    @abstractmethod
    def obtain_trajectories(self, n_episodes: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def pretrain_behavior_policy(self, n_episodes: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def calc_ground_truth_policy_value(self, n_episodes: int) -> float:
        raise NotImplementedError


@dataclass
class SyntheticDataset(BaseDataset):
    env: gym.Env
    behavior_policy: BasePolicy
    random_state: int = 12345

    def __post_init__(self):
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def obtain_trajectories(self, n_episodes: int) -> Dict[str, Any]:
        """rollout behavior policy on environment and collect logged dataset"""
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
            "n_trajectories": n_episodes,
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
            "terminal": dones,
            "pscore": action_probs,
        }

        return logged_dataset

    def pretrain_behavior_policy(self, n_episodes: int) -> None:
        """pretrain behavior policy on environment"""
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

    def calc_ground_truth_policy_value(self, n_episodes: int) -> float:
        """rollout behavior policy and calculate mean episodic reward"""
        return self.env.calc_ground_truth_policy_value(self.behavior_policy, n_episodes)
