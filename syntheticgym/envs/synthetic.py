"""Reinforcement Learning (RL) Environment for Simulation System."""
from typing import Tuple, Optional, Any

import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .simulator.base import BaseStateTransition
from .simulator.function import StateTransition
from .simulator.base import BaseRewardFunction
from .simulator.function import RewardFunction

from ..types import Action


class SyntheticEnv(gym.Env):
    def __init__(
        self,
        StateTransition: BaseStateTransition = StateTransition,
        RewardFunction: BaseRewardFunction = RewardFunction,
        state_dim: int = 10,
        n_actions: int = 100,
        action_context_dim: int = 10,
        action_context: Optional[np.ndarray] = None,
        reward_type: str = "continuous",  # "binary"
        reward_std: float = 0.0,
        obs_std: float = 0.0,
        step_per_episode=10,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(random_state)

        check_scalar(
            state_dim,
            name="state_dim",
            target_type=int,
            min_val=1,
        )
        self.state_dim = state_dim

        check_scalar(
            n_actions,
            name="n_actions",
            target_type=int,
            min_val=1,
        )
        self.n_actions = n_actions

        check_scalar(
            action_context_dim,
            name="action_context_dim",
            target_type=int,
            min_val=1,
        )
        self.action_context_dim = action_context_dim

        check_scalar(
            obs_std,
            name="obs_std",
            target_type=float,
            min_val=0.0,
        )
        self.obs_std = obs_std

        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        self.step_per_episode = step_per_episode

        # initialize action_context
        if action_context is None:
            action_context = self.random_.normal(
            loc=0.0, scale=1.0, size=(self.n_actions, self.action_context_dim)
        )

        check_scalar(
            action_context,
            name="action_context",
            target_type=np.ndarray,
        )

        self.state_transition = StateTransition(
            state_dim=state_dim,
            action_context_dim=action_context_dim,
            action_context=action_context,
            random_state=random_state,
        )   

        self.reward_function = RewardFunction(
            reward_type=reward_type,
            reward_std=reward_std,
            state_dim=state_dim,
            action_context_dim=action_context_dim,
            action_context=action_context,
            random_state=random_state,
        )   
        
        # define observation space
        self.observation_space = Box(
            low=np.full(self.state_dim, -np.inf),
            high=np.full(self.state_dim, np.inf),
            dtype=float,
        )

        # define action space
        self.action_type = "discrete"
        self.action_dim = 1
        self.action_space = Discrete(self.n_actions)

        # define reward range
        self.reward_range = (-np.inf, np.inf)

    def _observation(self, state):
        # add noise to state
        obs = state + self.random_.normal(
            loc=0.0, scale=self.obs_std, size=self.state_dim
        )
        return obs

    def step(self, action: Action) -> Tuple[Any]:
        """Simulate a recommender interaction with a user.

        Note
        -------
        The simulation procedure is given as follows.

        1. Sample reward (i.e., feedback on user engagement) for the given item.

        2. Update user state with user_preference_dynamics

        3. Return the user feedback to the RL agent.

        Parameters
        -------
        action: {int, array-like of shape (1, )} (>= 0)
            Indicating which item to present to the user.

        Returns
        -------
        feedbacks: Tuple
            obs: ndarray of shape (1,)
                Generated based on state.
                (e.g. complete with the state, noise added to the state, or only part of the state)

            reward: float
                User engagement signal. Either binary or continuous.

            done: bool
                Wether the episode end or not.

            truncated: False
                For API consistency.

            info: dict
                Additional feedbacks (user_id, state) for analysts.
                Note that those feedbacks are unobservable to the agent.

        """
        # 1. sample reward for the given item.
        reward = self.reward_function.reward_function(self.state, action)

        # 2. update user state with state_transition
        self.state = self.state_transition.state_transition(self.state, action)

        done = self.t == self.step_per_episode - 1

        if done:
            obs, info = self.reset()

        else:
            self.t += 1
            obs = self._observation(self.state)

        info = {"state": self.state}

        return obs, reward, done, False, info

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Initialize the environment.

        Returns
        -------
        obs: ndarray of shape (1,)
            Generated based on state.
            (e.g. complete with the state, noise added to the state, or only part of the state)

        info: dict
            Additional feedbacks (state) for analysts.
            Note that those feedbacks are unobservable to the agent.

        """
        if seed is not None:
            self.random_ = check_random_state(seed)

        # initialize internal env state
        self.t = 0
        # initialize state
        state = self.random_.normal(loc=0.0, scale=1.0, size=self.state_dim)
        self.state = state
        obs = self._observation(self.state)

        info = {"state": self.state}

        return obs, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
