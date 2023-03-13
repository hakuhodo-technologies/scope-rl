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
        state_dim: int = 5,
        action_type: str = "continuous",  # "discrete"
        n_actions: int = 10,  # Applicable only when action_type is "discrete"
        action_dim: int = 3,
        action_context: Optional[
            np.ndarray
        ] = None,  # Applicable only when action_type is "discrete"
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
            action_dim,
            name="action_dim",
            target_type=int,
            min_val=1,
        )
        self.action_dim = action_dim

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

        # define observation space
        self.observation_space = Box(
            low=np.full(state_dim, -np.inf),
            high=np.full(state_dim, np.inf),
            dtype=float,
        )

        # action space
        if action_type not in ["continuous", "discrete"]:
            raise ValueError(
                f'action_type must be either "continuous" or "discrete", but {action_type} is given'
            )

        if action_type == "continuous":
            self.action_type = "continuous"
            self.action_space = Box(low=-0.1, high=10, shape=(action_dim,), dtype=float)

        elif action_type == "discrete":
            self.action_type = "discrete"
            self.action_space = Discrete(n_actions)

            # initialize action_context
            if action_context is None:
                action_context = self.random_.normal(
                    loc=0.0, scale=1.0, size=(n_actions, action_dim)
                )
            check_scalar(
                action_context,
                name="action_context",
                target_type=np.ndarray,
            )

        self.state_transition = StateTransition(
            state_dim=state_dim,
            action_type=action_type,
            action_dim=action_dim,
            action_context=action_context,
            random_state=random_state,
        )

        self.reward_function = RewardFunction(
            reward_type=reward_type,
            reward_std=reward_std,
            state_dim=state_dim,
            action_type=action_type,
            action_dim=action_dim,
            action_context=action_context,
            random_state=random_state,
        )

        # define reward range
        self.reward_range = (-np.inf, np.inf)

    def _observation(self, state):
        # add noise to state
        obs = state + self.random_.normal(
            loc=0.0, scale=self.obs_std, size=self.state_dim
        )
        return obs

    def step(self, action: Action) -> Tuple[Any]:
        """Simulate a action interaction with a context.

        Note
        -------
        The simulation procedure is given as follows.

        1. Sample reward for the given action.

        2. Update state with state_transition

        3. Return the feedback to the RL agent.

        Parameters
        -------
        action: {int, array-like of shape (action_dim, )} (>= 0)
            Indicating which action to present to the context.

        Returns
        -------
        feedbacks: Tuple
            obs: ndarray of shape (state_dim,)
                Generated based on state.
                (e.g. complete with the state, noise added to the state, or only part of the state)

            reward: float
                Either binary or continuous.

            done: bool
                Wether the episode end or not.

            truncated: False
                For API consistency.

            info: dict
                Additional feedbacks (state) for analysts.
                Note that those feedbacks are unobservable to the agent.

        """
        # 1. sample reward for the given action.
        reward = self.reward_function.sample(self.state, action)

        # 2. update state with state_transition
        self.state = self.state_transition.step(self.state, action)

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
        obs: ndarray of shape (state_dim,)
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
        state = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim,))
        state = state / np.linalg.norm(state, ord=2)
        self.state = state
        obs = self._observation(self.state)

        info = {"state": self.state}

        return obs, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
