"""Reinforcement Learning (RL) Environment for Simulation System."""
from typing import Tuple, Optional, Any

import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .simulator.base import BaseStateTransitionFunction
from .simulator.function import StateTransitionFunction
from .simulator.base import BaseRewardFunction
from .simulator.function import RewardFunction

from ..utils import check_array
from ..types import Action


class SyntheticEnv(gym.Env):
    def __init__(
        self,
        StateTransitionFunction: BaseStateTransitionFunction = StateTransitionFunction,
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
            low=np.full(state_dim, -1.0),
            high=np.full(state_dim, 1.0),
            dtype=float,
        )

        # define action space
        if action_type not in ["continuous", "discrete"]:
            raise ValueError(
                f'action_type must be either "continuous" or "discrete", but {action_type} is given'
            )

        if action_type == "continuous":
            self.action_type = "continuous"
            self.action_space = Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=float
            )

        elif action_type == "discrete":
            self.action_type = "discrete"
            self.action_space = Discrete(n_actions)

            if action_context is None:
                action_context = self.random_.normal(
                    loc=0.0, scale=1.0, size=(n_actions, action_dim)
                )

            check_array(
                action_context,
                name="action_context",
                expected_dim=2,
            )
            if action_context.shape != (n_actions, action_dim):
                raise ValueError(
                    f"The shape of action_context must be (n_actions, action_dim), but found {action_context.shape}."
                )
            self.action_context = action_context

        if isinstance(StateTransitionFunction, BaseStateTransitionFunction):
            self.state_transition_function = StateTransitionFunction
        elif issubclass(StateTransitionFunction, BaseStateTransitionFunction):
            self.state_transition_function = StateTransitionFunction(
                state_dim=state_dim,
                action_dim=action_dim,
                random_state=random_state,
            )
        else:
            raise ValueError(
                "StateTransitionFunction must be a child class of BaseStateTransitionFunction"
            )

        if isinstance(RewardFunction, BaseRewardFunction):
            self.reward_function = RewardFunction
        elif issubclass(RewardFunction, BaseRewardFunction):
            self.reward_function = RewardFunction(
                state_dim=state_dim,
                action_dim=action_dim,
                random_state=random_state,
            )
        else:
            raise ValueError(
                "RewardFunction must be a child class of BaseRewardFunction"
            )

        # define reward range
        if reward_type == "continuous":
            self.reward_range = (-np.inf, np.inf)
        else:
            self.reward_range = (0, 1)

        check_scalar(
            reward_std,
            name="reward_std",
            target_type=float,
            min_val=0.0,
        )
        self.reward_type = reward_type
        self.reward_std = reward_std

    def _observation(self, state):
        obs = self.random_.normal(loc=state, scale=self.obs_std)
        return obs

    def _sample_reward(self, mean_reward_function):
        if self.reward_type == "continuous":
            reward = self.random_.normal(
                loc=mean_reward_function, scale=self.reward_std
            )
        else:
            reward = self.random_.binominal(1, p=mean_reward_function)
        return reward

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
        if self.action_type == "discrete":
            action = self.action_context[action]

        # 1. sample reward for the given action.
        mean_reward_function = self.reward_function.mean_reward_function(
            self.state, action
        )
        reward = self._sample_reward(mean_reward_function)

        # 2. update state with state_transition
        self.state = self.state_transition_function.step(self.state, action)

        done = self.t == self.step_per_episode - 1

        if done:
            obs, _ = self.reset()

        else:
            self.t += 1
            obs = self._observation(self.state)

        return obs, reward, done, False, {}

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

        self.t = 0
        state = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim,))
        self.state = state / np.linalg.norm(state, ord=2)
        obs = self._observation(self.state)

        return obs, {}

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
