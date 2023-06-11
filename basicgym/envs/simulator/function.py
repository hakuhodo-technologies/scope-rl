# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Mathematical Functions used in Synthetic System ."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .base import BaseRewardFunction
from .base import BaseStateTransitionFunction
from ...utils import sigmoid


@dataclass
class StateTransitionFunction(BaseStateTransitionFunction):
    """Class to define the state transition function.

    Bases: :class:`basicgym.BaseStateTransitionFunction`

    Imported as: :class:`basicgym.envs.simulator.StateTransitionFunction`

    Tip
    -------
    Use :class:`BaseStateTransitionFunction` to define a custom StateTransitionFunction.

    Parameters
    -------
    state_dim: int
        Dimension of the state.

    action_dim: int
        Dimension of the action (context).

    random_state: int, default=None (>= 0)
        Random state.

    """

    state_dim: int
    action_dim: int
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(
            self.state_dim,
            name="state_dim",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            self.action_dim,
            name="action_dim",
            target_type=int,
            min_val=1,
        )
        self.random_ = check_random_state(self.random_state)

        self.state_coef = self.random_.normal(
            loc=0.0, scale=1.0, size=(self.state_dim, self.state_dim)
        )
        self.action_coef = self.random_.normal(
            loc=0.0, scale=1.0, size=(self.state_dim, self.action_dim)
        )
        self.state_action_coef = self.random_.normal(
            loc=0.0, scale=1.0, size=(self.state_dim, self.action_dim)
        )

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Update the state based on the presented action.

        Parameters
        -------
        state: array-like of shape (state_dim, )
            Current state.

        action: array-like of shape (action_dim, )
            Indicating the action chosen by the agent.

        Return
        -------
        state: array-like of shape (state_dim, )
            Next state.

        """
        state = state / self.state_dim
        action = action / self.action_dim

        state = (
            self.state_coef @ state
            + self.action_coef @ action
            + (self.state_action_coef @ action).T @ state
        )
        state = state / np.linalg.norm(state, ord=2)
        return state


@dataclass
class RewardFunction(BaseRewardFunction):
    """Class to define the reward function.

    Bases: :class:`basicgym.BaseRewardFunction`

    Imported as: :class:`basicgym.envs.simulator.RewardFunction`

    Tip
    -------
    Use :class:`BaseRewardFunction` to define a custom RewardFunction.

    Parameters
    -------
    state_dim: int
        Dimension of the state.

    action_dim: int
        Dimension of the action (context).

    reward_type: {"continuous", "binary"}, default="continuous"
        Reward type.

    reward_std: float, default=0.0 (>=0)
        Noise level of the reward. Applicable only when reward_type is "continuous".

    random_state: int, default=None (>= 0)
        Random state.

    """

    state_dim: int
    action_dim: int
    reward_type: str = "continuous"  # "binary"
    reward_std: float = 0.0
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(
            self.state_dim,
            name="state_dim",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            self.action_dim,
            name="action_dim",
            target_type=int,
            min_val=1,
        )
        check_scalar(self.reward_std, name="reward_std", target_type=float, min_val=0.0)
        if self.reward_type not in ["continuous", "binary"]:
            raise ValueError(
                f'reward_type must be either "continuous" or "binary", but {self.reward_type} is given'
            )
        self.random_ = check_random_state(self.random_state)

        self.state_coef = self.random_.normal(
            loc=0.0, scale=1.0, size=(self.state_dim,)
        )
        self.action_coef = self.random_.normal(
            loc=0.0, scale=1.0, size=(self.action_dim,)
        )
        self.state_action_coef = self.random_.normal(
            loc=0.0, scale=1.0, size=(self.state_dim, self.action_dim)
        )

    def mean_reward_function(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Linear expected immediate reward function.

        Parameters
        -------
        state: array-like of shape (state_dim, )
            State in the RL environment.

        action: array-like of shape (action_dim, )
            Indicating the action chosen by the agent.

        Return
        -------
        mean_reward_function: float
            Expected immediate reward function conditioned on the state and action.

        """
        state = state / self.state_dim
        action = action / self.action_dim

        logit = (
            self.state_coef.T @ state
            + self.action_coef.T @ action
            + state.T @ self.state_action_coef @ action
        )
        mean_reward_function = (
            logit if self.reward_type == "continuous" else sigmoid(logit)
        )
        return mean_reward_function
