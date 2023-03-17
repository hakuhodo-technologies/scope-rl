"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseStateTransitionFunction(metaclass=ABCMeta):
    """Base class to define the state transition function.

    Imported as: :class:`syntheticgym.envs.BaseStateTransitionFunction`

    """

    @abstractmethod
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
            Indicating the action presented by the agent.

        Return
        -------
        state: array-like of shape (state_dim, )
            Next state.

        """
        raise NotImplementedError


@dataclass
class BaseRewardFunction(metaclass=ABCMeta):
    """Base class to define the mean reward function.

    Imported as: :class:`syntheticgym.envs.RewardFunction`

    """

    @abstractmethod
    def mean_reward_function(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Mean reward function.

        Parameters
        -------
        state: array-like of shape (state_dim, )
            State in the RL environment.

        action: array-like of shape (action_dim, )
            Indicating the action presented by the agent.

        Return
        -------
        mean_reward_function: float
            Mean reward function conditioned on the state and action.

        """
        raise NotImplementedError
