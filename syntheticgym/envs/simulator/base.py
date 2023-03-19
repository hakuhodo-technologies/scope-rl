"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
from ...types import Action


@dataclass
class BaseStateTransition(metaclass=ABCMeta):
    """Base class to define state_transition

    Imported as: :class:`syntheticgym.envs.`

    """

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        action: Action,
    ) -> np.ndarray:
        """Function that determines how to update the state based on the presented action.

        Parameters
        -------
        state: array-like of shape (state_dim, )
            When the true state is unobservable, you can gain observation instead of state.
        
        action: {int, array-like of shape (action_dim, )} (>= 0)
            Indicating which action to present to the context.

        Returns
        -------
        state: array-like of shape (state_dim, )
            When the true state is unobservable, you can gain observation instead of state.

        """
        raise NotImplementedError

@dataclass
class BaseRewardFunction(metaclass=ABCMeta):
    """Base class to define reward_function.

    Imported as: :class:`syntheticgym.envs.RewardFunction`

    """

    @abstractmethod
    def sample(
        self,
        state: np.ndarray,
        action: Action,
    ) -> float:
        """Reward function.

        Parameters
        -------
        state: array-like of shape (state_dim, )
            When the true state is unobservable, you can gain observation instead of state.

        action: {int, array-like of shape (action_dim, )} (>= 0)
            Indicating which action to present to the context.

        Returns
        -------
        reward: float
            Either binary or continuous.

        """
        raise NotImplementedError
