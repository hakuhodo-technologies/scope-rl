"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
from ...types import Action


@dataclass
class BaseUserModel(metaclass=ABCMeta):
    """Base class to define user_preference_dynamics and reward_function.

    Imported as: class:`recgym.UserModel`

    """

    @abstractmethod
    def user_preference_dynamics(
        self,
        state: np.ndarray,
        action: Action,
    ) -> np.ndarray:
        """Function that determines how to update the state (i.e., user preference) based on the recommended item.

        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        action: {int, array-like of shape (1, )} (>= 0)
            selected an item to recommendation from n_items.

        Returns
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        """
        raise NotImplementedError

    @abstractmethod
    def reward_function(
        self,
        state: np.ndarray,
        action: Action,
    ) -> float:
        """Reward function.

        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        action: {int, array-like of shape (1, )} (>= 0)
            selected an item to recommendation from n_items.

        Returns
        -------
        reward: float
            User engagement signal. Either binary or continuous.

        """
        raise NotImplementedError
