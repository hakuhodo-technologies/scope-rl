"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class BaseUserModel(metaclass=ABCMeta):
    """Base class to define user_preference_dynamics and reward_model.
    
    Imported as: class:`recgym.UserModel` 
    
    """

    @abstractmethod
    def user_preference_dynamics(
        self,
    ) -> np.ndarray:
        """Function that determines how to update the state (i.e., user preference) based on the recommended item.

        Returns
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.
        """
        raise NotImplementedError

    @abstractmethod
    def reward_model(
        self,
    ) -> float:
        """Reward function.

        Returns
        -------
        reward: float
            User engagement signal. Either binary or continuous.
        """
        raise NotImplementedError
