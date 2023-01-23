"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class UserModel(metaclass=ABCMeta):
    """Base class to determine user_preference_dynamics and reward_model.
    
    Imported as: class:`recgym.UserModel` 
    
    """

    @abstractmethod
    def user_preference_dynamics(
        self,
        state: np.ndarray,
        action: np.ndarray,
        item_feature_vector: np.ndarray,
    ) -> np.ndarray:
      
        """update state
        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            user_feature_vector
        action: {int, array-like of shape (1, )} (>= 0)
            selected an item to recommendation from n_items.
        item_feature_vector: ndarray of shape (n_items, item_feature_dim), default=None
            Feature vectors that characterizes each item.
        alpha: float, default = 1.0 (0=<alpha=<1)
            stepsize

        Returns
        -------
        state: array-like of shape (user_feature_dim, )
            make the initial state the user_feature of the chosen user.
            user_feature is amplified/attenuated by the recommended item_feature.
        """
        raise NotImplementedError

    @abstractmethod
    def reward_model(
        self,
        state: np.ndarray,
        action: np.ndarray,
        item_feature_vector:np.ndarray,
    ) -> float:
        """determine reward
        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            user_feature_vector
        action: {int, array-like of shape (1, )} (>= 0)
            selected an item to recommendation from n_items.
        item_feature_vector: ndarray of shape (n_items, item_feature_dim), default=None
            Feature vectors that characterizes each item.

        Returns
        -------
        reward: float
            user engagement gained.
        """
        raise NotImplementedError
