# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
from ...types import Action


@dataclass
class BaseUserModel(metaclass=ABCMeta):
    """Base class to define user_preference_dynamics and reward_function.

    Imported as: class:`recgym.BaseUserModel`

    """

    @abstractmethod
    def user_preference_dynamics(
        self,
        state: np.ndarray,
        action: Action,
        item_feature_vector: np.ndarray,
    ) -> np.ndarray:
        """Function that determines the user state transition (i.e., user preference) based on the recommended item.

        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        action: int or array-like of shape (1, )
            Indicating which item to present to the user.

        item_feature_vector: ndarray of shape (n_items, item_feature_dim), default=None
            Feature vectors that characterizes each item.

        Returns
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        """
        raise NotImplementedError

    @abstractmethod
    def reward_function(
        self,
        state: np.ndarray,
        action: Action,
        item_feature_vector: np.ndarray,
    ) -> float:
        """Reward function.

        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        action: int or array-like of shape (1, )
            Indicating which item to present to the user.

        item_feature_vector: ndarray of shape (n_items, item_feature_dim), default=None
            Feature vectors that characterizes each item.

        Returns
        -------
        reward: bool or float
            User engagement signal as a reward. Either binary or continuous.

        """
        raise NotImplementedError
