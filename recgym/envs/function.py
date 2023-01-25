"""Mathematical Functions used in Recommendation System (REC) ."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .base import BaseUserModel


@dataclass
class UserModel(BaseUserModel):
    """Class to define user_preference_dynamics and reward_model.

    Imported as: :class:`recgym.envs.UserModel`

    Tip
    -------
    Use :class:`BaseUserModel` to define custom UserModel.

    Parameters
    -------
    state: array-like of shape (user_feature_dim, )
        A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
        When the true state is unobservable, you can gain observation instead of state.

    action: {int, array-like of shape (1, )} (>= 0)
        selected an item to recommendation from n_items.

    item_feature_vector: ndarray of shape (n_items, item_feature_dim), default=None
        Feature vectors that characterizes each item.
    
    References
    -------
    Sarah Dean, Jamie Morgenstern.
    "Preference Dynamics Under Personalized Recommendations." 2022.

    """

    state: np.ndarray
    action: int
    item_feature_vector: Optional[np.ndarray] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(
            self.state,
            name="state",
            target_type=np.ndarray,
        )
        check_scalar(
            self.action,
            name="action",
            target_type=np.int64,
        )
        check_scalar(
            self.item_feature_vector,
            name="item_feature_vector",
            target_type=np.ndarray,
        )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def user_preference_dynamics(
        self,
        alpha: float = 1.0,
    )-> np.ndarray:
        """Function that determines how to update the state (i.e., user preference) based on the recommended item. user_feature is amplified by the recommended item_feature
        
        Parameters
        -------
        alpha: float, default = 1.0 (0=<alpha=<1)
            stepsize

        Returns
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        """
        self.state = (
            self.state
            + alpha * self.state @ self.item_feature_vector[self.action] * self.item_feature_vector[self.action]
        )
        self.state = self.state / np.linalg.norm(self.state, ord=2)
        return self.state


    def reward_model(
        self,
    )-> float:
        """Reward function. inner product of state and recommended item_feature

        Returns
        -------
        reward: float
            User engagement signal. Either binary or continuous.

        """
        reward = self.state @ self.item_feature_vector[self.action]
        return reward
