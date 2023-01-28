"""Mathematical Functions used in Recommender System (REC) ."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .base import BaseUserModel
from ...types import Action


@dataclass
class UserModel(BaseUserModel):
    """Class to define user_preference_dynamics and reward_function.

    Imported as: :class:`recgym.envs.UserModel`

    Tip
    -------
    Use :class:`BaseUserModel` to define custom UserModel.

    Parameters
    -------
    reward_type: str = "continuous"
        Reward type (i.e., countinuous / binary).

    reward_std: float, default=0.0 (>=0)
        Standard deviation of the reward distribution. Applicable only when reward_type is "continuous".

    item_feature_vector: ndarray of shape (n_items, item_feature_dim), default=None
        Feature vectors that characterizes each item.

    random_state: int, default=None (>= 0)
        Random state.

    References
    -------
    Sarah Dean, Jamie Morgenstern.
    "Preference Dynamics Under Personalized Recommendations." 2022.

    """

    reward_type: str = "continuous"  # "binary"
    reward_std: float = 0.0
    item_feature_vector: Optional[np.ndarray] = (None,)
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(
            self.reward_std,
            name="reward_std",
            target_type=float,
        )

        if self.reward_type not in ["continuous", "binary"]:
            raise ValueError(
                f'reward_type must be either "continuous" or "binary", but {self.reward_type} is given'
            )

        self.random_ = check_random_state(self.random_state)

    def user_preference_dynamics(
        self,
        state: np.ndarray,
        action: Action,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Function that determines how to update the state (i.e., user preference) based on the recommended item. user_feature is amplified by the recommended item_feature

        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        action: {int, array-like of shape (1, )} (>= 0)
            selected an item to recommendation from n_items.

        alpha: float, default = 1.0 (0=<alpha=<1)
            stepsize

        Returns
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        """
        state = (
            state
            + alpha
            * state
            @ self.item_feature_vector[action]
            * self.item_feature_vector[action]
        )
        state = state / np.linalg.norm(state, ord=2)
        return state

    def reward_function(
        self,
        state: np.ndarray,
        action: Action,
    ) -> float:
        """Reward function. inner product of state and recommended item_feature

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
        reward = state @ self.item_feature_vector[action]

        if self.reward_type is "continuous":
            reward = reward + self.random_.normal(loc=0.0, scale=self.reward_std)

        return reward
