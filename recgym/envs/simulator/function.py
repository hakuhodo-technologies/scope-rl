# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Mathematical Functions used in Recommender System (REC) ."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .base import BaseUserModel
from ...utils import sigmoid
from ...types import Action


@dataclass
class UserModel(BaseUserModel):
    """Class to define a user model based on user_preference_dynamics and reward_function.

    Bases: :class:`recgym.BaseUserModel`

    Imported as: :class:`recgym.envs.UserModel`

    Tip
    -------
    Use :class:`BaseUserModel` to define a custom UserModel.

    Parameters
    -------
    user_feature_dim: int
        Dimension of the user feature vectors. (API consistency.)

    item_feature_dim: int
        Dimension of the item feature vectors.

    reward_type: {"continuous", "binary"}, default="continuous"
        Reward type.

    reward_std: float, default=0.0 (>=0)
        Noise level of the reward. Applicable only when reward_type is "continuous".

    random_state: int, default=None (>= 0)
        Random state.

    References
    -------
    Sarah Dean, Jamie Morgenstern.
    "Preference Dynamics Under Personalized Recommendations." 2022.

    """

    user_feature_dim: int
    item_feature_dim: int
    reward_type: str = "continuous"  # "binary"
    reward_std: float = 0.0
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(
            self.user_feature_dim,
            name="user_feature_dim",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            self.item_feature_dim,
            name="item_feature_dim",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            self.reward_std,
            name="reward_std",
            target_type=float,
            min_val=0.0,
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
        item_feature_vector: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Function that determines the user state transition (i.e., user preference) based on the recommended item. user_feature is amplified by the recommended item_feature

        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        action: int or array-like of shape (1, )
            Indicating which item to present to the user.

        item_feature_vector: array-like of shape (n_items, item_feature_dim), default=None
            Feature vectors that characterize each item.

        alpha: float, default = 1.0 (0=<alpha=<1)
            Step size controlling how fast the user preference evolves over time.

        Returns
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        """
        coefficient = state @ item_feature_vector[action]
        state = state + alpha * coefficient * item_feature_vector[action]
        state = state / np.linalg.norm(state, ord=2)
        return state

    def reward_function(
        self,
        state: np.ndarray,
        action: Action,
        item_feature_vector: np.ndarray,
    ) -> float:
        """Reward function. inner product of state and recommended item_feature

        Parameters
        -------
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        action: int or array-like of shape (1, )
            Indicating which item to present to the user.

        item_feature_vector: array-like of shape (n_items, item_feature_dim), default=None
            Feature vectors that characterize each item.

        Returns
        -------
        reward: float
            User engagement signal as a reward. Either binary or continuous.

        """
        logit = state @ item_feature_vector[action]
        mean_reward_function = (
            logit if self.reward_type == "continuous" else sigmoid(logit)
        )

        if self.reward_type == "continuous":
            reward = self.random_.normal(
                loc=mean_reward_function, scale=self.reward_std
            )
        else:
            reward = self.random_.binominal(1, p=mean_reward_function)

        return reward
