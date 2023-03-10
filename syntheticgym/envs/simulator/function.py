"""Mathematical Functions used in Synthetic System ."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .base import BaseRewardFunction
from .base import BaseStateTransition
from ...types import Action


@dataclass
class StateTransition(BaseStateTransition):
    """Class to define state_transition.

    Imported as: :class:`syntheticgym.envs.`

    Tip
    -------
    Use :class:`BaseStateTransition` to define custom StateTransition.

    Parameters
    -------
    action_context: ndarray of shape (n_items, item_feature_dim), default=None
        Feature vectors that characterizes each item.

    state_dim:

    action_context_dim:
        
    random_state: int, default=None (>= 0)
        Random state.

    References
    -------
    Sarah Dean, Jamie Morgenstern.
    "Preference Dynamics Under Personalized Recommendations." 2022.

    """
    state_dim: int = 10
    action_type: str = "continuous",  # "discrete"
    action_context_dim: int = 10
    action_context: Optional[np.ndarray] = (None,)
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)

        self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.state_dim))
        self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_context_dim))
        self.state_action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_context_dim))


    def step(
        self,
        state: np.ndarray,
        action: Action,
    ) -> np.ndarray:
        """Function that determines how to update the state (i.e., user preference) based on the recommended item. user_feature is amplified by the recommended item_feature

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
        if self.action_type == "continuous":
            state = self.state_coef @ state +  self.action_coef @ action + (self.state_action_coef @ action).T @ state
        
        elif self.action_type == "discrete":
            state = self.state_coef @ state + self.action_coef @ self.action_context[action] +  (self.state_action_coef @ self.action_context[action]).T @ state
            
        state = state / np.linalg.norm(state, ord=2)

        return state


@dataclass
class RewardFunction(BaseRewardFunction):
    """Class to define reward_function.

    Imported as: :class:`syntheticgym.envs.RewardFunction`

    Tip
    -------
    Use :class:`BaseRewardFunction` to define custom RewardFunction.

    Parameters
    -------
    reward_type: str = "continuous"
        Reward type (i.e., countinuous / binary).

    reward_std: float, default=0.0 (>=0)
        Standard deviation of the reward distribution. Applicable only when reward_type is "continuous".

    action_context: ndarray of shape (n_items, item_feature_dim), default=None
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
    state_dim: int = 10
    action_type: str = "continuous",  # "discrete"
    action_context_dim: int = 10
    action_context: Optional[np.ndarray] = (None,)
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

        self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, ))
        self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.action_context_dim, ))
        self.state_action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_context_dim))

    def sample(
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
        if self.action_type == "continuous":
            reward = self.state_coef.T @ state / self.state_dim + self.action_coef.T @ action / self.action_context_dim + state.T @ (self.state_action_coef @ action) / self.state_dim / self.action_context_dim
        
        elif self.action_type == "discrete":
            reward = self.state_coef.T @ state / self.state_dim + self.action_coef.T @ self.action_context[action] / self.action_context_dim + state.T @ (self.state_action_coef @ self.action_context[action]) / self.state_dim / self.action_context_dim

        if self.reward_type == "continuous":
            reward = reward + self.random_.normal(loc=0.0, scale=self.reward_std)

        return reward
