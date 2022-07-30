"""Abstract Base Class for Off-Policy Estimator."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm, truncnorm
from sklearn.utils import check_scalar


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_trajectory_value(self) -> np.ndarray:
        """Estimate the trajectory-wise expected reward."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of the evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate the confidence intervals of the policy value."""
        raise NotImplementedError


@dataclass
class BaseCumulativeDistributionalOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for Cumulative Distributional OPE estimators."""

    @abstractmethod
    def estimate_cumulative_distribution_function(self) -> Tuple[np.ndarray]:
        """Estimate the cumulative distribution function (cdf) of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_mean(self) -> float:
        """Estimate the mean of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_variance(self) -> float:
        """Estimate the variance of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_conditional_value_at_risk(self) -> float:
        """Estimate the conditional value at risk (cVaR) of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interquartile_range(self) -> Dict[str, float]:
        """Estimate the interquartile range of the policy value."""
        raise NotImplementedError

    def obtain_reward_scale(self, gamma: float = 1.0) -> np.ndarray:
        """Obtain the reward scale of the cumulative distribution function.

        Parameters
        -------
        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        reward_scale: ndarray of shape (n_partition, )
            Reward scale of the cumulative distribution function.

        """
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)

        if self.use_observations_as_reward_scale:
            behavior_policy_reward = self.input_dict_["reward"].reshape(
                (-1, self.step_per_episode)
            )
            discount = np.full(behavior_policy_reward.shape[1], gamma).cumprod()
            behavior_policy_trajectory_wise_reward = (
                behavior_policy_reward * discount
            ).sum(axis=1)
            reward_scale = np.sort(np.unique(behavior_policy_trajectory_wise_reward))
        else:
            reward_scale = np.linspace(
                self.scale_min, self.scale_max, num=self.n_partition
            )

        return reward_scale

    def _aggregate_trajectory_wise_statistics_discrete(
        self,
        step_per_episode: int,
        reward: Optional[np.ndarray] = None,
        behavior_policy_trajectory_wise_pscore: Optional[np.ndarray] = None,
        evaluation_policy_trajectory_wise_pscore: Optional[np.ndarray] = None,
        initial_state_value_prediction: Optional[np.ndarray] = None,
        gamma: float = 1.0,
    ):
        """Aggregate step-wise observations into trajectory wise statistics for the discrete action setup.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: ndarray of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi(a_t \\mid s_t)`

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        trajectory_wise_reward: ndarray of shape (n_episodes, )
            Trajectory wise reward observed under the behavior policy.

        trajectory_wise_importance_weight: ndarray of shape (n_episodes, )
            Trajectory wise importance weight.

        initial_state_value_prediction: ndarray of shape (n_episodes, )
            Estimated initial state value.

        """
        trajectory_wise_importance_weight = None
        trajectory_wise_reward = None

        if reward is not None:
            reward = reward.reshape((-1, step_per_episode))
            discount = np.full(reward.shape[1], gamma).cumprod()
            trajectory_wise_reward = (reward * discount).sum(axis=1)

        if (
            behavior_policy_trajectory_wise_pscore is not None
            and evaluation_policy_trajectory_wise_pscore is not None
        ):
            behavior_policy_trajectory_wise_pscore = (
                behavior_policy_trajectory_wise_pscore.reshape((-1, step_per_episode))[
                    :, 0
                ]
            )
            evaluation_policy_trajectory_wise_pscore = (
                evaluation_policy_trajectory_wise_pscore.reshape(
                    (-1, step_per_episode)
                )[:, 0]
            )
            trajectory_wise_importance_weight = (
                evaluation_policy_trajectory_wise_pscore
                / behavior_policy_trajectory_wise_pscore
            )

        return (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        )

    def _aggregate_trajectory_wise_statistics_continuous(
        self,
        step_per_episode: int,
        action: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None,
        behavior_policy_trajectory_wise_pscore: Optional[np.ndarray] = None,
        evaluation_policy_action: Optional[np.ndarray] = None,
        initial_state_value_prediction: Optional[np.ndarray] = None,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
    ):
        """Aggregate step-wise observations into trajectory wise statistics for the continuous action setup.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: ndarray of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: ndarray of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        trajectory_wise_reward: ndarray of shape (n_episodes, )
            Trajectory wise reward observed under the behavior policy.

        trajectory_wise_importance_weight: ndarray of shape (n_episodes, )
            Trajectory wise importance weight.

        initial_state_value_prediction: ndarray of shape (n_episodes, )
            Estimated initial state value.

        """
        trajectory_wise_importance_weight = None
        trajectory_wise_reward = None

        if reward is not None:
            reward = reward.reshape((-1, step_per_episode))
            discount = np.full(reward.shape[1], gamma).cumprod()
            trajectory_wise_reward = (reward * discount).sum(axis=1)

        if (
            action is not None
            and behavior_policy_trajectory_wise_pscore is not None
            and evaluation_policy_action is not None
        ):
            action_dim = action.shape[1]
            action = action.reshape((-1, step_per_episode, action_dim))
            evaluation_policy_action = evaluation_policy_action.reshape(
                (-1, step_per_episode, action_dim)
            )
            behavior_policy_trajectory_wise_pscore = (
                behavior_policy_trajectory_wise_pscore.reshape((-1, step_per_episode))
            )

            if sigma is None:
                sigma = np.ones(action.shape[1])

            if use_truncated_kernel:
                similarity_weight = truncnorm.pdf(
                    evaluation_policy_action,
                    a=(action_min - action) / sigma,
                    b=(action_max - action) / sigma,
                    loc=action,
                    scale=sigma,
                ).cumprod(axis=1)[:, -1, 0]
            else:
                similarity_weight = norm.pdf(
                    evaluation_policy_action,
                    loc=action,
                    scale=sigma,
                ).cumprod(axis=1)[:, -1, 0]

            trajectory_wise_importance_weight = (
                similarity_weight / behavior_policy_trajectory_wise_pscore[:, 0]
            )

        return (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        )
