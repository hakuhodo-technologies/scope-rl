"""Abstract Base class for Off-Policy Estimator."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_trajectory_value(self) -> np.ndarray:
        """Estimate trajectory-wise reward."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value."""
        raise NotImplementedError


@dataclass
class BaseCumulativeDistributionalOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for Cumulative Distributional OPE estimators."""

    @abstractmethod
    def estimate_cumulative_distribution_function(self) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function (cdf) of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_mean(self) -> float:
        """Estimate mean of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_variance(self) -> float:
        """Estimate variance of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_conditional_value_at_risk(self) -> float:
        """Estimate conditional value at risk (cVaR) of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interquartile_range(self) -> Dict[str, float]:
        """Estimate interquartile range of the policy value."""
        raise NotImplementedError

    def _aggregate_trajectory_wise_statistics_discrete(
        self,
        step_per_episode: int,
        reward: Optional[np.ndarray] = None,
        behavior_policy_trajectory_wise_pscore: Optional[np.ndarray] = None,
        evaluation_policy_trajectory_wise_pscore: Optional[np.ndarray] = None,
        initial_state_value_prediction: Optional[np.ndarray] = None,
        gamma: float = 1.0,
    ):
        """Aggregate step-wise observation into trajectory wise statistics for the discrete action setup.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
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
        reward: Optional[np.ndarray] = None,
        behavior_policy_trajectory_wise_pscore: Optional[np.ndarray] = None,
        evaluation_policy_trajectory_wise_pscore: Optional[np.ndarray] = None,
        initial_state_value_prediction: Optional[np.ndarray] = None,
        gamma: float = 1.0,
    ):
        """Aggregate step-wise observation into trajectory wise statistics for the continuous action setup.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        """


@dataclass
class BaseDistributionallyRobustOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for Worst Case Distributional OPE estimators."""

    @abstractmethod
    def estimate_worst_case_policy_value(self) -> float:
        """Estimate the worst case policy value of evaluation policy."""
        raise NotImplementedError

    def _aggregate_trajectory_wise_statistics_discrete(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
    ):
        """Aggregate step-wise observation into trajectory wise statistics for the discrete action setup.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        """
        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(reward.shape[1], gamma).cumprod()
        trajectory_wise_reward = (reward * discount).sum(axis=1)

        behavior_policy_trajectory_wise_pscore = (
            behavior_policy_trajectory_wise_pscore.reshape((-1, step_per_episode))[:, 0]
        )
        evaluation_policy_trajectory_wise_pscore = (
            evaluation_policy_trajectory_wise_pscore.reshape((-1, step_per_episode))[
                :, 0
            ]
        )
        trajectory_wise_importance_weight = (
            evaluation_policy_trajectory_wise_pscore
            / behavior_policy_trajectory_wise_pscore
        )

        return (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
        )

    def _aggregate_trajectory_wise_statistics_continuous(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
    ):
        """Aggregate step-wise observation into trajectory wise statistics for the continuous action setup.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        """
