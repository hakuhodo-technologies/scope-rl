"""Abstract Base Class for Off-Policy Estimator."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.utils import check_scalar

from ..utils import (
    gaussian_kernel,
    estimate_confidence_interval_by_bootstrap,
    estimate_confidence_interval_by_hoeffding,
    estimate_confidence_interval_by_empirical_bernstein,
    estimate_confidence_interval_by_t_test,
)


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

    @property
    def _estimate_confidence_interval(self) -> Dict[str, Callable]:
        return {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _calc_behavior_policy_pscore(
        self,
        step_per_episode: int,
        pscore: np.ndarray,
        pscore_type: str,
    ):
        pscore = pscore.reshape((-1, step_per_episode))

        # step-wise pscore
        behavior_policy_pscore = np.cumprod(pscore, axis=1)

        if pscore_type == "trajectory_wise":
            behavior_policy_pscore = np.tile(
                behavior_policy_pscore[:, -1], (step_per_episode, 1)
            ).T

        return behavior_policy_pscore

    def _calc_evaluation_policy_pscore(
        self,
        step_per_episode: int,
        action: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        pscore_type: str,
    ):
        """Transform the evaluation policy action distribution into the evaluation policy pscore (action choice probability).

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        pscore_type: {"trajectory-wise", "step-wise"}
            Indicates wether to return trajectory-wise pscore or step-wise pscore.

        Return
        -------
        evaluation_policy_trajectory_wise_pscore: array-like of shape (n_episodesstep_, per_episode)
            Trajectory-wise action choice probability of the evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi(a_t \\mid s_t)`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodesstep_per_episode),
            Step-wise action choice probability of the evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi(a_{t'} \\mid s_{t'})`

        """
        evaluation_policy_base_pscore = evaluation_policy_action_dist[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))

        # step-wise pscore
        evaluation_policy_pscore = np.cumprod(evaluation_policy_base_pscore, axis=1)

        if pscore_type == "trajectory_wise":
            evaluation_policy_pscore = np.tile(
                evaluation_policy_pscore[:, -1], (step_per_episode, 1)
            ).T

        return evaluation_policy_pscore

    def _calc_similarity_weight(
        self,
        step_per_episode: int,
        action: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float,
        pscore_type: str,
    ):
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )

        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action_scaler[np.newaxis, :],
            action / action_scaler[np.newaxis, :],
            sigma=sigma,
        )

        similarity_weight = similarity_weight.reshape((-1, step_per_episode))
        similarity_weight = np.cumprod(similarity_weight, axis=1)

        if pscore_type == "trajectory_wise":
            similarity_weight = np.tile(
                similarity_weight[:, -1], (step_per_episode, 1)
            ).T

        return similarity_weight


@dataclass
class BaseStateMarginalOffPolicyEstimator(BaseOffPolicyEstimator):
    def _calc_behavior_policy_pscore(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        pscore: np.ndarray,
    ):
        pscore = pscore.reshape((-1, step_per_episode))
        pscore_ = np.roll(pscore, n_step_pdis + 1, axis=1)
        pscore_[:, : n_step_pdis + 1] = 1

        numerator = np.cumprod(pscore, axis=1)
        denominator = np.cumprod(pscore_, axis=1)
        return numerator / denominator

    def _calc_evaluation_policy_pscore(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        evaluation_policy_pscore = evaluation_policy_action_dist[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))
        evaluation_policy_pscore_ = np.roll(
            evaluation_policy_pscore, n_step_pdis + 1, axis=1
        )
        evaluation_policy_pscore_[:, : n_step_pdis + 1] = 1

        numerator = np.cumprod(evaluation_policy_pscore, axis=1)
        denominator = np.cumprod(evaluation_policy_pscore_, axis=1)
        return numerator / denominator

    def _calc_similarity_weight(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float,
    ):
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )

        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action_scaler[np.newaxis, :],
            action / action_scaler[np.newaxis, :],
            sigma=sigma,
        )

        similarity_weight = similarity_weight.reshape((-1, step_per_episode))
        similarity_weight_ = np.roll(similarity_weight, n_step_pdis + 1, axis=1)
        similarity_weight_[:, : n_step_pdis + 1] = 1

        numerator = np.cumprod(similarity_weight, axis=1)
        denominator = np.cumprod(similarity_weight_, axis=1)
        return numerator / denominator

    def _calc_marginal_importance_weight(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        state_marginal_importance_weight: np.ndarray,
    ):
        state_marginal_importance_weight = state_marginal_importance_weight.reshape(
            (-1, step_per_episode)
        )
        state_marginal_importance_weight = np.roll(
            state_marginal_importance_weight, n_step_pdis, axis=1
        )
        state_marginal_importance_weight[:, :n_step_pdis] = 1

        return state_marginal_importance_weight


@dataclass
class BaseStateActionMarginalOffPolicyEstimator(BaseStateMarginalOffPolicyEstimator):
    def _calc_behavior_policy_pscore(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        pscore: np.ndarray,
    ):
        pscore = pscore.reshape((-1, step_per_episode))
        pscore_ = np.roll(pscore, n_step_pdis, axis=1)
        pscore_[:, :n_step_pdis] = 1

        numerator = np.cumprod(pscore, axis=1)
        denominator = np.cumprod(pscore_, axis=1)
        return numerator / denominator

    def _calc_evaluation_policy_pscore(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        evaluation_policy_pscore = evaluation_policy_action_dist[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))
        evaluation_policy_pscore_ = np.roll(
            evaluation_policy_pscore, n_step_pdis, axis=1
        )
        evaluation_policy_pscore_[:, :n_step_pdis] = 1

        numerator = np.cumprod(evaluation_policy_pscore, axis=1)
        denominator = np.cumprod(evaluation_policy_pscore_, axis=1)
        return numerator / denominator

    def _calc_similarity_weight(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float,
    ):
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )

        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action_scaler[np.newaxis, :],
            action / action_scaler[np.newaxis, :],
            sigma=sigma,
        )

        action_dim = action.shape[1]
        similarity_weight = similarity_weight.reshape(
            (-1, step_per_episode, action_dim)
        )
        similarity_weight_ = np.roll(similarity_weight, n_step_pdis, axis=1)
        similarity_weight_[:, :n_step_pdis] = 1

        numerator = np.cumprod(similarity_weight, axis=1)
        denominator = np.cumprod(similarity_weight_, axis=1)
        return numerator / denominator

    def _calc_marginal_importance_weight(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        state_action_marginal_importance_weight: np.ndarray,
    ):
        state_action_marginal_importance_weight = (
            state_action_marginal_importance_weight.reshape((-1, step_per_episode))
        )
        state_action_marginal_importance_weight = np.roll(
            state_action_marginal_importance_weight, n_step_pdis, axis=1
        )
        state_action_marginal_importance_weight[:, :n_step_pdis] = 1

        return state_action_marginal_importance_weight


@dataclass
class BaseCumulativeDistributionOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for Cumulative Distribution OPE estimators."""

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

    def _target_value_given_idx(idx_: int, reward_scale: np.ndarray):
        if len(idx_):
            target_idx = idx_[0]
            target_value = (reward_scale[target_idx] + reward_scale[target_idx + 1]) / 2
        else:
            target_value = reward_scale[-1]
        return target_value

    def _aggregate_trajectory_wise_statistics_discrete(
        self,
        step_per_episode: int,
        action: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None,
        pscore: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        initial_state_value_prediction: Optional[np.ndarray] = None,
        gamma: float = 1.0,
    ):
        """Aggregate step-wise observations into trajectory wise statistics for the discrete action setup.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: ndarray of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
        n_samples = len(action)

        if reward is not None:
            reward = reward.reshape((-1, step_per_episode))
            discount = np.full(reward.shape[1], gamma).cumprod()
            trajectory_wise_reward = (reward * discount).sum(axis=1)

        if (
            action is not None
            and pscore is not None
            and evaluation_policy_action_dist is not None
        ):
            pscore = pscore.reshape((-1, step_per_episode))
            behavior_policy_pscore = np.cumprod(pscore, axis=1)[:, -1]

            evaluation_policy_pscore = evaluation_policy_action_dist[
                np.arange(n_samples), action
            ].reshape((-1, step_per_episode))

            evaluation_policy_pscore = np.cumprod(evaluation_policy_pscore, axis=1)[
                :, -1
            ]

            trajectory_wise_importance_weight = (
                evaluation_policy_pscore / behavior_policy_pscore
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
        pscore: Optional[np.ndarray] = None,
        evaluation_policy_action: Optional[np.ndarray] = None,
        initial_state_value_prediction: Optional[np.ndarray] = None,
        action_scaler: Optional[np.ndarray] = None,
        sigma: float = 1.0,
        gamma: float = 1.0,
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

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

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
            action is not None
            and pscore is not None
            and evaluation_policy_action is not None
        ):

            action_dim = action.shape[1]
            action = action.reshape((-1, step_per_episode, action_dim))
            evaluation_policy_action = evaluation_policy_action.reshape(
                (-1, step_per_episode, action_dim)
            )

            pscore = pscore.reshape((-1, step_per_episode))
            behavior_policy_pscore = np.cumprod(pscore, axis=1)[:, -1]

            similarity_weight = gaussian_kernel(
                evaluation_policy_action / action_scaler[np.newaxis, :],
                action / action_scaler[np.newaxis, :],
                sigma=sigma,
            )

            trajectory_wise_importance_weight = (
                similarity_weight / behavior_policy_pscore
            )

        return (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        )
