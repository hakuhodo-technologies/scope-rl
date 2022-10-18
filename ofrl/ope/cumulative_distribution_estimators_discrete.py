"""Cumulative Distribution Off-Policy Estimators for Discrete action."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.utils import check_scalar

from .estimators_base import (
    BaseCumulativeDistributionOffPolicyEstimator,
)
from ..utils import check_array


@dataclass
class DiscreteCumulativeDistributionDirectMethod(
    BaseCumulativeDistributionOffPolicyEstimator,
):
    """Direct Method (DM) for estimating cumulative distribution function (CDF) in discrete-action OPE.

    Note
    -------
    DM estimates CDF using the initial state value given by Fitted Q Evaluation (FQE) as follows.

    .. math::

        \\hat{F}_{\\mathrm{DM}}(m, \\pi; \\mathcal{D}) := \\mathbb{E}_{n} \\left[ \\mathbb{E}_{a_0 \\sim \\pi(a_0 \\mid s_0)} \\hat{G}(m; s_0, a_0) \\right]

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function and :math:`\\hat{G}(\\cdot)` is the estimated conditional distribution.

    Parameters
    -------
    estimator_name: str, default="cdf_dm"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Alina Beygelzimer and John Langford.
    "The Offset Tree for Learning with Partial Labels.", 2009.

    """

    estimator_name: str = "cdf_dm"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_cumulative_distribution_function: array-like of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if (
            reward.shape[0] // step_per_episode
            != initial_state_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] // step_per_episode == initial_state_value_prediction`, but found False"
            )

        density = np.histogram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0]

        return np.insert(density, 0, 0).cumsum()

    def estimate_mean(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alphas: np.ndarray = np.linspace(0, 1, 20),
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alphas: array-like of default=np.linspace(0, 1, 20)
            Set of proportions of the sided region.

        Return
        -------
        estimated_conditional_value_at_risk: NDArray
            Estimated conditional value at risk (CVaR) of the policy value.

        """
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        alphas = np.sort(alphas)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density > alpha)[0]
            lower_idx_ = idx_[0] if len(idx_) else -2
            cvar[i] = (np.diff(cumulative_density) * reward_scale[1:])[
                : lower_idx_ + 1
            ].sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        estimated_interquartile_range: Dict[str, float]
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        def target_value_given_idx(idx_):
            if len(idx_):
                target_idx = idx_[0]
                target_value = (
                    reward_scale[target_idx] + reward_scale[target_idx + 1]
                ) / 2
            else:
                target_value = reward_scale[-1]
            return target_value

        lower_idx_ = np.nonzero(cumulative_density > alpha)[0]
        median_idx_ = np.nonzero(cumulative_density > 0.5)[0]
        upper_idx_ = np.nonzero(cumulative_density > 1 - alpha)[0]

        estimated_interquartile_range = {
            "median": target_value_given_idx(median_idx_),
            f"{100 * (1. - alpha)}% quartile (lower)": target_value_given_idx(
                lower_idx_
            ),
            f"{100 * (1. - alpha)}% quartile (upper)": target_value_given_idx(
                upper_idx_
            ),
        }

        return estimated_interquartile_range


@dataclass
class DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling(
    BaseCumulativeDistributionOffPolicyEstimator,
):
    """Trajectory-wise Importance Sampling (TIS) for estimating cumulative distribution function (CDF) in discrete-action OPE.

    Note
    -------
    TIS estimates CDF using importance sampling techniques as follows.

    .. math::

        \\hat{F}_{\\mathrm{TIS}}(m, \\pi; \\mathcal{D}) := \\mathbb{E}_{n} \\left[ w_{1:T-1} \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} \\right]

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function,
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}` is the trajectory-wise importance weight,
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.

    Parameters
    -------
    estimator_name: str, default="cdf_tis"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "cdf_tis"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_cumulative_distribution_function: array-like of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action_dist.shape[1] - 1,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action_dist.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, "
                "but found False"
            )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        )

        n = len(trajectory_wise_reward)

        sort_idxes = trajectory_wise_reward.argsort()
        sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
        cumulative_density = np.clip(sorted_importance_weight.cumsum() / n, 0, 1)

        histogram = np.histogram(
            trajectory_wise_reward, bins=reward_scale, density=False
        )[0]
        cumulative_density = cumulative_density[histogram.cumsum().astype(int) - 1]

        return np.insert(cumulative_density, 0, 0)

    def estimate_mean(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alphas: np.ndarray = np.linspace(0, 1, 20),
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alphas: array-like of default=np.linspace(0, 1, 20)
            Set of proportions of the sided region.

        Return
        -------
        estimated_conditional_value_at_risk: NDArray
            Estimated conditional value at risk (CVaR) of the policy value.

        """
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        alphas = np.sort(alphas)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density > alpha)[0]
            lower_idx_ = idx_[0] if len(idx_) else -2
            cvar[i] = (np.diff(cumulative_density) * reward_scale[1:])[
                : lower_idx_ + 1
            ].sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        estimated_interquartile_range: Dict[str, float]
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        def target_value_given_idx(idx_):
            if len(idx_):
                target_idx = idx_[0]
                target_value = (
                    reward_scale[target_idx] + reward_scale[target_idx + 1]
                ) / 2
            else:
                target_value = reward_scale[-1]
            return target_value

        lower_idx_ = np.nonzero(cumulative_density > alpha)[0]
        median_idx_ = np.nonzero(cumulative_density > 0.5)[0]
        upper_idx_ = np.nonzero(cumulative_density > 1 - alpha)[0]

        estimated_interquartile_range = {
            "median": target_value_given_idx(median_idx_),
            f"{100 * (1. - alpha)}% quartile (lower)": target_value_given_idx(
                lower_idx_
            ),
            f"{100 * (1. - alpha)}% quartile (upper)": target_value_given_idx(
                upper_idx_
            ),
        }

        return estimated_interquartile_range


@dataclass
class DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust(
    BaseCumulativeDistributionOffPolicyEstimator,
):
    """Trajectory-wise Doubly Robust (TDR) for estimating cumulative distribution function (CDF) in discrete-action OPE.

    Note
    -------
    TDR estimates CDF using importance sampling techniques as follows.

    .. math::

        \\hat{F}_{\\mathrm{TDR}}(m, \\pi; \\mathcal{D})
        := \\mathbb{E}_{n} \\left[ w_{1:T-1} \\left( \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} - \\hat{G}(m; s_0, a_0) \\right) \\right]
        + \\hat{F}_{\\mathrm{DM}}(m, \\pi; \\mathcal{D})

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function,
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}` is the trajectory-wise importance weight,
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.

    Parameters
    -------
    estimator_name: str, default="cdf_tdr"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "cdf_tdr"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_cumulative_distribution_function: array-like of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action_dist.shape[1] - 1,
        )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action_dist.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, "
                "but found False"
            )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )

        weighted_residual = np.zeros_like(reward_scale)
        for i, threshold in enumerate(reward_scale):
            observation = (trajectory_wise_reward <= threshold).astype(int)
            prediction = (initial_state_value_prediction <= threshold).astype(int)
            weighted_residual[i] = (
                trajectory_wise_importance_weight * (observation - prediction)
            ).mean()

        histogram_baseline = np.histogram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0].cumsum()
        histogram_baseline = np.insert(histogram_baseline, 0, 0)

        cumulative_density = weighted_residual + histogram_baseline
        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1)

    def estimate_mean(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alphas: np.ndarray = np.linspace(0, 1, 20),
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alphas: array-like of default=np.linspace(0, 1, 20)
            Set of proportions of the sided region.

        Return
        -------
        estimated_conditional_value_at_risk: NDArray
            Estimated conditional value at risk (CVaR) of the policy value.

        """
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        alphas = np.sort(alphas)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density > alpha)[0]
            lower_idx_ = idx_[0] if len(idx_) else -2
            cvar[i] = (np.diff(cumulative_density) * reward_scale[1:])[
                : lower_idx_ + 1
            ].sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        estimated_interquartile_range: Dict[str, float]
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        def target_value_given_idx(idx_):
            if len(idx_):
                target_idx = idx_[0]
                target_value = (
                    reward_scale[target_idx] + reward_scale[target_idx + 1]
                ) / 2
            else:
                target_value = reward_scale[-1]
            return target_value

        lower_idx_ = np.nonzero(cumulative_density > alpha)[0]
        median_idx_ = np.nonzero(cumulative_density > 0.5)[0]
        upper_idx_ = np.nonzero(cumulative_density > 1 - alpha)[0]

        estimated_interquartile_range = {
            "median": target_value_given_idx(median_idx_),
            f"{100 * (1. - alpha)}% quartile (lower)": target_value_given_idx(
                lower_idx_
            ),
            f"{100 * (1. - alpha)}% quartile (upper)": target_value_given_idx(
                upper_idx_
            ),
        }

        return estimated_interquartile_range


@dataclass
class DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling(
    DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling,
):
    """Self Normalized Trajectory-wise Importance Sampling (SNTIS) for estimating cumulative distribution function (CDF) in discrete-action OPE.

    Note
    -------
    SNTIS estimates CDF using importance sampling techniques as follows.

    .. math::

        \\hat{F}_{\\mathrm{SNTIS}}(m, \\pi; \\mathcal{D}))
        := \\mathbb{E}_{n} \\left[ \\frac{w_{1:T-1}}{\\sum_{n} [w_{1:T-1}]} \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} \\right]

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function,
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}` is the trajectory-wise importance weight,
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.

    Parameters
    -------
    estimator_name: str, default="cdf_sntis"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "cdf_sntis"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_cumulative_distribution_function: array-like of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action_dist.shape[1] - 1,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action_dist.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, "
                "but found False"
            )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        )

        weight_sum = trajectory_wise_importance_weight.sum()

        sort_idxes = trajectory_wise_reward.argsort()
        sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
        cumulative_density = np.clip(
            sorted_importance_weight.cumsum() / weight_sum, 0, 1
        )

        histogram = np.histogram(
            trajectory_wise_reward, bins=reward_scale, density=False
        )[0]
        cumulative_density = cumulative_density[histogram.cumsum().astype(int) - 1]

        return np.insert(cumulative_density, 0, 0)


@dataclass
class DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust(
    DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust,
):
    """Self Normalized Trajectory-wise Doubly Robust (SNTDR) for estimating cumulative distribution function (CDF) in discrete-action OPE.

    Note
    -------
    SNTDR estimates CDF using importance sampling techniques as follows.

    .. math::

        \\hat{F}_{\\mathrm{SNTDR}}(,m \\pi; \\mathcal{D}))
        := \\mathbb{E}_{n} \\left[ \\frac{w_{1:T-1}}{\\sum_{n} [w_{1:T-1}]} \\left( \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq t \\right \\} - \\hat{G}(m; s_0, a_0) \\right) \\right]
        + \\hat{F}_{\\mathrm{DM}}(m, \\pi; \\mathcal{D}))

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function,
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}` is the trajectory-wise importance weight,
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.

    Parameters
    -------
    estimator_name: str, default="cdf_sntdr"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "cdf_sntdr"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_cumulative_distribution_function: array-like of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action_dist.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, "
                "but found False"
            )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )

        weighted_residual = np.zeros_like(reward_scale)
        for i, threshold in enumerate(reward_scale):
            observation = (trajectory_wise_reward <= threshold).astype(int)
            prediction = (initial_state_value_prediction <= threshold).astype(int)
            weighted_residual[i] = (
                trajectory_wise_importance_weight * (observation - prediction)
            ).sum() / trajectory_wise_importance_weight.sum()

        histogram_baseline = np.histogram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0].cumsum()
        histogram_baseline = np.insert(histogram_baseline, 0, 0)

        cumulative_density = weighted_residual + histogram_baseline
        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1)
