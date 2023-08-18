# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Cumulative Distribution Off-Policy Estimators for discrete action cases."""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
from sklearn.utils import check_scalar

from ..estimators_base import (
    BaseCumulativeDistributionOPEEstimator,
)
from ...utils import check_array


@dataclass
class CumulativeDistributionDM(
    BaseCumulativeDistributionOPEEstimator,
):
    """Direct Method (DM) for estimating the cumulative distribution function (CDF) for discrete action spaces.

    Bases: :class:`scope_rl.ope.BaseCumulativeDistributionOPEEstimator`

    Imported as: :class:`scope_rl.ope.discrete.CumulativeDistributionDM`

    Note
    -------
    DM estimates the CDF using the initial state value as follows.

    .. math::

        \\hat{F}_{\\mathrm{DM}}(m, \\pi; \\mathcal{D}) := \\frac{1}{n} \\sum_{i=1}^n \\sum_{a \\in \\mathcal{A}} \\pi(a \\mid s_0^{(i)}) \\hat{G}(m; s_0^{(i)}, a)

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function and :math:`\\hat{G}(\\cdot)` is an estimator for :math:`\\mathbb{E} \\left[ \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} \\mid s,a \\right]`.

    DM has low variance compared to other estimators, but can produce larger bias due to approximation errors.

    There are several methods to estimate :math:`\\hat{Q}(s, a)` such as Fitted Q Evaluation (FQE) (Le et al., 2019) and
    Minimax Q-Function Learning (MQL) (Uehara et al., 2020).

    .. seealso::

        The implementation of FQE is provided by `d3rlpy <https://d3rlpy.readthedocs.io/en/latest/references/off_policy_evaluation.html>`_.
        The implementations of Minimax Learning is available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="cdf_dm"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation." 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints." 2019.

    """

    estimator_name: str = "cdf_dm"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate the cumulative distribution function (CDF) of the reward distribution under the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
        )
        check_array(
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if not (
            reward.shape[0]
            == evaluation_policy_action_dist.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == evaluation_policy_action_dist.shape[0] == state_action_value_prediction.shape[0]`"
                ", but found False"
            )
        if reward.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if (
            evaluation_policy_action_dist.shape[1]
            != state_action_value_prediction.shape[1]
        ):
            raise ValueError(
                "Expected evaluation_policy_action_dist.shape[1] == state_action_value_prediction.shape[1], but found False"
            )
        if not np.allclose(
            evaluation_policy_action_dist.sum(axis=1),
            np.ones(evaluation_policy_action_dist.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action_dist.sum(axis=1) == np.ones(evaluation_policy_action_dist.shape[0])`"
                ", but found it False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_trajectory=step_per_trajectory,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
        )

        initial_state_value_prediction = np.clip(
            initial_state_value_prediction, reward_scale.min(), reward_scale.max()
        )
        density = np.histogram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0]
        probability_density_function = density * np.diff(reward_scale)

        return np.insert(probability_density_function, 0, 0).cumsum()

    def estimate_mean(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_mean: float
            Estimated mean of the reward under the evaluation policy.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_variance: float
            Estimated variance of the reward under the evaluation policy.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alphas: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alphas: array-like of shape (n_alpha, ), default=None
            Set of proportions of the shaded region. The values should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        Return
        -------
        estimated_conditional_value_at_risk: ndarray of (n_alpha, )
            Estimated conditional value at risk (CVaR) of the reward under the evaluation policy.

        """
        if alphas is None:
            alphas = np.linspace(0, 1, 21)
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        alphas = np.sort(alphas)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density[1:] > alpha)[0]
            if len(idx_) == 0:
                cvar[i] = (
                    np.diff(cumulative_density) * reward_scale[1:]
                ).sum() / cumulative_density[-1]
            elif idx_[0] == 0:
                cvar[i] = reward_scale[1]
            else:
                lower_idx_ = idx_[0]
                relative_probability_density = (
                    np.diff(cumulative_density)[: lower_idx_ + 1]
                    / cumulative_density[lower_idx_ + 1]
                )
                cvar[i] = (
                    relative_probability_density * reward_scale[1 : lower_idx_ + 2]
                ).sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alpha: float, default=0.05
            Proportion of the shaded region.

        Return
        -------
        estimated_interquartile_range: dict
            Estimated interquartile range of the reward under the evaluation policy.

            .. code-block:: python

                key: [
                    mean,
                    {100 * (1. - alpha)}% quartile (lower),
                    {100 * (1. - alpha)}% quartile (upper),
                ]

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        lower_idx_ = np.nonzero(cumulative_density > alpha)[0]
        median_idx_ = np.nonzero(cumulative_density > 0.5)[0]
        upper_idx_ = np.nonzero(cumulative_density > 1 - alpha)[0]

        estimated_interquartile_range = {
            "median": self._target_value_given_idx(
                median_idx_, reward_scale=reward_scale
            ),
            f"{100 * (1. - alpha)}% quartile (lower)": self._target_value_given_idx(
                lower_idx_,
                reward_scale=reward_scale,
            ),
            f"{100 * (1. - alpha)}% quartile (upper)": self._target_value_given_idx(
                upper_idx_,
                reward_scale=reward_scale,
            ),
        }

        return estimated_interquartile_range


@dataclass
class CumulativeDistributionTIS(
    BaseCumulativeDistributionOPEEstimator,
):
    """Trajectory-wise Importance Sampling (TIS) for estimating the cumulative distribution function (CDF) for discrete action spaces.

    Bases: :class:`scope_rl.ope.BaseCumulativeDistributionOPEEstimator`

    Imported as: :class:`scope_rl.ope.discrete.CumulativeDistributionTIS`

    Note
    -------
    TIS estimates the CDF via trajectory-wise importance weighting as follows.

    .. math::

        \\hat{F}_{\\mathrm{TIS}}(m, \\pi; \\mathcal{D}) := \\frac{1}{n} \\sum_{i=1}^n w_{0:T-1}^{(i)} \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t^{(i)} \\leq m \\right \\}

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function,
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t \\mid s_t) / \\pi_0(a_t \\mid s_t))` is the trajectory-wise importance weight,
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.

    TIS enables an unbiased estimation of the policy value. However, when the trajectory length (:math:`T`) is large,
    TIS suffers from high variance due to the product of importance weights over the entire horizon.

    Parameters
    -------
    estimator_name: str, default="cdf_tis"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation." 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits." 2021.

    """

    estimator_name: str = "cdf_tis"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate the cumulative distribution function (CDF) of the reward distribution under the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            pscore,
            name="pscore",
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, "
                "but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if not np.allclose(
            evaluation_policy_action_dist.sum(axis=1),
            np.ones(evaluation_policy_action_dist.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action_dist.sum(axis=1) == np.ones(evaluation_policy_action_dist.shape[0])`"
                ", but found it False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        )

        n = len(trajectory_wise_reward)

        sort_idxes = trajectory_wise_reward.argsort()
        sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
        cumulative_density = np.clip(sorted_importance_weight.cumsum() / n, 0, 1)

        trajectory_wise_reward = np.clip(
            trajectory_wise_reward, reward_scale.min(), reward_scale.max()
        )
        histogram = np.histogram(
            trajectory_wise_reward, bins=reward_scale, density=False
        )[0]

        idx = histogram.cumsum().astype(int) - 1
        idx = np.where(idx < 0, 0, idx)

        cumulative_density = cumulative_density[idx]

        return np.insert(cumulative_density, 0, 0)

    def estimate_mean(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_mean: float
            Estimated mean of the reward under the evaluation policy.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_variance: float
            Estimated variance of the reward under the evaluation policy.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alphas: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alphas: array-like of shape (n_alpha, ), default=None
            Set of proportions of the shaded region. The values should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        Return
        -------
        estimated_conditional_value_at_risk: ndarray of (n_alpha, )
            Estimated conditional value at risk (CVaR) of the reward under the evaluation policy.

        """
        if alphas is None:
            alphas = np.linspace(0, 1, 21)
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        alphas = np.sort(alphas)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density[1:] > alpha)[0]
            if len(idx_) == 0:
                cvar[i] = (
                    np.diff(cumulative_density) * reward_scale[1:]
                ).sum() / cumulative_density[-1]
            elif idx_[0] == 0:
                cvar[i] = reward_scale[1]
            else:
                lower_idx_ = idx_[0]
                relative_probability_density = (
                    np.diff(cumulative_density)[: lower_idx_ + 1]
                    / cumulative_density[lower_idx_ + 1]
                )
                cvar[i] = (
                    relative_probability_density * reward_scale[1 : lower_idx_ + 2]
                ).sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alpha: float, default=0.05
            Proportion of the shaded region.

        Return
        -------
        estimated_interquartile_range: dict
            Estimated interquartile range of the reward under the evaluation policy.

            .. code-block:: python

                key: [
                    mean,
                    {100 * (1. - alpha)}% quartile (lower),
                    {100 * (1. - alpha)}% quartile (upper),
                ]

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        lower_idx_ = np.nonzero(cumulative_density > alpha)[0]
        median_idx_ = np.nonzero(cumulative_density > 0.5)[0]
        upper_idx_ = np.nonzero(cumulative_density > 1 - alpha)[0]

        estimated_interquartile_range = {
            "median": self._target_value_given_idx(
                median_idx_, reward_scale=reward_scale
            ),
            f"{100 * (1. - alpha)}% quartile (lower)": self._target_value_given_idx(
                lower_idx_,
                reward_scale=reward_scale,
            ),
            f"{100 * (1. - alpha)}% quartile (upper)": self._target_value_given_idx(
                upper_idx_,
                reward_scale=reward_scale,
            ),
        }

        return estimated_interquartile_range


@dataclass
class CumulativeDistributionTDR(
    BaseCumulativeDistributionOPEEstimator,
):
    """Trajectory-wise Doubly Robust (TDR) for estimating the cumulative distribution function (CDF) for discrete action spaces.

    Bases: :class:`scope_rl.ope.BaseCumulativeDistributionOPEEstimator`

    Imported as: :class:`scope_rl.ope.discrete.CumulativeDistributionTrajectoryWiseDR`

    Note
    -------
    TDR estimates the CDF via trajectory-wise importance weighting and estimated Q-function :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{F}_{\\mathrm{TDR}}(m, \\pi; \\mathcal{D})
        &:= \\frac{1}{n} \\sum_{i=1}^n \\sum_{a \\in \\mathcal{A}} \\pi(a \\mid s_0^{(i)}) \\hat{G}(m; s_0^{(i)}, a) \\\\
        & \quad \quad + \\frac{1}{n} \\sum_{i=1}^n w_{0:T-1}^{(i)} \\left( \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t^{(i)} \\leq m \\right \\} - \\hat{G}(m; s_0^{(i)}, a_0^{(i)}) \\right)

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function and :math:`\\hat{G}(\\cdot;s,a)` is an estimator for :math:`\\mathbb{E} \\left[ \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} \\mid s,a \\right]`.
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t \\mid s_t) / \\pi_0(a_t \\mid s_t))` is the trajectory-wise importance weight
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.

    TDR is unbiased and has lower variance than TIS when :math:`\\hat{Q}(\\cdot)` is reasonably accurate and satisfies :math:`0 < \\hat{Q}(\\cdot) < 2 Q(\\cdot)`.
    However, when the importance weight is quite large, it may still suffer from a high variance.

    Parameters
    -------
    estimator_name: str, default="cdf_tdr"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation." 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits." 2021.

    """

    estimator_name: str = "cdf_tdr"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate the cumulative distribution function (CDF) of the reward distribution under the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            pscore,
            name="pscore",
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
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0] "
                "== state_action_value_prediction.shape[0]`, but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if (
            evaluation_policy_action_dist.shape[1]
            != state_action_value_prediction.shape[1]
        ):
            raise ValueError(
                "Expected evaluation_policy_action_dist.shape[1] == state_action_value_prediction.shape[1], but found False"
            )
        if not np.allclose(
            evaluation_policy_action_dist.sum(axis=1),
            np.ones(evaluation_policy_action_dist.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action_dist.sum(axis=1) == np.ones(evaluation_policy_action_dist.shape[0])`"
                ", but found it False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
        )

        trajectory_wise_reward = np.clip(
            trajectory_wise_reward, reward_scale.min(), reward_scale.max()
        )
        initial_state_value_prediction = np.clip(
            initial_state_value_prediction, reward_scale.min(), reward_scale.max()
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
        )[0]
        histogram_baseline = (histogram_baseline * np.diff(reward_scale)).cumsum()
        histogram_baseline = np.insert(histogram_baseline, 0, 0)

        cumulative_density = weighted_residual + histogram_baseline
        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1)

    def estimate_mean(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_mean: float
            Estimated mean of the reward under the evaluation policy.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_variance: float
            Estimated variance of the reward under the evaluation policy.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alphas: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alphas: array-like of shape (n_alpha, ), default=None
            Set of proportions of the shaded region. The values should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        Return
        -------
        estimated_conditional_value_at_risk: ndarray of (n_alpha, )
            Estimated conditional value at risk (CVaR) of the reward under the evaluation policy.

        """
        if alphas is None:
            alphas = np.linspace(0, 1, 21)
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        alphas = np.sort(alphas)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density[1:] > alpha)[0]
            if len(idx_) == 0:
                cvar[i] = (
                    np.diff(cumulative_density) * reward_scale[1:]
                ).sum() / cumulative_density[-1]
            elif idx_[0] == 0:
                cvar[i] = reward_scale[1]
            else:
                lower_idx_ = idx_[0]
                relative_probability_density = (
                    np.diff(cumulative_density)[: lower_idx_ + 1]
                    / cumulative_density[lower_idx_ + 1]
                )
                cvar[i] = (
                    relative_probability_density * reward_scale[1 : lower_idx_ + 2]
                ).sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alpha: float, default=0.05
            Proportion of the shaded region.

        Return
        -------
        estimated_interquartile_range: dict
            Estimated interquartile range of the reward under the evaluation policy.

            .. code-block:: python

                key: [
                    mean,
                    {100 * (1. - alpha)}% quartile (lower),
                    {100 * (1. - alpha)}% quartile (upper),
                ]

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        lower_idx_ = np.nonzero(cumulative_density > alpha)[0]
        median_idx_ = np.nonzero(cumulative_density > 0.5)[0]
        upper_idx_ = np.nonzero(cumulative_density > 1 - alpha)[0]

        estimated_interquartile_range = {
            "median": self._target_value_given_idx(
                median_idx_, reward_scale=reward_scale
            ),
            f"{100 * (1. - alpha)}% quartile (lower)": self._target_value_given_idx(
                lower_idx_,
                reward_scale=reward_scale,
            ),
            f"{100 * (1. - alpha)}% quartile (upper)": self._target_value_given_idx(
                upper_idx_,
                reward_scale=reward_scale,
            ),
        }

        return estimated_interquartile_range


@dataclass
class CumulativeDistributionSNTIS(
    CumulativeDistributionTIS,
):
    """Self Normalized Trajectory-wise Importance Sampling (SNTIS) for estimating the cumulative distribution function (CDF) for discrete action spaces.

    Bases: :class:`scope_rl.ope.discrete.CumulativeDistributionTIS` :class:`scope_rl.ope.BaseCumulativeDistributionOPEEstimator`

    Imported as: :class:`scope_rl.ope.discrete.CumulativeDistributionSNTIS`

    Note
    -------
    SNTIS estimates the CDF via trajectory-wise importance weighting as follows.

    .. math::

        \\hat{F}_{\\mathrm{SNTIS}}(m, \\pi; \\mathcal{D}))
        := \\sum_{i=1}^n \\frac{w_{0:T-1}^{(i)}}{\\sum_{i'=1}^n w_{0:T-1}^{(i')}} \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t^{(i)} \\leq m \\right \\}

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function,
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t \\mid s_t) / \\pi_0(a_t \\mid s_t))` is the trajectory-wise importance weight,
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.

    The self-normalized estimator is no longer unbiased, but has a bounded variance while also remaining consistent.

    Parameters
    -------
    estimator_name: str, default="cdf_sntis"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation." 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits." 2021.

    """

    estimator_name: str = "cdf_sntis"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate the cumulative distribution function (CDF) of the reward distribution under the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            pscore,
            name="pscore",
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, "
                "but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if not np.allclose(
            evaluation_policy_action_dist.sum(axis=1),
            np.ones(evaluation_policy_action_dist.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action_dist.sum(axis=1) == np.ones(evaluation_policy_action_dist.shape[0])`"
                ", but found it False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        )

        weight_sum = trajectory_wise_importance_weight.sum()

        sort_idxes = trajectory_wise_reward.argsort()
        sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
        cumulative_density = np.clip(
            sorted_importance_weight.cumsum() / weight_sum, 0, 1
        )

        trajectory_wise_reward = np.clip(
            trajectory_wise_reward, reward_scale.min(), reward_scale.max()
        )

        histogram = np.histogram(
            trajectory_wise_reward, bins=reward_scale, density=False
        )[0]

        idx = histogram.cumsum().astype(int) - 1
        idx = np.where(idx < 0, 0, idx)

        cumulative_density = cumulative_density[idx]

        return np.insert(cumulative_density, 0, 0)


@dataclass
class CumulativeDistributionSNTDR(
    CumulativeDistributionTDR,
):
    """Self Normalized Trajectory-wise Doubly Robust (SNTDR) for estimating the cumulative distribution function (CDF) for discrete action spaces.

    Bases: :class:`scope_rl.ope.discrete.CumulativeDistributionTDR` :class:`scope_rl.ope.BaseCumulativeDistributionOPEEstimator`

    Imported as: :class:`scope_rl.ope.discrete.CumulativeDistributionSNTDR`

    Note
    -------
    SNTDR estimates the CDF via trajectory-wise importance weighting and estimated Q-function :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{F}_{\\mathrm{SNTDR}}(m, \\pi; \\mathcal{D}))
        &:= \\frac{1}{n} \\sum_{i=1}^n \\sum_{a \\in \\mathcal{A}} \\pi(a \\mid s_0^{(t)}) \\hat{G}(m; s_0^{(t)}, a) \\\\
        & \quad \quad + \\sum_{i=1}^n \\frac{w_{0:T-1}^{(i)}}{\\sum_{i'=1}^n w_{0:T-1}^{(i')}} \\left( \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t^{(i)} \\leq m \\right \\} - \\hat{G}(m; s_0^{(i)}, a_0^{(i)}) \\right)

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function and :math:`\\hat{G}(\\cdot)` is an estimator for :math:`\\mathbb{E} \left[ \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} \\mid s,a \\right]`.
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t \\mid s_t) / \\pi_0(a_t \\mid s_t))` is the trajectory-wise importance weight and
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.

    The self-normalized estimator is no longer unbiased, but has a bounded variance while also remaining consistent.

    Parameters
    -------
    estimator_name: str, default="cdf_sntdr"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation." 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits." 2021.

    """

    estimator_name: str = "cdf_sntdr"

    def __post_init__(self):
        self.action_type = "discrete"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate the cumulative distribution function (CDF) of the reward distribution under the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            pscore,
            name="pscore",
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
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0] "
                "== state_action_value_prediction.shape[0]`, but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if (
            evaluation_policy_action_dist.shape[1]
            != state_action_value_prediction.shape[1]
        ):
            raise ValueError(
                "Expected evaluation_policy_action_dist.shape[1] == state_action_value_prediction.shape[1], but found False"
            )
        if not np.allclose(
            evaluation_policy_action_dist.sum(axis=1),
            np.ones(evaluation_policy_action_dist.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action_dist.sum(axis=1) == np.ones(evaluation_policy_action_dist.shape[0])`"
                ", but found it False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
        )

        trajectory_wise_reward = np.clip(
            trajectory_wise_reward, reward_scale.min(), reward_scale.max()
        )
        initial_state_value_prediction = np.clip(
            initial_state_value_prediction, reward_scale.min(), reward_scale.max()
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
        )[0]
        histogram_baseline = (histogram_baseline * np.diff(reward_scale)).cumsum()
        histogram_baseline = np.insert(histogram_baseline, 0, 0)

        cumulative_density = weighted_residual + histogram_baseline
        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1)
