"""Cumulative Distribution Off-Policy Estimators for continuous action cases (designed for deterministic policies)."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.utils import check_scalar

from d3rlpy.preprocessing import ActionScaler

from .estimators_base import (
    BaseCumulativeDistributionOffPolicyEstimator,
)
from ..utils import check_array


@dataclass
class ContinuousCumulativeDistributionDirectMethod(
    BaseCumulativeDistributionOffPolicyEstimator,
):
    """Direct Method (DM) for estimating cumulative distribution function (CDF) in continuous-action OPE.

    Bases: :class:`ofrl.ope.BaseCumulativeDistributionOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.ContinuousCumulativeDistributionDirectMethod`

    Note
    -------
    DM estimates CDF using the initial state value as follows.

    .. math::

        \\hat{F}_{\\mathrm{DM}}(m, \\pi; \\mathcal{D}) := \\mathbb{E}_{n} [\\hat{G}(m; s_0, \\pi(a_0))]

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function and :math:`\\hat{G}(\\cdot)` is the estimated conditional distribution.

    DM has low variance, but can incur bias due to approximation errors.

    There are several ways to estimate :math:`\\hat{Q}(s, a)` such as Fitted Q Evaluation (FQE) (Le et al., 2019) and
    Minimax Q-Function Learning (MQL) (Uehara et al., 2020).

    .. seealso::

        The implementation of FQE is provided by `d3rlpy <https://d3rlpy.readthedocs.io/en/latest/references/off_policy_evaluation.html>`_.
        The implementations of Minimax Learning is available at :class:`ofrl.ope.weight_value_learning`.

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
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

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
        check_array(reward, name="reward", expected_dim=1)
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
        if reward.shape[0] != state_action_value_prediction.shape[0]:
            raise ValueError(
                "Expected `reward.shape[0] == state_action_value_prediction.shape[0]`, but found False"
            )
        if reward.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_trajectory=step_per_trajectory,
            state_action_value_prediction=state_action_value_prediction,
        )

        initial_state_value_prediction = np.clip(
            initial_state_value_prediction, reward_scale.min(), reward_scale.max()
        )

        density = np.histogram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0]

        return np.insert(density, 0, 0).cumsum()

    def estimate_mean(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
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
            Reward observation.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
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
            Reward observation.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
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
            Reward observation.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alphas: array-like of shape (n_alpha, ), default=None
            Set of proportions of the sided region. The values should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        Return
        -------
        estimated_conditional_value_at_risk: ndarray of (n_alpha, )
            Estimated conditional value at risk (CVaR) of the policy value.

        """
        if alphas is None:
            alphas = np.linspace(0, 1, 21)
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        alphas = np.sort(alphas)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density > alpha)[0]
            lower_idx_ = (
                -2 if len(idx_) == 0 or idx_[0] == len(reward_scale) - 1 else idx_[0]
            )
            cvar[i] = (np.diff(cumulative_density) * reward_scale[1:])[
                : lower_idx_ + 1
            ].sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        estimated_interquartile_range: dict
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
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
class ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling(
    BaseCumulativeDistributionOffPolicyEstimator,
):
    """Trajectory-wise Importance Sampling (TIS) for estimating cumulative distribution function (CDF) in continuous-action OPE.

    Bases: :class:`ofrl.ope.BaseCumulativeDistributionOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.ContinuousCumulativeTrajectoryWiseImportanceSampling`

    Note
    -------
    TIS estimates CDF using importance sampling techniques as follows.

    .. math::

        \\hat{F}_{\\mathrm{TIS}}(m, \\pi; \\mathcal{D})
        := \\mathbb{E}_{n} \\left[ w_{1:T-1} \\delta(\\pi, a_{0:T-1}) \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} \\right]

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function,
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t \\mid s_t) / \\pi_0(a_t \\mid s_t))` is the trajectory-wise importance weight,
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.
    :math:`\\delta(\\pi, a_{0:T-1}) = \\prod_{t=0}^{T-1} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    TIS enables an unbiased estimation of the policy value. However, when the trajectory length (:math:`T`) is large,
    TIS suffers from high variance due to the product of importance weights.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    """

    estimator_name: str = "cdf_tis"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            pscore,
            name="pscore",
            expected_dim=2,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        check_array(
            action,
            name="action",
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
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, "
                "but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
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
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            reward_scale=reward_scale,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            reward_scale=reward_scale,
            action_scaler=action_scaler,
            sigma=sigma,
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
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alphas: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        alphas: array-like of shape (n_alpha, ), default=None
            Set of proportions of the sided region. The values should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        Return
        -------
        estimated_conditional_value_at_risk: ndarray of (n_alpha, )
            Estimated conditional value at risk (CVaR) of the policy value.

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
            evaluation_policy_action=evaluation_policy_action,
            reward_scale=reward_scale,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density > alpha)[0]
            lower_idx_ = (
                -2 if len(idx_) == 0 or idx_[0] == len(reward_scale) - 1 else idx_[0]
            )
            cvar[i] = (np.diff(cumulative_density) * reward_scale[1:])[
                : lower_idx_ + 1
            ].sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        estimated_interquartile_range: dict
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            reward_scale=reward_scale,
            action_scaler=action_scaler,
            sigma=sigma,
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
class ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust(
    BaseCumulativeDistributionOffPolicyEstimator,
):
    """Trajectory-wise Doubly Robust (TDR) for estimating cumulative distribution function (CDF) in continuous-action OPE.

    Bases: :class:`ofrl.ope.BaseCumulativeDistributionOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.ContinuousCumulativeTrajectoryWiseDoublyRobust`


    Note
    -------
    TDR estimates CDF using importance sampling techniques as follows.

    .. math::

        \\hat{F}_{\\mathrm{TDR}}(m, \\pi; \\mathcal{D})
        &:= \\mathbb{E}_{n} [\\hat{G}(m; s_0, \\pi(a_0))] \\\\
        & \quad \quad + \\mathbb{E}_{n} \\left[ w_{0:T-1} \\delta(\\pi, a_{0:T-1}) \\left( \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} - \\hat{G}(m; s_0, a_0) \\right) \\right]

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function and :math:`\\hat{G}(\\cdot)` is the estimated conditional distribution.
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t \\mid s_t) / \\pi_0(a_t \\mid s_t))` is the trajectory-wise importance weight and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.
    :math:`\\delta(\\pi, a_{0:T-1}) = \\prod_{t=0}^{T-1} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    TDR is unbiased and reduces the variance of TIS when :math:`\\hat{Q}(\\cdot)` is reasonably accurate to satisfy :math:`0 < \\hat{Q}(\\cdot) < 2 Q(\\cdot)`. 
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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    """

    estimator_name: str = "cdf_tdr"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            pscore,
            name="pscore",
            expected_dim=2,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        check_array(
            action,
            name="action",
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
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            action_scaler=action_scaler,
            sigma=sigma,
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
        )[0].cumsum()
        histogram_baseline = np.insert(histogram_baseline, 0, 0)

        cumulative_density = weighted_residual + histogram_baseline
        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1)

    def estimate_mean(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            action_scaler=action_scaler,
            sigma=sigma,
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
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alphas: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        alphas: array-like of shape (n_alpha, ), default=None
            Set of proportions of the sided region. The values should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        Return
        -------
        estimated_conditional_value_at_risk: ndarray of (n_alpha, )
            Estimated conditional value at risk (CVaR) of the policy value.

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
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        )

        cvar = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            idx_ = np.nonzero(cumulative_density > alpha)[0]
            lower_idx_ = (
                -2 if len(idx_) == 0 or idx_[0] == len(reward_scale) - 1 else idx_[0]
            )
            cvar[i] = (np.diff(cumulative_density) * reward_scale[1:])[
                : lower_idx_ + 1
            ].sum()

        return cvar

    def estimate_interquartile_range(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        estimated_interquartile_range: dict
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            reward_scale=reward_scale,
            action_scaler=action_scaler,
            sigma=sigma,
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
class ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling(
    ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling,
):
    """Self Normalized Trajectory-wise Importance Sampling (SNTIS) for estimating cumulative distribution function (CDF) in continuous-action OPE.

    Bases: :class:`ofrl.ope.ContinuousCumulativeTrajectoryWiseImportanceSampling` -> :class:`ofrl.ope.BaseCumulativeDistributionOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.ContinuousCumulativeSelfNormalizedTrajectoryWiseImportanceSampling`


    Note
    -------
    SNTIS estimates CDF using importance sampling techniques as follows.

    .. math::

        \\hat{F}_{\\mathrm{SNTIS}}(m, \\pi; \\mathcal{D}))
        := \\mathbb{E}_{n} \\left[ \\frac{w_{0:T-1} \\delta(\\pi, a_{0:T-1})}{\\sum_{n} w_{0:T-1} \\delta(\\pi, a_{0:T-1})} \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} \\right]

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function,
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t \\mid s_t) / \\pi_0(a_t \\mid s_t))` is the trajectory-wise importance weight,
    and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.
    :math:`\\delta(\\pi, a_{0:T-1}) = \\prod_{t=0}^{T-1} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    The self-normalized estimator is no longer unbiased, but has a bounded variance while also being consistent.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    """

    estimator_name: str = "cdf_sntis"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            pscore,
            name="pscore",
            expected_dim=2,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        check_array(
            action,
            name="action",
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
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`"
                ", but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
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
class ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust(
    ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust,
):
    """Self Normalized Trajectory-wise Doubly Robust (SNTDR) for estimating cumulative distribution function (CDF) in continuous-action OPE.

    Bases: :class:`ofrl.ope.ContinuousCumulativeTrajectoryWiseDoublyRobust` -> :class:`ofrl.ope.BaseCumulativeDistributionOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.ContinuousCumulativeSelfNormalizedTrajectoryWiseDoublyRobust`

    Note
    -------
    SNTDR estimates CDF using importance sampling techniques as follows.

    .. math::

        \\hat{F}_{\\mathrm{SNTDR}}(m, \\pi; \\mathcal{D}))
        &:= \\mathbb{E}_{n} [\\hat{G}(m; s_0, \\pi(a_0))] \\\\
        & \\quad \\quad + \\mathbb{E}_{n} \\left[ \\frac{w_{0:T-1} \\delta(\\pi, a_{0:T-1})}{\\sum_{n} w_{0:T-1} \\delta(\\pi, a_{0:T-1})} \\left( \\mathbb{I} \\left \\{\\sum_{t=0}^{T-1} \\gamma^t r_t \\leq t \\right \\} - \\hat{G}(m; s_0, a_0) \\right) \\right]
        

    where :math:`\\hat{F}(\\cdot)` is the estimated cumulative distribution function and :math:`\\hat{G}(\\cdot)` is the estimated conditional distribution.
    :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t \\mid s_t) / \\pi_0(a_t \\mid s_t))` is the trajectory-wise importance weight and :math:`\\mathbb{I} \\{ \\cdot \\}` is the indicator function.
    :math:`\\delta(\\pi, a_{0:T-1}) = \\prod_{t=0}^{T-1} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    The self-normalized estimator is no longer unbiased, but has a bounded variance while also being consistent.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    """

    estimator_name: str = "cdf_sntdr"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        sigma: float, default=1.0 (> 0)
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_cumulative_distribution_function: ndarray of shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            pscore,
            name="pscore",
            expected_dim=2,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
            min_val=0.0,
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
            == evaluation_policy_action.shape[0]
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            action_scaler=action_scaler,
            sigma=sigma,
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
        )[0].cumsum()
        histogram_baseline = np.insert(histogram_baseline, 0, 0)

        cumulative_density = weighted_residual + histogram_baseline
        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1)
