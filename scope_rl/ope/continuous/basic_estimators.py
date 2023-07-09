# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators for continuous action cases (designed for deterministic evaluation policies)."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.utils import check_scalar

from d3rlpy.preprocessing import ActionScaler

from ..estimators_base import BaseOffPolicyEstimator
from ...utils import check_array


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) (designed for deterministic evaluation policies) for continuous action spaces.

    Bases: :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.DirectMethod`

    Note
    -------
    DM estimates the policy value using an estimated initial state value as follows.

    .. math::

        \\hat{J}_{\\mathrm{DM}} (\\pi; \\mathcal{D})
        := \\frac{1}{n} \\sum_{i=1}^n \\hat{Q}(s_0^{(i)}, \\pi(s_0^{(i)}))
        = \\frac{1}{n} \\sum_{i=1}^n \\hat{V}(s_0^{(i)}),

    where :math:`\\mathcal{D}=\\{\\{(s_t, a_t, r_t)\\}_{t=0}^{T-1}\\}_{i=1}^n` is the logged dataset with :math:`n` trajectories.
    :math:`T` indicates step per episode. :math:`\\hat{Q}(s_t, a_t)` is the estimated Q value given a state-action pair.
    :math:`\\hat{V}(s_t)` is the estimated value function given a state.

    DM has low variance compared to other estimators, but can produce larger bias due to approximation errors.

    There are several methods to estimate :math:`\\hat{Q}(s, a)` such as Fitted Q Evaluation (FQE) (Le et al., 2019) and
    Minimax Q-Function Learning (MQL) (Uehara et al., 2020).

    .. seealso::

        The implementation of FQE is provided by `d3rlpy <https://d3rlpy.readthedocs.io/en/latest/references/off_policy_evaluation.html>`_.
        The implementations of Minimax Learning is available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="dm"
        Name of the estimator.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints." 2019.

    """

    estimator_name = "dm"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        state_action_value_prediction: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value (expected reward under the evaluation policy) estimated for each trajectory.

        """
        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_trajectory, 2)
        )
        return state_action_value_prediction[:, 0, 1]

    def estimate_policy_value(
        self,
        step_per_trajectory: int,
        state_action_value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        Return
        -------
        V_hat: float
            Estimated policy value.

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        if state_action_value_prediction.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            state_action_value_prediction=state_action_value_prediction,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_trajectory: int,
        state_action_value_prediction: np.ndarray,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Name of the method to estimate the confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: dict
            Dictionary storing the estimated mean and upper-lower confidence bounds.

            .. code-block:: python

                key: [
                    mean,
                    {100 * (1. - alpha)}% CI (lower),
                    {100 * (1. - alpha)}% CI (upper),
                ]

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        if state_action_value_prediction.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )
        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            state_action_value_prediction=state_action_value_prediction,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class TrajectoryWiseImportanceSampling(BaseOffPolicyEstimator):
    """Trajectory-wise Importance Sampling (TIS) (designed for deterministic evaluation policies) for continuous action spaces.

    Bases: :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.TrajectoryWiseImportanceSampling`

    Note
    -------
    TIS estimates the policy value via trajectory-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{TIS}} (\\pi; \\mathcal{D})
        := \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=0}^{T-1} \\gamma^t w_{0:T-1}^{(i)} \\delta(\\pi, a_{0:T-1}^{(i)}) r_t^{(i)},

    where :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (1 / \\pi_0(a_t | s_t))` is the (trajectory-wise) importance weight.
    :math:`\\delta(\\pi, a_{0:T-1}) = \\prod_{t=0}^{T-1} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy where :math:`K(\\cdot)` is a kernel function.
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.

    TIS is able to correct the distribution shift between the behavior and evaluation policies. However, when the trajectory length (:math:`T`) is large,
    TIS suffers from high variance due to the product of importance weights over the entire horizon.

    Parameters
    -------
    estimator_name: str, default="tis"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name: str = "tis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value (expected reward under the evaluation policy) estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_continuous(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="trajectory_wise",
        )
        similarity_weight = self._calc_similarity_weight(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            pscore_type="trajectory_wise",
        )

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        estimated_trajectory_value = (
            discount[np.newaxis, :]
            * reward
            * similarity_weight
            / behavior_policy_pscore
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        V_hat: float
            Estimated policy value.

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        return self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            kernel=kernel,
            bandwidth=bandwidth,
            action_scaler=action_scaler,
        ).mean()

    def estimate_interval(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Name of the method to estimate the confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: dict
            Dictionary storing the estimated mean and upper-lower confidence bounds.

            .. code-block:: python

                key: [
                    mean,
                    {100 * (1. - alpha)}% CI (lower),
                    {100 * (1. - alpha)}% CI (upper),
                ]

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            kernel=kernel,
            bandwidth=bandwidth,
            action_scaler=action_scaler,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class PerDecisionImportanceSampling(BaseOffPolicyEstimator):
    """Per-Decision Importance Sampling (PDIS) (designed for deterministic evaluation policies) for continuous action spaces.

    Bases: :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.PerDecisionImportanceSampling`

    Note
    -------
    PDIS estimates the policy value via step-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{PDIS}} (\\pi; \\mathcal{D})
        := \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=0}^{T-1} \\gamma^t w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)}) r_t^{(i)},

    where :math:`w_{0:t} := \\prod_{t'=0}^t (1 / \\pi_0(a_{t'} | s_{t'}))`. is the importance weight for each time step wrt the previous actions (referred to as the per-decision or step-wise importance weight).
    :math:`\\delta(\\pi, a_{0:t}) = \\prod_{t'=0}^t K(\\pi(s_{t'}), a_{t'})` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy where :math:`K(\\cdot)` is a kernel function.
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.

    By using per-decision importance weighting instead of trajectory-wise importance weighting of TIS, PDIS has lower variance than TIS while still correcting the distribution shift between the behavior and evaluation policies.
    However, when the trajectory length (:math:`T`) is large, PDIS still suffers from high variance.

    Parameters
    -------
    estimator_name: str, default="pdis"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name: str = "pdis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value (expected reward under the evaluation policy) estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_continuous(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="step_wise",
        )
        similarity_weight = self._calc_similarity_weight(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            pscore_type="step_wise",
        )

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        estimated_trajectory_value = (
            discount[np.newaxis, :]
            * reward
            * similarity_weight
            / behavior_policy_pscore
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        V_hat: float
            Estimated policy value.

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        return self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            kernel=kernel,
            bandwidth=bandwidth,
            action_scaler=action_scaler,
        ).mean()

    def estimate_interval(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Name of the method to estimate the confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: dict
            Dictionary storing the estimated mean and upper-lower confidence bounds.

            .. code-block:: python

                key: [
                    mean,
                    {100 * (1. - alpha)}% CI (lower),
                    {100 * (1. - alpha)}% CI (upper),
                ]

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            kernel=kernel,
            bandwidth=bandwidth,
            action_scaler=action_scaler,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust (DR) (designed for deterministic evaluation policies) for continuous action spaces.

    Bases: :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.DoublyRobust`

    Note
    -------
    DR estimates the policy value via step-wise importance weighting and estimated Q-function :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{J}_{\\mathrm{DR}} (\\pi; \\mathcal{D})
        := \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=0}^{T-1} \\gamma^t \\left(w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)}) (r_t^{(i)} - \\hat{Q}(s_t^{(i)}, a_t^{(i)}))
        + w_{0:t-1}^{(i)} \\delta(\\pi, a_{0:t-1}^{(i)}) \\hat{Q}(s_t^{(i)}, \\pi(s_t^{(i)})) \\right),

    where :math:`w_{0:t} := \\prod_{t'=0}^t (1 / \\pi_0(a_{t'} | s_{t'}))` is the per-decision importance weight.
    :math:`\\delta(\\pi, a_{0:t}) = \\prod_{t'=1}^t K(\\pi(s_{t'}), a_{t'})` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy where :math:`K(\\cdot)` is a kernel function.
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.

    DR corrects distribution shift between the behavior and evaluation policies and has lower variance than PDIS when :math:`\\hat{Q}(\\cdot)` is reasonably accurate and satisfies :math:`0 < \\hat{Q}(\\cdot) < 2 Q(\\cdot)`.
    However, when the importance weight is quite large, it may still suffer from a high variance.

    Parameters
    -------
    estimator_name: str, default="dr"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning." 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning." 2016.

    """

    estimator_name = "dr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value (expected reward under the evaluation policy) estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_continuous(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="step_wise",
        )
        similarity_weight = self._calc_similarity_weight(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            pscore_type="step_wise",
        )
        weight = similarity_weight / behavior_policy_pscore
        weight_prev = np.roll(weight, 1, axis=1)
        weight_prev[:, 0] = 1

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_trajectory, 2)
        )
        state_value_prediction = state_action_value_prediction[:, :, 1]
        state_action_value_prediction = state_action_value_prediction[:, :, 0]

        estimated_trajectory_value = (
            discount[np.newaxis, :]
            * (
                weight * (reward - state_action_value_prediction)
                + weight_prev * state_value_prediction
            )
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        V_hat: float
            Estimated policy value.

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action.shape[0] "
                "== state_action_value_prediction.shape[0]`, but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        return self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
            kernel=kernel,
            bandwidth=bandwidth,
            action_scaler=action_scaler,
        ).mean()

    def estimate_interval(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Name of the method to estimate the confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: dict
            Dictionary storing the estimated mean and upper-lower confidence bounds.

            .. code-block:: python

                key: [
                    mean,
                    {100 * (1. - alpha)}% CI (lower),
                    {100 * (1. - alpha)}% CI (upper),
                ]

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action.shape[0] "
                "== state_action_value_prediction.shape[0]`, but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if action.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
            kernel=kernel,
            bandwidth=bandwidth,
            action_scaler=action_scaler,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SelfNormalizedTIS(TrajectoryWiseImportanceSampling):
    """Self-Normalized Trajectory-wise Importance Sampling (SNTIS) (designed for deterministic evaluation policies) for continuous action spaces.

    Bases: :class:`scope_rl.ope.continuous.TrajectoryWiseImportanceSampling` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.SelfNormalizedTIS`

    Note
    -------
    SNTIS estimates the policy value via self-normalized trajectory-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNTIS}} (\\pi; \\mathcal{D})
        := \\sum_{i=1}^n \\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{0:T-1}^{(i)} \\delta(\\pi, a_{0:T-1}^{(i)})}{\\sum_{i'=1}^n w_{1:T-1}^{(i')} \\delta(\\pi, a_{0:T-1}^{(i')})} r_t^{(i)},

    where :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (1 / \\pi_0(a_t | s_t))` is the trajectory-wise importance weight.
    :math:`\\delta(\\pi, a_{0:T}) = \\prod_{t=0}^{T-1} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy where :math:`K(\\cdot)` is a kernel function.
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.

    The self-normalized estimator has variance bounded by :math:`r_{max}^2`.

    Parameters
    -------
    estimator_name: str, default="sntis"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning." 2019.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name: str = "sntis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value (expected reward under the evaluation policy) estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_continuous(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="trajectory_wise",
        )
        similarity_weight = self._calc_similarity_weight(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            pscore_type="trajectory_wise",
        )
        weight = similarity_weight / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class SelfNormalizedPDIS(PerDecisionImportanceSampling):
    """Self-Normalized Per-Decision Importance Sampling (SNPDIS) (designed for deterministic evaluation policies) for continuous action spaces.

    Bases: :class:`scope_rl.ope.continuous.PerDecisionImportanceSampling` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.SelfNormalizedPDIS`

    Note
    -------
    SNPDIS estimates the policy value via self-normalized step-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNPDIS}} (\\pi; \\mathcal{D})
        := \\sum_{i=1}^n \\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{1:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)})}{\\sum_{i'=1}^n w_{1:t}^{(i')} \\delta(\\pi, a_{0:t}^{(i')})} r_t^{(i)},

    where :math:`w_{0:t} := \\prod_{t'=1}^t (1 / \\pi_0(a_{t'} | s_{t'}))` is the per-decision importance weight.
    :math:`\\delta(\\pi, a_{0:t}) = \\prod_{t'=1}^t K(\\pi(s_{t'}), a_{t'})` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy where :math:`K(\\cdot)` is a kernel function.
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.

    The self-normalized estimator has variance bounded by :math:`r_{max}^2`.

    Parameters
    -------
    estimator_name: str, default="snpdis"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning." 2019.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name: str = "snpdis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value (expected reward under the evaluation policy) estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_continuous(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="step_wise",
        )
        similarity_weight = self._calc_similarity_weight(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            pscore_type="step_wise",
        )
        weight = similarity_weight / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class SelfNormalizedDR(DoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) (designed for deterministic evaluation policies) for continuous action spaces.

    Bases: :class:`scope_rl.ope.continuous.DoublyRobust` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.SelfNormalizedDR`

    Note
    -------
    SNDR estimates the policy value via self-normalized step-wise importance weighting and estimated Q-function :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNDR}} (\\pi; \\mathcal{D})
        := \\sum_{i=1}^n \\sum_{t=0}^{T-1} \\gamma^t \\left( \\frac{w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)})}{\\sum_{i'=1}^n w_{0:t}^{(i')} \\delta(\\pi, a_{0:t}^{(i')})} (r_t^{(i)} - \\hat{Q}(s_t^{(i)}, a_t^{(i)}))
        + \\frac{w_{0:t-1}^{(i)} \\delta(\\pi, a_{0:t-1}^{(i)})}{\\sum_{i'=1}^n w_{0:t-1}^{(i')} \\delta(\\pi, a_{0:t-1}^{(i')})} \\hat{Q}(s_t^{(i)}, \\pi(s_t^{(i)})) \\right),

    where :math:`w_{0:t} := \\prod_{t'=0}^t (1 / \\pi_0(a_{t'} | s_{t'}))` is the per-decision importance weight.
    :math:`\\delta(\\pi, a_{0:t}) = \\prod_{t'=1}^t K(\\pi(s_{t'}), a_{t'})` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy where :math:`K(\\cdot)` is a kernel function.
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.

    The self-normalized estimator has variance bounded by :math:`r_{max}^2`.

    Parameters
    -------
    estimator_name: str, default="sndr"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning." 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning." 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning." 2016.

    """

    estimator_name = "sndr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value (expected reward under the evaluation policy) estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_continuous(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="step_wise",
        )
        similarity_weight = self._calc_similarity_weight(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            pscore_type="step_wise",
        )
        weight = similarity_weight / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)
        self_normalized_weight_prev = np.roll(self_normalized_weight, 1, axis=1)
        self_normalized_weight_prev[:, 0] = 1

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_trajectory, 2)
        )
        state_value_prediction = state_action_value_prediction[:, :, 1]
        state_action_value_prediction = state_action_value_prediction[:, :, 0]

        estimated_trajectory_value = (
            discount[np.newaxis, :]
            * (
                self_normalized_weight * (reward - state_action_value_prediction)
                + self_normalized_weight_prev * state_value_prediction
            )
        ).sum(axis=1)

        return estimated_trajectory_value
