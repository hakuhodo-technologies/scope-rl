"""Off-Policy Estimators for discrete action cases."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.utils import check_scalar

from .estimators_base import BaseOffPolicyEstimator
from ..utils import check_array


@dataclass
class DiscreteDirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) for discrete-action OPE.

    Bases: :class:`ofrl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.DiscreteDirectMethod`

    Note
    -------
    DM estimates the policy value using the initial state value as follows.

    .. math::

        \\hat{J}_{\\mathrm{DM}} (\\pi; \\mathcal{D}) 
        := \\mathbb{E}_n [\\mathbb{E}_{a_0 \\sim \\pi(a_0 | s_0)} [\\hat{Q}(s_0, a_0)] ] 
        = \\mathbb{E}_n [\\hat{V}(s_0)],

    where :math:`\\mathcal{D}=\\{\\{(s_t, a_t, r_t)\\}_{t=0}^{T-1}\\}_{i=1}^n` is the logged dataset with :math:`n` trajectories of data.
    :math:`T` indicates step per episode. :math:`\\hat{Q}(s_t, a_t)` is the estimated Q value given a state-action pair.
    :math:`\\hat{V}(s_t)` is the estimated value function given a state.

    DM has low variance, but can incur bias due to approximation errors.

    There are several ways to estimate :math:`\\hat{Q}(s, a)` such as Fitted Q Evaluation (FQE) (Le et al., 2019) and
    Minimax Q-Function Learning (MQL) (Uehara et al., 2020). 

    .. seealso::

        The implementation of FQE is provided by `d3rlpy <https://d3rlpy.readthedocs.io/en/latest/references/off_policy_evaluation.html>`_.
        The implementations of Minimax Learning is available at :class:`ofrl.ope.weight_value_learning`.

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

    estimator_name: str = "dm"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value estimated for each trajectory.

        """
        state_value = (
            (state_action_value_prediction * evaluation_policy_action_dist)
            .sum(axis=1)
            .reshape((-1, step_per_trajectory))
        )
        return state_value[:, 0]

    def estimate_policy_value(
        self,
        step_per_trajectory: int,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        Return
        -------
        V_hat: ndarray of shape (n_trajectories, )
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
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        if (
            state_action_value_prediction.shape[0]
            != evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `state_action_value_prediction.shape[0] == evaluation_policy_action_dist.shape[0]`"
                ", but found False"
            )
        if (
            state_action_value_prediction.shape[1]
            != evaluation_policy_action_dist.shape[1]
        ):
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == evaluation_policy_action_dist.shape[1]`"
                ", but found False"
            )
        if state_action_value_prediction.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if not np.allclose(
            evaluation_policy_action_dist.sum(axis=1),
            np.ones(evaluation_policy_action_dist.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action_dist.sum(axis=1) == np.ones(evaluation_policy_action_dist.shape[0])`"
                ", but found it False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_trajectory: int,
        evaluation_policy_action_dist: np.ndarray,
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

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence interval.

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
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        if (
            state_action_value_prediction.shape[0]
            != evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `state_action_value_prediction.shape[0] == evaluation_policy_action_dist.shape[0]`"
                ", but found False"
            )
        if (
            state_action_value_prediction.shape[1]
            != evaluation_policy_action_dist.shape[1]
        ):
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == evaluation_policy_action_dist.shape[1]`"
                ", but found False"
            )
        if state_action_value_prediction.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[0] \\% step_per_trajectory == 0`, but found False"
            )
        if not np.allclose(
            evaluation_policy_action_dist.sum(axis=1),
            np.ones(evaluation_policy_action_dist.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action_dist.sum(axis=1) == np.ones(evaluation_policy_action_dist.shape[0])`"
                ", but found it False"
            )
        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DiscreteTrajectoryWiseImportanceSampling(BaseOffPolicyEstimator):
    """Trajectory-wise Important Sampling (TIS) for discrete-action OPE.

    Bases: :class:`ofrl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.DiscreteTrajectoryWiseImportanceSampling`

    Note
    -------
    TIS estimates the policy value via trajectory-wise importance weight as follows.

    .. math::

        \\hat{J}_{\\mathrm{TIS}} (\\pi; \\mathcal{D}) := \\mathbb{E}_{n} \\left[\\sum_{t=0}^{T-1} \\gamma^t w_{0:T-1} r_t \\right],

    where :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` is the trajectory-wise importance weight.

    TIS enables an unbiased estimation of the policy value. However, when the trajectory length (:math:`T`) is large, 
    TIS suffers from high variance due to the product of importance weights.

    Parameters
    -------
    estimator_name: str, default="tis"
        Name of the estimator.

    References
    -------
    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name = "tis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="trajectory_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="trajectory_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
        estimated_trajectory_value = (discount[np.newaxis, :] * weight * reward).sum(
            axis=1
        )

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        V_hat: ndarray of shape (n_trajectories, )
            Estimated policy value.

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action_dist.shape[0]`"
                ", but found False"
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
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        gamma: float = 1.0,
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

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence interval.

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
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action_dist.shape[0]`"
                ", but found False"
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
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DiscretePerDecisionImportanceSampling(BaseOffPolicyEstimator):
    """Per-Decision Importance Sampling (PDIS) for discrete-action OPE.

    Bases: :class:`ofrl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.DiscretePerDecisionImportanceSampling`

    Note
    -------
    PDIS estimates the policy value via step-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{PDIS}} (\\pi; \\mathcal{D}) := \\mathbb{E}_{n} \\left[\\sum_{t=0}^{T-1} \\gamma^t w_{0:t} r_t \\right],

    where :math:`w_{0:t} := \\prod_{t'=0}^t (\\pi(a_{t'} | s_{t'}) / \\pi_0(a_{t'} | s_{t'}))` is the importance weight of past interactions 
    (referred to as per-decision importance weight).

    By using per-decision importance weight instead of the trajectory-wise importance weight, PDIS reduces the variance of TIS while remaining unbiased. 
    However, when the length of a trajectory (:math:`T`) is large, PDIS still suffers from high variance.

    Parameters
    -------
    estimator_name: str, default="pdis"
        Name of the estimator.

    References
    -------
    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name = "pdis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="step_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="step_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
        estimated_trajectory_value = (discount[np.newaxis, :] * weight * reward).sum(
            axis=1
        )

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        V_hat: ndarray of shape (n_trajectories, )
            Estimated policy value.

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action_dist.shape[0]`"
                ", but found False"
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
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        return self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        gamma: float = 1.0,
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

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence interval.

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
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action_dist.shape[0]`"
                ", but found False"
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
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DiscreteDoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust (DR) for discrete-action OPE.

    Bases: :class:`ofrl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.DiscreteDoublyRobust`

    Note
    -------
    DR estimates the policy value via step-wise importance weighting and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{J}_{\\mathrm{DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} \\left[\\sum_{t=0}^{T-1} \\gamma^t (w_{0:t} (r_t - \\hat{Q}(s_t, a_t)) + w_{0:t-1} \\mathbb{E}_{a \\sim \\pi(a | s_t)}[\\hat{Q}(s_t, a)]) \\right],

    where :math:`w_{0:t} := \\prod_{t'=0}^t (\\pi(a_{t'} | s_{t'}) / \\pi_0(a_{t'} | s_{t'}))` is the per-decision importance weight.

    DR is unbiased and reduces the variance of PDIS when :math:`\\hat{Q}(\\cdot)` is reasonably accurate to satisfy :math:`0 < \\hat{Q}(\\cdot) < 2 Q(\\cdot)`. 
    However, when the importance weight is quite large, it may still suffer from a high variance.

    Parameters
    -------
    estimator_name: str, default="dr"
        Name of the estimator.

    References
    -------
    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning." 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning." 2016.

    """

    estimator_name = "dr"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="step_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="step_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore
        weight_prev = np.roll(weight, 1, axis=1)
        weight_prev[:, 0] = 1

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        state_value_prediction = (
            (state_action_value_prediction * evaluation_policy_action_dist)
            .sum(axis=1)
            .reshape((-1, step_per_trajectory))
        )

        state_action_value_prediction = state_action_value_prediction[
            np.arange(len(action)), action
        ].reshape((-1, step_per_trajectory))

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
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        V_hat: ndarray of shape (n_trajectories, )
            Estimated policy value.

        """
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
        check_array(
            pscore,
            name="pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == state_action_value_prediction.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] "
                "== state_action_value_prediction.shape[0] == evaluation_policy_action_dist.shape[0]`"
                ", but found False"
            )
        if (
            state_action_value_prediction.shape[1]
            != evaluation_policy_action_dist.shape[1]
        ):
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == evaluation_policy_action_dist.shape[1]`"
                ", but found False"
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
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        return self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
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

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence interval.

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
            reward,
            name="reward",
            expected_dim=1,
        )
        check_array(
            pscore,
            name="pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == state_action_value_prediction.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0] "
                "== state_action_value_prediction.shape[0] == evaluation_policy_action_dist.shape[0]`"
                ", but found False"
            )
        if (
            state_action_value_prediction.shape[1]
            != evaluation_policy_action_dist.shape[1]
        ):
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == evaluation_policy_action_dist.shape[1]`"
                ", but found False"
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
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DiscreteSelfNormalizedTrajectoryWiseImportanceSampling(
    DiscreteTrajectoryWiseImportanceSampling
):
    """Self-Normalized Trajectory-wise Important Sampling (SNTIS) for discrete-action OPE.

    Bases: :class:`ofrl.ope.DiscreteTrajectoryWiseImportanceSampling` -> :class:`ofrl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.DiscreteSelfNormalizedTrajectoryWiseImportanceSampling`

    Note
    -------
    SNTIS estimates the policy value via self-normalized trajectory-wise importance weight as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNTIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} \\left[\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{0:T-1}}{\\sum_{n} w_{0:T-1}} r_t \\right],

    where :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` is the trajectory-wise importance weight.

    The self-normalized estimator is no longer unbiased, but has variance bounded by :math:`r_{max}^2` while also being consistent.

    Parameters
    -------
    estimator_name: str, default="sntis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation." 2021.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning." 2019.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name = "sntis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="trajectory_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="trajectory_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class DiscreteSelfNormalizedPerDecisionImportanceSampling(
    DiscretePerDecisionImportanceSampling
):
    """Self-Normalized Per-Decision Importance Sampling (SNPDIS) for discrete-action OPE.

    Bases: :class:`ofrl.ope.DiscretePerDecisionImportanceSampling` -> :class:`ofrl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.DiscreteSelfNormalizedPerDecisionImportanceSampling`

    Note
    -------
    SNPDIS estimates the policy value via self-normalized step-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNPDIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} \\left[\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{1:t}}{\\sum_{n} w_{1:t}} r_t \\right],

    where :math:`w_{0:t} := \\prod_{t'=1}^t (\\pi(a_{t'} | s_{t'}) / \\pi_0(a_{t'} | s_{t'}))` is the per-decision importance weight.

    The self-normalized estimator is no longer unbiased, but has variance bounded by :math:`r_{max}^2` while also being consistent.

    Parameters
    -------
    estimator_name: str, default="snpdis"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning." 2019.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name = "snpdis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="step_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="step_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class DiscreteSelfNormalizedDoublyRobust(DiscreteDoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) for discrete-action OPE.

    Bases: :class:`ofrl.ope.DiscreteDoublyRobust` -> :class:`ofrl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`ofrl.ope.DiscreteSelfNormalizedDoublyRobust`

    Note
    -------
    SNDR estimates the policy value via self-normalized step-wise importance weighting and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNDR}} (\\pi; \\mathcal{D})
        &:= \\mathbb{E}_{n} \\Biggl[\\sum_{t=0}^{T-1} \\gamma^t \\biggl( \\frac{w_{0:t}}{\\sum_{n} w_{0:t}} (r_t - \\hat{Q}(s_t, a_t)) \\\\
        & \quad \quad \quad \quad \quad + \\frac{w_{0:t-1}}{\\sum_{n} w_{0:t-1}} \\mathbb{E}_{a \\sim \\pi(a | s_t)}[\\hat{Q}(s_t, a)] \\biggr) \\Biggr],

    where :math:`w_{0:t} := \\prod_{t'=0}^t (\\pi(a_{t'} | s_{t'}) / \\pi_0(a_{t'} | s_{t'}))` is the per-decision importance weight.

    The self-normalized estimator is no longer unbiased, but has variance bounded by :math:`r_{max}^2` while also being consistent.

    Parameters
    -------
    estimator_name: str, default="sndr"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning." 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning." 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning." 2016.

    """

    estimator_name = "sndr"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observation.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a | s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a | s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
            pscore_type="step_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="step_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)
        self_normalized_weight_prev = np.roll(self_normalized_weight, 1, axis=1)
        self_normalized_weight_prev[:, 0] = 1

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        state_value_prediction = (
            (state_action_value_prediction * evaluation_policy_action_dist)
            .sum(axis=1)
            .reshape((-1, step_per_trajectory))
        )

        state_action_value_prediction = state_action_value_prediction[
            np.arange(len(action)), action
        ].reshape((-1, step_per_trajectory))

        estimated_trajectory_value = (
            discount[np.newaxis, :]
            * (
                self_normalized_weight * (reward - state_action_value_prediction)
                + self_normalized_weight_prev * state_value_prediction
            )
        ).sum(axis=1)

        return estimated_trajectory_value