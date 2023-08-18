# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""State(-Action) Marginal Off-Policy Estimators for continuous action cases (designed for deterministic evaluation policies)."""
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from sklearn.utils import check_scalar

from d3rlpy.preprocessing import ActionScaler

from ..estimators_base import (
    BaseOffPolicyEstimator,
    BaseStateMarginalOPEEstimator,
    BaseStateActionMarginalOPEEstimator,
)
from ...utils import check_array


@dataclass
class DoubleReinforcementLearning(BaseOffPolicyEstimator):
    """Double Reinforcement Learning (DRL) estimator for continuous action space.

    Bases: :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.DoubleReinforcementLearning`

    Note
    -------
    DRL estimates the policy value using state-action marginal importance weight and Q-function estimated by cross-fitting.

    .. math::

        \\hat{J}_{\\mathrm{DRL}} (\\pi; \\mathcal{D})
        := \\frac{1}{n} \\sum_{k=1}^K \\sum_{i=1}^{n_k} \\sum_{t=0}^{T-1} ( \\rho^j(s_{t}^{(i)}, a_{t}^{(i)}) (r_{t}^{(i)} - Q^j(s_{t}^{(i)}, a_{t}^{(i)}))
        + \\rho^j(s_{t-1}^{(i)}, a_{t-1}^{(i)}) Q^j(s_{t}^{(i)}, \\pi(s_{t}^{(i)})) )

    where :math:`\\rho(s, a) \\approx d^{\\pi}(s, a) / d^{\\pi_b}(s, a)` is the state-action marginal importance weight,
    where :math:`d^{\\pi}(s)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`(s, a)`.
    :math:`Q(s, a)` is the Q-function.
    :math:`K` is the number of folds and :math:`\\mathcal{D}_j` is the :math:`j`-th split of logged data consisting of :math:`n_k` samples.
    :math:`\\rho^j` and :math:`Q^j` are estimated on the subset of data used for OPE, i.e., :math:`\\mathcal{D} \\setminus \\mathcal{D}_j`.

    DRL achieves the semiparametric efficiency bound with a consistent value predictor.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="drl"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Masatoshi Uehara.
    "Double Reinforcement Learning for Efficient Off-Policy Evaluation in Markov Decision Processes." 2020.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    """

    estimator_name: str = "drl"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        state_action_value_prediction: np.ndarray,
    ):
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

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
        state_value_prediction = state_action_value_prediction[:, :, 1]
        state_action_value_prediction = state_action_value_prediction[:, :, 0]

        reward = reward.reshape((-1, step_per_trajectory))

        weight = state_action_marginal_importance_weight.reshape(
            (-1, step_per_trajectory)
        )
        weight_prev = np.roll(weight, 1, axis=1)
        weight_prev[:, 0] = 1

        estimated_trajectory_value = (
            weight * (reward - state_action_value_prediction)
            + weight_prev * state_value_prediction
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        state_action_value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

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
            state_action_marginal_importance_weight,
            name="state_action_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        if not (
            reward.shape[0]
            == state_action_marginal_importance_weight.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == state_action_marginal_importance_weight.shape[0] == state_action_value_prediction.shape[0]`"
                ", but found False"
            )
        if state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_trajectory=step_per_trajectory,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            state_action_value_prediction=state_action_value_prediction,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_trajectory: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
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

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

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
            state_action_marginal_importance_weight,
            name="state_action_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            state_action_value_prediction,
            name="state_action_value_prediction",
            expected_dim=2,
        )
        if not (
            reward.shape[0]
            == state_action_marginal_importance_weight.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == state_action_marginal_importance_weight.shape[0] == state_action_value_prediction.shape[0]`"
                ", but found False"
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
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            state_action_value_prediction=state_action_value_prediction,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class StateMarginalDM(BaseStateMarginalOPEEstimator):
    """Direct Method (DM) for continuous-action and stationary OPE (designed for deterministic evaluation policies).

    Bases: :class:`scope_rl.ope.BaseStateMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateMarginalDM`

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

    There are several methods to estimate :math:`\\hat{Q}(s, a)` such as Fitted Q Evaluation (FQE) (Le et al., 2019),
    Minimax Q-Function Learning (MQL) (Uehara et al., 2020), and Augmented Lagrangian Method (ALM) (Yang et al., 2020).

    .. seealso::

        The implementation of FQE is provided by `d3rlpy <https://d3rlpy.readthedocs.io/en/latest/references/off_policy_evaluation.html>`_.
        The implementations of Minimax Weight and Value Learning (including ALM) is available at :class:`scope_rl.ope.weight_value_learning`.

    Note
    -------
    This function is different from :class:`DirectMethod` in that
    the initial state is sampled from the stationary distribution :math:`d^{\pi}(s_0)`.

    Parameters
    -------
    estimator_name: str, default="sm_dm"
        Name of the estimator.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints." 2019.

    """

    estimator_name = "sm_dm"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        initial_state_value_prediction: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_trajectories, )
            Policy value (expected reward under the evaluation policy) estimated for each trajectory.
            (Equivalent to initial_state_value_prediction.)

        """
        return initial_state_value_prediction

    def estimate_policy_value(
        self,
        initial_state_value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

        Return
        -------
        V_hat: float
            Estimated policy value.

        """
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        estimated_policy_value = self._estimate_trajectory_value(
            initial_state_value_prediction,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        initial_state_value_prediction: np.ndarray,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`

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
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            initial_state_value_prediction,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class StateMarginalIS(BaseStateMarginalOPEEstimator):
    """State Marginal Importance Sampling (SM-IS) for continuous action spaces.

    Bases: :class:`scope_rl.ope.BaseStateMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateMarginalIS`

    Note
    -------
    SM-IS estimates the policy value using state marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we combine State Marginal Importance Sampling and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-IS}} (\\pi; \\mathcal{D})
        := \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=0}^{k-1} \\gamma^t w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)}) r_t^{(i)}
        + \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=k}^{T-1} \\gamma^t \\rho(s_{t-k}^{(i)}) w_{t-k:t}^{(i)} \\delta(\\pi, a_{t-k:t}^{(i)}) r_t^{(i)},

    where :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` and :math:`\\rho(s) \\approx d^{\\pi}(s) / d^{\\pi_b}(s)` is the state-marginal importance weight,
    where :math:`d^{\\pi}(s)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`s`.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.
    Additionally, when :math:`k=0`, this estimator is reduced to the vanilla state marginal IS.


    SM-IS corrects distribution shift between the behavior and evaluation policies.
    Moreover, SM-IS reduces the variance caused by trajectory-wise or per-decision importance weighting by considering the marginal distribution across various timesteps.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sm_is"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name: str = "sm_is"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
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
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d^{\\pi}(s) / d^{\\pi_b}(s)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        state_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            state_marginal_importance_weight=state_marginal_importance_weight,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        estimated_trajectory_value = (discount[np.newaxis, :] * weight * reward).sum(
            axis=1
        )

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
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
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d^{\\pi}(s) / d^{\\pi_b}(s)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
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
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
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
            state_marginal_importance_weight,
            name="state_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
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
            action,
            name="action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_marginal_importance_weight.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d^{\\pi}(s) / d^{\\pi_b}(s)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
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
            Method to estimate the confidence interval.

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
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
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
            state_marginal_importance_weight,
            name="state_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
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
            action,
            name="action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_marginal_importance_weight.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class StateMarginalDR(BaseStateMarginalOPEEstimator):
    """State Marginal Doubly Robust (SM-DR) for continuous action spaces.

    Bases: :class:`scope_rl.ope.BaseStateMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateMarginalDR`

    Note
    -------
    SM-DR estimates the policy value using state marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we combine state-marginal importance weighting and :math:`k`-step DR as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-DR}} (\\pi; \\mathcal{D})
        &:= \\frac{1}{n} \\sum_{i=1}^n \\hat{Q}(s_0^{(i)}, \\pi(s_0^{(i)})) \\\\
        & \quad \quad + \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=0}^{k-1} \\gamma^t w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)}) \\left(r_t^{(i)} + \\gamma \\hat{Q}(s_{t+1}^{(i)}, \\pi(s_{t+1}^{(i)})) - \\hat{Q}(s_t^{(i)}, a_t^{(i)}) \\right) \\\\
        & \quad \quad + \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=k}^{T-1} \\gamma^t \\rho(s_{t-k}^{(i)}) w_{t-k:t}^{(i)} \\delta(\\pi, a_{t-k:t}^{(i)}) \\left(r_t^{(i)} + \\gamma \\hat{Q}(s_{t+1}^{(i)}, \\pi(s_{t+1}^{(i)})) - \\hat{Q}(s_t^{(i)}, a_t^{(i)}) \\right),

    where :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` and :math:`\\rho(s) \\approx d^{\\pi}(s) / d^{\\pi_b}(s)` is the state-marginal importance weight,
    where :math:`d^{\\pi}(s)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`s`.
    :math:`Q(s, a)` is the state-action value.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.
    Additionally, when :math:`k=0`, this estimator is reduced to the vanilla state marginal DR.

    SM-DR corrects the distribution shift between the behavior and evaluation policies.
    Moreover, SM-DR reduces the variance caused by the trajectory-wise or per-decision importance weight by considering the marginal distribution across various timesteps.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sm_dr"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning." 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning." 2016.

    """

    estimator_name: str = "sm_dr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d^{\\pi}(s) / d^{\\pi_b}(s)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        state_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            state_marginal_importance_weight=state_marginal_importance_weight,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        ).reshape((-1, step_per_trajectory))

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_trajectory, 2)
        )
        state_value_prediction = state_action_value_prediction[:, :, 1]
        state_action_value_prediction = state_action_value_prediction[:, :, 0]

        estimated_trajectory_value = initial_state_value_prediction + (
            discount[np.newaxis, :]
            * weight
            * (reward + gamma * state_value_prediction - state_action_value_prediction)
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d^{\\pi}(s) / d^{\\pi_b}(s)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

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
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
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
            state_marginal_importance_weight,
            name="state_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
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
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_marginal_importance_weight.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0]`== state_action_value_prediction.shape[0]"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
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

        estimated_policy_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            initial_state_value_prediction=initial_state_value_prediction,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d^{\\pi}(s) / d^{\\pi_b}(s)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

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
            Method to estimate the confidence interval.

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
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
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
            state_marginal_importance_weight,
            name="state_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
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
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_marginal_importance_weight.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0] == state_action_value_prediction.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            initial_state_value_prediction=initial_state_value_prediction,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class StateMarginalSNIS(StateMarginalIS):
    """State Marginal Self-Normalized Importance Sampling (SM-SNIS) for continuous action spaces.

    Bases: :class:`scope_rl.ope.continuous.StateMarginalIS` :class:`scope_rl.ope.BaseStateMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateMarginalSNIS`

    Note
    -------
    SM-SNIS estimates the policy value using state marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we combine state-marginal importance weighting and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-SNIS}} (\\pi; \\mathcal{D})
        &:= \\sum_{i=1}^n \\sum_{t=0}^{k-1} \\gamma^t \\frac{w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)})}{\\sum_{i'=1}^n w_{0:t}^{(i')} \\delta(\\pi, a_{0:t}^{(i')})} r_t^{(i)} \\\\
        & \quad \quad + \\sum_{i=1}^n \\sum_{t=k}^{T-1} \\gamma^t \\frac{\\rho(s_{t-k}^{(i)}) w_{t-k:t}^{(i)} \\delta(\\pi, a_{t-k:t}^{(i+1)})}{\\sum_{i'=1}^n \\rho(s_{t-k}^{(i')}) w_{t-k:t}^{(i')} \\delta(\\pi, a_{t-k:t}^{(i')})} r_t^{(i)},

    where :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` and :math:`\\rho(s) \\approx d^{\\pi}(s) / d^{\\pi_b}(s)` is the state-marginal importance weight,
    where :math:`d^{\\pi}(s)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`s`.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.
    Additionally, when :math:`k=0`, this estimator is reduced to the vanilla state marginal SNIS.

    SM-SNIS corrects the distribution shift between the behavior and evaluation policies.
    Moreover, SM-SNIS reduces the variance caused by trajectory-wise or per-decision importance weighting by considering the marginal distribution across various timesteps.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sm_snis"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name: str = "sm_snis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
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
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d^{\\pi}(s) / d^{\\pi_b}(s)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        state_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            state_marginal_importance_weight=state_marginal_importance_weight,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class StateMarginalSNDR(StateMarginalDR):
    """State Marginal Self-Normalized Doubly Robust (SM-SNDR) for continuous action spaces.

    Bases: :class:`scope_rl.continuous.StateMarginalDR` :class:`scope_rl.BaseStateMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateMarginalSNDR`

    Note
    -------
    SM-SNDR estimates the policy value using state marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we combine state-marginal importance weighting and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-SNDR}} (\\pi; \\mathcal{D})
        &:= \\frac{1}{n} \\sum_{i=1}^n \\hat{Q}(s_0^{(i)}, \\pi(s_0^{(i)}) \\\\
        & \quad \quad + \\sum_{i=1}^n \\sum_{t=0}^{k-1} \\gamma^t \\frac{w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)})}{\\sum_{i'=1}^n w_{0:t}^{(i')} \\delta(\\pi, a_{0:t}^{(i')})} \\left(r_t^{(i)} + \\gamma \\hat{Q}(s_{t+1}^{(i)}, \\pi(s_{t+1}^{(i)}))) - \\hat{Q}(s_t^{(i)}, a_t^{(i)}) \\right) \\\\
        & \quad \quad + \\sum_{i=1}^n \\sum_{t=k}^{T-1} \\gamma^t \\frac{\\rho(s_{t-k}^{(i)}) w_{t-k:t}^{(i)} \\delta(\\pi, a_{t-k:t}^{(i)})}{\\sum_{i'=1}^n \\rho(s_{t-k}^{(i')}) w_{t-k:t}^{(i')} \\delta(\\pi, a_{t-k:t}^{(i')})} \\left(r_t^{(i)} + \\gamma \\hat{Q}(s_{t+1}^{(i)}, \\pi(s_{t+1}^{(i)})) - \\hat{Q}(s_t^{(i)}, a_t^{(i)}) \\right),

    where :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` and :math:`\\rho(s) \\approx d^{\\pi}(s) / d^{\\pi_b}(s)` is the state-marginal importance weight,
    where :math:`d^{\\pi}(s)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`s`.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.
    Additionally, when :math:`k=0`, this estimator is reduced to the vanilla state marginal SNDR.

    SM-SNDR corrects the distribution shift between the behavior and evaluation policies.
    Moreover, SM-SNDR reduces the variance caused by trajectory-wise or per-decision importance weighting by considering the marginal distribution across various timesteps.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sm_sndr"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning." 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning." 2016.

    """

    estimator_name: str = "sm_sndr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d^{\\pi}(s) / d^{\\pi_b}(s)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        state_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            state_marginal_importance_weight=state_marginal_importance_weight,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_trajectory, 2)
        )
        state_value_prediction = state_action_value_prediction[:, :, 1]
        state_action_value_prediction = state_action_value_prediction[:, :, 0]

        estimated_trajectory_value = initial_state_value_prediction + (
            discount[np.newaxis, :]
            * self_normalized_weight
            * (reward + gamma * state_value_prediction - state_action_value_prediction)
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class StateActionMarginalIS(BaseStateActionMarginalOPEEstimator):
    """State-Action Marginal Importance Sampling (SAM-IS) for continuous action spaces.

    Bases: :class:`scope_rl.ope.BaseStateActionMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateActionMarginalIS`

    Note
    -------
    SAM-IS estimates the policy value using state-action marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we combine state-marginal importance weighting and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-IS}} (\\pi; \\mathcal{D})
        &:= \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=0}^{k-1} \\gamma^t w_{0:t}^{(i)} \\delta(\\pi, a_{t_1:t_2}^{(i)}) r_t^{(i)} \\\\
        & \quad \quad + \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=k}^{T-1} \\gamma^t \\rho(s_{t-k}^{(i)}, a_{t-k}^{(i)}) w_{t-k+1:t}^{(i)} \\delta(\\pi, a_{t_1:t_2}^{(i)}) r_t^{(i)},

    where :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` and :math:`\\rho(s, a) \\approx d^{\\pi}(s, a) / d^{\\pi_b}(s, a)` is the state-marginal importance weight,
    where :math:`d^{\\pi}(s, a)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`(s, a)`.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.
    Additionally, when :math:`k=0`, this estimator is reduced to the vanilla state-action marginal IS.

    SAM-IS corrects the distribution shift between the behavior and evaluation policies.
    Moreover, SAM-IS reduces the variance caused by trajectory-wise or per-decision importance weighting by considering the marginal distribution across various timesteps.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sam_is"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name: str = "sam_is"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
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
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        state_action_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
        )
        weight = state_action_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
        estimated_trajectory_value = (discount[np.newaxis, :] * weight * reward).sum(
            axis=1
        )

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
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
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
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
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
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
            state_action_marginal_importance_weight,
            name="state_action_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
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
            action,
            name="action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_action_marginal_importance_weight.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if action_scaler is not None and not isinstance(action_scaler, ActionScaler):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
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
            Method to estimate the confidence interval.

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
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
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
            state_action_marginal_importance_weight,
            name="state_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
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
            action,
            name="action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_action_marginal_importance_weight.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class StateActionMarginalDR(BaseStateActionMarginalOPEEstimator):
    """State-Action Marginal Doubly Robust (SAM-DR) for continuous action spaces.

    Bases: :class:`scope_rl.ope.BaseStateActionMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateActionMarginalDR`

    Note
    -------
    SAM-DR estimates the policy value using state-action marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we combine state-marginal importance weighting and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-DR}} (\\pi; \\mathcal{D})
        &:= \\frac{1}{n} \\sum_{i=1}^n \\hat{Q}(s_0^{(i)}, \\pi(s_0^{(i)})) \\\\
        & \quad \quad + \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=0}^{k-1} \\gamma^t w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)}) \\left( r_t^{(i)} + \\gamma \\hat{Q}(s_{t+1}^{(i)}, \\pi(s_{t+1}^{(i)})) - \\hat{Q}(s_t^{(i)}, a_t^{(i)}) \\right) \\\\
        & \quad \quad + \\frac{1}{n} \\sum_{i=1}^n \\sum_{t=k}^{T-1} \\gamma^t \\rho(s_{t-k}^{(i)}, a_{t-k}^{(i)}) w_{t-k+1:t}^{(i)} \\delta(\\pi, a_{t-k+1:t}^{(i)}) \\left( r_t^{(i)} + \\gamma \\hat{Q}(s_{t+1}^{(i)}, \\pi(s_{t+1}^{(i)})) - \\hat{Q}(s_t^{(i)}, a_t^{(i)}) \\right),

    where :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` and :math:`\\rho(s, a) \\approx d^{\\pi}(s, a) / d^{\\pi_b}(s, a)` is the state-marginal importance weight,
    where :math:`d^{\\pi}(s, a)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`(s, a)`.
    :math:`Q(s, a)` is the state-action value.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.
    Additionally, when :math:`k=0`, this estimator is reduced to the vanilla state-action marginal DR.

    SAM-DR corrects the distribution shift between the behavior and evaluation policies.
    Moreover, SAM-DR reduces the variance caused by trajectory-wise or per-decision importance weighting by considering the marginal distribution across various timesteps.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sam_dr"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning." 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning." 2016.

    """

    estimator_name: str = "sam_dr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        state_action_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
        )
        weight = state_action_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_trajectory, 2)
        )
        state_value_prediction = state_action_value_prediction[:, :, 1]
        state_action_value_prediction = state_action_value_prediction[:, :, 0]

        estimated_trajectory_value = initial_state_value_prediction + (
            discount[np.newaxis, :]
            * weight
            * (reward + gamma * state_value_prediction - state_action_value_prediction)
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

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
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
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
            state_action_marginal_importance_weight,
            name="state_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
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
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_action_marginal_importance_weight.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0] == state_action_value_prediction.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
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

        estimated_policy_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            initial_state_value_prediction=initial_state_value_prediction,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

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
            Method to estimate the confidence interval.

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
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
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
            state_action_marginal_importance_weight,
            name="state_marginal_importance_weight",
            expected_dim=1,
            min_val=0.0,
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
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_action_marginal_importance_weight.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
            == state_action_value_prediction.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0] == state_action_value_prediction.shape[0]`"
                ", but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            reward=reward,
            pscore=pscore,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            initial_state_value_prediction=initial_state_value_prediction,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class StateActionMarginalSNIS(StateActionMarginalIS):
    """State-Action Marginal Self-Normalized Importance Sampling (SAM-SNIS) for continuous action spaces.

    Bases: :class:`scope_rl.ope.continuous.StateActionMarginalIS` :class:`scope_rl.ope.BaseStateActionMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateActionMarginalSNIS`

    Note
    -------
    SAM-SNIS estimates the policy value using state-action marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we combine state-marginal importance weighting and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-SNIS}} (\\pi; \\mathcal{D})
        &:= \\sum_{i=1}^n \\sum_{t=0}^{k-1} \\gamma^t \\frac{w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)})}{\\sum_{i'=1} w_{0:t}^{(i')} \\delta(\\pi, a_{0:t}^{(i')})} r_t^{(i)} \\\\
        & \quad \quad + \\sum_{i=1}^n \\sum_{t=k}^{T-1} \\gamma^t \\frac{\\rho(s_{t-k}^{(i)}, a_{t-k}^{(i)}) w_{t-k+1:t}^{(i)} \\delta(\\pi, a_{t-l+1:t}^{(i)})}{\\sum_{i'=1}^n \\rho(s_{t-k}^{(i')}, a_{t-k}^{(i')}) w_{t-k+1:t}^{(i')} \\delta(\\pi, a_{t-l+1:t}^{(i')})} r_t^{(i)},

    where :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` and :math:`\\rho(s, a) \\approx d^{\\pi}(s, a) / d^{\\pi_b}(s, a)` is the state-marginal importance weight,
    where :math:`d^{\\pi}(s, a)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`(s, a)`.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.
    Additionally, when :math:`k=0`, this estimator is reduced to the vanilla state-action marginal SNIS.

    SAM-SNIS corrects the distribution shift between the behavior and evaluation policies.
    Moreover, SAM-SNIS reduces the variance caused by trajectory-wise or per-decision importance weighting by considering the marginal distribution across various timesteps.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sam_snis"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation." 2000.

    """

    estimator_name: str = "sam_snis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
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
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        state_action_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
        )
        weight = state_action_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class StateActionMarginalSNDR(StateActionMarginalDR):
    """State-Action Marginal Self-Normalized Doubly Robust (SAM-SNDR) for continuous action spaces.

    Bases: :class:`scope_rl.ope.continuous.StateActionMarginalDR` :class:`scope_rl.ope.BaseStateActionMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.continuous.StateActionMarginalSNDR`

    Note
    -------
    SAM-SNDR estimates the policy value using state-action marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we combine state-marginal importance weighting and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-SNDR}} (\\pi; \\mathcal{D})
        &:= \\frac{1}{n} \\sum_{i=1}^n \\hat{Q}(s_0^{(i)}, \\pi(s_0^{(i)})) \\\\
        & \quad \quad + \\sum_{i=1}^n \\sum_{t=0}^{k-1} \\gamma^t \\frac{w_{0:t}^{(i)} \\delta(\\pi, a_{0:t}^{(i)})}{\\sum_{i'=1}^n w_{0:t}^{(i')} \\delta(\\pi, a_{0:t}^{(i')})} (r_t^{(i)} + \\gamma \\hat{Q}(s_{t+1}^{(i)}, \\pi(s_{t+1}^{(i)})) - \\hat{Q}(s_t^{(i)}, a_t^{(i)})) \\\\
        & \quad \quad + \\sum_{i=1}^n \\sum_{t=k}^{T-1} \\gamma^t \\frac{w(s_{t-k}^{(i)}, a_{t-k}^{(i)}) w_{t-k+1:t}^{(i)} \\delta(\\pi, a_{t-k+1:t}^{(i)})}{\\sum_{i'=1}^n w(s_{t-k}^{(i')}, a_{t-k}^{(i')}) w_{t-k+1:t}^{(i')} \\delta(\\pi, a_{t-k+1:t}^{(i')})} (r_t^{(i)} + \\gamma \\hat{Q}(s_{t+1}^{(i)}, \\pi(s_{t+1}^{(i)})) - \\hat{Q}(s_t^{(i)}, a_t^{(i)})),

    where :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} (\\pi(a_t | s_t) / \\pi_0(a_t | s_t))` and :math:`\\rho(s, a) \\approx d^{\\pi}(s, a) / d^{\\pi_b}(s, a)` is the state-marginal importance weight,
    where :math:`d^{\\pi}(s, a)` is the marginal visitation probability of the policy :math:`\\pi` on :math:`(s, a)`.
    :math:`Q(s, a)` is the state-action value.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that the bandwidth of the kernel is an important hyperparameter; the variance of the above estimator often becomes small when the bandwidth of the kernel is large, while the bias often becomes large in those cases.
    Additionally, when :math:`k=0`, this estimator is reduced to the vanilla state-action marginal SNDR.

    SAM-SNDR corrects the distribution shift between the behavior and evaluation policies.
    Moreover, SAM-SNDR reduces the variance caused by trajectory-wise or per-decision importance weighting by considering the marginal distribution across various timesteps.

    There are several ways to estimate the state(-action) marginal importance weight such as Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).

    .. seealso::

        The implementations of such weight learning methods are available at :class:`scope_rl.ope.weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sam_sndr"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators." 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian." 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments." 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning." 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning." 2016.

    """

    estimator_name: str = "sam_sndr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of initial steps whose rewards are estimated by step-wise importance weighting,
            rewards of the later time steps are estimated by state marginal importance weighting.
            When set to zero, the estimator is reduced to the vanilla state marginal IS.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d^{\\pi}(s, a) / d^{\\pi_b}(s, a)`

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_b(a | s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a | s_t))`.

        initial_state_value_prediction: array-like of shape (n_trajectories, )
            Estimated initial state value.

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
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        state_action_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_trajectory=step_per_trajectory,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
        )
        weight = state_action_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_trajectory))
        discount = np.full(step_per_trajectory, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_trajectory, 2)
        )
        state_value_prediction = state_action_value_prediction[:, :, 1]
        state_action_value_prediction = state_action_value_prediction[:, :, 0]

        estimated_trajectory_value = initial_state_value_prediction + (
            discount[np.newaxis, :]
            * self_normalized_weight
            * (reward + gamma * state_value_prediction - state_action_value_prediction)
        ).sum(axis=1)

        return estimated_trajectory_value
