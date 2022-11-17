from dataclasses import dataclass
from typing import Optional, Union, Dict

import numpy as np
from sklearn.utils import check_scalar

from .estimators_base import (
    BaseOffPolicyEstimator,
    BaseStateMarginalOffPolicyEstimator,
    BaseStateActionMarginalOffPolicyEstimator,
)
from ..utils import check_array


@dataclass
class ContinuousDoubleReinforcementLearning(BaseOffPolicyEstimator):
    """Double Reinforcement Learning (DRL) estimator for continuous action space.

    Note
    -------
    DRL estimates the policy value using state-action marginal importance weight and Q function estimated by cross-fitting.

    .. math::

        \\hat{J}_{\\mathrm{DRL}} (\\pi; \\mathcal{D})
        := \\frac{1}{n} \\sum{k=1}^K \\sum_{i=1}^{n_k} \\sum_{t=0}^{T-1} ( w^j(s_{i,t}, a_{i, t}) (r_{i, t} - Q^j(s_{i, t}, a_{i, t}))
            + w^j(s_{i, t-1}, a_{i, t-1}) Q^j(s_{i, t}, \\pi(s_{i, t})) )

    where :math:`w(s, a) \\approx d^{\\pi}(s, a) / d^{\\pi_0}(s, a)` is the state-action marginal importance weight and :math:`Q(s, a)` is the Q function.
    :math:`K` is the number of folds and :math:`\\mathcal{D}_j` is the :math:`j`-th split of logged data consisting of :math:`n_k` samples.
    :math:`w^j` and :math:`Q^j` are trained on the subset of data used for OPE, i.e., :math:`\\mathcal{D} \\setminus \\mathcal{D}_j`.

    There are several ways to estimate state marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="drl"
        Name of the estimator.

    References
    -------
    Nathan Kallus and Masatoshi Uehara.
    "Double Reinforcement Learning for Efficient Off-Policy Evaluation in Markov Decision Processes.", 2020.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    """

    estimator_name: str = "drl"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        state_action_value_prediction: np.ndarray,
    ):
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_episode, 2)
        )
        weight = np.insert(state_action_marginal_importance_weight, 0, 1, axis=1)

        estimated_trajectory_value = (
            weight[:, 1:] * (reward - state_action_value_prediction[:, :, 0])
            + weight[:, :-1] * state_action_value_prediction[:, :, 1]
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        state_action_value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
        if not state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            state_action_value_prediction=state_action_value_prediction,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

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

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
        if not state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
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
class ContinuousStateMarginalImportanceSampling(BaseStateMarginalOffPolicyEstimator):
    """State Marginal Importance Sampling (SM-IS) for continuous-action OPE.

    Note
    -------
    SM-IS estimates the policy value using state marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we consider the combination of marginalized OPE and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-IS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{k-1} \\gamma^t w_{0:t} \\delta(\\pi, a_{0:t}) r_t ] + \\mathbb{E}_{n} [\\sum_{t=k}^{T-1} \\gamma^t w_s(s_{t-k}) w_{t-k:t} \\delta(\\pi, a_{t-k:t}) r_t],

    where :math:`w_s(s) := \\frac{d_{\\pi}(s)}{d_{\\pi_0}(s)}` and :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that, when :math:`k=0`, this estimator is identical to the (pure) state marginal IS.

    There are several ways to estimate state marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sm_is"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation.", 2018

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "sm_is"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
        )
        state_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            state_marginal_importance_weight=state_marginal_importance_weight,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma

        estimated_trajectory_value = (discount[np.newaxis, :] * weight * reward).sum(
            axis=1
        )

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_scalar(
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        sigma: float = 1.0,
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
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

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

        """
        check_scalar(
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousStateMarginalDoublyRobust(BaseStateMarginalOffPolicyEstimator):
    """State Marginal Doubly Robust (SM-DR) for continuous-action OPE.

    Note
    -------
    SM-DR estimates the policy value using state marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we consider the combination of marginalized OPE and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\hat{Q}(s_0, a_0)]
            + \\mathbb{E}_{n} [\\sum_{t=0}^{k-1} \\gamma^t w_{0:t} \\delta(\\pi, a_{0:t}) (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))]
            + \\mathbb{E}_{n} [\\sum_{t=k}^{T-1} \\gamma^t w_s(s_{t-k}) w_{t-k:t} \\delta{\\pi, a_{t-k:t}} (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))],

    where :math:`w_s(s) := \\frac{d_{\\pi}(s)}{d_{\\pi_0}(s)}` and :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`. :math:`Q(s, a)` is the state-action value.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that, when :math:`k=0`, this estimator is identical to the (pure) state marginal DR.

    There are several ways to estimate state marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sm_dr"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation.", 2018

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "sm_dr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
        )
        state_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            state_marginal_importance_weight=state_marginal_importance_weight,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        ).reshape((-1, step_per_episode))

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_episode, 2)
        )
        state_action_value_prediction = np.insert(
            state_action_value_prediction, -1, 0, axis=1
        )
        state_value_prediction = state_action_value_prediction[:, 1:, 1]
        state_action_value_prediction = state_action_value_prediction[:, :-1, 0]

        estimated_trajectory_value = state_value_prediction[:, 0] + (
            discount[np.newaxis, :]
            * weight
            * (reward + gamma * state_value_prediction - state_action_value_prediction)
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_scalar(
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if not state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        sigma: float = 1.0,
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
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

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

        """
        check_scalar(
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if not state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousStateMarginalSelfNormalizedImportanceSampling(
    ContinuousStateMarginalImportanceSampling
):
    """State Marginal Self-Normalized Importance Sampling (SM-SNIS) for continuous-action OPE.

    Note
    -------
    SM-SNIS estimates the policy value using state marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we consider the combination of marginalized OPE and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-SNIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{k-1} \\gamma^t \\frac{w_{0:t} \\delta(\\pi, a_{0:t})}{\\sum_{n} w_{*} \\delta_{*}} r_t ] + \\mathbb{E}_{n} [\\sum_{t=k}^{T-1} \\gamma^t \\frac{w_s(s_{t-k}) w_{t-k:t} \\delta{t-k:t}}{\\sum_n w_{*} \\delta_{*}} r_t],

    where :math:`w_s(s) := \\frac{d_{\\pi}(s)}{d_{\\pi_0}(s)}` and :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`. w_{*} is the abstruction of any weights.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that, when :math:`k=0`, this estimator is identical to the (pure) state marginal SNIS.

    There are several ways to estimate state marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sm_snis"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation.", 2018

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "sm_snis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
        )
        state_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            state_marginal_importance_weight=state_marginal_importance_weight,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma
        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class ContinuousStateMarginalSelfNormalizedDoublyRobust(
    ContinuousStateMarginalDoublyRobust
):
    """State Marginal Self-Normalized Doubly Robust (SM-SNDR) for continuous-action OPE.

    Note
    -------
    SM-SNDR estimates the policy value using state marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we consider the combination of marginalized OPE and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-SNDR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\hat{Q}(s_0, a_0)]
            + \\mathbb{E}_{n} [\\sum_{t=0}^{k-1} \\gamma^t \\frac{w_{0:t} \\delta(\\pi, a_{0:t})}{\\sum_{n} w_{*} \\delta_{*}} (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))]
            + \\mathbb{E}_{n} [\\sum_{t=k}^{T-1} \\gamma^t \\frac{w_s(s_{t-k}) w_{t-k:t} \\delta(\\pi, a_{t-k:t})}{\\sum_{n} w_{*} \\delta_{*}} (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))],

    where :math:`w_s(s) := \\frac{d_{\\pi}(s)}{d_{\\pi_0}(s)}` and :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`.
    :math:`w_{*}` is the abstruction of any weights and :math:`Q(s, a)` is the state-action value.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that, when :math:`k=0`, this estimator is identical to the (pure) state marginal SNDR.

    There are several ways to estimate state marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sm_sndr"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation.", 2018

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "sm_sndr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
        )
        state_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            state_marginal_importance_weight=state_marginal_importance_weight,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_episode, 2)
        )
        state_action_value_prediction = np.insert(
            state_action_value_prediction, -1, 0, axis=1
        )
        state_value_prediction = state_action_value_prediction[:, 1:, 1]
        state_action_value_prediction = state_action_value_prediction[:, :-1, 0]

        estimated_trajectory_value = state_value_prediction[:, 0] + (
            discount[np.newaxis, :]
            * self_normalized_weight
            * (reward + gamma * state_value_prediction - state_action_value_prediction)
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class ContinuousStateActionMarginalImportanceSampling(
    BaseStateActionMarginalOffPolicyEstimator
):
    """State-Action Marginal Importance Sampling (SAM-IS) for continuous-action OPE.

    Note
    -------
    SAM-IS estimates the policy value using state-action marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we consider the combination of marginalized OPE and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-IS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{k-1} \\gamma^t w_{0:t} \\delta(\\pi, a_{t_1:t_2}) r_t ] + \\mathbb{E}_{n} [\\sum_{t=k}^{T-1} \\gamma^t w(s_{t-k}, a_{t-k}) w_{t-k+1:t} \\delta(\\pi, a_{t_1:t_2}) r_t],

    where :math:`w(s, a) := \\frac{d_{\\pi}(s, a)}{d_{\\pi_0}(s, a)}` and :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that, when :math:`k=0`, this estimator is identical to the (pure) state-action marginal IS.

    There are several ways to estimate state-action marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state-action marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sam_is"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation.", 2018

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "sam_is"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
        )
        state_action_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
        )
        weight = state_action_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma
        estimated_trajectory_value = (discount[np.newaxis, :] * weight * reward).sum(
            axis=1
        )

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_scalar(
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        sigma: float = 1.0,
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
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

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

        """
        check_scalar(
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousStateActionMarginalDoublyRobust(
    BaseStateActionMarginalOffPolicyEstimator
):
    """State-Action Marginal Doubly Robust (SAM-DR) for continuous-action OPE.

    Note
    -------
    SAM-DR estimates the policy value using state-action marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we consider the combination of marginalized OPE and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\hat{Q}(s_0, a_0)]
            + \\mathbb{E}_{n} [\\sum_{t=0}^{k-1} \\gamma^t w_{0:t} \\delta(\\pi, a_{0:t}) (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))]
            + \\mathbb{E}_{n} [\\sum_{t=k}^{T-1} \\gamma^t w(s_{t-k}, a_{t-k}) w_{t-k+1:t} \\delta(\\pi, a_{t-k+1:t}) (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))],

    where :math:`w(s, a) := \\frac{d_{\\pi}(s, a)}{d_{\\pi_0}(s, a)}` and :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`. :math:`Q(s, a)` is the state-action value.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that, when :math:`k=0`, this estimator is identical to the (pure) state-action marginal DR.

    There are several ways to estimate state-action marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state-action marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sam_dr"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation.", 2018

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "sam_dr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
        )
        state_action_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
        )
        weight = state_action_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction
        state_value_prediction = np.insert(state_value_prediction, -1, 0, axis=1)
        state_value_prediction = state_action_value_prediction[:, 1:, 1]
        state_action_value_prediction = state_action_value_prediction[:, :-1, 0]

        estimated_trajectory_value = state_value_prediction[:, 0] + (
            discount[np.newaxis, :]
            * weight
            * (reward + gamma * state_value_prediction - state_action_value_prediction)
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_scalar(
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if not state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        estimated_policy_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        sigma: float = 1.0,
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
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

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

        """
        check_scalar(
            n_step_pdis,
            name="n_step_pdis",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
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
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
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
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if not state_action_value_prediction.shape[1] != 2:
            raise ValueError(
                "Expected `state_action_value_prediction.shape[1] == 2`, but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            action_scaler=action_scaler,
            sigma=sigma,
            gamma=gamma,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousStateActionMarginalSelfNormalizedImportanceSampling(
    ContinuousStateActionMarginalImportanceSampling
):
    """State-Action Marginal Self-Normalized Importance Sampling (SAM-SNIS) for continuous-action OPE.

    Note
    -------
    SAM-SNIS estimates the policy value using state-action marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we consider the combination of marginalized OPE and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-SNIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{k-1} \\gamma^t \\frac{w_{0:t} \\delta(\\pi, a_{0:t})}{\\sum_{n} w_{*} \\delta_{*}} r_t ] + \\mathbb{E}_{n} [\\sum_{t=k}^{T-1} \\gamma^t \\frac{w(s_{t-k}, a_{t-k}) w_{t-k+1:t} \\delta(\\pi, a_{t-l+1:t})}{\\sum_n w_{*} \\delta_{*}} r_t],

    where :math:`w(s, a) := \\frac{d_{\\pi}(s, a)}{d_{\\pi_0}(s, a)}` and :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`. w_{*} is the abstruction of any weights.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that, when :math:`k=0`, this estimator is identical to the (pure) state-action marginal SNIS.

    There are several ways to estimate state-action marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state-action marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sam_snis"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation.", 2018

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "sam_snis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
        )
        state_action_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
        )
        weight = state_action_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )
        self_normalized_weight = weight / (weight.mean(axis=1)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma
        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class ContinuousStateActionMarginalSelfNormalizedDoublyRobust(
    ContinuousStateActionMarginalDoublyRobust
):
    """State-Action Marginal Self-Normalized Doubly Robust (SAM-SNDR) for continuous-action OPE.

    Note
    -------
    SAM-SNDR estimates the policy value using state-action marginal importance weighting.
    Following SOPE (Yuan et al., 2021), we consider the combination of marginalized OPE and :math:`k`-step PDIS as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-SNDR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\hat{Q}(s_0, a_0)]
            + \\mathbb{E}_{n} [\\sum_{t=0}^{k-1} \\gamma^t \\frac{w_{0:t} \\delta(\\pi, a_{0:t})}{\\sum_{n} w_{*} \\delta_{*}} (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))]
            + \\mathbb{E}_{n} [\\sum_{t=k}^{T-1} \\gamma^t \\frac{w(s_{t-k}, a_{t-k}) w_{t-k+1:t} \\delta(\\pi, a_{t-k+1:t})}{\\sum_{n} w_{*} \\delta_{*}} (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))],

    where :math:`w(s, a) := \\frac{d_{\\pi}(s, a)}{d_{\\pi_0}(s, a)}` and :math:`w_{t_1:t_2} := \\prod_{t=t_1}^{t_2} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`.
    :math:`w_{*}` is the abstruction of any weights and :math:`Q(s, a)` is the state-action value.
    :math:`\\delta(\\pi, a_{t_1:t_2}) = \\prod_{t=t_1}^{t_2} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy
    (:math:`K(\\cdot, \\cdot)` is a kernel function).
    Note that, when :math:`k=0`, this estimator is identical to the (pure) state-action marginal SNDR.

    There are several ways to estimate state-action marginal importance weight including Augmented Lagrangian Method (ALM) (Yang et al., 2020) and Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of the estimation methods of state-action marginal importance weight is available at `ofrl/ope/weight_value_learning`.

    Parameters
    -------
    estimator_name: str, default="sam_sndr"
        Name of the estimator.

    References
    -------
    Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum.
    "SOPE: Spectrum of Off-Policy Estimators.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
    "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation.", 2018

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "sam_sndr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        n_step_pdis: int,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: np.ndarray,
        sigma: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        n_step_pdis: int (>= 0)
            Number of previous steps to use per-decision importance weight.
            When zero is given, the estimator corresponds to the pure state marginal IS.

        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, 2)
            :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            pscore=pscore,
        )
        similarity_weight = self._calc_similarity_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            action_scaler=action_scaler,
            sigma=sigma,
        )
        state_action_marginal_importance_weight = self._calc_marginal_importance_weight(
            n_step_pdis=n_step_pdis,
            step_per_episode=step_per_episode,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
        )
        weight = state_action_marginal_importance_weight * (
            similarity_weight / behavior_policy_pscore
        )
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma

        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_episode, 2)
        )
        state_action_value_prediction = np.insert(
            state_action_value_prediction, -1, 0, axis=1
        )
        state_value_prediction = state_action_value_prediction[:, 1:, 1]
        state_action_value_prediction = state_action_value_prediction[:, :-1, 0]

        estimated_trajectory_value = state_value_prediction[:, 0] + (
            discount[np.newaxis, :]
            * self_normalized_weight
            * (reward + gamma * state_value_prediction - state_action_value_prediction)
        ).sum(axis=1)

        return estimated_trajectory_value
