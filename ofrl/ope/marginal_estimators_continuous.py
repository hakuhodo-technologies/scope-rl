from dataclasses import dataclass
from typing import Optional, Union, Dict

import numpy as np
from sklearn.utils import check_scalar

from .estimators_base import (
    BaseStateMarginalOffPolicyEstimator,
    BaseStateActionMarginalOffPolicyEstimator,
)
from ..utils import check_array, gaussian_kernel


@dataclass
class ContinuousStateMarginalImportanceSampling(BaseStateMarginalOffPolicyEstimator):
    """State Marginal Importance Sampling (SM-IS) for continuous-action OPE.

    Note
    -------
    SM-IS estimates the policy value via state marginal importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-IS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t w_s(s_t) w_a(s_t, a_t) \\delta(\\pi(s_t), a_t) r_t],

    where :math:`w_s(s) := d_{\\pi}(s) / d_{\\pi_0}(s)` and :math:`w_a(s, a) := \\pi(a \\mid s) / \\pi_0(a \\mid s)`.
    and :math:`\\delta(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    There are several ways to estimate state marginal importance weight including Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        behavior_policy_base_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        behavior_policy_base_pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action_scaler[np.newaxis, :],
            action / action_scaler[np.newaxis, :],
            sigma=sigma,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_base_pscore
        )
        undiscounted_value = (reward * weight).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)
        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        behavior_policy_base_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        behavior_policy_base_pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

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
            behavior_policy_base_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action.shape[1] - 1,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_marginal_importance_weight.shape[0]
            == behavior_policy_base_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== behavior_policy_base_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not np.allclose(
            evaluation_policy_action.sum(axis=1),
            np.ones(evaluation_policy_action.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action.sum(axis=1) == np.ones(evaluation_policy_action.shape[0])`"
                ", but found it False"
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
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            behavior_policy_base_pscore=behavior_policy_base_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        behavior_policy_base_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
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

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        behavior_policy_base_pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

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
            behavior_policy_base_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action.shape[1] - 1,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_marginal_importance_weight.shape[0]
            == behavior_policy_base_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== behavior_policy_base_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not np.allclose(
            evaluation_policy_action.sum(axis=1),
            np.ones(evaluation_policy_action.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action.sum(axis=1) == np.ones(evaluation_policy_action.shape[0])`"
                ", but found it False"
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
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            behavior_policy_base_pscore=behavior_policy_base_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
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
    SM-DR estimates the policy value via state marginal importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\hat{Q}(s_0, a_0)]
            + \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t w_s(s_t) w_a(s_t, a_t) \\delta(\\pi(s_t), a_t)
                (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))],

    where :math:`w_s(s) := d_{\\pi}(s) / d_{\\pi_0}(s)` and :math:`w_a(s, a) := \\pi(a \\mid s) / \\pi_0(a \\mid s)`.
    :math:`\\delta(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    There are several ways to estimate state marginal importance weight :math:`w_s(s)`
    including Minimax Weight Learning (MWL) (Uehara et al., 2020).

    In addition, :math:`\\hat{Q}` can also be estimated using Minimax Q-Function Learning (MQL) (Uehara et al., 2020).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        behavior_policy_base_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        behavior_policy_base_pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action_scaler[np.newaxis, :],
            action / action_scaler[np.newaxis, :],
            sigma=sigma,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_base_pscore
        ).reshape((-1, step_per_episode))

        state_value_prediction = (
            (state_action_value_prediction * evaluation_policy_action)
            .sum(axis=1)
            .reshape((-1, step_per_episode))
        )
        state_action_value_prediction = state_action_value_prediction[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))

        reward = reward.reshape((-1, step_per_episode))

        discount = np.full(step_per_episode, gamma).cumprod()
        discount = np.roll(discount, 1)
        discount[0] = 1

        weight = weight[:, :-1]
        discount = discount[:-1]
        reward = reward[:, :-1]
        state_value_prediction = state_action_value_prediction[:, 1:]
        state_action_value_prediction = state_action_value_prediction[:, :-1]

        estimated_trajectory_value = state_value_prediction[:, 0].mean()
        estimated_trajectory_value += (
            (
                weight
                * discount[np.newaxis, :]
                * (
                    reward
                    + gamma * state_value_prediction
                    - state_action_value_prediction
                )
            )
            .sum(axis=1)
            .mean()
        )
        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        behavior_policy_base_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        behavior_policy_base_pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

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
            behavior_policy_base_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
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
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action.shape[1] - 1,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_marginal_importance_weight.shape[0]
            == behavior_policy_base_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== behavior_policy_base_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if evaluation_policy_action.shape != state_action_value_prediction.shape:
            raise ValueError(
                "Expected `evaluation_policy_action.shape == state_action_value_prediction.shape`, but found False"
            )
        if not np.allclose(
            evaluation_policy_action.sum(axis=1),
            np.ones(evaluation_policy_action.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action.sum(axis=1) == np.ones(evaluation_policy_action.shape[0])`"
                ", but found it False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            behavior_policy_base_pscore=behavior_policy_base_pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
            sigma=sigma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        behavior_policy_base_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
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

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        behavior_policy_base_pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

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
            behavior_policy_base_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
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
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action.shape[1] - 1,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_marginal_importance_weight.shape[0]
            == behavior_policy_base_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] "
                "== behavior_policy_base_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if evaluation_policy_action.shape != state_action_value_prediction.shape:
            raise ValueError(
                "Expected `evaluation_policy_action.shape == state_action_value_prediction.shape`, but found False"
            )
        if not np.allclose(
            evaluation_policy_action.sum(axis=1),
            np.ones(evaluation_policy_action.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action.sum(axis=1) == np.ones(evaluation_policy_action.shape[0])`"
                ", but found it False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_marginal_importance_weight=state_marginal_importance_weight,
            behavior_policy_base_pscore=behavior_policy_base_pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
            sigma=sigma,
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
    SM-SNIS estimates the policy value via state marginal importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-SNIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_s(s_t) w_a(s_t, a_t) \\delta(\\pi(s_t), a_t)}{\\sum_{n} [w_s(s_t) w_a(s_t, a_t) \\delta(\\pi(s_t), a_t)]} r_t],

    where :math:`w_s(s) := d_{\\pi}(s) / d_{\\pi_0}(s)` and :math:`w_a(s, a) := \\pi(a \\mid s) / \\pi_0(a \\mid s)`.
    :math:`\\delta(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    There are several ways to estimate state marginal importance weight including Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        behavior_policy_base_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        action_scaler: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        behavior_policy_base_pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action_scaler[np.newaxis, :],
            action / action_scaler[np.newaxis, :],
            sigma=sigma,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_base_pscore
        )
        weight = weight / weight.mean()

        undiscounted_value = (reward * weight).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)
        return estimated_trajectory_value


@dataclass
class ContinuousStateMarginalSelfNormalizedDoublyRobust(
    ContinuousStateMarginalDoublyRobust
):
    """State Marginal Self-Normalized Doubly Robust (SM-SNDR) for continuous-action OPE.

    Note
    -------
    SM-SNDR estimates the policy value via state marginal importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SM-DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\hat{Q}(s_0, a_0)]
            + \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_s(s_t) w_a(s_t, a_t) \\delta(\\pi(s_t), a_t)}{\\sum_{n} [w_s(s_t) w_a(s_t, a_t) \\delta(\\pi(s_t), a_t)]}
                (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))],

    where :math:`w_s(s) := d_{\\pi}(s) / d_{\\pi_0}(s)` and :math:`w_a(s, a) := \\pi(a \\mid s) / \\pi_0(a \\mid s)`.
    :math:`\\delta(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    There are several ways to estimate state marginal importance weight :math:`w_s(s)`
    including Minimax Weight Learning (MWL) (Uehara et al., 2020).

    In addition, :math:`\\hat{Q}` can also be estimated using Minimax Q-Function Learning (MQL) (Uehara et al., 2020).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_marginal_importance_weight: np.ndarray,
        behavior_policy_base_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        action_scaler: np.ndarray,
        gamma: float = 1.0,
        sigma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state, i.e., :math:`d_{\\pi}(s) / d_{\\pi_0}(s)`

        behavior_policy_base_pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: array-like of shape (action_dim, )
            Scaling factor of action.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action[np.newaxis, :],
            action / action_scaler[np.newaxis, :],
            sigma=sigma,
        )
        weight = state_marginal_importance_weight * (
            similarity_weight / behavior_policy_base_pscore
        ).reshape((-1, step_per_episode))
        weight = weight / weight.mean()

        state_value_prediction = (
            (state_action_value_prediction * evaluation_policy_action)
            .sum(axis=1)
            .reshape((-1, step_per_episode))
        )
        state_action_value_prediction = state_action_value_prediction[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))

        reward = reward.reshape((-1, step_per_episode))

        discount = np.full(step_per_episode, gamma).cumprod()
        discount = np.roll(discount, 1)
        discount[0] = 1

        weight = weight[:, :-1]
        discount = discount[:-1]
        reward = reward[:, :-1]
        state_value_prediction = state_action_value_prediction[:, 1:]
        state_action_value_prediction = state_action_value_prediction[:, :-1]

        estimated_trajectory_value = state_value_prediction[:, 0].mean()
        estimated_trajectory_value += (
            (
                weight
                * discount[np.newaxis, :]
                * (
                    reward
                    + gamma * state_value_prediction
                    - state_action_value_prediction
                )
            )
            .sum(axis=1)
            .mean()
        )
        return estimated_trajectory_value


@dataclass
class ContinuousStateActionMarginalImportanceSampling(
    BaseStateActionMarginalOffPolicyEstimator
):
    """State-Action Marginal Importance Sampling (SAM-IS) for continuous-action OPE.

    Note
    -------
    SAM-IS estimates the policy value via state marginal importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-IS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t w(s_t, a_t) \\delta(\\pi(s_t), a_t) r_t],

    where :math:`w(s, a) := d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`.
    :math:`\\delta(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    There are several ways to estimate state-action marginal importance weight including Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        step_per_episode: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        undiscounted_value = (reward * state_action_marginal_importance_weight).reshape(
            (-1, step_per_episode)
        )
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)
        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        gamma: float = 1.0,
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

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

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
        if reward.shape[0] != state_action_marginal_importance_weight.shape[0]:
            raise ValueError(
                "Expected `reward.shape[0] == state_action_marginal_importance_weight.shape[0], but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

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
        if reward.shape[0] != state_action_marginal_importance_weight.shape[0]:
            raise ValueError(
                "Expected `reward.shape[0] == state_action_marginal_importance_weight.shape[0], but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
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
    SAM-DR estimates the policy value via state marginal importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\hat{Q}(s_0, a_0)]
            + \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t w(s_t, a_t) \\delta(\\pi(s_t), a_t)
                (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))],

    where :math:`w(s, a) := d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`.
    :math:`\\delta(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    There are several ways to estimate state marginal importance weight :math:`w(s, a)`
    including Minimax Weight Learning (MWL) (Uehara et al., 2020).

    In addition, :math:`\\hat{Q}` can also be estimated using Minimax Q-Function Learning (MQL) (Uehara et al., 2020).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        state_value_prediction = state_action_value_prediction[
            np.arange(len(evaluation_policy_action)), evaluation_policy_action
        ].reshape((-1, step_per_episode))
        state_action_value_prediction = state_action_value_prediction[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))

        weight = state_action_marginal_importance_weight.reshape((-1, step_per_episode))

        reward = reward.reshape((-1, step_per_episode))

        discount = np.full(step_per_episode, gamma).cumprod()
        discount = np.roll(discount, 1)
        discount[0] = 1

        weight = weight[:, :-1]
        discount = discount[:-1]
        reward = reward[:, :-1]
        state_value_prediction = state_action_value_prediction[:, 1:]
        state_action_value_prediction = state_action_value_prediction[:, :-1]

        estimated_trajectory_value = state_value_prediction[:, 0].mean()
        estimated_trajectory_value += (
            (
                weight
                * discount[np.newaxis, :]
                * (
                    reward
                    + gamma * state_value_prediction
                    - state_action_value_prediction
                )
            )
            .sum(axis=1)
            .mean()
        )
        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

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
            evaluation_policy_action,
            name="evaluation_policy_action",
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
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action.shape[1] - 1,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_action_marginal_importance_weight.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_marginal_importance_weight.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if evaluation_policy_action.shape != state_action_value_prediction.shape:
            raise ValueError(
                "Expected `evaluation_policy_action.shape == state_action_value_prediction.shape`, but found False"
            )
        if not np.allclose(
            evaluation_policy_action.sum(axis=1),
            np.ones(evaluation_policy_action.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action.sum(axis=1) == np.ones(evaluation_policy_action.shape[0])`"
                ", but found it False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        evaluation_policy_action: np.ndarray,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

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
            evaluation_policy_action,
            name="evaluation_policy_action",
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
            action,
            name="action",
            expected_dim=1,
            min_val=0,
            max_val=evaluation_policy_action.shape[1] - 1,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == state_action_marginal_importance_weight.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == state_action_marginal_importance_weight.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if evaluation_policy_action.shape != state_action_value_prediction.shape:
            raise ValueError(
                "Expected `evaluation_policy_action.shape == state_action_value_prediction.shape`, but found False"
            )
        if not np.allclose(
            evaluation_policy_action.sum(axis=1),
            np.ones(evaluation_policy_action.shape[0]),
        ):
            raise ValueError(
                "Expected `evaluation_policy_action.sum(axis=1) == np.ones(evaluation_policy_action.shape[0])`"
                ", but found it False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            state_action_marginal_importance_weight=state_action_marginal_importance_weight,
            evaluation_policy_action=evaluation_policy_action,
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
class ContinuousStateActionMarginalSelfNormalizedImportanceSampling(
    ContinuousStateActionMarginalImportanceSampling
):
    """State-Action Marginal Self-Normalized Importance Sampling (SAM-SNIS) for continuous-action OPE.

    Note
    -------
    SAM-SNIS estimates the policy value via state marginal importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-SNIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w(s_t, a_t) \\delta(\\pi(s_t), a_t)}{\\sum_n w(s_t, a_t) \\pi(s_t), a_t} r_t],

    where :math:`w(s, a) := d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`.
    :math:`\\delta(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    There are several ways to estimate state-action marginal importance weight including Minimax Weight Learning (MWL) (Uehara et al., 2020).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        step_per_episode: int,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        weight = (
            state_action_marginal_importance_weight
            / state_action_marginal_importance_weight.mean()
        )
        undiscounted_value = (reward * weight).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)
        return estimated_trajectory_value


@dataclass
class ContinuousStateActionMarginalSelfNormalizedDoublyRobust(
    ContinuousStateActionMarginalDoublyRobust
):
    """State-Action Marginal Self-Normalized Doubly Robust (SAM-SNDR) for continuous-action OPE.

    Note
    -------
    SAM-SNDR estimates the policy value via state marginal importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SAM-SNDR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\hat{Q}(s_0, a_0)]
            + \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w(s_t, a_t) \\delta(\\pi(s_t), a_t)}{\\sum_n w(s_t, a_t) \\delta(\\pi(s_t), a_t)}
                (r_t + \\gamma \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_{t+1}, a)] - \\hat{Q}(s_t, a_t))],

    where :math:`w(s, a) := d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`.
    :math:`\\delta(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.

    There are several ways to estimate state marginal importance weight :math:`w(s, a)`
    including Minimax Weight Learning (MWL) (Uehara et al., 2020).

    In addition, :math:`\\hat{Q}` can also be estimated using Minimax Q-Function Learning (MQL) (Uehara et al., 2020).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        state_action_marginal_importance_weight: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        state_action_marginal_importance_weight: array-like of shape (n_episodes * step_per_episode, )
            Marginal importance weight of the state-action pair, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_0}(s, a)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by the evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        state_value_prediction = state_action_value_prediction[
            np.arange(len(evaluation_policy_action)), evaluation_policy_action
        ].reshape((-1, step_per_episode))
        state_action_value_prediction = state_action_value_prediction[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))

        weight = state_action_marginal_importance_weight.reshape((-1, step_per_episode))
        weight = weight / weight.mean()

        reward = reward.reshape((-1, step_per_episode))

        discount = np.full(step_per_episode, gamma).cumprod()
        discount = np.roll(discount, 1)
        discount[0] = 1

        weight = weight[:, :-1]
        discount = discount[:-1]
        reward = reward[:, :-1]
        state_value_prediction = state_action_value_prediction[:, 1:]
        state_action_value_prediction = state_action_value_prediction[:, :-1]

        estimated_trajectory_value = state_value_prediction[:, 0].mean()
        estimated_trajectory_value += (
            (
                weight
                * discount[np.newaxis, :]
                * (
                    reward
                    + gamma * state_value_prediction
                    - state_action_value_prediction
                )
            )
            .sum(axis=1)
            .mean()
        )
        return estimated_trajectory_value
