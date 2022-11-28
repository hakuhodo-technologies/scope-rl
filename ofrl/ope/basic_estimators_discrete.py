"""Off-Policy Estimators for Discrete Action."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.utils import check_scalar

from .estimators_base import BaseOffPolicyEstimator
from ..utils import check_array


@dataclass
class DiscreteDirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) for discrete-action OPE.

    Note
    -------
    DM estimates the policy value using the initial state value as follows.

    .. math::

        \\hat{J}_{\\mathrm{DM}} (\\pi; \\mathcal{D}) := \\mathbb{E}_n [\\mathbb{E}_{a_0 \\sim \\pi(a_0 \\mid s_0)} [\\hat{Q}(s_0, a_0)] ],

    .. math::

        \\hat{J}_{\\mathrm{DM}} (\\pi; \\mathcal{D}) := \\mathbb{E}_n [\\hat{V}(s_0)],

    where :math:`\\mathcal{D}=\\{\\{(s_t, a_t, r_t)\\}_{t=0}^{T-1}\\}_{i=1}^n` is the logged dataset with :math:`n` trajectories of data.
    :math:`T` indicates step per episode. :math:`\\hat{Q}(s_t, a_t)` is the estimated Q value given a state-action pair.
    \\hat{V}(s_t) is the estimated value function given a state.

    There are several ways to estimate :math:`\\hat{Q}(s, a)` such as Fitted Q Evaluation (FQE) (Le et al., 2019) and
    Minimax Q-Function Learning (MQL) (Uehara et al., 2020). :math:`\\hat{V}(s)` is estimated in a similar manner using
    Minimax Value Learning (MVL) (Uehara et al., 2020).

    We use the implementation of FQE provided by d3rlpy (Seno et al., 2021).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_discrete.py`.

    Parameters
    -------
    estimator_name: str, default="dm"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Takuma Seno and Michita Imai.
    "d3rlpy: An Offline Deep Reinforcement Library.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Alina Beygelzimer and John Langford.
    "The Offset Tree for Learning with Partial Labels.", 2009.

    """

    estimator_name: str = "dm"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        initial_state_value_prediction,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.
            (Equivalent to initial_state_value_prediction.)

        """
        return initial_state_value_prediction

    def estimate_policy_value(
        self, initial_state_value_prediction: np.ndarray, **kwargs
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        estimated_policy_value = self._estimate_trajectory_value(
            initial_state_value_prediction
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
        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

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
            initial_state_value_prediction
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

    Note
    -------
    TIS estimates the policy value via trajectory-wise importance weight as follows.

    .. math::

        \\hat{J}_{\\mathrm{TIS}} (\\pi; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t w_{0:T-1} r_t],

    where :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}` is the importance weight.

    Parameters
    -------
    estimator_name: str, default="tis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name = "tis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            pscore=pscore,
            pscore_type="trajectory_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="trajectory_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma
        estimated_trajectory_value = (discount[np.newaxis, :] * weight * reward).sum(
            axis=1
        )

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
            step_per_episode=step_per_episode,
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

    Note
    -------
    PDIS estimates the policy value via step-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{PDIS}} (\\pi; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t w_{0:t} r_t],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi(a_{t'} \\mid s_{t'})}{\\pi_0(a_{t'} \\mid s_{t'})}`

    Parameters
    -------
    estimator_name: str, default="pdis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name = "pdis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            pscore=pscore,
            pscore_type="step_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="step_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma
        estimated_trajectory_value = (discount[np.newaxis, :] * weight * reward).sum(
            axis=1
        )

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            gamma=gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
            step_per_episode=step_per_episode,
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

    Note
    -------
    DR estimates the policy value via step-wise importance weighting and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{J}_{\\mathrm{DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t (w_{0:t} (r_t - \\hat{Q}(s_t, a_t)) + w_{0:t-1} \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi(a_{t'} \\mid s_{t'})}{\\pi_0(a_{t'} \\mid s_{t'})}`

    Parameters
    -------
    estimator_name: str, default="dr"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name = "dr"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            pscore=pscore,
            pscore_type="step_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="step_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore
        weight_prev = np.roll(weight, 1, axis=1)
        weight_prev[:, 0] = 1

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma

        state_value_prediction = (
            (state_action_value_prediction * evaluation_policy_action_dist)
            .sum(axis=1)
            .reshape((-1, step_per_episode))
        )

        state_action_value_prediction = state_action_value_prediction[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))

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
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
            step_per_episode=step_per_episode,
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

    Note
    -------
    SNTIS estimates the policy value via self-normalized trajectory-wise importance weight as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNTIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{0:T-1}}{\\sum_{n} [w_{0:T-1}]} r_t],

    where :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`

    Parameters
    -------
    estimator_name: str, default="sntis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name = "sntis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            pscore=pscore,
            pscore_type="trajectory_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="trajectory_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma
        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class DiscreteSelfNormalizedPerDecisionImportanceSampling(
    DiscretePerDecisionImportanceSampling
):
    """Self-Normalized Per-Decision Importance Sampling (SNPDIS) for discrete-action OPE.

    Note
    -------
    SNPDIS estimates the policy value via self-normalized step-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNPDIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{1:t}}{\\sum_{n} [w_{1:t}]} r_t],

    where :math:`w_{0:t} := \\prod_{t'=1}^t \\frac{\\pi(a_{t'} \\mid s_{t'})}{\\pi_0(a_{t'} \\mid s_{t'})}`

    Parameters
    -------
    estimator_name: str, default="snpdis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name = "snpdis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            pscore=pscore,
            pscore_type="step_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="step_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma
        estimated_trajectory_value = (
            discount[np.newaxis, :] * self_normalized_weight * reward
        ).sum(axis=1)

        return estimated_trajectory_value


@dataclass
class DiscreteSelfNormalizedDoublyRobust(DiscreteDoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) for discrete-action OPE.

    Note
    -------
    SNDR estimates the policy value via self-normalized step-wise importance weighting and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{J}_{\\mathrm{DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{0:t-1}}{\\sum_{n} [w_{0:t-1}]}
                (w_t (r_t - \\hat{Q}(s_t, a_t)) + \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi(a_{t'} \\mid s_{t'})}{\\pi_0(a_{t'} \\mid s_{t'})}`
    and :math:`w_{t}} := \\frac{\\pi(a_t \\mid s_t)}{\\pi_0(a_t \\mid s_t)}`

    Parameters
    -------
    estimator_name: str, default="sndr"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name = "sndr"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        pscore: array-like of shape (n_episodes * step_per_episode, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
        behavior_policy_pscore = self._calc_behavior_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            pscore=pscore,
            pscore_type="step_wise",
        )
        evaluation_policy_pscore = self._calc_evaluation_policy_pscore_discrete(
            step_per_episode=step_per_episode,
            action=action,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            pscore_type="step_wise",
        )
        weight = evaluation_policy_pscore / behavior_policy_pscore
        self_normalized_weight = weight / (weight.mean(axis=0)[np.newaxis, :] + 1e-10)
        self_normalized_weight_prev = np.roll(self_normalized_weight, 1, axis=1)
        self_normalized_weight_prev[:, 0] = 1

        reward = reward.reshape((-1, step_per_episode))
        discount = np.full(step_per_episode, gamma).cumprod() / gamma

        state_value_prediction = (
            (state_action_value_prediction * evaluation_policy_action_dist)
            .sum(axis=1)
            .reshape((-1, step_per_episode))
        )

        state_action_value_prediction = state_action_value_prediction[
            np.arange(len(action)), action
        ].reshape((-1, step_per_episode))

        estimated_trajectory_value = (
            discount[np.newaxis, :]
            * (
                self_normalized_weight * (reward - state_action_value_prediction)
                + self_normalized_weight_prev * state_value_prediction
            )
        ).sum(axis=1)

        return estimated_trajectory_value
