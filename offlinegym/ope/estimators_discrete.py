"""Off-Policy Estimators for Discrete Action."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.utils import check_scalar

from .estimators_base import BaseOffPolicyEstimator
from ..utils import (
    estimate_confidence_interval_by_bootstrap,
    estimate_confidence_interval_by_hoeffding,
    estimate_confidence_interval_by_empirical_bernstein,
    estimate_confidence_interval_by_t_test,
    check_array,
)


@dataclass
class DiscreteDirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) for discrete OPE.

    Note
    -------
    DM estimates policy value using initial state value given by Fitted Q Evaluation (FQE) as follows.

    .. math::

        \\hat{V}_{\\mathrm{DM}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_n [\\mathbb{E}_{a_0 \\sim \\pi_e(a_0 \\mid s_0)} [\\hat{Q}(s_0, a_0)] ],

    where :math:`\\mathcal{D}=\\{\\{(s_t, a_t, r_t)\\}_{t=0}^T\\}_{i=1}^n` is logged dataset with :math:`n` trajectories of data.
    :math:`T` indicates step per episode. :math:`\\hat{Q}(s_t, a_t)` is estimated Q value given state-action pair.

    Parameters
    -------
    estimator_name: str, default="dm"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Alina Beygelzimer and John Langford.
    "The Offset Tree for Learning with Partial Labels.", 2009.

    """

    estimator_name: str = "dm"

    def __post_init__(self):
        self.action_type = "discrete"

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _estimate_trajectory_value(
        self,
        initial_state_value_prediction,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Estimated policy value for each trajectory.
            (Equivalent to initial_state_value_prediction.)

        """
        return initial_state_value_prediction

    def estimate_policy_value(
        self, initial_state_value_prediction: np.ndarray, **kwargs
    ) -> float:
        """Estimate policy value of evaluation policy.

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
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        alpha: float, default=0.05
            Significant level. The value should be within `[0, 1)`.

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
    """Trajectory-wise Important Sampling (TIS) for discrete OPE.

    Note
    -------
    TIS estimates policy value using trajectory-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{TIS}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t w_{1:T} r_t],

    where :math:`w_{0:T} := \\prod_{t=1}^T \\frac{\\pi_e(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}` is the importance weight.

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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        weight = (
            evaluation_policy_trajectory_wise_pscore
            / behavior_policy_trajectory_wise_pscore
        )
        undiscounted_value = (reward * weight).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)
        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate policy value of evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

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
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_trajectory_wise_pscore,
            name="evaluation_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        if not (
            reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_trajectory_wise_pscore.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`"
                ", but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        estimated_policy_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alpha: float, default=0.05
            Significant level. The value should be within `[0, 1)`.

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
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_trajectory_wise_pscore,
            name="evaluation_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        if not (
            reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_trajectory_wise_pscore.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`"
                ", but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
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
    """Per-Decision Importance Sampling (PDIS) for discrete OPE.

    Note
    -------
    PDIS estimates policy value using step-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{PDIS}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t w_{0:t} r_t],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`

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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_e(a_{t'} \\mid s_{t'})`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        weight = evaluation_policy_step_wise_pscore / behavior_policy_step_wise_pscore
        undiscounted_value = (reward * weight).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)
        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate policy value of evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_e(a_{t'} \\mid s_{t'})`

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
            behavior_policy_step_wise_pscore,
            name="behavior_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_step_wise_pscore,
            name="evaluation_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        if not (
            reward.shape[0]
            == behavior_policy_step_wise_pscore.shape[0]
            == evaluation_policy_step_wise_pscore.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_step_wise_pscore.shape[0] == evaluation_policy_step_wise_pscore.shape[0]`"
                ", but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        return self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_step_wise_pscore=behavior_policy_step_wise_pscore,
            evaluation_policy_step_wise_pscore=evaluation_policy_step_wise_pscore,
            gamma=gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_e(a_{t'} \\mid s_{t'})`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alpha: float, default=0.05
            Significant level. The value should be within `[0, 1)`.

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
            behavior_policy_step_wise_pscore,
            name="behavior_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_step_wise_pscore,
            name="evaluation_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        if not (
            reward.shape[0]
            == behavior_policy_step_wise_pscore.shape[0]
            == evaluation_policy_step_wise_pscore.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_step_wise_pscore.shape[0] == evaluation_policy_step_wise_pscore.shape[0]`"
                ", but found False"
            )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_step_wise_pscore=behavior_policy_step_wise_pscore,
            evaluation_policy_step_wise_pscore=evaluation_policy_step_wise_pscore,
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
    """Doubly Robust (DR) for discrete OPE.

    Note
    -------
    DR estimates policy value using step-wise importance weight and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t (w_{0:t} (r_t - \\hat{Q}(s_t, a_t)) + w_{0:t-1} \\mathbb{E}_{a \\sim \\pi_e(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`

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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Action choice probability of evaluation policy for all action,
            i.e., :math:`\\pi_e(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        baselines = (state_action_value_prediction * evaluation_policy_action_dist).sum(
            axis=1
        )
        estimated_value = np.empty_like(reward, dtype=float)
        for i in range(len(action)):
            estimated_value[i] = state_action_value_prediction[i, action[i]]

        weight = (
            evaluation_policy_step_wise_pscore / behavior_policy_step_wise_pscore
        ).reshape((-1, step_per_episode))
        weight_prev = np.roll(weight, 1, axis=1)
        weight_prev[:, 0] = 1

        weight = weight.flatten()
        weight_prev = weight_prev.flatten()

        undiscounted_value = (
            weight * (reward - estimated_value) + weight_prev * baselines
        ).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate policy value of evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Action choice probability of evaluation policy for all action,
            i.e., :math:`\\pi_e(a \\mid s_t) \\forall a \\in \\mathcal{A}`

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
            behavior_policy_step_wise_pscore,
            name="behavior_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_step_wise_pscore,
            name="evaluation_policy_step_wise_pscore",
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
            == behavior_policy_step_wise_pscore.shape[0]
            == evaluation_policy_step_wise_pscore.shape[0]
            == state_action_value_prediction.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_step_wise_pscore.shape[0] == evaluation_policy_step_wise_pscore.shape[0] "
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
            behavior_policy_step_wise_pscore=behavior_policy_step_wise_pscore,
            evaluation_policy_step_wise_pscore=evaluation_policy_step_wise_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Action choice probability of evaluation policy for all action,
            i.e., :math:`\\pi_e(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        alpha: float, default=0.05
            Significant level. The value should be within `[0, 1)`.

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
            behavior_policy_step_wise_pscore,
            name="behavior_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_step_wise_pscore,
            name="evaluation_policy_step_wise_pscore",
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
            == behavior_policy_step_wise_pscore.shape[0]
            == evaluation_policy_step_wise_pscore.shape[0]
            == state_action_value_prediction.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_step_wise_pscore.shape[0] == evaluation_policy_step_wise_pscore.shape[0] "
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
            behavior_policy_step_wise_pscore=behavior_policy_step_wise_pscore,
            evaluation_policy_step_wise_pscore=evaluation_policy_step_wise_pscore,
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
    """Self-Normalized Trajectory-wise Important Sampling (SNTIS) for discrete OPE.

    Note
    -------
    SNTIS estimates policy value using self-normalized trajectory-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SNTIS}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t \\frac{w_{1:T}}{\\mathbb{E}_n [w_{1:T}]} r_t],

    where :math:`w_{0:T} := \\prod_{t=1}^T \\frac{\\pi_e(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}`

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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        weight = (
            evaluation_policy_trajectory_wise_pscore
            / behavior_policy_trajectory_wise_pscore
        )
        weight_mean = weight.reshape((-1, step_per_episode)).mean(axis=0)
        self_normalized_weight = weight / np.tile(
            weight_mean + 1e-10, len(weight) // step_per_episode
        )

        undiscounted_value = (reward * self_normalized_weight).reshape(
            (-1, step_per_episode)
        )
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)
        return estimated_trajectory_value


@dataclass
class DiscreteSelfNormalizedPerDecisionImportanceSampling(
    DiscretePerDecisionImportanceSampling
):
    """Self-Normalized Per-Decision Importance Sampling (SNPDIS) for discrete OPE.

    Note
    -------
    SNPDIS estimates policy value using self-normalized step-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SNPDIS}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t \\frac{w_{1:t}}{\\mathbb{E}_n [w_{1:t}]} r_t],

    where :math:`w_{0:t} := \\prod_{t'=1}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`

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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        weight = evaluation_policy_step_wise_pscore / behavior_policy_step_wise_pscore
        weight_mean = weight.reshape((-1, step_per_episode)).mean(axis=0)
        self_normalized_weight = weight / np.tile(
            weight_mean + 1e-10, len(weight) // step_per_episode
        )

        undiscounted_value = (reward * self_normalized_weight).reshape(
            (-1, step_per_episode)
        )
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)
        return estimated_trajectory_value


@dataclass
class DiscreteSelfNormalizedDoublyRobust(DiscreteDoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) for discrete OPE.

    Note
    -------
    SNDR estimates policy value using self-normalized step-wise importance weight and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t \\frac{w_{0:t-1}}{\\mathbb{E}_n [w_{0:t-1}]}
                (w_t (r_t - \\hat{Q}(s_t, a_t)) + \\mathbb{E}_{a \\sim \\pi_e(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`
    and :math:`w_{t}} := \\frac{\\pi_e(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}`

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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, )
            Action chosen by behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action_dist: array-like of shape (n_episodes * step_per_episode, n_action)
            Action choice probability of evaluation policy for all action,
            i.e., :math:`\\pi_e(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, n_action)
            :math:`\\hat{Q}` for all action,
            i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        baselines = (state_action_value_prediction * evaluation_policy_action_dist).sum(
            axis=1
        )
        estimated_value = np.empty_like(reward, dtype=float)
        for i in range(len(action)):
            estimated_value[i] = state_action_value_prediction[i, action[i]]

        weight = (
            evaluation_policy_step_wise_pscore / behavior_policy_step_wise_pscore
        ).reshape((-1, step_per_episode))
        weight_prev = np.roll(weight, 1, axis=1)
        weight_prev[:, 0] = 1

        weight_prev_mean = weight_prev.mean(axis=0)
        weight_prev = weight_prev.flatten() / np.tile(
            weight_prev_mean + 1e-10, len(weight)
        )

        weight_mean = weight.mean(axis=0)
        weight = weight.flatten() / np.tile(weight_mean + 1e-10, len(weight))

        undiscounted_value = (
            weight * (reward - estimated_value) + weight_prev * baselines
        ).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_value.shape[1], gamma).cumprod()
        estimated_trajectory_value = (undiscounted_value * discount).sum(axis=1)

        return estimated_trajectory_value
