"""Off-Policy Estimators for Discrete Actions."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

from ..utils import estimate_confidence_interval_by_bootstrap


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators for discrete actions."""

    @abstractmethod
    def _estimate_trajectory_values(self) -> np.ndarray:
        """Estimate trajectory-wise rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        raise NotImplementedError


@dataclass
class DiscreteDirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) for discrete OPE.

    Note
    -------
    DM estimates policy value using initial state value given by Fitted Q Evaluation (FQE) as follows.

    .. math::

        \\hat{V}_{\\mathrm{DM}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_n [\\mathbb{E}_{a_0 \\sim \\pi_e(a_0 \\mid s_0)} [\\hat{Q}(x_0, a_0)] ],

    where :math:`\\mathcal{D}=\\{\\{(s_t, a_t, r_t)\\}_{t=0}^T\\}_{i=1}^n` is logged dataset with :math:`n` trajectories of data.
    :math:`T` indicates step per episode. :math:`\\hat{Q}(x_t, a_t)` is estimated Q value given state-action pair.

    Parameters
    -------
    estimator_name: str, default="dm"
        Name of the estimator.

    """

    estimator_name: str = "dm"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_values(
        self,
        initial_state_value,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        initial_state_value: NDArray, shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.
            (Equivalent to initial_state_value in DM.)

        """
        return initial_state_value

    def estimate_policy_value(self, initial_state_value: np.ndarray, **kwargs) -> float:
        estimated_policy_value = self._estimate_trajectory_values(
            initial_state_value
        ).mean()
        """Estimate policy value of evaluation policy.
        
        Parameters
        -------
        initial_state_value: NDArray, shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        V_hat: NDArray, shape (n_episodes, )
            Estimated policy value.
        
        """
        return estimated_policy_value

    def estimate_interval(
        self,
        initial_state_value: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        initial_state_value: NDArray, shape (n_episodes, )
            Estimated initial state value.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_trajectory_values = self._estimate_trajectory_values(
            initial_state_value
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
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

    where :math:`w_{0:T} := \\prod_{t=1}^T \\frac{\\pi_e(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}`

    Parameters
    -------
    estimator_name: str, default="tis"
        Name of the estimator.

    """

    estimator_name = "tis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        rewards: np.ndarray,
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

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        weights = (
            evaluation_policy_trajectory_wise_pscore
            / behavior_policy_trajectory_wise_pscore
        )
        undiscounted_values = (rewards * weights).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_values.shape[1], gamma).cumprod()
        estimated_trajectory_values = (undiscounted_values * discount).sum(axis=1)
        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        step_per_episode: int,
        rewards: np.ndarray,
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

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        V_hat: NDArray, shape (n_episodes, )
            Estimated policy value.

        """
        estimated_policy_value = self._estimate_trajectory_values(
            step_per_episode,
            rewards,
            behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore,
            gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        step_per_episode: int,
        rewards: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_trajectory_values = self._estimate_trajectory_values(
            step_per_episode,
            rewards,
            behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DiscreteStepWiseImportanceSampling(BaseOffPolicyEstimator):
    """Step-wise Importance Sampling (SIS) for discrete OPE.

    Note
    -------
    SIS estimates policy value using step-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SIS}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t w_{0:t} r_t],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`

    Parameters
    -------
    estimator_name: str, default="sis"
        Name of the estimator.

    """

    estimator_name = "sis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        rewards: np.ndarray,
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

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_e(a_{t'} \\mid s_{t'})`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        weights = evaluation_policy_step_wise_pscore / behavior_policy_step_wise_pscore
        undiscounted_values = (rewards * weights).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_values.shape[1], gamma).cumprod()
        estimated_trajectory_values = (undiscounted_values * discount).sum(axis=1)
        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        step_per_episode: int,
        rewards: np.ndarray,
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

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_e(a_{t'} \\mid s_{t'})`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        V_hat: NDArray, shape (n_episodes, )
            Estimated policy value.

        """
        return self._estimate_trajectory_values(
            step_per_episode,
            rewards,
            behavior_policy_step_wise_pscore,
            evaluation_policy_step_wise_pscore,
            gamma=gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_e(a_{t'} \\mid s_{t'})`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_trajectory_values = self._estimate_trajectory_values(
            step_per_episode,
            rewards,
            behavior_policy_step_wise_pscore,
            evaluation_policy_step_wise_pscore,
            gamma=gamma,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
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

    """

    estimator_name = "dr"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        counterfactual_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        counterfactual_state_action_value: NDArray, shape (n_episodes * step_per_episode, n_actions)
            :math:`\\hat{Q}` for all actions.

        counterfactual_pscore: NDArray, shape (n_episodes * step_per_episode, n_actions)
            Action choice probability of evaluation policy for all actions.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        baselines = (counterfactual_state_action_value * counterfactual_pscore).sum(
            axis=1
        )
        estimated_values = np.empty_like(rewards, dtype=float)
        for i in range(len(actions)):
            estimated_values[i] = counterfactual_state_action_value[i, actions[i]]

        weights = (
            evaluation_policy_step_wise_pscore / behavior_policy_step_wise_pscore
        ).reshape((-1, step_per_episode))
        weights_prev = np.roll(weights, 1, axis=1)
        weights_prev[:, 0] = 1

        weights = weights.flatten()
        weights_prev = weights_prev.flatten()

        undiscounted_values = (
            weights * (rewards - estimated_values) + weights_prev * baselines
        ).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_values.shape[1], gamma).cumprod()
        estimated_trajectory_values = (undiscounted_values * discount).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        counterfactual_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate policy value of evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        counterfactual_state_action_value: NDArray, shape (n_episodes * step_per_episode, n_actions)
            :math:`\\hat{Q}` for all actions.

        counterfactual_pscore: NDArray, shape (n_episodes * step_per_episode, n_actions)
            Action choice probability of evaluation policy for all actions.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        V_hat: NDArray, shape (n_episodes, )
            Estimated policy value.

        """
        return self._estimate_trajectory_values(
            step_per_episode,
            actions,
            rewards,
            behavior_policy_step_wise_pscore,
            evaluation_policy_step_wise_pscore,
            counterfactual_state_action_value,
            counterfactual_pscore,
            gamma=gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        counterfactual_pscore: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        counterfactual_state_action_value: NDArray, shape (n_episodes * step_per_episode, n_actions)
            :math:`\\hat{Q}` for all actions.

        counterfactual_pscore: NDArray, shape (n_episodes * step_per_episode, n_actions)
            Action choice probability of evaluation policy for all actions.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_trajectory_values = self._estimate_trajectory_values(
            step_per_episode,
            actions,
            rewards,
            behavior_policy_step_wise_pscore,
            evaluation_policy_step_wise_pscore,
            counterfactual_state_action_value,
            counterfactual_pscore,
            gamma=gamma,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
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

    """

    estimator_name = "sntis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        rewards: np.ndarray,
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

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        weights = (
            evaluation_policy_trajectory_wise_pscore
            / behavior_policy_trajectory_wise_pscore
        )
        weight_mean = weights.reshape((-1, step_per_episode)).mean(axis=0)
        self_normalized_weights = weights / np.tile(
            weight_mean + 1e-10, len(weights) // step_per_episode
        )

        undiscounted_values = (rewards * self_normalized_weights).reshape(
            (-1, step_per_episode)
        )
        discount = np.full(undiscounted_values.shape[1], gamma).cumprod()
        estimated_trajectory_values = (undiscounted_values * discount).sum(axis=1)
        return estimated_trajectory_values


@dataclass
class DiscreteSelfNormalizedStepWiseImportanceSampling(
    DiscreteStepWiseImportanceSampling
):
    """Self-Normalized Step-wise Importance Sampling (SNSIS) for discrete OPE.

    Note
    -------
    SNSIS estimates policy value using self-normalized step-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SNSIS}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t \\frac{w_{1:t}}{\\mathbb{E}_n [w_{1:t}]} r_t],

    where :math:`w_{0:t} := \\prod_{t'=1}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`

    Parameters
    -------
    estimator_name: str, default="snsis"
        Name of the estimator.

    """

    estimator_name = "snsis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        rewards: np.ndarray,
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

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        weights = evaluation_policy_step_wise_pscore / behavior_policy_step_wise_pscore
        weight_mean = weights.reshape((-1, step_per_episode)).mean(axis=0)
        self_normalized_weights = weights / np.tile(
            weight_mean + 1e-10, len(weights) // step_per_episode
        )

        undiscounted_values = (rewards * self_normalized_weights).reshape(
            (-1, step_per_episode)
        )
        discount = np.full(undiscounted_values.shape[1], gamma).cumprod()
        estimated_trajectory_values = (undiscounted_values * discount).sum(axis=1)
        return estimated_trajectory_values


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

    """

    estimator_name = "sndr"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        counterfactual_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        rewards: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability by evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        counterfactual_state_action_value: NDArray, shape (n_episodes * step_per_episode, n_actions)
            :math:`\\hat{Q}` for all actions.

        counterfactual_pscore: NDArray, shape (n_episodes * step_per_episode, n_actions)
            Action choice probability of evaluation policy for all actions.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        baselines = (counterfactual_state_action_value * counterfactual_pscore).sum(
            axis=1
        )
        estimated_values = np.empty_like(rewards, dtype=float)
        for i in range(len(actions)):
            estimated_values[i] = counterfactual_state_action_value[i, actions[i]]

        weights = (
            evaluation_policy_step_wise_pscore / behavior_policy_step_wise_pscore
        ).reshape((-1, step_per_episode))
        weights_prev = np.roll(weights, 1, axis=1)
        weights_prev[:, 0] = 1

        weights_prev_mean = weights_prev.mean(axis=0)
        weights_prev = weights_prev.flatten() / np.tile(
            weights_prev_mean + 1e-10, len(weights)
        )
        weights = weights.flatten() / np.tile(weights_prev_mean + 1e-10, len(weights))

        undiscounted_values = (
            weights * (rewards - estimated_values) + weights_prev * baselines
        ).reshape((-1, step_per_episode))
        discount = np.full(undiscounted_values.shape[1], gamma).cumprod()
        estimated_trajectory_values = (undiscounted_values * discount).sum(axis=1)

        return estimated_trajectory_values
