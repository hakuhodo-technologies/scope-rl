"""Off-Policy Estimators for Continuous Actions (designed for deterministic policies)."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..utils import estimate_confidence_interval_by_bootstrap
from ..ope.estimators_discrete import BaseOffPolicyEstimator


def gaussian_kernel(dist):
    return (np.exp(-(dist ** 2) / 2) / np.sqrt(2 * np.pi)).sum(axis=-1)


kernel_functions = {
    "gaussian": gaussian_kernel,
}


@dataclass
class ContinuousDirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) for continuous OPE (assume deterministic policies).

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

    estimator_name = "dm"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_values(
        self,
        initial_state_value: np.ndarray,
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

    def estimate_policy_value(
        self,
        initial_state_value: np.ndarray,
        **kwargs,
    ) -> float:
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
        estimated_policy_value = self._estimate_trajectory_values(
            initial_state_value,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        initial_state_value: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
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
            initial_state_value,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousTrajectoryWiseImportanceSampling(BaseOffPolicyEstimator):
    """Trajectory-wise Importance Sampling (TIS) for continuous OPE (assume deterministic policies).

    Note
    -------
    TIS estimates policy value using trajectory-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{TIS}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t w_{1:T} r_t],

    where :math:`w_{0:T} := \\prod_{t=1}^T \\frac{\\pi_e(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}`

    Parameters
    -------
    action_dim: int (> 0)
        Dimensions of actions.

    kernel: str, default="gaussian"
        Choice of kernel function.
        "gaussian" is acceptable.

    band_width: Optional[np.ndarray]
        A bandwidth hyperparameter for each action dimension.
        A larger value increases bias instead of reducing variance.
        A smaller value increased variance instead of reducing bias.

    estimator_name: str, default="tis"
        Name of the estimator.

    """

    action_dim: int
    kernel: str = "gaussian"
    band_width: Optional[np.ndarray] = None
    estimator_name: str = "tis"

    def __post_init__(self):
        self.action_type = "continuous"
        if self.band_width is None:
            self.band_width = np.ones(self.action_dim)
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        actions = actions.reshape((-1, step_per_episode, self.action_dim))
        evaluation_policy_actions = evaluation_policy_actions.reshape(
            (-1, step_per_episode, self.action_dim)
        )
        rewards = rewards.reshape((-1, step_per_episode))
        behavior_policy_trajectory_wise_pscore = (
            behavior_policy_trajectory_wise_pscore.reshape((-1, step_per_episode))
        )

        discount = np.full(rewards.shape[1], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.band_width
        similarity_weight = (self.kernel_function(distance) / self.band_width).cumprod(
            axis=1
        )[:, -1]
        similarity_weight = np.tile(
            similarity_weight.reshape((-1, 1)), step_per_episode
        )

        estimated_trajectory_values = (
            discount
            * rewards
            * similarity_weight
            / behavior_policy_trajectory_wise_pscore
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
            evaluation_policy_actions,
            gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
            evaluation_policy_actions,
            gamma,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousStepWiseImportanceSampling(BaseOffPolicyEstimator):
    """Step-wise Importance Sampling (SIS) for continuous OPE (assume deterministic policies).

    Note
    -------
    SIS estimates policy value using step-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SIS}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t w_{0:t} r_t],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`

    Parameters
    -------
    action_dim: int (> 0)
        Dimensions of actions.

    kernel: str, default="gaussian"
        Choice of kernel function.
        "gaussian" is acceptable.

    band_width: Optional[np.ndarray]
        A bandwidth hyperparameter for each action dimension.
        A larger value increases bias instead of reducing variance.
        A smaller value increased variance instead of reducing bias.

    estimator_name: str, default="sis"
        Name of the estimator.

    """

    action_dim: int
    kernel: str = "gaussian"
    band_width: Optional[np.ndarray] = None
    estimator_name: str = "sis"

    def __post_init__(self):
        self.action_type = "continuous"
        if self.band_width is None:
            self.band_width = np.ones(self.action_dim)
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        actions = actions.reshape((-1, step_per_episode, self.action_dim))
        evaluation_policy_actions = evaluation_policy_actions.reshape(
            (-1, step_per_episode, self.action_dim)
        )
        rewards = rewards.reshape((-1, step_per_episode))
        behavior_policy_step_wise_pscore = behavior_policy_step_wise_pscore.reshape(
            (-1, step_per_episode)
        )

        discount = np.full(rewards.shape[1], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.band_width
        similarity_weight = (self.kernel_function(distance) / self.band_width).cumprod(
            axis=1
        )
        estimated_trajectory_values = (
            discount * rewards * similarity_weight / behavior_policy_step_wise_pscore
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
            evaluation_policy_actions,
            gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
            evaluation_policy_actions,
            gamma,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousDoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust (DR) for continuous OPE (assume deterministic policies).

    Note
    -------
    DR estimates policy value using step-wise importance weight and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t (w_{0:t} (r_t - \\hat{Q}(s_t, a_t)) + w_{0:t-1} \\mathbb{E}_{a \\sim \\pi_e(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`

    Parameters
    -------
    action_dim: int (> 0)
        Dimensions of actions.

    kernel: str, default="gaussian"
        Choice of kernel function.
        "gaussian" is acceptable.

    band_width: Optional[np.ndarray]
        A bandwidth hyperparameter for each action dimension.
        A larger value increases bias instead of reducing variance.
        A smaller value increased variance instead of reducing bias.

    estimator_name: str, default="dr"
        Name of the estimator.

    """

    action_dim: int
    kernel: str = "gaussian"
    band_width: Optional[np.ndarray] = None
    estimator_name = "dr"

    def __post_init__(self):
        self.action_type = "continuous"
        if self.band_width is None:
            self.band_width = np.ones(self.action_dim)
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        counterfactual_state_action_value: NDArray, shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by evaluation policy.

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        actions = actions.reshape((-1, step_per_episode, self.action_dim))
        evaluation_policy_actions = evaluation_policy_actions.reshape(
            (-1, step_per_episode, self.action_dim)
        )

        rewards = rewards.reshape((-1, step_per_episode))
        counterfactual_state_action_value = counterfactual_state_action_value.reshape(
            (-1, step_per_episode)
        )

        pscores = behavior_policy_step_wise_pscore.reshape((-1, step_per_episode))
        pscores_prev = np.roll(pscores, 1, axis=1)
        pscores_prev[:, 0] = 1

        discount = np.full(rewards.shape[1], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.band_width
        similarity_weight = (self.kernel_function(distance) / self.band_width).cumprod(
            axis=1
        )
        similarity_weight_prev = np.roll(similarity_weight, 1, axis=1)
        similarity_weight_prev[:, 0] = 1

        estimated_trajectory_values = (
            discount
            * (
                (rewards - counterfactual_state_action_value)
                * similarity_weight
                / pscores
                + counterfactual_state_action_value
                * similarity_weight_prev
                / pscores_prev
            )
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        counterfactual_state_action_value: NDArray, shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by evaluation policy.

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
            counterfactual_state_action_value,
            evaluation_policy_actions,
            gamma,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        counterfactual_state_action_value: NDArray, shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by evaluation policy.

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
            counterfactual_state_action_value,
            evaluation_policy_actions,
            gamma,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousSelfNormalizedTrajectoryWiseImportanceSampling(
    ContinuousTrajectoryWiseImportanceSampling
):
    """Self-Normalized Trajectory-wise Importance Sampling (SNTIS) for continuous OPE (assume deterministic policies).

    Note
    -------
    SNTIS estimates policy value using self-normalized trajectory-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SNTIS}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t \\frac{w_{1:T}}{\\mathbb{E}_n [w_{1:T}]} r_t],

    where :math:`w_{0:T} := \\prod_{t=1}^T \\frac{\\pi_e(a_t \\mid s_t)}{\\pi_b(a_t \\mid s_t)}`

    Parameters
    -------
    action_dim: int (> 0)
        Dimensions of actions.

    kernel: str, default="gaussian"
        Choice of kernel function.
        "gaussian" is acceptable.

    band_width: Optional[np.ndarray]
        A bandwidth hyperparameter for each action dimension.
        A larger value increases bias instead of reducing variance.
        A smaller value increased variance instead of reducing bias.

    estimator_name: str, default="sntis"
        Name of the estimator.

    """

    action_dim: int
    kernel: str = "gaussian"
    band_width: Optional[np.ndarray] = None
    estimator_name: str = "sntis"

    def __post_init__(self):
        self.action_type = "continuous"
        if self.band_width is None:
            self.band_width = np.ones(self.action_dim)
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        actions = actions.reshape((-1, step_per_episode, self.action_dim))
        evaluation_policy_actions = evaluation_policy_actions.reshape(
            (-1, step_per_episode, self.action_dim)
        )
        rewards = rewards.reshape((-1, step_per_episode))
        importance_weight = 1 / (
            behavior_policy_trajectory_wise_pscore.reshape((-1, step_per_episode))
        )
        importance_weight_mean = importance_weight.mean(axis=0)
        importance_weight_mean = np.tile(
            importance_weight_mean, len(importance_weight)
        ).reshape((-1, step_per_episode))
        self_normalized_importance_weight = importance_weight / (
            importance_weight_mean + 1e-10
        )

        discount = np.full(rewards.shape[1], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.band_width
        similarity_weight = (self.kernel_function(distance) / self.band_width).cumprod(
            axis=1
        )[:, -1]
        similarity_weight = np.tile(
            similarity_weight.reshape((-1, 1)), step_per_episode
        )

        estimated_trajectory_values = (
            discount * rewards * similarity_weight * self_normalized_importance_weight
        ).sum(axis=1)

        return estimated_trajectory_values


@dataclass
class ContinuousSelfNormalizedStepWiseImportanceSampling(
    ContinuousStepWiseImportanceSampling
):
    """Self-Normalized Step-wise Importance Sampling (SNSIS) for continuous OPE (assume deterministic policies).

    Note
    -------
    SNSIS estimates policy value using self-normalized step-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SNSIS}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t \\frac{w_{1:t}}{\\mathbb{E}_n [w_{1:t}]} r_t],

    where :math:`w_{0:t} := \\prod_{t'=1}^t \\frac{\\pi_e(a_{t'} \\mid s_{t'})}{\\pi_b(a_{t'} \\mid s_{t'})}`

    Parameters
    -------
    action_dim: int (> 0)
        Dimensions of actions.

    kernel: str, default="gaussian"
        Choice of kernel function.
        "gaussian" is acceptable.

    band_width: Optional[np.ndarray]
        A bandwidth hyperparameter for each action dimension.
        A larger value increases bias instead of reducing variance.
        A smaller value increased variance instead of reducing bias.

    estimator_name: str, default="snsis"
        Name of the estimator.

    """

    action_dim: int
    kernel: str = "gaussian"
    band_width: Optional[np.ndarray] = None
    estimator_name: str = "snsis"

    def __post_init__(self):
        self.action_type = "continuous"
        if self.band_width is None:
            self.band_width = np.ones(self.action_dim)
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        actions = actions.reshape((-1, step_per_episode, self.action_dim))
        evaluation_policy_actions = evaluation_policy_actions.reshape(
            (-1, step_per_episode, self.action_dim)
        )
        rewards = rewards.reshape((-1, step_per_episode))
        importance_weight = 1 / (
            behavior_policy_step_wise_pscore.reshape((-1, step_per_episode))
        )
        importance_weight_mean = importance_weight.mean(axis=0)
        importance_weight_mean = np.tile(
            importance_weight_mean, len(importance_weight)
        ).reshape((-1, step_per_episode))
        self_normalized_importance_weight = importance_weight / (
            importance_weight_mean + 1e-10
        )

        discount = np.full(rewards.shape[1], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.band_width
        similarity_weight = (self.kernel_function(distance) / self.band_width).cumprod(
            axis=1
        )

        estimated_trajectory_values = (
            discount * rewards * similarity_weight * self_normalized_importance_weight
        ).sum(axis=1)

        return estimated_trajectory_values


@dataclass
class ContinuousSelfNormalizedDoublyRobust(ContinuousDoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) for continuous OPE (assume deterministic policies).

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
    action_dim: int (> 0)
        Dimensions of actions.

    kernel: str, default="gaussian"
        Choice of kernel function.
        "gaussian" is acceptable.

    band_width: Optional[np.ndarray]
        A bandwidth hyperparameter for each action dimension.
        A larger value increases bias instead of reducing variance.
        A smaller value increased variance instead of reducing bias.

    estimator_name: str, default="sndr"
        Name of the estimator.

    """

    action_dim: int
    kernel: str = "gaussian"
    band_width: Optional[np.ndarray] = None
    estimator_name = "sndr"

    def __post_init__(self):
        self.action_type = "continuous"
        if self.band_width is None:
            self.band_width = np.ones(self.action_dim)
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        step_per_episode: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
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

        counterfactual_state_action_value: NDArray, shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by evaluation policy.

        evaluation_policy_actions: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

        """
        actions = actions.reshape((-1, step_per_episode, self.action_dim))
        evaluation_policy_actions = evaluation_policy_actions.reshape(
            (-1, step_per_episode, self.action_dim)
        )

        rewards = rewards.reshape((-1, step_per_episode))
        counterfactual_state_action_value = counterfactual_state_action_value.reshape(
            (-1, step_per_episode)
        )

        pscores = behavior_policy_step_wise_pscore.reshape((-1, step_per_episode))
        pscores_prev = np.roll(pscores, 1, axis=1)
        pscores_prev[:, 0] = 1

        importance_weight = 1 / pscores
        importance_weight_prev = 1 / pscores_prev
        importance_weight_prev_mean = importance_weight_prev.mean(axis=0)
        importance_weight_prev_mean = np.tile(
            importance_weight_prev_mean, len(importance_weight)
        ).reshape((-1, step_per_episode))
        self_normalized_importance_weight_prev = importance_weight_prev / (
            importance_weight_prev_mean + 1e-10
        )
        self_normalized_importance_weight = importance_weight / (
            importance_weight_prev_mean + 1e-10
        )

        discount = np.full(rewards.shape[1], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.band_width
        similarity_weight = (self.kernel_function(distance) / self.band_width).cumprod(
            axis=1
        )
        similarity_weight_prev = np.roll(similarity_weight, 1, axis=1)
        similarity_weight_prev[:, 0] = 1

        estimated_trajectory_values = (
            discount
            * (
                (rewards - counterfactual_state_action_value)
                * similarity_weight
                * self_normalized_importance_weight
                + counterfactual_state_action_value
                * similarity_weight_prev
                * self_normalized_importance_weight_prev
            )
        ).sum(axis=1)

        return estimated_trajectory_values
