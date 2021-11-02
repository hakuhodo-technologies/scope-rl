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
    def _estimate_trajectory_values(self) -> Union[np.ndarray]:
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
    """Direct Method (DM) for discrete OPE."""

    estimator_name: str = "dm"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_trajectory_values(
        self,
        initial_state_value,
        **kwargs,
    ) -> np.ndarray:
        return initial_state_value

    def estimate_policy_value(self, initial_state_value: np.ndarray, **kwargs) -> float:
        estimated_policy_value = self._estimate_trajectory_values(
            initial_state_value
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        initial_state_value: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
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
    """Trajectory-wise Important Sampling (TIS)."""

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
    """Step-wise Importance Sampling (SIS)."""

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
    """Doubly Robust (DR) for a stochastic policy."""

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
    """Self-Normalized Trajectory-wise Important Sampling (SNTIS)."""

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
    """Self-Normalized Step-wise Importance Sampling (SNSIS)."""

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
    """Self-Normalized Doubly Robust (SNDR)."""

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
