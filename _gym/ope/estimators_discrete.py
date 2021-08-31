"""Off-Policy Estimators for Discrete Actions."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

from _gym.utils import estimate_confidence_interval_by_bootstrap


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators for discrete actions."""

    @abstractmethod
    def _estimate_trajectory_rewards(self) -> Union[np.ndarray]:
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

    estimator_name = "dm"

    def _estimate_trajectory_values(
        self,
        estimated_state_action_value: np.ndarray,
        evaluation_policy_state_action_pscore: np.ndarray,
    ) -> np.ndarray:
        initial_state_estimated_state_action_value = estimated_state_action_value[
            :, 0, :
        ]
        initial_state_evaluation_policy_state_action_pscore = (
            evaluation_policy_state_action_pscore[:, 0, :]
        )

        estimated_trajectory_values = (
            initial_state_estimated_state_action_value
            * initial_state_evaluation_policy_state_action_pscore
        ).sum(axis=1)
        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        estimated_state_action_value: np.ndarray,
        evaluation_policy_state_action_pscore: np.ndarray,
        **kwargs,
    ) -> float:
        estimated_policy_value = self._estimate_trajectory_values(
            estimated_state_action_value,
            evaluation_policy_state_action_pscore,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        estimated_state_action_value: np.ndarray,
        evaluation_policy_state_action_pscore: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_trajectory_values = self._estimate_trajectory_values(
            estimated_state_action_value,
            evaluation_policy_state_action_pscore,
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

    def _estimate_trajectory_values(
        self,
        rewards: np.ndarray,
        behavior_policy_trajectory_pscore: np.ndarray,
        evaluation_policy_trajectory_pscore: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        discount = np.full(rewards.shape[0], gamma).cumprod()
        weights = (
            evaluation_policy_trajectory_pscore / behavior_policy_trajectory_pscore
        )
        estimated_trajectory_values = ((rewards * weights) * discount).sum(axis=1)
        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        rewards: np.ndarray,
        behavior_policy_trajectory_pscore: np.ndarray,
        evaluation_policy_trajectory_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        estimated_policy_value = self._estimate_trajectory_values(
            rewards,
            behavior_policy_trajectory_pscore,
            evaluation_policy_trajectory_pscore,
            gamma,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        rewards: np.ndarray,
        behavior_policy_trajectory_pscore: np.ndarray,
        evaluation_policy_trajectory_pscore: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_trajectory_values = self._estimate_trajectory_values(
            rewards,
            behavior_policy_trajectory_pscore,
            evaluation_policy_trajectory_pscore,
            gamma,
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

    def _estimate_trajectory_values(
        self,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_step_pscore: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        discount = np.full(rewards.shape[0], gamma).cumprod()
        weights = evaluation_policy_step_pscore / behavior_policy_step_pscore

        estimated_trajectory_values = ((rewards * weights) * discount).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_step_pscore: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        return self._estimate_trajectory_values(
            rewards,
            behavior_policy_step_pscore,
            evaluation_policy_step_pscore,
            gamma,
        ).mean()

    def estimate_interval(
        self,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_step_pscore: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_trajectory_values = self._estimate_trajectory_values(
            rewards,
            behavior_policy_step_pscore,
            evaluation_policy_step_pscore,
            gamma,
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

    def _estimate_trajectory_values(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_step_pscore: np.ndarray,
        estimated_state_action_value: np.ndarray,
        evaluation_policy_state_action_pscore: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        discount = np.full(rewards.shape[0], gamma).cumprod()

        baselines = (
            estimated_state_action_value * evaluation_policy_state_action_pscore
        ).sum(axis=2)

        estimated_values = np.empty_like(rewards, dtype=float)
        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                estimated_values[i, j] = estimated_state_action_value[
                    i, j, actions[i, j]
                ]

        weights = evaluation_policy_step_pscore / behavior_policy_step_pscore
        weights_prev = np.roll(weights, 1, axis=1)
        weights_prev[:, 0] = 1

        estimated_trajectory_values = (
            (weights * (rewards - estimated_values) + weights_prev * baselines)
            * discount
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_step_pscore: np.ndarray,
        estimated_state_action_value: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        return self._estimate_trajectory_values(
            rewards,
            behavior_policy_step_pscore,
            evaluation_policy_step_pscore,
            estimated_state_action_value,
            gamma,
        ).mean()

    def estimate_interval(
        self,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_step_pscore: np.ndarray,
        estimated_state_action_value: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_trajectory_values = self._estimate_trajectory_values(
            rewards,
            behavior_policy_step_pscore,
            evaluation_policy_step_pscore,
            estimated_state_action_value,
            gamma,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
