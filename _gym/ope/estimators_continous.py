"""Off-Policy Estimators for Continuous Actions (designed for deterministic policies)."""
from dataclasses import dataclass
from typing import Dict

import numpy as np

from _gym.utils import estimate_confidence_interval_by_bootstrap, action_scaler
from _gym.ope import BaseOffPolicyEstimator


def gaussian_kernel():
    raise NotImplementedError()


kernel_functions = {
    "gaussian": gaussian_kernel,
}


@dataclass
class ContinuousDirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) for continuous OPE (assume deterministic policies)."""

    estimator_name = "dm"

    def _estimate_trajectory_values(
        self,
        estimated_state_action_value: np.ndarray,
    ) -> np.ndarray:
        return estimated_state_action_value[:, 0]

    def estimate_policy_value(
        self,
        estimated_state_action_value: np.ndarray,
        **kwargs,
    ) -> float:
        estimated_policy_value = self._estimate_trajectory_values(
            estimated_state_action_value,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        estimated_state_action_value: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_trajectory_values = self._estimate_trajectory_values(
            estimated_state_action_value,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousStepWiseImportanceSampling(BaseOffPolicyEstimator):
    """Step-wise Importance Sampling (SIS) for continuous OPE (assume deterministic policies)."""

    kernel: str = "gaussian"
    band_width: float = 1.0
    estimator_name: str = "sis"

    def __post_init__(self):
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')

        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        discount = np.full(rewards.shape[0], gamma).cumprod()

        behavior_policy_actions = action_scaler(actions)
        evaluation_policy_actions = action_scaler(actions)
        distance = (
            behavior_policy_actions - evaluation_policy_actions
        ) / self.band_width

        estimated_trajectory_values = (
            (
                (self.kernel_function(distance) * rewards / self.band_width)
                / behavior_policy_step_pscore
            )
            * discount
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        return self._estimate_trajectory_values(
            actions,
            rewards,
            behavior_policy_step_pscore,
            evaluation_policy_actions,
            gamma,
        ).mean()

    def estimate_interval(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_trajectory_values = self._estimate_trajectory_values(
            actions,
            rewards,
            behavior_policy_step_pscore,
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
    """Doubly Robust (DR) for continuous OPE (assume deterministic policies)."""

    kernel: str = "gaussian"
    band_width: float = 1.0
    estimator_name = "dr"

    def _estimate_trajectory_values(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        estimated_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        discount = np.full(rewards.shape[0], gamma).cumprod()

        behavior_policy_actions = action_scaler(actions)
        evaluation_policy_actions = action_scaler(evaluation_policy_actions)
        distance = (
            behavior_policy_actions - evaluation_policy_actions
        ) / self.band_width

        pscores = behavior_policy_step_pscore
        pscores_prev = np.roll(pscores, 1, axis=1)
        pscores_prev[:, 0] = 1

        estimated_trajectory_values = (
            (
                (
                    (
                        kernel_functions(distance)
                        * (rewards - estimated_state_action_value)
                    )
                    / self.band_width
                )
                / pscores
                + estimated_state_action_value / pscores_prev
            )
            * discount
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        estimated_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        return self._estimate_trajectory_values(
            actions,
            rewards,
            behavior_policy_step_pscore,
            estimated_state_action_value,
            evaluation_policy_actions,
            gamma,
        ).mean()

    def estimate_interval(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        estimated_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_trajectory_values = self._estimate_trajectory_values(
            actions,
            rewards,
            behavior_policy_step_pscore,
            estimated_state_action_value,
            evaluation_policy_actions,
            gamma,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_trajectory_values,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
