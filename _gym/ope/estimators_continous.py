"""Off-Policy Estimators for Continuous Actions (designed for deterministic policies)."""
from dataclasses import dataclass
from typing import Dict

import numpy as np

from _gym.utils import estimate_confidence_interval_by_bootstrap
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

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_values(
        self,
        initial_state_value: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return initial_state_value

    def estimate_policy_value(
        self,
        initial_state_value: np.ndarray,
        **kwargs,
    ) -> float:
        estimated_policy_value = self._estimate_trajectory_values(
            initial_state_value,
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
    """Trajectory-wise Importance Sampling (TIS) for continuous OPE (assume deterministic policies)."""

    kernel: str = "gaussian"
    band_width: float = 1.0
    estimator_name: str = "tis"

    def __post_init__(self):
        self.action_type = "continuous"
        self.scaling_factor = self.band_width * (
            self.action_space.high - self.action_space.low
        )
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        discount = np.full(rewards.shape[0], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.scaling_factor

        estimated_trajectory_values = (
            (
                (self.kernel_function(distance) * rewards / self.scaling_factor)
                / behavior_policy_trajectory_wise_pscore
            )
            * discount
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        return self._estimate_trajectory_values(
            actions,
            rewards,
            behavior_policy_step_wise_pscore,
            evaluation_policy_actions,
            gamma,
        ).mean()

    def estimate_interval(
        self,
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
        estimated_trajectory_values = self._estimate_trajectory_values(
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
    """Step-wise Importance Sampling (SIS) for continuous OPE (assume deterministic policies)."""

    kernel: str = "gaussian"
    band_width: float = 1.0
    estimator_name: str = "sis"

    def __post_init__(self):
        self.action_type = "continuous"
        self.scaling_factor = self.band_width * (
            self.action_space.high - self.action_space.low
        )
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        discount = np.full(rewards.shape[0], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.scaling_factor

        estimated_trajectory_values = (
            (
                (self.kernel_function(distance) * rewards / self.scaling_factor)
                / behavior_policy_step_wise_pscore
            )
            * discount
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        return self._estimate_trajectory_values(
            actions,
            rewards,
            behavior_policy_step_wise_pscore,
            evaluation_policy_actions,
            gamma,
        ).mean()

    def estimate_interval(
        self,
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
        estimated_trajectory_values = self._estimate_trajectory_values(
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
    """Doubly Robust (DR) for continuous OPE (assume deterministic policies)."""

    kernel: str = "gaussian"
    band_width: float = 1.0
    estimator_name = "dr"

    def __post_init__(self):
        self.action_type = "continuous"
        self.scaling_factor = self.band_width * (
            self.action_space.high - self.action_space.low
        )
        if self.kernel not in ["gaussian"]:
            raise ValueError('kernel must be "gaussian", but {self.kernel} is given')
        self.kernel_function = kernel_functions[self.kernel]

    def _estimate_trajectory_values(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        discount = np.full(rewards.shape[0], gamma).cumprod()
        distance = (actions - evaluation_policy_actions) / self.scaling_factor

        pscores = behavior_policy_step_pscore
        pscores_prev = np.roll(pscores, 1, axis=1)
        pscores_prev[:, 0] = 1

        estimated_trajectory_values = (
            (
                (
                    (
                        kernel_functions(distance)
                        * (rewards - counterfactual_state_action_value)
                    )
                    / self.scaling_factor
                )
                / pscores
                + counterfactual_state_action_value / pscores_prev
            )
            * discount
        ).sum(axis=1)

        return estimated_trajectory_values

    def estimate_policy_value(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        counterfactual_state_action_value: np.ndarray,
        evaluation_policy_actions: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        return self._estimate_trajectory_values(
            actions,
            rewards,
            behavior_policy_step_wise_pscore,
            counterfactual_state_action_value,
            evaluation_policy_actions,
            gamma,
        ).mean()

    def estimate_interval(
        self,
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
        estimated_trajectory_values = self._estimate_trajectory_values(
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
