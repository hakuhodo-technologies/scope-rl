"""Off-Policy Estimators for Continuous Actions (designed for deterministic policies)."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from gym.spaces import Box

from _gym.utils import estimate_confidence_interval_by_bootstrap
from _gym.ope import BaseOffPolicyEstimator


def gaussian_kernel(dist):
    return (np.exp(-(dist ** 2) / 2) / np.sqrt(2 * np.pi)).sum(axis=-1)


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
    """Step-wise Importance Sampling (SIS) for continuous OPE (assume deterministic policies)."""

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
    """Doubly Robust (DR) for continuous OPE (assume deterministic policies)."""

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
