"""Off-Policy Estimators."""
"""Off-Policy Estimators."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any

import numpy as np

from _gym.utils import estimate_confidence_interval_by_bootstrap


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

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
class DirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM)."""

    estimator_name = "dm"

    def _estimate_trajectory_values(
        self,
        estimated_state_action_value: np.ndarray,
        evaluation_policy_state_action_pscore: np.ndarray,
    ) -> Union[np.ndarray]:
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
