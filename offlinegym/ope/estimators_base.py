"""Abstract Base class for Off-Policy Estimator."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_trajectory_value(self) -> np.ndarray:
        """Estimate trajectory-wise reward."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value."""
        raise NotImplementedError


@dataclass
class BaseCumulativeDistributionalOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for Cumulative Distributional OPE estimators."""

    @abstractmethod
    def estimate_cumulative_distribution_function(self) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function (cdf) of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_mean(self) -> float:
        """Estimate mean of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_variance(self) -> float:
        """Estimate variance of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_conditional_value_at_risk(self) -> float:
        """Estimate conditional value at risk (cVaR) of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interquartile_range(self) -> Dict[str, float]:
        """Estimate interquartile range of the policy value."""
        raise NotImplementedError


@dataclass
class BaseWorstCaseDistributionalOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for Worst Case Distributional OPE estimators."""

    @abstractmethod
    def estimate_worst_case_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError
