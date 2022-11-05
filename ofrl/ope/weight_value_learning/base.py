"""Abstract Base class for Weight Value Learning."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseWeightValueLearner(metaclass=ABCMeta):
    """Base class for weight/value learning."""

    @abstractmethod
    def save(self):
        """Save models."""
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """Load models."""
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        """Fit function approximation models."""
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """Predict weights/values for the given state/state-action pair."""
        raise NotImplementedError
