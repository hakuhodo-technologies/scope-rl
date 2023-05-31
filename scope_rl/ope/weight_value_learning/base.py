# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Abstract base class for weight and value learning."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseWeightValueLearner(metaclass=ABCMeta):
    """Base class for weight/value learning.

    Imported as: :class:`scope_rl.ope.weight_value_learning.BaseWeightValueLearner`

    """

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
