"""Abstract Base Class for Policy."""
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import numpy as np


# fix later
class BasePolicy(metaclass=ABCMeta):
    "Base class for reinforcement learning (RL) agent (i.e., policy)"

    @abstractmethod
    def act(self, state: np.ndarray) -> Tuple[Union[int, float]]:
        """Choose action to take."""
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """Update the policy."""
        raise NotImplementedError
