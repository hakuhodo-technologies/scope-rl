"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BaseSimulator(metaclass=ABCMeta):
    """Base class for bidding auction simulators."""

    @abstractmethod
    def simulate_auction(
        self,
        timestep: int,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        contexts: np.ndarray,
        bid_prices: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Simulate bidding auction for given queries and return outcome."""
        raise NotImplementedError
