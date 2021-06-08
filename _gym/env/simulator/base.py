"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BaseSimulator(metaclass=ABCMeta):
    """Base class to calculate outcome probability and stochastically determine auction result."""

    @abstractmethod
    def calc_and_sample_outcome(
        self,
        timestep: int,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        bid_prices: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Simulate bidding auction for given queries.
           (calculate outcome probability and stochastically determine auction result.)"""
        raise NotImplementedError

    @abstractmethod
    def map_idx_to_contexts(
        self, ad_ids: np.ndarray, user_ids: np.ndarray
    ) -> np.ndarray:
        """Map the ad and the user index into context vectors."""
        raise NotImplementedError
