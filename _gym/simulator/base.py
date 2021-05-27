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
        adjust_rate: int,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Simulate bidding auction for given queries and return outcome."""
        raise NotImplementedError

    @abstractmethod
    def fit_reward_predictor(self, n_samples: int) -> None:
        """Fit reward predictor in advance to use prediction in bidding price determination."""
        raise NotImplementedError()

    @abstractmethod
    def _predict_reward(self, timestep: int, contexts: np.ndarray) -> np.ndarray:
        """Predict reward (i.e., auction outcome) to determine bidding price."""
        raise NotImplementedError()

    @abstractmethod
    def _calc_ground_truth_reward(
        self, timestep: int, contexts: np.ndarray
    ) -> np.ndarray:
        """Calculate ground-truth reward (i.e., auction outcome) to determine bidding price."""
        raise NotImplementedError()

    @abstractmethod
    def _map_idx_to_contexts(
        self, ad_ids: np.ndarray, user_ids: np.ndarray
    ) -> np.ndarray:
        """Map the ad and the user index into context vectors."""
        raise NotImplementedError()

    @abstractmethod
    def _determine_bid_price(
        self, timestep: int, adjust_rate: int, contexts: np.ndarray
    ) -> np.ndarray:
        """Determine the bidding price using given adjust rate and the predicted/ground-truth rewards."""
        raise NotImplementedError()

    @abstractmethod
    def _calc_and_sample_outcome(
        self, timestep: int, bid_prices: np.ndarray, contexts: np.ndarray
    ) -> Tuple[np.ndarray]:
        """Calculate pre-determined probabilities from contexts and stochastically sample the outcome."""
        raise NotImplementedError()
