"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class BaseSimulator(metaclass=ABCMeta):
    """Base class to calculate outcome probability and stochastically determine auction result."""

    @abstractmethod
    def generate_auction(self, search_volume: int) -> Tuple[np.ndarray]:
        """Sample ad and user pair for each auction."""
        raise NotImplementedError

    @abstractmethod
    def map_idx_to_features(
        self, ad_ids: np.ndarray, user_ids: np.ndarray
    ) -> np.ndarray:
        """Map the ad and the user index into feature vectors."""
        raise NotImplementedError

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


@dataclass
class BaseWinningPriceDistribution(metaclass=ABCMeta):
    """Base class to sample the winning price (i.e., second price) and compare with the given bid price."""

    @abstractmethod
    def sample_outcome(
        self,
        bid_prices: np.ndarray,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> Tuple[np.ndarray]:
        """Stochastically determine impression and second price for each auction."""
        raise NotImplementedError


@dataclass
class BaseClickAndConversionRate(metaclass=ABCMeta):
    """Base class to Class to define ground-truth CTR/CVR."""

    @abstractmethod
    def calc_prob(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Calculate CTR/CVR."""
        raise NotImplementedError

    @abstractmethod
    def sample_outcome(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Stochastically determine whether click/conversion occurs or not."""
        raise NotImplementedError
