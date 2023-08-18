# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Abstract Base Class for Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class BaseSimulator(metaclass=ABCMeta):
    """Base class to calculate the outcome probability and stochastically determine auction result.

    Imported as: :class:`rtbgym.envs.simulator.BaseSimulator`

    """

    @abstractmethod
    def generate_auction(self, search_volume: int) -> Tuple[np.ndarray]:
        """Sample ad and user pair for each auction.

        Parameters
        -------
        search_volume: int, default=None (> 0)
            Total number of auctions to generate.

        Returns
        -------
        ad_ids: ndarray of shape (search_volume, )
            IDs of the ads.

        user_ids: ndarray of shape (search_volume, )
            IDs of the users.

        """
        raise NotImplementedError

    @abstractmethod
    def map_idx_to_features(
        self, ad_ids: np.ndarray, user_ids: np.ndarray
    ) -> np.ndarray:
        """Map the ad and the user index into feature vectors.

        Parameters
        -------
        ad_ids: array-like of shape (search_volume, )
            IDs of the ads.
            (search_volume is determined in RL environment.)

        user_ids: array-like of shape (search_volume, )
            IDs of the users.
            (search_volume is determined in RL environment.)

        Returns
        -------
        ad_feature_vector: ndarray of shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: ndarray of shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        """
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

        Parameters
        -------
        timestep: int (> 0)
            Timestep in the RL environment.

        ad_ids: array-like of shape (search_volume, )
            IDs of the ads.

        user_ids: array-like of shape (search_volume, )
            IDs of the users.

        bid_prices: array-like of shape(search_volume, )
            Bid price for each action.

        Returns
        -------
        costs: ndarray of shape (search_volume, )
            Cost raised (i.e., second price) for each auction.

        impressions: ndarray of shape (search_volume, )
            Binary indicator of whether impression occurred or not for each auction.

        clicks: ndarray of shape (search_volume, )
            Binary indicator of whether click occurred or not for each auction.

        conversions: ndarray of shape (search_volume, )
            Binary indicator of whether conversion occurred or not for each auction.

        """
        raise NotImplementedError


@dataclass
class BaseWinningPriceDistribution(metaclass=ABCMeta):
    """Base class to sample the winning price (i.e., second price) and compare it with the given bid price.

    Imported as: class:`rtbgym.BaseWinningPriceDistribution`

    """

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
        """Stochastically determine impression and second price for each auction.

        Parameters
        -------
        bid_prices: array-like of shape (search_volume, )
            Bid price for each auction.

        ad_ids: array-like of shape (search_volume/n_samples, )
            Ad ids used for each auction.

        user_ids: array-like of shape (search_volume/n_samples, )
            User ids used for each auction.

        ad_feature_vector: array-like of shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: array-like of shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        timestep: {int, array-like of shape (n_samples, )}
            Timestep in the RL environment.

        Returns
        -------
        impressions: ndarray of shape (search_volume, )
            Whether impression occurred for each auction.

        winning_prices: ndarray of shape (search_volume, )
            Sampled winning price for each auction.

        """
        raise NotImplementedError


@dataclass
class BaseClickAndConversionRate(metaclass=ABCMeta):
    """Base class to Class to define ground-truth CTR/CVR.

    Imported as: class:`rtbgym.BaseClickAndConversionRate`

    """

    @abstractmethod
    def calc_prob(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Calculate Click Through Rate (CTR) / Conversion Rate (CVR).

        Parameters
        -------
        ad_ids: array-like of shape (search_volume/n_samples, )
            Ad ids used for each auction.

        user_ids: array-like of shape (search_volume/n_samples, )
            User ids used for each auction.

        ad_feature_vector: array-like of shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: array-like of shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        timestep: {int, array-like of shape (n_samples, )}
            Timestep in the RL environment.

        Returns
        -------
        ctrs/cvrs: ndarray of shape (search_volume/n_samples, )
            Ground-truth CTR (i.e., click per impression) or CVR (i.e., conversion per click) for each auction.

        """
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
        """Stochastically determine whether click/conversion occurs or not.

        Parameters
        -------
        ad_ids: array-like of shape (search_volume/n_samples, )
            Ad ids used for each auction.

        user_ids: array-like of shape (search_volume/n_samples, )
            User ids used for each auction.

        ad_feature_vector: array-like of shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: array-like of shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        timestep: {int, array-like of shape (n_samples, )}
            Timestep in the RL environment.

        Returns
        -------
        clicks/conversions: array-like of shape (search_volume/n_samples, )
            Whether click occurs (when impression=True) or whether conversion occurs (when click=True).

        """
        raise NotImplementedError
