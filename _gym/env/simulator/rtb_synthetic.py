"""Synthetic Bidding Auction Simulation."""
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import numpy as np
from sklearn.utils import check_random_state

from _gym.utils import NormalDistribution

from .base import (
    BaseSimulator,
    BaseWinningPriceDistribution,
    BaseClickAndConversionRate,
)
from .function import WinningPriceDistribution, ClickThroughRate, ConversionRate


@dataclass
class RTBSyntheticSimulator(BaseSimulator):
    """Class to calculate outcome probability and stochastically determine auction result
       in Real-Time Bidding (RTB) setting for display advertising.

    Parameters
    -------
    cost_indicator: str, default="click"
        Defines when the cost arises.
        Choose either from "impression", "click" or "conversion".

    step_per_episode: int, default=7
        Number of timesteps in an episode.

    n_ads: int, default=100
        Number of ads used for fitting the reward predictor.

    n_users: int, default=100
        Number of users used for fitting the reward predictor.

    ad_feature_dim: int, default=5
        Dimensions of the ad feature vectors.

    user_feature_dim: int, default=5
        Dimensions of the user feature vectors.

    ad_sampling_rate: Optional[Union[NDArray[int], NDArray[float]]], shape (n_ads, ), default=None
        Sampling probalities to determine which ad (id) is used in each auction.

    user_sampling_rate: Optional[Union[NDArray[int], NDArray[float]]], shape (n_users, ), default=None
        Sampling probalities to determine which user (id) is used in each auction.

    WinningPriceDistribution:

    ClickTroughRate:

    ConversionRate:

    standard_bid_price_distribution: NormalDistribution, default=NormalDistribution(mean=100, std=20)
        Distribution of the bid price whose average impression probability is expected to be 0.5.

    minimum_standard_bid_price: Optional[int], default=None
        Minimum value for standard bid price.
        If None, minimum_standard_bid_price is set to standard_bid_price_distribution.mean / 2.

    random_state: int, default=12345
        Random state.

    References
    -------

    """

    cost_indicator: str = "click"
    step_per_episode: int = 7
    n_ads: int = 100
    n_users: int = 100
    ad_feature_dim: int = 5
    user_feature_dim: int = 5
    ad_feature_vector: Optional[np.ndarray] = None
    user_feature_vector: Optional[np.ndarray] = None
    ad_sampling_rate: Optional[np.ndarray] = None
    user_sampling_rate: Optional[np.ndarray] = None
    WinningPriceDistribution: BaseWinningPriceDistribution = WinningPriceDistribution
    ClickThroughRate: BaseClickAndConversionRate = ClickThroughRate
    ConversionRate: BaseClickAndConversionRate = ConversionRate
    standard_bid_price_distribution: NormalDistribution = NormalDistribution(
        mean=50,
        std=5,
        random_state=12345,
    )
    minimum_standard_bid_price: Optional[Union[int, float]] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        if not (isinstance(self.n_ads, int) and self.n_ads > 0):
            raise ValueError(
                f"n_ads must be a positive interger, but {self.n_ads} is given"
            )
        if not (isinstance(self.n_users, int) and self.n_users > 0):
            raise ValueError(
                f"n_users must be a positive interger, but {self.n_users} is given"
            )
        if not (isinstance(self.ad_feature_dim, int) and self.ad_feature_dim > 0):
            raise ValueError(
                f"ad_feature_dim must be a positive interger, but {self.ad_feature_dim} is given"
            )
        if not (isinstance(self.user_feature_dim, int) and self.user_feature_dim > 0):
            raise ValueError(
                f"user_feature_dim must be a positive interger, but {self.user_feature_dim} is given"
            )
        if not (
            self.ad_feature_vector is None
            or (
                isinstance(self.ad_feature_vector, np.ndarray)
                and self.ad_feature_dim.shape == (self.n_ads, self.ad_feature_dim)
            )
        ):
            raise ValueError(
                "ad_feature_vector must be an 2-dimensional NDArray with shape (n_ads, ad_feature_dim)"
            )
        if not (
            self.user_feature_vector is None
            or (
                isinstance(self.user_feature_vector, np.ndarray)
                and self.user_feature_dim.shape == (self.n_users, self.user_feature_dim)
            )
        ):
            raise ValueError(
                "user_feature_vector must be an 2-dimensional NDArray with shape (n_users, user_feature_dim)"
            )
        if not (
            self.ad_sampling_rate is None
            or (
                isinstance(self.ad_sampling_rate, np.ndarray)
                and self.ad_sampling_rate.ndim == 1
                and self.ad_sampling_rate.min() >= 0
                and self.ad_sampling_rate.max() > 0
            )
        ):
            raise ValueError(
                "ad_sampling_rate must be an 1-dimensional NDArray of non-negative float values"
            )
        if not (
            self.user_sampling_rate is None
            or (
                isinstance(self.user_sampling_rate, np.ndarray)
                and self.user_sampling_rate.ndim == 1
                and self.user_sampling_rate.min() >= 0
                and self.user_sampling_rate.max() > 0
            )
        ):
            raise ValueError(
                "user_sampling_rate must be an 1-dimensional NDArray of non-negative float values"
            )
        if self.ad_sampling_rate is not None and self.n_ads != len(
            self.ad_sampling_rate
        ):
            raise ValueError("length of ad_sampling_rate must be equal to n_ads")
        if self.user_sampling_rate is not None and self.n_users != len(
            self.user_sampling_rate
        ):
            raise ValueError("length of user_sampling_rate must be equal to n_users")
        if not isinstance(self.standard_bid_price_distribution, NormalDistribution):
            raise ValueError(
                "standard_bid_price_distribution must be a NormalDistribution"
            )
        if not isinstance(self.standard_bid_price_distribution.mean, (int, float)):
            raise ValueError(
                "standard_bid_price_distribution must have a single parameter for mean and std"
            )
        if self.minimum_standard_bid_price is not None and not (
            isinstance(self.minimum_standard_bid_price, (int, float))
            and 0
            <= self.minimum_standard_bid_price
            <= self.standard_bid_price_distribution.mean
        ):
            raise ValueError(
                f"minimum_standard_bid_price must be a float value within [0, standard_bid_price_distribution.mean], but {self.minimum_standard_bid_price} is given"
            )
        if not (
            self.step_per_episode is None
            or (isinstance(self.step_per_episode, int) and self.step_per_episode > 0)
        ):
            raise ValueError(
                f"step_per_episode must be a positive interger, but {self.step_per_episode} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        if self.ad_feature_vector is None:
            self.ad_feature_vector = self.random_.normal(
                size=(self.n_ads, self.ad_feature_dim)
            )
        if self.user_feature_vector is None:
            self.user_feature_vector = self.random_.normal(
                size=(self.n_users, self.user_feature_dim)
            )

        self.ad_ids = np.arange(self.n_ads)
        self.user_ids = np.arange(self.n_users)

        if self.ad_sampling_rate is None:
            self.ad_sampling_rate = np.full(
                (self.step_per_episode, self.n_ads), 1 / self.n_ads
            )
        else:
            self.ad_sampling_rate = self.ad_sampling_rate / np.sum(
                self.ad_sampling_rate, axis=1
            )

        if self.user_sampling_rate is None:
            self.user_sampling_rate = np.full(
                (self.step_per_episode, self.n_users), 1 / self.n_users
            )
        else:
            self.user_sampling_rate = self.user_sampling_rate / np.sum(
                self.user_sampling_rate, axis=1
            )

        # define winning function
        self.winning_price_distribution = self.WinningPriceDistribution(
            n_ads=self.n_ads,
            n_users=self.n_users,
            ad_feature_dim=self.ad_feature_dim,
            user_feature_dim=self.user_feature_dim,
            step_per_episode=self.step_per_episode,
            standard_bid_price_distribution=self.standard_bid_price_distribution,
            minimum_standard_bid_price=self.minimum_standard_bid_price,
            random_state=self.random_state,
        )
        # define click/imp and conversion/click rate function
        self.ctr = self.ClickThroughRate(
            n_ads=self.n_ads,
            n_users=self.n_users,
            ad_feature_dim=self.ad_feature_dim,
            user_feature_dim=self.user_feature_dim,
            step_per_episode=self.step_per_episode,
            random_state=self.random_state,
        )
        self.cvr = self.ConversionRate(
            n_ads=self.n_ads,
            n_users=self.n_users,
            ad_feature_dim=self.ad_feature_dim,
            user_feature_dim=self.user_feature_dim,
            step_per_episode=self.step_per_episode,
            random_state=self.random_state
            + 1,  # to differenciate the coef with that of CTR
        )

    @property
    def standard_bid_price(self):
        return self.winning_price_distribution.standard_bid_price

    def generate_auction(self, volume: int, timestep: Optional[int] = None):
        """Sample ad and user pair for each auction.

        Parameters
        -------
        volume: int
            Total numbers of auction to generate.

        timestep: Optional[int], default=None
            Timestep of the RL environment.

        Returns
        -------
        ad_ids: NDArray[int], shape (volume, )
            IDs of the ads used for the auction bidding.

        user_ids: NDArray[int], shape (volume, )
            IDs of the users who receives the winning ads.

        """
        if not (isinstance(volume, (int, np.integer)) and volume > 0):
            raise ValueError(
                f"volume must be a positive interger, but {volume} is given"
            )

        if isinstance(timestep, int):
            ad_ids = self.random_.choice(
                self.ad_ids,
                size=volume,
                p=self.ad_sampling_rate[timestep],
            )
            user_ids = self.random_.choice(
                self.user_ids,
                size=volume,
                p=self.user_sampling_rate[timestep],
            )

        else:
            ad_ids = self.random_.choice(
                self.ad_ids,
                size=volume,
                p=self.ad_sampling_rate.mean(axis=0),
            )
            user_ids = self.random_.choice(
                self.user_ids,
                size=volume,
                p=self.user_sampling_rate.mean(axis=0),
            )

        return ad_ids, user_ids

    def map_idx_to_features(
        self, ad_ids: np.ndarray, user_ids: np.ndarray
    ) -> np.ndarray:
        """Map the ad and the user index into feature vectors.

        Parameters
        -------
        ad_ids: NDArray[int], shape (search_volume, )
            IDs of the ads used for the auction bidding.
            (search_volume is determined in RL environment.)

        user_ids: NDArray[int], shape (search_volume, )
            IDs of the users who receives the winning ads.
            (search_volume is determined in RL environment.)

        Returns
        -------
        ad_feature_vector: Union[NDArray[int], NDArray[float]], shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: Union[NDArray[int], NDArray[float]], shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        """
        if not (
            isinstance(ad_ids, np.ndarray)
            and ad_ids.ndim == 1
            and 0 <= ad_ids.min()
            and ad_ids.max() < self.n_ads
        ):
            raise ValueError(
                "ad_ids must be 1-dimensional NDArray with integers within [0, n_ads)"
            )
        if not (
            isinstance(user_ids, np.ndarray)
            and user_ids.ndim == 1
            and 0 <= user_ids.min()
            and user_ids.max() < self.n_users
        ):
            raise ValueError(
                "user_ids must be 1-dimensional NDArray with integers within [0, n_users)"
            )
        if not (len(ad_ids) == len(user_ids)):
            raise ValueError("ad_ids and user_ids must have same length")

        ad_features = self.ad_feature_vector[ad_ids]
        user_features = self.user_feature_vector[user_ids]
        return ad_features, user_features

    def calc_and_sample_outcome(
        self,
        timestep: int,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        bid_prices: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Simulate bidding auction for given queries.
           (calculate outcome probability and stochastically determine auction result.)

        Parameters
        -------
        timestep: int
            Corresponds to the timestep of the RL environment.

        ad_ids: NDArray[int], shape (search_volume, )
            IDs of the ads used for the auction bidding.
            (search_volume is determined in RL environment.)

        user_ids: NDArray[int], shape (search_volume, )
            IDs of the users who receives the winning ads.
            (search_volume is determined in RL environment.)

        bid_prices: NDArray[int], shape(search_volume, )
            Bid price for each action.
            (search_volume is determined in RL environment.)

        Returns
        -------
        auction_results: Tuple
            costs: NDArray[int], shape (search_volume, )
                Cost raised (i.e., second price) for each auction.

            impressions: NDArray[int], shape (search_volume, )
                Binary indicator of whether impression occurred or not for each auction.

            clicks: NDArray[int], shape (search_volume, )
                Binary indicator of whether click occurred or not for each auction.

            conversions: NDArray[int], shape (search_volume, )
                Binary indicator of whether conversion occurred or not for each auction.

        """
        if not (isinstance(timestep, int) and timestep >= 0):
            raise ValueError(
                f"timestep must be a non-negative interger, but {timestep} is given"
            )
        if not (
            isinstance(bid_prices, np.ndarray)
            and bid_prices.ndim == 1
            and 0 <= bid_prices.min()
        ):
            raise ValueError(
                "bid_prices must be 1-dimensional NDArray with non-negative integers"
            )
        if not (len(ad_ids) == len(user_ids) == len(bid_prices)):
            raise ValueError("ad_ids, user_ids, and bid_prices must have same length")

        ad_feature_vector = self.ad_feature_vector[ad_ids]
        user_feature_vector = self.user_feature_vector[user_ids]

        impressions, winning_prices = self.winning_price_distribution.sample_outcome(
            bid_prices=bid_prices,
            ad_ids=ad_ids,
            user_ids=user_ids,
            ad_feature_vector=ad_feature_vector,
            user_feature_vector=user_feature_vector,
            timestep=timestep,
        )
        clicks = (
            self.ctr.sample_outcome(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timestep,
            )
            * impressions
        )
        conversions = (
            self.cvr.sample_outcome(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timestep,
            )
            * clicks
        )

        if self.cost_indicator == "impression":
            costs = winning_prices * impressions
        elif self.cost_indicator == "click":
            costs = winning_prices * clicks
        elif self.cost_indicator == "conversion":
            costs = winning_prices * conversions

        return costs, impressions, clicks, conversions
