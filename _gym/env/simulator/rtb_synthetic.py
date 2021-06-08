"""Synthetic Bidding Auction Simulation."""
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import numpy as np
from sklearn.utils import check_random_state

from _gym.utils import NormalDistribution

from .base import BaseSimulator
from .function import WinningFunction, CTR, CVR


@dataclass
class RTBSyntheticSimulator(BaseSimulator):
    """Class to calculate outcome probability and stochastically determine auction result
       in Real-Time Bidding (RTB) setting for display advertising.

    Parameters
    -------
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

    standard_bid_price_distribution: NormalDistribution, default=NormalDistribution(mean=100, std=20)
        Parameter in RTBSyntheticSimulator class.
        Distribution of the bid price whose average impression probability is expected to be 0.5.

    minimum_standard_bid_price: Optional[int], default=None
        Parameter in RTBSyntheticSimulator class.
        Minimum value for standard bid price.
        If None, minimum_standard_bid_price is set to standard_bid_price_distribution.mean / 2.

    trend_interval: Optional[int], default=None
        Length of the ctr/cvr trend cycle.
        For example, trend_interval=24 indicates that the trend cycles every 24h.
        If None, trend_interval is set to step_per_episode.

    random_state: int, default=12345
        Random state.

    References
    -------

    """

    n_ads: int = 100
    n_users: int = 100
    ad_feature_dim: int = 5
    user_feature_dim: int = 5
    ad_sampling_rate: Optional[np.ndarray] = None
    user_sampling_rate: Optional[np.ndarray] = None
    standard_bid_price_distribution: NormalDistribution = NormalDistribution(
        mean=50, std=5
    )
    minimum_standard_bid_price: Optional[Union[int, float]] = None
    trend_interval: int = 24
    random_state: int = 12345

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
                "user_sampling_rate must be an NDArray of non-negative float values"
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
        if not (
            self.trend_interval is None
            or (isinstance(self.trend_interval, int) and self.trend_interval > 0)
        ):
            raise ValueError(
                f"trend_interval must be a positive interger, but {self.trend_interval} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        self.ads = self.random_.normal(size=(self.n_ads, self.ad_feature_dim))
        self.users = self.random_.normal(size=(self.n_users, self.user_feature_dim))

        self.ad_ids = np.arange(self.n_ads)
        self.user_ids = np.arange(self.n_users)

        if self.ad_sampling_rate is None:
            self.ad_sampling_rate = np.full(self.n_ads, 1 / self.n_ads)
        else:
            self.ad_sampling_rate = self.ad_sampling_rate / np.sum(
                self.ad_sampling_rate
            )

        if self.user_sampling_rate is None:
            self.user_sampling_rate = np.full(self.n_users, 1 / self.n_users)
        else:
            self.user_sampling_rate = self.user_sampling_rate / np.sum(
                self.user_sampling_rate
            )

        # define standard bid price for each ads
        if self.minimum_standard_bid_price is None:
            self.minimum_standard_bid_price = (
                self.standard_bid_price_distribution.mean / 2
            )
        standard_bid_prices = np.clip(
            self.standard_bid_price_distribution.sample(self.n_ads),
            self.minimum_standard_bid_price,
            None,
        )

        # define winning function
        self.winning_function = WinningFunction(self.random_state)
        # winning function parameter for each ad
        self.wf_ks = self.random_.normal(
            loc=50,
            scale=5,
            size=self.n_ads,
        )
        self.wf_thetas = self.random_.normal(
            loc=standard_bid_prices * 0.02,
            scale=(standard_bid_prices * 0.02) / 5,
            size=self.n_ads,
        )

        # define click/imp and conversion/click rate function
        self.ctr = CTR(
            ad_feature_dim=self.ad_feature_dim,
            user_feature_dim=self.user_feature_dim,
            trend_interval=self.trend_interval,
            random_state=self.random_state,
        )
        self.cvr = CVR(
            ad_feature_dim=self.ad_feature_dim,
            user_feature_dim=self.user_feature_dim,
            trend_interval=self.trend_interval,
            random_state=self.random_state + 1,
        )

        # define impression difficulty on users
        # the more likely the users click, the higher the bid prices they have
        self.ks_coef = 1 + self.ads @ self.ctr.coef[: self.ad_feature_dim]

    @property
    def standard_bid_price(self):
        return self.standard_bid_price_distribution.mean

    def generate_auction(self, volume: int):
        """Sample ad and user pair for each auction.

        Parameters
        -------
        volume: int
            Total numbers of auction to generate.

        Returns
        -------
        ad_ids: NDArray[int], shape (volume, )
            IDs of the ads used for the auction bidding.

        user_ids: NDArray[int], shape (volume, )
            IDs of the users who receives the winning ads.

        """
        if not (isinstance(volume, int) and 0 <= volume):
            raise ValueError(
                f"volume must be a non-negative interger, but {volume} is given"
            )
        ad_ids = self.random_.choice(
            self.ad_ids,
            size=volume,
            p=self.ad_sampling_rate,
        )
        user_ids = self.random_.choice(
            self.user_ids,
            size=volume,
            p=self.user_sampling_rate,
        )
        return ad_ids, user_ids

    def map_idx_to_contexts(
        self, ad_ids: np.ndarray, user_ids: np.ndarray
    ) -> np.ndarray:
        """Map the ad and the user index into context vectors.

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
        contexts: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim)
            Context vector (contain both the ad and the user features) for each auction.

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
        if not (len(ad_ids) == len(user_ids) == len(bid_prices)):
            raise ValueError(
                "ad_ids, user_ids, contexts, and bid_prices must have same length"
            )

        ad_features = self.ads[ad_ids]
        user_features = self.users[user_ids]
        contexts = np.concatenate([ad_features, user_features], axis=1)

        return contexts

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
            raise ValueError(
                "ad_ids, user_ids, contexts, and bid_prices must have same length"
            )

        contexts = self.map_idx_to_contexts(ad_ids, user_ids)
        ks, thetas = self.wf_ks[ad_ids], self.wf_thetas[ad_ids]
        ks_coef = self.ks_coef[user_ids]

        impressions, winning_prices = self.winning_function.sample_outcome(
            ks * ks_coef, thetas, bid_prices
        )
        clicks = self.ctr.sample_outcome(timestep, contexts) * impressions
        conversions = self.cvr.sample_outcome(timestep, contexts) * clicks
        costs = winning_prices * clicks

        return costs, impressions, clicks, conversions
