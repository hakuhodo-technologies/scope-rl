# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Synthetic Bidding Auction Simulation."""
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .base import (
    BaseSimulator,
    BaseWinningPriceDistribution,
    BaseClickAndConversionRate,
)
from .function import (  # noqa: F401
    WinningPriceDistribution,
    ClickThroughRate,
    ConversionRate,
)
from ...utils import NormalDistribution, check_array
from ...types import Numeric


@dataclass
class RTBSyntheticSimulator(BaseSimulator):
    """Class to calculate the outcome probability and stochastically determine auction result in Real-Time Bidding (RTB) setting for display advertising.

    Imported as: :class:`rtbgym.envs.simulator.RTBSyntheticSimulator`

    Parameters
    -------
    cost_indicator: {"impression", "click", "conversion"}, default="click"
        Defines when the cost arises.

    step_per_episode: int, default=7 (> 0)
        Number of timesteps in an episode.

    n_ads: int, default=100 (> 0)
        Number of (candidate) ads used for auction bidding.

    n_users: int, default=100 (> 0)
        Number of (candidate) users used for auction bidding.

    ad_feature_vector: array-like of shape (n_ads, ad_feature_dim), default=None
        Feature vectors that characterizes each ad.

    user_feature_vector: array-like of shape (n_users, user_feature_dim), default=None
        Feature vectors that characterizes each user.

    ad_sampling_rate: array-like of shape (step_per_episode, n_ads) or (n_ads, ), default=None
        Sampling probabilities to determine which ad (id) is used in each auction.

    user_sampling_rate: array-like of shape (step_per_episode, n_users) or (n_uses, ), default=None
        Sampling probabilities to determine which user (id) is used in each auction.

    WinningPriceDistribution: BaseWinningPriceDistribution
        Winning price distribution of auctions.
        Both class and instance are acceptable.

    ClickThroughRate: BaseClickAndConversionRate
        Click through rate (i.e., click / impression).
        Both class and instance are acceptable.

    ConversionRate: BaseClickAndConversionRate
        Conversion rate (i.e., conversion / click).
        Both class and instance are acceptable.

    standard_bid_price_distribution: NormalDistribution, default=None
        Distribution of the bid price whose average impression probability is expected to be 0.5.

    minimum_standard_bid_price: int, default=None (> 0)
        Minimum value for standard bid price.
        If `None`, minimum_standard_bid_price is set to :class:`standard_bid_price_distribution.mean / 2`.

    search_volume_distribution: NormalDistribution, default=None
        Search volume distribution for each timestep.

    minimum_search_volume: int, default = 10 (> 0)
        Minimum search volume at each timestep.

    random_state: int, default=None (>= 0)
        Random state.

    References
    -------
    Di Wu, Xiujun Chen, Xun Yang, Hao Wang, Qing Tan, Xiaoxun Zhang, Jian Xu, and Kun Gai.
    "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising." 2018.

    Jun Zhao, Guang Qiu, Ziyu Guan, Wei Zhao, and Xiaofei He.
    "Deep Reinforcement Learning for Sponsored Search Real-time Bidding." 2018.

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
    WinningPriceDistribution: BaseWinningPriceDistribution = (  # noqa: F811
        WinningPriceDistribution
    )
    ClickThroughRate: BaseClickAndConversionRate = ClickThroughRate  # noqa: F811
    ConversionRate: BaseClickAndConversionRate = ConversionRate  # noqa: F811
    standard_bid_price_distribution: Optional[NormalDistribution] = (None,)
    minimum_standard_bid_price: Optional[Union[int, float]] = None
    search_volume_distribution: Optional[NormalDistribution] = (None,)
    minimum_search_volume: int = 10
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.cost_indicator not in ["impression", "click", "conversion"]:
            raise ValueError(
                f"cost_indicator must be 'impression', 'click', or 'conversion', but {self.click_indicator} is given"
            )
        check_scalar(
            self.step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )

        check_scalar(
            self.n_ads,
            name="n_ads",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            self.n_users,
            name="n_users",
            target_type=int,
            min_val=1,
        )
        self.ad_ids = np.arange(self.n_ads)
        self.user_ids = np.arange(self.n_users)

        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        check_scalar(
            self.ad_feature_dim,
            name="ad_feature_dim",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            self.user_feature_dim,
            name="user_feature_dim",
            target_type=int,
            min_val=1,
        )
        if self.ad_feature_vector is None:
            self.ad_feature_vector = self.random_.normal(
                size=(self.n_ads, self.ad_feature_dim)
            )
        check_array(
            self.ad_feature_vector,
            name="ad_feature_vector",
            expected_dim=2,
        )
        if self.ad_feature_vector.shape != (self.n_ads, self.ad_feature_dim):
            raise ValueError(
                "The shape of ad_feature_vector must be (n_ads, ad_feature_dim)"
            )
        if self.user_feature_vector is None:
            self.user_feature_vector = self.random_.normal(
                size=(self.n_users, self.user_feature_dim)
            )
        check_array(
            self.user_feature_vector,
            name="user_feature_vector",
            expected_dim=2,
        )
        if self.user_feature_vector.shape != (self.n_users, self.user_feature_dim):
            raise ValueError(
                "The shape of user_feature_vector must be (n_users, user_feature_dim)"
            )

        if self.ad_sampling_rate is None:
            self.ad_sampling_rate = np.full(
                (self.step_per_episode, self.n_ads), 1 / self.n_ads
            )
        elif isinstance(self.ad_sampling_rate, np.ndarray):
            if self.ad_sampling_rate.ndim == 1:
                self.ad_sampling_rate = np.tile(
                    self.ad_sampling_rate / self.ad_sampling_rate.sum(),
                    (self.step_per_episode, 1),
                )
            else:
                self.ad_sampling_rate = (
                    self.ad_sampling_rate
                    / np.tile(np.sum(self.ad_sampling_rate, axis=1), (self.n_ads, 1)).T
                )
        check_array(
            self.ad_sampling_rate,
            name="ad_sampling_rate",
            expected_dim=2,
            min_val=0,
        )
        if self.ad_sampling_rate.shape != (self.step_per_episode, self.n_ads):
            raise ValueError(
                "The shape of ad_sampling_rate must be (step_per_episode, n_ads)"
            )
        if not np.allclose(
            self.ad_sampling_rate.sum(axis=1), np.ones(self.step_per_episode)
        ):
            raise ValueError(
                "Expected `ad_sampling_rate.sum(axis=1) == np.ones(step_per_episode)`, but found False"
            )
        if self.user_sampling_rate is None:
            self.user_sampling_rate = np.full(
                (self.step_per_episode, self.n_users), 1 / self.n_users
            )
        elif isinstance(self.user_sampling_rate, np.ndarray):
            if self.user_sampling_rate.ndim == 1:
                self.user_sampling_rate = np.tile(
                    self.user_sampling_rate / self.user_sampling_rate.sum(),
                    (self.step_per_episode, 1),
                )
            else:
                self.user_sampling_rate = (
                    self.user_sampling_rate
                    / np.tile(
                        np.sum(self.user_sampling_rate, axis=1), (self.n_users, 1)
                    ).T
                )
        check_array(
            self.user_sampling_rate,
            name="user_sampling_rate",
            expected_dim=2,
        )
        if self.user_sampling_rate.shape != (self.step_per_episode, self.n_users):
            raise ValueError(
                "The shape of user_sampling_rate must be (step_per_episode, n_users)"
            )
        if not np.allclose(
            self.user_sampling_rate.sum(axis=1), np.ones(self.step_per_episode)
        ):
            raise ValueError(
                "Expected `user_sampling_rate.sum(axis=1) == np.ones(step_per_episode)`, but found False"
            )

        if self.standard_bid_price_distribution is None:
            self.standard_bid_price_distribution = NormalDistribution(
                mean=50,
                std=5,
                random_state=self.random_state,
            )
        if not isinstance(self.standard_bid_price_distribution, NormalDistribution):
            raise ValueError(
                "standard_bid_price_distribution must be a NormalDistribution"
            )
        if not isinstance(self.standard_bid_price_distribution.mean, Numeric):
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

        if self.search_volume_distribution is None:
            self.search_volume_distribution = NormalDistribution(
                mean=200,
                std=20,
                random_state=self.random_state,
            )
        if isinstance(self.search_volume_distribution.mean, Numeric):
            self.search_volume_distribution = NormalDistribution(
                mean=np.full(
                    self.step_per_episode, self.search_volume_distribution.mean
                ),
                std=np.full(self.step_per_episode, self.search_volume_distribution.std),
                random_state=self.random_state,
            )
        check_array(
            self.search_volume_distribution.mean,
            name="search_volume_distribution.mean",
            expected_dim=1,
            min_val=0,
        )

        if not (
            isinstance(self.search_volume_distribution.mean, (int, float))
            or len(self.search_volume_distribution.mean) == self.step_per_episode
        ):
            raise ValueError(
                "length of search_volume_distribution must be equal to step_per_episode"
            )
        check_scalar(
            self.minimum_search_volume,
            name="minimum_search_volume",
            target_type=int,
            min_val=1,
        )

        # define winning function
        if isinstance(self.WinningPriceDistribution, BaseWinningPriceDistribution):
            self.winning_price_distribution = self.WinningPriceDistribution
        elif issubclass(self.WinningPriceDistribution, BaseWinningPriceDistribution):
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
        else:
            raise ValueError(
                "WinningPriceDistribution must be a child class of BaseWinningPriceDistribution"
            )

        # define click/imp and conversion/click rate function
        if isinstance(self.ClickThroughRate, BaseClickAndConversionRate):
            self.ctr = self.ClickThroughRate
        elif issubclass(self.ClickThroughRate, BaseClickAndConversionRate):
            self.ctr = self.ClickThroughRate(
                n_ads=self.n_ads,
                n_users=self.n_users,
                ad_feature_dim=self.ad_feature_dim,
                user_feature_dim=self.user_feature_dim,
                step_per_episode=self.step_per_episode,
                random_state=self.random_state,
            )
        else:
            raise ValueError(
                "ClickThroughRate must be a child class of BaseClickAndConversionRate"
            )

        if isinstance(self.ConversionRate, BaseClickAndConversionRate):
            self.cvr = self.ConversionRate
        elif issubclass(self.ConversionRate, BaseClickAndConversionRate):
            self.cvr = self.ConversionRate(
                n_ads=self.n_ads,
                n_users=self.n_users,
                ad_feature_dim=self.ad_feature_dim,
                user_feature_dim=self.user_feature_dim,
                step_per_episode=self.step_per_episode,
                random_state=self.random_state
                + 1,  # to differentiate the coef with that of CTR
            )
        else:
            raise ValueError(
                "ConversionRate must be a child class of BaseClickAndConversionRate"
            )

    @property
    def standard_bid_price(self):
        return self.winning_price_distribution.standard_bid_price

    def generate_auction(
        self, volume: Optional[int] = None, timestep: Optional[int] = None
    ):
        """Sample ad and user pair for each auction.

        Parameters
        -------
        volume: int, default=None (> 0)
            Total number of auctions to generate.

        timestep: int, default=None (> 0)
            Timestep in the RL environment.

        Returns
        -------
        ad_ids: ndarray of shape (volume, )
            IDs of the ads.

        user_ids: ndarray of shape (volume, )
            IDs of the users.

        """
        # stochastically determine search volume
        if volume is None:
            if timestep is None:
                volume = np.clip(
                    self.search_volume_distribution.sample(),
                    self.minimum_search_volume,
                    None,
                ).astype(int)[0][0]

            else:
                volume = np.clip(
                    self.search_volume_distribution.sample(),
                    self.minimum_search_volume,
                    None,
                ).astype(int)[0][timestep - 1]
        check_scalar(
            volume,
            name="volume",
            target_type=(int, np.integer),
            min_val=1,
        )

        if timestep is not None:
            check_scalar(timestep, name="timestep", target_type=int, min_val=0)
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
        check_array(
            ad_ids,
            name="ad_ids",
            expected_dim=1,
            expected_dtype=int,
            min_val=0,
            max_val=self.n_ads - 1,
        )
        check_array(
            user_ids,
            name="user_ids",
            expected_dim=1,
            expected_dtype=int,
            min_val=0,
            max_val=self.n_users - 1,
        )
        if ad_ids.shape[0] != user_ids.shape[0]:
            raise ValueError("ad_ids and user_ids must have the same length")

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
        """Simulate bidding auction for given queries. (Calculate outcome probability and stochastically determine auction result.)

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
            (search_volume is determined in RL environment.)

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
        check_scalar(
            timestep,
            name="timestep",
            target_type=int,
            min_val=0,
        )
        check_array(
            ad_ids,
            name="ad_ids",
            expected_dim=1,
            expected_dtype=int,
            min_val=0,
            max_val=self.n_ads - 1,
        )
        check_array(
            user_ids,
            name="user_ids",
            expected_dim=1,
            expected_dtype=int,
            min_val=0,
            max_val=self.n_users - 1,
        )
        if ad_ids.shape[0] != user_ids.shape[0]:
            raise ValueError("ad_ids and user_ids must have the same length")
        check_array(
            bid_prices,
            name="bid_prices",
            expected_dim=1,
            min_val=0,
        )

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
