# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Mathematical Functions used in Real-Time Bidding (RTB) Simulation."""
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .base import (
    BaseWinningPriceDistribution,
    BaseClickAndConversionRate,
)
from ...utils import NormalDistribution
from ...utils import sigmoid
from ...utils import check_array


@dataclass
class WinningPriceDistribution(BaseWinningPriceDistribution):
    """Class to sample the winning price (i.e., second price) and compare it with the given bid price.

    Imported as: :class:`rtbgym.envs.simulator.WinningDistribution`

    Note
    -------
    Winning price distribution follows gamma distribution.

    .. math::

        p(x) = x^{k-1} \\frac{\\exp(- x / \\theta)}{\\theta^k \\Gamma(k)},

    where :math:`\\Gamma(k) := (k-1)!` and :math:`k` and :math:`\\theta` are hyperparameters.

    Tip
    -------
    Use :class:`BaseWinningPriceDistribution` to define a custom WinningPriceDistribution.

    Parameters
    -------
    n_ads: int (> 0)
        Number of ads.

    n_users: int (> 0)
        Number of users. (This is for API consistency)

    ad_feature_dim: int (> 0)
        Dimension of the ad feature vectors. (This is for API consistency)

    user_feature_dim: int (> 0)
        Dimension of the user feature vectors. (This is for API consistency)

    step_per_episode: int (> 0)
        Length of the CTR trend cycle. (This is for API consistency)

    standard_bid_price_distribution: NormalDistribution, default=None
        Distribution of the bid price whose average impression probability is expected to be 0.5.

    minimum_standard_bid_price: {int, float}, default=None (> 0)
        Minimum value for standard bid price.
        If None, minimum_standard_bid_price is set to standard_bid_price_distribution.mean / 2.

    random_state: int, default=None (>= 0)
        Random state.

    References
    -------
    Wen-Yuan Zhu, Wen-Yueh Shih, Ying-Hsuan Lee, Wen-Chih Peng, and Jiun-Long Huang.
    "A Gamma-based Regression for Winning Price Estimation in Real-Time Bidding Advertising." 2017.

    """

    n_ads: int
    n_users: int
    ad_feature_dim: int
    user_feature_dim: int
    step_per_episode: int
    standard_bid_price_distribution: Optional[NormalDistribution] = (None,)
    minimum_standard_bid_price: Optional[Union[int, float]] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(self.n_ads, name="n_ads", target_type=int, min_val=1)
        if not isinstance(self.standard_bid_price_distribution, NormalDistribution):
            raise ValueError(
                "standard_bid_price_distribution must be a child class of NormalDistribution"
            )
        if self.minimum_standard_bid_price is None:
            self.minimum_standard_bid_price = (
                self.standard_bid_price_distribution.mean / 2
            )
        check_scalar(
            self.minimum_standard_bid_price,
            name="minimum_standard_bid_price",
            target_type=(int, float),
            min_val=0,
        )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        if self.standard_bid_price_distribution is None:
            self.standard_bid_price_distribution = NormalDistribution(
                mean=50,
                std=5,
                random_state=self.random_state,
            )
        standard_bid_prices = np.clip(
            self.standard_bid_price_distribution.sample(self.n_ads),
            self.minimum_standard_bid_price,
            None,
        )
        self.ks = self.random_.normal(
            loc=50,
            scale=5,
            size=self.n_ads,
        )
        self.thetas = self.random_.normal(
            loc=standard_bid_prices * 0.02,
            scale=(standard_bid_prices * 0.02) / 5,
            size=self.n_ads,
        )

    @property
    def standard_bid_price(self):
        return self.standard_bid_price_distribution.mean

    def sample_outcome(
        self,
        bid_prices: np.ndarray,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> Tuple[np.ndarray]:
        """Calculate impression probability for given bid price.

        Parameters
        -------
        bid_prices: array-like of shape (search_volume, )
            Bid price for each auction.

        ad_ids: array-like of shape (search_volume/n_samples, )
            Ad ids used for each auction. (This is for API consistency)

        user_ids: array-like of shape (search_volume/n_samples, )
            User ids used for each auction. (This is for API consistency)

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
        check_array(
            bid_prices,
            name="bid_prices",
            expected_dim=1,
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
        winning_prices = np.clip(
            self.random_.gamma(shape=self.ks[ad_ids], scale=self.thetas[ad_ids]),
            1,
            None,
        )
        impressions = winning_prices < bid_prices
        return impressions.astype(int), winning_prices.astype(int)


@dataclass
class ClickThroughRate(BaseClickAndConversionRate):
    """Class to calculate ground-truth CTR (i.e., click per impression).

    Imported as: :class:`rtbgym.envs.simulator.ClickThroughRate`

    Note
    -------
    We define two coefficient, context coefficient (`coef`) and time coefficient (`time_coef`).
    First, the value is calculated linearly from context vector and coef by inner product.
    Then, we multiply the value with `time_coef` and gain (ground-truth) CTR.

    In short, CTR is calculated as follows.
        CTR = (context @ coef) * time_coef, where @ denotes inner product.

    Tip
    -------
    Use :class:`BaseClickAndConversionRate` to define a custom ClickThroughRate.

    Parameters
    -------
    n_ads: int (> 0)
        Number of ads. (This is for API consistency)

    n_users: int (> 0)
        Number of users. (This is for API consistency)

    ad_feature_dim: int (> 0)
        Dimension of the ad feature vectors.

    user_feature_dim: int (> 0)
        Dimension of the user feature vectors.

    step_per_episode: int (> 0)
        Length of the CTR trend cycle.

    random_state: int, default=None (>= 0)
        Random state.

    """

    n_ads: int
    n_users: int
    ad_feature_dim: int
    user_feature_dim: int
    step_per_episode: int
    random_state: Optional[int] = None

    def __post_init__(self):
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
        check_scalar(
            self.step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        coef_dim = self.ad_feature_dim + self.user_feature_dim
        self.coef = self.random_.normal(loc=0.0, scale=0.5 / coef_dim, size=coef_dim)

        # define intermittent time_coef using trigonometric function
        n_wave = 10
        time_coef_weight = self.random_.beta(5, 20, size=n_wave)
        start_point = self.random_.uniform(size=n_wave)

        time_coef = np.zeros(self.step_per_episode + 20)
        for i in range(10):
            time_coef += time_coef_weight[i] * (
                np.cos(
                    (
                        np.arange(self.step_per_episode + 20) * (i + 1) * np.pi
                        + start_point[i] * 2 * np.pi
                    )
                    / self.step_per_episode
                )
                + 1
            )

        start_idx = self.random_.randint(5, 15)
        self.time_coef = (
            time_coef[start_idx : start_idx + self.step_per_episode] / n_wave
        )

    def calc_prob(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Calculate CTR (i.e., click per impression).

        Note
        -------
        CTR is calculated using both context coefficient (`coef`) and time coefficient (`time_coef`).
            CTR = (context @ coef) * time_coef, where @ denotes inner product.

        Parameters
        -------
        ad_ids: array-like of shape (search_volume/n_samples, )
            Ad ids used for each auction. (not used, but for API consistency)

        user_ids: array-like of shape (search_volume/n_samples, )
            User ids used for each auction. (not used, but for API consistency)

        ad_feature_vector: array-like of shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: array-like of shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        timestep: {int, array-like of shape (n_samples, )}
            Timestep in the RL environment.

        Returns
        -------
        ctrs: ndarray of shape (search_volume/n_samples, )
            Ground-truth CTR (i.e., click per impression) for each auction.

        """
        check_array(
            ad_feature_vector,
            name="ad_feature_vector",
            expected_dim=2,
        )
        check_array(
            user_feature_vector,
            name="user_feature_vector",
            expected_dim=2,
        )
        if ad_feature_vector.shape[1] != self.ad_feature_dim:
            raise ValueError(
                "Expected `ad_feature_dim.shape[1] == ad_feature_dim`, but found False"
            )
        if user_feature_vector.shape[1] != self.user_feature_dim:
            raise ValueError(
                "Expected `user_feature_dim.shape[1] == user_feature_dim`, but found False"
            )
        if ad_feature_vector.shape[0] != user_feature_vector.shape[0]:
            raise ValueError(
                "Expected ad_feature_dim and user_feature_dim must have the same length"
            )
        if not (isinstance(timestep, int) and timestep >= 0) and not (
            isinstance(timestep, np.ndarray)
            and np.issubsctype(timestep, int)
            and timestep.ndim == 1
            and timestep.min() >= 0
        ):
            raise ValueError(
                "timestep must be an non-negative integer or an 1-dimensional NDArray of non-negative integers"
            )
        contexts = np.concatenate([ad_feature_vector, user_feature_vector], axis=1)
        ctrs = sigmoid(contexts @ self.coef.T) * self.time_coef[timestep].flatten()
        return ctrs

    def sample_outcome(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Stochastically determine whether click occurs in impression=True case.

        Parameters
        -------
        ad_ids: array-like of shape (search_volume/n_samples, )
            Ad ids used for each auction. (not used, but for API consistency)

        user_ids: array-like of shape (search_volume/n_samples, )
            User ids used for each auction. (not used, but for API consistency)

        ad_feature_vector: array-like of shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: array-like of shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        timestep: {int, array-like of shape (n_samples, )}
            Timestep in the RL environment.

        Returns
        -------
        clicks: array-like of shape (search_volume/n_samples, )
            Whether click occurs when impression=True.

        """
        ctrs = self.calc_prob(
            timestep=timestep,
            ad_ids=ad_ids,
            user_ids=user_ids,
            ad_feature_vector=ad_feature_vector,
            user_feature_vector=user_feature_vector,
        )
        clicks = self.random_.rand(len(ad_ids)) < ctrs
        return clicks.astype(int)


@dataclass
class ConversionRate(BaseClickAndConversionRate):
    """Class to calculate ground-truth CVR (i.e., conversion per click).

    Imported as: :class:`rtbgym.envs.simulator.ConversionRate`

    Note
    -------
    We define two coefficient, context coefficient (`coef`) and time coefficient (`time_coef`).
    First, the value is calculated linearly from context vector and coef by inner product.
    Then, we multiply the value with `time_coef` and gain (ground-truth) CVR.

    In short, CVR is calculated as follows.
        CVR = (context @ coef) * time_coef, where @ denotes inner product.

    Tip
    -------
    Use :class:`BaseClickAndConversionRate` to define a custom ConversionRate.

    Parameters
    -------
    n_ads: int (> 0)
        Number of ads. (This is for API consistency)

    n_users: int (> 0)
        Number of users. (This is for API consistency)

    ad_feature_dim: int (> 0)
        Dimension of the ad feature vectors.

    user_feature_dim: int (> 0)
        Dimension of the user feature vectors.

    step_per_episode: int (> 0)
        Length of the CVR trend cycle.

    random_state: int, default=None (>= 0)
        Random state.

    """

    n_ads: int
    n_users: int
    ad_feature_dim: int
    user_feature_dim: int
    step_per_episode: int
    random_state: Optional[int] = None

    def __post_init__(self):
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
        check_scalar(
            self.step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        coef_dim = self.ad_feature_dim + self.user_feature_dim
        self.coef = self.random_.normal(loc=0.0, scale=0.5 / coef_dim, size=coef_dim)

        # define intermittent time_coef using trigonometric function
        n_wave = 10
        time_coef_weight = self.random_.beta(10, 15, size=n_wave)
        start_point = self.random_.uniform(size=n_wave)

        time_coef = np.zeros(self.step_per_episode + 20)
        for i in range(10):
            time_coef += time_coef_weight[i] * (
                np.cos(
                    (
                        np.arange(self.step_per_episode + 20) * (i + 1) * np.pi
                        + start_point[i] * 2 * np.pi
                    )
                    / self.step_per_episode
                )
                + 1
            )

        start_idx = self.random_.randint(5, 15)
        self.time_coef = (
            time_coef[start_idx : start_idx + self.step_per_episode] / n_wave
        )

    def calc_prob(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Calculate CVR (i.e., conversion per click) using context vectors.

        Note
        -------
        CVR is calculated using both context coefficient (`coef`) and time coefficient (`time_coef`).
            CVR = (context @ coef) * time_coef, where @ denotes inner product.


        Parameters
        -------
        ad_ids: array-like of shape (search_volume/n_samples, )
            Ad ids used for each auction. (not used, but for API consistency)

        user_ids: array-like of shape (search_volume/n_samples, )
            User ids used for each auction. (not used, but for API consistency)

        ad_feature_vector: array-like of shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: array-like of shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        timestep: {int, array-like of shape (n_samples, )}
            Timestep in the RL environment.

        Returns
        -------
        cvrs: ndarray of shape (search_volume/n_samples, )
            Ground-truth CVR (i.e., conversion per click) for each auction.

        """
        check_array(
            ad_feature_vector,
            name="ad_feature_vector",
            expected_dim=2,
        )
        check_array(
            user_feature_vector,
            name="user_feature_vector",
            expected_dim=2,
        )
        if ad_feature_vector.shape[1] != self.ad_feature_dim:
            raise ValueError(
                "Expected `ad_feature_dim.shape[1] == ad_feature_dim`, but found False"
            )
        if user_feature_vector.shape[1] != self.user_feature_dim:
            raise ValueError(
                "Expected `user_feature_dim.shape[1] == user_feature_dim`, but found False"
            )
        if ad_feature_vector.shape[0] != user_feature_vector.shape[0]:
            raise ValueError(
                "Expected ad_feature_dim and user_feature_dim must have the same length"
            )
        if not (isinstance(timestep, int) and timestep >= 0) and not (
            isinstance(timestep, np.ndarray)
            and np.issubsctype(timestep, int)
            and timestep.ndim == 1
            and timestep.min() >= 0
        ):
            raise ValueError(
                "timestep must be an non-negative integer or an 1-dimensional NDArray of non-negative integers"
            )
        contexts = np.concatenate([ad_feature_vector, user_feature_vector], axis=1)
        cvrs = sigmoid(contexts @ self.coef.T) * self.time_coef[timestep].flatten()
        return cvrs

    def sample_outcome(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Stochastically determine whether conversion occurs in click=True case.

        Parameters
        -------
        ad_ids: array-like of shape (search_volume/n_samples, )
            Ad ids used for each auction. (not used, but for API consistency)

        user_ids: array-like of shape (search_volume/n_samples, )
            User ids used for each auction. (not used, but for API consistency)

        ad_feature_vector: array-like of shape (search_volume/n_samples, ad_feature_dim)
            Ad feature vector for each auction.

        user_feature_vector: array-like of shape (search_volume/n_samples, user_feature_dim)
            User feature vector for each auction.

        timestep: {int, array-like of shape (n_samples, )}
            Timestep in the RL environment.

        Returns
        -------
        conversions: ndarray of shape (search_volume/n_samples, )
            Whether conversion occurs when click=True.

        """
        cvrs = self.calc_prob(
            ad_ids=ad_ids,
            user_ids=user_ids,
            ad_feature_vector=ad_feature_vector,
            user_feature_vector=user_feature_vector,
            timestep=timestep,
        )
        conversions = self.random_.rand(len(ad_ids)) < cvrs
        return conversions.astype(int)
