"""Mathematical Functions used in Real-Time Bidding (RTB) Simulation."""
from dataclasses import dataclass
from typing import Tuple, Union
from nptyping import NDArray

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class WinningFunction:
    """Class to sample the winning price (i.e., second price) and compare with the given bid price.

    Note
    -------
    Winning price distribution follows gamma distribution.

    .. math::

        p(x) = x^{k-1} \\frac{\\mathrm{e}^{- x / \\theta}}{\\theta^k \\Gamma(k)},

    where :math:`\\Gamma(k) := (k-1)!` and :math:`k` and :math:`\\theta` are hyperparameters.

    Parameters
    -------
    random_state: int = 12345
        Random state.

    References
    -------
    Wen-Yuan Zhu, Wen-Yueh Shih, Ying-Hsuan Lee, Wen-Chih Peng, and  Jiun-Long Huang.
    "A Gamma-based Regression for Winning Price Estimation in Real-Time Bidding Advertising.", 2017.

    """
    random_state: int = 12345

    def __post_init__(self):
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def sample_outcome(
        self, ks: NDArray[float], thetas: NDArray[float], bid_prices: NDArray[int]
    ) -> Tuple[NDArray[int]]:
        """Calculate impression probability for given bid price.

        Parameters
        -------
        ks: NDArray[int], shape (search_volume, )
            Pre-defined shape hyperparameter for winning price (gamma) distribution for each ad.
            (search_volume is determined in RL environment.)

        thetas: NDArray[int], shape (search_volume, )
            Pre-defined scale hyperparameter for winning price (gamma) distribution for each ad.
            (search_volume is determined in RL environment.)

        bid_prices: NDArray[int], shape (search_volume, )
            Bid price for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        impressions: NDArray[int], shape (search_volume, )
            Whether impression occurred for each auction.

        winning_prices: NDArray[int], shape (search_volume, )
            Sampled winning price for each auction.

        """
        if not (isinstance(ks, NDArray[float]) and ks.min() > 0):
            raise ValueError("ks must be an NDArray of positive float values")
        if not (isinstance(thetas, NDArray[float]) and thetas.min() > 0):
            raise ValueError("thetas must be an NDArray of positive float values")
        if not (isinstance(bid_prices, NDArray[int]) and bid_prices.min() >= 0):
            raise ValueError("bid_prices must be an NDArray of non-negative integers")

        winning_prices = self.random_.gamma(shape=ks, scale=thetas)
        impressions = winning_prices < bid_prices

        return impressions.astype(int), winning_prices.astype(int)


@dataclass
class CTR:
    """Class to calculate ground-truth CTR (i.e., click per impression).

    Note
    -------
    We define two coefficient, context coefficient (coef) and time coefficient (time_coef).
    First, the value is calculated linearly from context vector and coef by inner product.
    Then, we multiply the value with time_coef and gain (ground-truth) CTR.

    In short, CTR is calculated as follows.
        CTR = (context @ coef) * time_coef, where @ denotes inner product.

    Parameters
    -------
    ad_feature_dim: int
        Dimensions of the ad feature vectors.

    user_feature_dim: int
        Dimensions of the user feature vectors.

    trend_interval: int
        Length of the CTR trend cycle.

    random_state: int, default=12345
        Random state.

    """
    ad_feature_dim: int
    user_feature_dim: int
    trend_interval: int
    random_state: int = 12345

    def __post_init__(self):
        if not (isinstance(self.ad_feature_dim, int) and self.ad_feature_dim > 0):
            raise ValueError(
                f"ad_feature_dim must be a positive interger, but {self.ad_feature_dim} is given"
            )
        if not (isinstance(self.user_feature_dim, int) and self.user_feature_dim > 0):
            raise ValueError(
                f"user_feature_dim must be a positive interger, but {self.user_feature_dim} is given"
            )
        if not (isinstance(self.trend_interval, int) and self.trend_interval > 0):
            raise ValueError(
                f"trend_interval must be a positive interger, but {self.trend_interval} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        self.coef = self.random_.normal(
            loc=0.5, scale=0.1, size=self.ad_feature_dim + self.user_feature_dim
        )
        self.coef = self.coef / (
            self.ad_feature_dim + self.user_feature_dim
        )  # to normalize

        self.time_coef = self.random_.beta(25, 25, size=self.trend_interval)
        self.time_coef = np.convolve(self.time_coef, np.ones(3) / 3, mode="same")

    def calc_prob(
        self, timestep: Union[int, NDArray[int]], contexts: NDArray[float]
    ) -> NDArray[float]:
        """Calculate CTR (i.e., click per impression) using context vectors.

        Note
        -------
        CTR is calculated using both context coefficient (coef) and time coefficient (time_coef).
            CTR = (context @ coef) * time_coef, where @ denotes inner product.


        Parameters
        -------
        timestep: Union[int, NDArray[int]], shape None/(n_samples, )
            Timestep of the RL environment.
            (n_samples is determined in fit_reward_estimator function in simulator.)

        contexts: NDArray[float], shape (search_volume/n_samples, ad_feature_dim + user_feature_dim)
            Context vector (both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)
            (n_samples is determined in fit_reward_estimator function in simulator.)

        Returns
        -------
        ctrs: NDArray[float], shape (search_volume/n_samples, )
            Ground-truth CTR (i.e., click per impression) for each auction.

        """
        if not (
            (isinstance(timestep, int) and timestep >= 0)
            or (isinstance(timestep, NDArray[int]) and timestep.min() >= 0)
        ):
            raise ValueError(
                "timestep must be non negative integer or an NDArray of non negative integers"
            )
        if not isinstance(contexts, NDArray[float]):
            raise ValueError("contexts must be an NDArray of float values")

        ctrs = (contexts @ self.coef.T) * self.time_coef[
            timestep % self.trend_interval
        ].flatten()
        return ctrs

    def sample_outcome(
        self, timestep: Union[int, NDArray[int]], contexts: NDArray[float]
    ) -> NDArray[int]:
        """Stochastically determine if click occurs or not in impression=True case.

        Parameters
        -------
        timestep: Union[int, NDArray[int]], shape None/(n_samples, )
            Timestep of the RL environment.
            (n_samples is determined in fit_reward_estimator function in simulator.)

        contexts: NDArray[float], shape (search_volume/n_samples, ad_feature_dim + user_feature_dim)
            Context vector (both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)
            (n_samples is determined in fit_reward_estimator function in simulator.)

        Returns
        -------
        clicks: NDArray[int], shape (search_volume/n_samples, )
            Whether click occurs in impression=True case.

        """
        ctrs = self.calc_prob(timestep, contexts)
        clicks = self.random_.rand(len(contexts)) < ctrs
        return clicks.astype(int)


@dataclass
class CVR:
    """
    Class to calculate ground-truth CVR (i.e., conversion per click).

    Note
    -------
    We define two coefficient, context coefficient (coef) and time coefficient (time_coef).
    First, the value is calculated linearly from context vector and coef by inner product.
    Then, we multiply the value with time_coef and gain (ground-truth) CVR.

    In short, CVR is calculated as follows.
        CVR = (context @ coef) * time_coef, where @ denotes inner product.

    To make correlation with CTR, we define coef of CVR by adding residuals sampled
    from normal distribution to that of CTR.

    Parameters
    -------
    ctr: CTR
        Pre-defined CTR function.

    """
    ctr: CTR

    def __post_init__(self):
        """
        # fix later, assertion fails.
        if not isinstance(self.ctr, CTR):
            print(type(self.ctr))  # <class '_gym.simulator.function.CTR'>
            raise ValueError("ctr must be the CTR or a child class of the CTR")
        """
        self.ad_feature_dim = self.ctr.ad_feature_dim
        self.user_feature_dim = self.ctr.user_feature_dim
        self.trend_interval = self.ctr.trend_interval
        self.random_ = self.ctr.random_

        residuals = self.random_.normal(
            loc=0.0, scale=0.1, size=self.ad_feature_dim + self.user_feature_dim
        )
        self.coef = self.ctr.coef + residuals

        self.time_coef = self.random_.beta(40, 10, size=self.trend_interval)
        self.time_coef = np.convolve(self.time_coef, np.ones(3) / 3, mode="same")

    def calc_prob(
        self, timestep: Union[int, NDArray[int]], contexts: NDArray[float]
    ) -> NDArray[float]:
        """Calculate CVR (i.e., conversion per click) using context vectors.

        Note
        -------
        CVR is calculated using both context coefficient (coef) and time coefficient (time_coef).
            CVR = (context @ coef) * time_coef, where @ denotes inner product.

        Parameters
        -------
        timestep: Union[int, NDArray[int]], shape None/(n_samples, )
            Timestep of the RL environment.
            (n_samples is determined in fit_reward_estimator function in simulator.)

        contexts: NDArray[float], shape (search_volume/n_samples, ad_feature_dim + user_feature_dim)
            Context vector (both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)
            (n_samples is determined in fit_reward_estimator function in simulator.)

        Returns
        -------
        cvrs: NDArray[float], shape (search_volume/n_samples, )
            Ground-truth CVR (i.e., conversion per click) for each auction.

        """
        if not (
            (isinstance(timestep, int) and timestep >= 0)
            or (isinstance(timestep, NDArray[int]) and timestep.min() >= 0)
        ):
            raise ValueError(
                "timestep must be non negative integer or an NDArray of non negative integers"
            )
        if not isinstance(contexts, NDArray[float]):
            raise ValueError("contexts must be an NDArray of float values")

        cvrs = (contexts @ self.coef.T) * self.time_coef[
            timestep % self.trend_interval
        ].flatten()
        return cvrs

    def sample_outcome(
        self, timestep: Union[int, NDArray[int]], contexts: NDArray[float]
    ) -> NDArray[int]:
        """Stochastically determine if conversion occurs or not in click=True case.

        Parameters
        -------
        timestep: Union[int, NDArray[int]], shape None/(n_samples, )
            Timestep of the RL environment.
            (n_samples is determined in fit_reward_estimator function in simulator.)

        contexts: NDArray[float], shape (search_volume/n_samples, ad_feature_dim + user_feature_dim)
            Context vector (both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)
            (n_samples is determined in fit_reward_estimator function in simulator.)

        Returns
        -------
        conversions: NDArray[int], shape (search_volume/n_samples, )
            Whether conversion occurs in click=True case.

        """
        cvrs = self.calc_prob(timestep, contexts)
        conversions = self.random_.rand(len(contexts)) < cvrs
        return conversions.astype(int)
