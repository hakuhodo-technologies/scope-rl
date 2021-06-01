"""Mathematical Functions used in Real-Time Bidding (RTB) Simulation."""
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from sklearn.utils import check_random_state

from _gym.utils import sigmoid


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
    random_state: int, default=12345
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
        self,
        ks: np.ndarray,
        thetas: np.ndarray,
        bid_prices: np.ndarray,
    ) -> Tuple[np.ndarray]:
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
        if not (isinstance(ks, np.ndarray) and ks.ndim == 1 and ks.min() > 0):
            raise ValueError(
                "ks must be an 1-dimensional NDArray of positive float values"
            )
        if not (
            isinstance(thetas, np.ndarray) and thetas.ndim == 1 and thetas.min() > 0
        ):
            raise ValueError(
                "thetas must be an 1-dimensional NDArray of positive float values"
            )
        if not (
            isinstance(bid_prices, np.ndarray)
            and bid_prices.ndim == 1
            and bid_prices.min() >= 0
        ):
            raise ValueError(
                "bid_prices must be an 1-dimensional NDArray of non-negative integers"
            )
        if not (len(ks) == len(thetas) == len(bid_prices)):
            raise ValueError("ks, thetas, and bid_prices must have same length")

        winning_prices = np.clip(self.random_.gamma(shape=ks, scale=thetas), 1, None)
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
            loc=0.0, scale=0.5, size=self.ad_feature_dim + self.user_feature_dim
        )
        self.coef = self.coef / (
            self.ad_feature_dim + self.user_feature_dim
        )  # to normalize

        # define intermittent time_coef using trigonometric function
        n_wave = 10
        time_coef_weight = self.random_.beta(5, 20, size=n_wave)
        start_point = self.random_.uniform(size=n_wave)

        time_coef = np.zeros(self.trend_interval + 20)
        for i in range(10):
            time_coef += time_coef_weight[i] * (
                np.cos(
                    (
                        np.arange(self.trend_interval + 20) * (i + 1) * np.pi
                        + start_point[i] * 2 * np.pi
                    )
                    / self.trend_interval
                )
                + 1
            )

        start_idx = np.random.randint(5, 15)
        self.time_coef = time_coef[start_idx : start_idx + self.trend_interval] / n_wave

    def calc_prob(
        self, timestep: Union[int, np.ndarray], contexts: np.ndarray
    ) -> np.ndarray:
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

        contexts: Union[NDArray[int], NDArray[float]], shape (search_volume/n_samples, ad_feature_dim + user_feature_dim)
            Context vector (both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)
            (n_samples is determined in fit_reward_estimator function in simulator.)

        Returns
        -------
        ctrs: NDArray[float], shape (search_volume/n_samples, )
            Ground-truth CTR (i.e., click per impression) for each auction.

        """
        if not (isinstance(timestep, int) and timestep >= 0) and not (
            isinstance(timestep, np.ndarray)
            and timestep.ndim == 1
            and timestep.min() >= 0
        ):
            raise ValueError(
                "timestep must be an non-negative integer or an 1-dimensional NDArray of non-negative integers"
            )
        if not (
            isinstance(contexts, np.ndarray)
            and contexts.ndim == 2
            and contexts.shape[1] == self.ad_feature_dim + self.user_feature_dim
        ):
            raise ValueError(
                "contexts must have (ad_feature_dim + user_feature_dim) columns"
            )
        if not isinstance(timestep, int) and len(timestep) != len(contexts):
            raise ValueError("timestep and contexts must have same length")

        ctrs = (
            sigmoid(contexts @ self.coef.T)
            * self.time_coef[timestep % self.trend_interval].flatten()
        )
        return ctrs

    def sample_outcome(
        self, timestep: Union[int, np.ndarray], contexts: np.ndarray
    ) -> np.ndarray:
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
    """Class to calculate ground-truth CVR (i.e., conversion per click).

    Note
    -------
    We define two coefficient, context coefficient (coef) and time coefficient (time_coef).
    First, the value is calculated linearly from context vector and coef by inner product.
    Then, we multiply the value with time_coef and gain (ground-truth) CVR.

    In short, CVR is calculated as follows.
        CVR = (context @ coef) * time_coef, where @ denotes inner product.

    Parameters
    -------
    ad_feature_dim: int
        Dimensions of the ad feature vectors.

    user_feature_dim: int
        Dimensions of the user feature vectors.

    trend_interval: int
        Length of the CVR trend cycle.

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
            loc=0.0, scale=0.5, size=self.ad_feature_dim + self.user_feature_dim
        )
        self.coef = self.coef / (
            self.ad_feature_dim + self.user_feature_dim
        )  # to normalize

        # define intermittent time_coef using trigonometric function
        n_wave = 10
        time_coef_weight = self.random_.beta(25, 25, size=n_wave)
        start_point = self.random_.uniform(size=n_wave)

        time_coef = np.zeros(self.trend_interval + 20)
        for i in range(10):
            time_coef += time_coef_weight[i] * (
                np.cos(
                    (
                        np.arange(self.trend_interval + 20) * (i + 1) * np.pi
                        + start_point[i] * 2 * np.pi
                    )
                    / self.trend_interval
                )
                + 1
            )

        start_idx = np.random.randint(5, 15)
        self.time_coef = time_coef[start_idx : start_idx + self.trend_interval] / n_wave

    def calc_prob(
        self, timestep: Union[int, np.ndarray], contexts: np.ndarray
    ) -> np.ndarray:
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

        contexts: Union[NDArray[int], NDArray[float]], shape (search_volume/n_samples, ad_feature_dim + user_feature_dim)
            Context vector (both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)
            (n_samples is determined in fit_reward_estimator function in simulator.)

        Returns
        -------
        cvrs: NDArray[float], shape (search_volume/n_samples, )
            Ground-truth CVR (i.e., conversion per click) for each auction.

        """
        if not (isinstance(timestep, int) and timestep >= 0) and not (
            isinstance(timestep, np.ndarray)
            and timestep.ndim == 1
            and timestep.min() >= 0
        ):
            raise ValueError(
                "timestep must be an non-negative integer or an 1-dimensional NDArray of non-negative integers"
            )
        if not (
            isinstance(contexts, np.ndarray)
            and contexts.ndim == 2
            and contexts.shape[1] == self.ad_feature_dim + self.user_feature_dim
        ):
            raise ValueError(
                "contexts must have (ad_feature_dim + user_feature_dim) columns"
            )
        if not isinstance(timestep, int) and len(timestep) != len(contexts):
            raise ValueError("timestep and contexts must have same length")

        cvrs = (
            sigmoid(contexts @ self.coef.T)
            * self.time_coef[timestep % self.trend_interval].flatten()
        )
        return cvrs

    def sample_outcome(
        self, timestep: Union[int, np.ndarray], contexts: np.ndarray
    ) -> np.ndarray:
        """Stochastically determine if click occurs or not in click=True case.

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
            Whether click occurs in click=True case.

        """
        cvrs = self.calc_prob(timestep, contexts)
        conversions = self.random_.rand(len(contexts)) < cvrs
        return conversions.astype(int)
