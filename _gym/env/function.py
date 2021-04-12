from dataclasses import dataclass
from typing import Union
from nptyping import NDArray

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class NormalDistribution:
    """Class to sample from normal distribution.

    Parameters
    -------
    mean: float.
        Mean value of the normal distribution.

    std: float.
        Standard deviation of the normal distribution.

    random_state: int, default=12345.
        Random state.

    """

    mean: float
    std: float
    random_state: int = 12345.0

    def __post_init__(self):
        if not isinstance(self.mean, float):
            raise ValueError(f"mean must be a float number, but {self.mean} is given")
        if not isinstance(self.std, float):
            raise ValueError("std must be a float number, but {self.std} is given")
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def sample(self, size: int = 1) -> NDArray[float]:
        """Sample random variables from the pre-determined normal distribution.

        Parameters
        -------
        size: int, default=1.
            Total numbers of the random variable to sample.

        Returns
        -------
        random_variables: NDArray[float], shape (size, ).
            Random variables sampled from the normal distribution.

        """
        if not (isinstance(size, int) and size > 0):
            raise ValueError(f"size must be a positive integer, but {size} is given")
        return self.random_.normal(loc=self.mean, scale=self.std, size=size)


@dataclass
class WinningFunction:
    """Class to calculate auction winning rate for given bid price.

    Note
    -------
    Calculate auction winning rate (i.e., impression probability) as follows.
        auction winning rate = (bid_price ** alpha) / (const + bid_price ** alpha),
        where alpha and const is parameters to determine the shape of winning function.

    Parameters
    -------
    alpha: float, default=2.0.
        Exponential coefficient parameter of the winning function.

    References
    -------

    """

    alpha: float = 2.0

    def __post_init__(self):
        if not (isinstance(self.alpha, float) and self.alpha > 0):
            raise ValueError(
                f"alpha must be a positive float number, but {self.alpha} is given"
            )

    def calc_prob(
        self, consts: NDArray[float], bid_prices: NDArray[int]
    ) -> NDArray[float]:
        """Calculate impression probability for given bid price.

        Parameters
        -------
        consts: NDArray[float], shape (search_volume, ).
            Parameter of the winning price function for each ad.
            (search_volume is determined in reinforcement learning (RL) environment.)

        bid_prices: NDArray[int], shape (search_volume, ).
            Bid price for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        impression_probabilities: NDArray[float], shape (search_volume, ).
            Auction winning probability for each auction.

        """
        if not (isinstance(consts, NDArray[float]) and consts.min() > 0):
            raise ValueError("consts must be an NDArray of positive float values")
        if not (isinstance(bid_prices, NDArray[int]) and bid_prices.min() > 0):
            raise ValueError("bid_prices must be an NDArray of positive integers")
        return (bid_prices ** self.alpha) / (consts + bid_prices ** self.alpha)


@dataclass
class SecondPrice:  # fix later
    """Class to stochastically determine second price.

    Note
    -------
    fix later


    Parameters
    -------
    n_dices: int.
        Number of assumed participant to the auction.

    random_state: int, default=12345.
        Random state.

    References
    -------

    """

    n_dices: int
    random_state: int = 12345

    def __post_init__(self):
        if not (isinstance(self.n_dices, int) and self.n_dices > 0):
            raise ValueError(
                f"n_dices must be a positive interger, but {self.n_dices} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def sample(self, consts: NDArray[float], probs: NDArray[int]) -> NDArray[int]:
        """Sample second price for each bid.

        Parameters
        -------
        consts: NDArray[float], shape (search_volume, ).
            Parameter of the winning price function for each ad.
            (search_volume is determined in RL environment.)

        probs: NDArray[int], shape (search_volume, ).
            Auction winning probability for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        second_prices: NDArray[int], shape (search_volume, ).
            Second price for each auction.

        """
        if not (isinstance(consts, NDArray[float]) and consts.min() > 0):
            raise ValueError("consts must be an NDArray of positive float values")
        if not (isinstance(probs, NDArray[float]) and probs.min() > 0):
            raise ValueError("probs must be an NDArray of positive float values")
        discounts = self.random_.rand((len(consts), self.n_dices)).max(axis=1)

        return self._inverse_winning_function(consts, probs * discounts).astype(int)

    def _inverse_winning_function(
        consts: NDArray[float], probs: NDArray[float]
    ) -> NDArray[float]:
        """Calculate second price for given auction winning probability using inverse winning function.

        Parameters
        -------
        consts: NDArray[float], shape (search_volume, ).
            Parameter of the winning price function for each ad.
            (search_volume is determined in RL environment.)

        probs: NDArray[int], shape (search_volume, ).
            Auction winning probability of the second place for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        second_prices: NDArray[float], shape (search_volume, ).
            Second price for each auction.

        """
        return (probs * consts) / (1 - probs)


@dataclass
class CTR:
    """Class to calculate ground-truth CTR (i.e., click per impression).

    Note
    -------
    We define two coefficient, context coefficient (coef) and time coefficient (time_coef).
    First, the value is calculated linearly from context vector and coef by inner product.
    Then, we multiply the value with time_coef and gain (ground-truth) CTR.

    Parameters
    -------
    ad_feature_dim: int.
        Dimentions of the ad feature vectors.

    user_feature_dim: int.
        Dimentions of the user feature vectors.

    trend_interval: int.
        Length of the CTR trend cycle.

    random_state: int, default=12345.
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

        Parameters
        -------
        timestep: Union[int, NDArray[int]], shape None/(n_samples, ).
            Timestep of the RL environment.
            (n_samples is determined in fit_reward_estimator function in simulator.)

        contexts: NDArray[float], shape (search_volume/n_samples, ad_feature_dim + user_feature_dim).
            Context vector (both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)
            (n_samples is determined in fit_reward_estimator function in simulator.)

        Returns
        -------
        ctrs: NDArray[float], shape (search_volume/n_samples, ).
            Ground-truth CTR (i.e., click per impression) for each auction.

        """
        if not (
            (isinstance(timestep, int) and timestep > 0)
            or (isinstance(timestep, NDArray[int]) and timestep.min() > 0)
        ):
            raise ValueError(
                "timestep must be non negative integer or an NDArray of non negative integers"
            )
        if not isinstance(contexts, NDArray[float]):
            raise ValueError("contexts must be an NDArray of float values")
        return (self.contexts @ self.coef.T) * self.time_coef[
            timestep % self.trend_interval
        ]


@dataclass
class CVR:
    """
    Class to calculate ground-truth CVR (i.e., conversion per click).

    Note
    -------
    We define two coefficient, context coefficient (coef) and time coefficient (time_coef).
    First, the value is calculated linearly from context vector and coef by inner product.
    Then, we multiply the value with time_coef and gain (ground-truth) CVR.

    To make correlation with CTR, we define coef of CVR by adding residuals sampled
    from normal distribution to that of CTR.

    Parameters
    -------
    ctr: CTR
        Pre-defined CTR function.

    """

    ctr: CTR

    def __post_init__(self):
        if not isinstance(ctr, CTR):
            raise ValueError("ctr must be the CTR or a child class of the CTR")

        self.trend_interval = self.ctr.trend_interval
        self.ad_feature_dim = self.ctr.ad_feature_dim
        self.user_feature_dim = self.ctr.user_feature_dim
        self.random_ = self.ctr.random_

        residuals = self.random_.normal(
            loc=0.0, scale=0.1, size=self.ad_feature_dim + self.user_feature_dim
        )
        self.coef = self.ctr.coef + residuals

        self.time_coef = self.random_.beta(40, 10, size=self.step_per_episode)
        self.time_coef = np.convolve(self.time_coef, np.ones(3) / 3, mode="same")

    def calc_prob(
        self, timestep: Union[int, NDArray[int]], contexts: NDArray[float]
    ) -> NDArray[float]:
        """Calculate CVR (i.e., conversion per click) using context vectors.

        Parameters
        -------
        timestep: Union[int, NDArray[int]], shape None/(n_samples, ).
            Timestep of the RL environment.
            (n_samples is determined in fit_reward_estimator function in simulator.)

        contexts: NDArray[float], shape (search_volume/n_samples, ad_feature_dim + user_feature_dim).
            Context vector (both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)
            (n_samples is determined in fit_reward_estimator function in simulator.)

        Returns
        -------
        cvrs: NDArray[float], shape (search_volume/n_samples, ).
            Ground-truth CVR (i.e., conversion per click) for each auction.

        """
        if not (
            (isinstance(timestep, int) and timestep > 0)
            or (isinstance(timestep, NDArray[int]) and timestep.min() > 0)
        ):
            raise ValueError(
                "timestep must be non negative integer or an NDArray of non negative integers"
            )
        if not isinstance(contexts, NDArray[float]):
            raise ValueError("contexts must be an NDArray of float values")
        return (self.contexts @ self.coef.T) * self.time_coef[
            timestep % self.trend_interval
        ]
