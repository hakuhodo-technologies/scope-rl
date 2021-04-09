from dataclasses import dataclass
from typing import Union
from nptyping import NDArray

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class NormalDistribution:
    mean: float
    std: float
    random_state: int

    def __post_init__(self):
        if not isinstance(self.mean, float):
            raise ValueError("mean must be a float number")
        if not isinstance(self.std, float):
            raise ValueError("std must be a float number")
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def sample(self, size: int = 1) -> NDArray[float]:
        return self.random_.normal(loc=self.mean, scale=self.std, size=size)


@dataclass
class WinningFunction:
    alpha: float

    def __post_init__(self):
        if not (isinstance(self.alpha, float) and self.alpha > 0):
            raise ValueError(
                f"alpha must be a positive float number, but {self.alpha} is given"
            )

    def calc_prob(self, consts: NDArray[float], bid_prices: NDArray[int]) -> NDArray[float]:
        """calc imp prob given winning price function"""
        return (bid_prices ** self.alpha) / (consts + bid_prices ** self.alpha)


@dataclass
class SecondPrice:  # fix later
    n_dices: int
    random_state: int

    def __post_init__(self):
        if not (isinstance(self.n_dices, int) and self.n_dices > 0):
            raise ValueError(
                f"n_dices must be a positive interger, but {self.n_dices} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def sample(self, consts: NDArray[float], probs: NDArray[int]) -> NDArray[int]:
        """sample second price for each bid"""
        discounts = self.random_.rand((len(consts), self.n_dices)).max(axis=1)
        return self.inverse_winning_func(consts, probs * discounts).astype(int)

    def inverse_winning_func(consts: NDArray[float], probs: NDArray[float]) -> NDArray[float]:
        return (probs * consts) / (1 - probs)


@dataclass
class CTR:
    trend_interval: int
    ad_feature_dim: int
    user_feature_dim: int
    random_state: int

    def __post_init__(self):
        if not (isinstance(self.step_per_espisode, int) and self.step_per_episode > 0):
            raise ValueError(
                f"step_per_episode must be a positive interger, but {self.step_per_episode} is given"
            )
        if not (isinstance(self.ad_feature_dim, int) and self.ad_feature_dim > 0):
            raise ValueError(
                f"ad_feature_dim must be a positive interger, but {self.ad_feature_dim} is given"
            )
        if not (isinstance(self.user_feature_dim, int) and self.user_feature_dim > 0):
            raise ValueError(
                f"user_feature_dim must be a positive interger, but {self.user_feature_dim} is given"
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
        """map context vector into click/imp prob"""
        return (self.contexts @ self.coef.T) * self.time_coef[timestep % self.trend_interval]


@dataclass
class CVR:
    ctr: CTR

    def __post_init__(self):
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
        """map context vector into conversion/click prob"""
        return (self.contexts @ self.coef.T) * self.time_coef[timestep % self.trend_interval]
