from dataclasses import dataclass
from typing import Union
from nptyping import Array

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class NormalDistribution:
    mean: float
    std: float
    random_state: int

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)

    def sample(self, size: int = 1) -> Array[float]:
        return self.random_.normal(loc=self.mean, scale=self.std, size=size)


@dataclass
class WinningFunction:
    alpha: float

    def calc_prob(self, consts: Array[float], bid_prices: Array[int]) -> Array[float]:
        """calc imp prob given winning price function"""
        return (bid_prices ** self.alpha) / (consts + bid_prices ** self.alpha)


@dataclass
class SecondPrice:  # fix later
    n_dices: int
    random_state: int

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)

    def sample(self, consts: Array[float], probs: Array[int]) -> Array[int]:
        """sample second price for each bid"""
        discounts = self.random_.rand((len(consts), self.n_dices)).max(axis=1)
        return self.inverse_winning_func(consts, probs * discounts).astype(int)

    def inverse_winning_func(consts: Array[float], probs: Array[float]) -> Array[float]:
        return (probs * consts) / (1 - probs)


@dataclass
class CTR:
    step_per_episode: int
    ad_feature_dim: int
    user_feature_dim: int
    random_state: int

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)

        self.coef = self.random_.normal(
            loc=0.5, scale=0.1, size=self.ad_feature_dim + self.user_feature_dim
        )
        self.coef = self.coef / (
            self.ad_feature_dim + self.user_feature_dim
        )  # to normalize

        self.time_coef = self.random_.beta(25, 25, size=self.step_per_episode)
        self.time_coef = np.convolve(self.time_step, np.ones(3) / 3, mode="same")

    def calc_prob(
        self, timestep: Union[int, Array[int]], contexts: Array[float]
    ) -> Array[float]:
        """map context vector into click/imp prob"""
        return (self.contexts @ self.coef.T) * self.time_coef[timestep]


@dataclass
class CVR:
    ctr: CTR

    def __post_init__(self):
        self.step_per_episode = self.ctr.step_per_episode
        self.ad_feature_dim = self.ctr.ad_feature_dim
        self.user_feature_dim = self.ctr.user_feature_dim
        self.random_ = self.ctr.random_

        residuals = self.random_.normal(
            loc=0.0, scale=0.1, size=self.ad_feature_dim + self.user_feature_dim
        )
        residuals = residuals / (self.ad_feature_dim + self.user_feature_dim)
        self.coef = self.ctr.coef + residuals

        self.time_coef = self.random_.beta(40, 10, size=self.step_per_episode)
        self.time_coef = np.convolve(self.time_step, np.ones(3) / 3, mode="same")

    def calc_prob(
        self, timestep: Union[int, Array[int]], contexts: Array[float]
    ) -> Array[float]:
        """map context vector into conversion/click prob"""
        return (self.contexts @ self.coef.T) * self.time_coef[timestep]
