from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
from nptyping import Array
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_X_y

from env.function import WinningFunction, SecondPrice, CTR, CVR


@dataclass
class BaseSimulator(metaclass=ABCMeta):
    @abstractmethod
    def simulate_auction(
        self, timestep: int, adjust_rate: int, ad_ids: Array[int], user_ids: Array[int]
    ) -> Tuple[Array[int]]:
        raise NotImplementedError

    @abstractmethod
    def fit_reward_predictor(self, n_samples: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict_reward(self, timestep: int, contexts: Array[float]) -> Array[float]:
        raise NotImplementedError

    @abstractmethod
    def _calc_ground_truth_reward(
        self, timestep: int, contexts: Array[float]
    ) -> Array[float]:
        raise NotImplementedError

    @abstractmethod
    def _map_idx_to_contexts(
        self, ad_ids: Array[int], user_ids: Array[int]
    ) -> Array[float]:
        raise NotImplementedError

    @abstractmethod
    def _determine_bid_price(
        self, timestep: int, adjust_rate: int, contexts: Array[float]
    ) -> Array[int]:
        raise NotImplementedError

    @abstractmethod
    def _calc_and_sample_outcome(
        self, timestep: int, bid_prices: Array[int], contexts: Array[float]
    ) -> Tuple[Array[float]]:
        raise NotImplementedError


@dataclass
class RTBSyntheticSimulator(BaseSimulator):
    objective: str = "conversion"
    use_reward_predictor: bool = False
    reward_predictor: Optional[BaseEstimator] = None
    step_per_espisode: int = 24
    n_ads: int = 100
    n_users: int = 100
    ad_feature_dim: int = 5
    user_feature_dim: int = 5
    standard_bid_price: int = 100
    n_dices: int = 10
    wf_alpha: float = 2.0
    random_state: int = 12345

    def __post_init__(self):
        if not (
            isinstance(self.objective, str)
            and self.objective in ["click", "conversion"]
        ):
            raise ValueError(
                f'objective must be either "click" or "conversion", but {self.objective} is given'
            )
        if self.use_reward_predictor and self.reward_predictor is None:
            raise ValueError(
                f"reward_predictor must be given when use_reward_predictor=True"
            )
        if not (isinstance(self.step_per_espisode, int) and self.step_per_episode > 0):
            raise ValueError(
                f"step_per_episode must be a positive interger, but {self.step_per_episode} is given"
            )
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
            isinstance(self.standard_bid_price, int) and self.standard_bid_price > 0
        ):
            raise ValueError(
                f"standard_bid_price must be a positive interger, but {self.standard_bid_price} is given"
            )
        if not (isinstance(self.n_dices, int) and self.n_dices > 0):
            raise ValueError(
                f"n_dices must be a positive interger, but {self.n_dices} is given"
            )
        if not (isinstance(self.wf_alpha, float) and self.wf_alpha > 0):
            raise ValueError(
                f"wf_alpha must be a positive float number, but {self.wf_alpha} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        # sample feature vectors for both ads and users
        self.ads = self.random_.rand((self.n_ads, self.ad_feature_dim))
        self.users = self.random_.rand((self.n_users, self.user_feature_dim))

        # define winningfunc parameters for each ads
        self.wf_consts = self.random_.normal(
            loc=(self.standard_bid_price) ** (1 / self.wf_alpha),
            scale=(self.standard_bid_price) ** (1 / self.wf_alpha) / 5,
            size=self.n_ads,
        )

        # define winning_func and second_price sampler
        self.winning_function = WinningFunction(self.wf_alpha)
        self.second_price = SecondPrice(self.n_dices, self.random_state)

        # define click/imp and conversion/click rate function
        self.ctr = CTR(
            self.step_per_episode,
            self.ad_feature_dim,
            self.user_feature_dim,
            self.random_state,
        )
        self.cvr = CVR(self.ctr)

        # fix later
        # define scaler for bidding function
        # numbers are from predicted reward:
        # click / impression ~= 1/4, conversion / impression ~= 1/12
        if self.objective == "click":
            self.scaler = 4
        else:  # "conversion"
            self.scaler = 12

    def simulate_auction(
        self, timestep: int, adjust_rate: int, ad_ids: Array[int], user_ids: Array[int]
    ) -> Tuple[Array[int]]:
        """simulate auction and return outcome"""
        contexts = self._map_idx_to_contexts(ad_ids, user_ids)
        wf_consts = self.wf_consts[ad_ids]
        bid_prices = self._determine_bid_price(timestep, adjust_rate, contexts)
        return self._calc_and_sample_outcome(timestep, wf_consts, bid_prices, contexts)

    def fit_reward_predictor(self, n_samples: int) -> None:
        """pre-train reward regression used for calculating bids"""

        if not self.use_reward_predictor:
            warnings.warn(
                "under use_reward_predictor=False mode, fitting does not take place"
            )
            return

        ad_ids = self.random_.choice(self.n_ads, n_samples)
        user_ids = self.random_.choice(self.n_ads, n_samples)
        contexts = self._map_idx_to_contexts(ad_ids, user_ids)

        timesteps = self.random_.choice(self.step_per_episode, n_samples)
        feature_vectors = np.concatenate([contexts, timesteps], axis=1)

        if self.objective == "click":
            probs = self.ctr.calc_prob(timesteps, contexts)

        else:  # "conversion"
            probs = self.ctr.calc_prob(timesteps, contexts) * self.cvr.calc_prob(
                timesteps, contexts
            )

        rewards = self.random_.rand(n_samples) < probs

        X, y = check_X_y(feature_vectors, rewards)
        self.reward_predictor.fit(X, y)

    def _predict_reward(self, timestep: int, contexts: Array[float]) -> Array[float]:
        """obtain reward prediction used for calculating bids"""
        feature_vectors = np.concatenate(
            [contexts, np.full(len(contexts), timestep)], axis=1
        )
        return self.reward_predictor.predict(feature_vectors)

    def _calc_ground_truth_reward(
        self, timestep: int, contexts: Array[float]
    ) -> Array[float]:
        return self.ctr.calc_prob(timestep, contexts) * self.cvr.calc_prob(
            timestep, contexts
        )

    def _map_idx_to_contexts(
        self, ad_ids: Array[int], user_ids: Array[int]
    ) -> Array[float]:
        """
        map ad and user ids to feature vector (and concat them into contexts)
        and sample winning func params for each ad
        """
        ad_features = self.ads[ad_ids]
        user_features = self.users[user_ids]
        contexts = np.concatenate([ad_features, user_features], axis=1)
        return contexts

    def _determine_bid_price(
        self, timestep: int, adjust_rate: int, contexts: Array[float]
    ) -> Array[int]:
        """calculate bids using predicted reward"""
        if self.use_reward_predictor:
            predicted_rewards = self._predict_reward(timestep, contexts)
            bid_prices = (
                adjust_rate * predicted_rewards * self.standard_bid_price * self.scaler
            ).astype(int)

        else:
            ground_truth_rewards = self._calc_ground_truth_reward(timestep, contexts)
            bid_prices = (
                adjust_rate
                * ground_truth_rewards
                * self.standard_bid_price
                * self.scaler
            ).astype(int)

        return bid_prices

    def _calc_and_sample_outcome(
        self,
        timestep: int,
        wf_consts: Array[float],
        bid_prices: Array[int],
        contexts: Array[float],
    ) -> Tuple(Array[int]):
        """calculate outcome occuring probabilities and sample results"""
        impression_probs = self.winning_function.calc_prob(wf_consts, bid_prices)
        second_prices = self.second_price.sample(wf_consts, impression_probs)
        ctrs = self.ctr.calc_prob(timestep, contexts)
        cvrs = self.cvr.calc_prob(timestep, contexts)

        random_variables = self.random_.rand(len(bid_prices), 3)

        impressions = (random_variables[:, 0] < impression_probs).astype(int)
        clicks = ((random_variables[:, 1] < ctrs) * impressions).astype(int)
        conversions = ((random_variables[:, 2] < cvrs) * clicks).astype(int)
        costs = (second_prices * clicks).astype(int)

        return bid_prices, costs, impressions, clicks, conversions
