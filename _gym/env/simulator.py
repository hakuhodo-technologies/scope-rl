"""Synthetic Bidding Auction Simulation."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
from nptyping import NDArray
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_X_y

from env.function import WinningFunction, SecondPrice, CTR, CVR


@dataclass
class BaseSimulator(metaclass=ABCMeta):
    """Base class for bidding auction simulators."""

    @abstractmethod
    def simulate_auction(
        """Simulate bidding auction for given queries and return outcome."""
        self, timestep: int, adjust_rate: int, ad_ids: NDArray[int], user_ids: NDArray[int]
    ) -> Tuple[NDArray[int]]:
        raise NotImplementedError

    @abstractmethod
    def fit_reward_predictor(self, n_samples: int) -> None:
        """Fit reward predictor in advance to use prediction in bidding price determination."""
        raise NotImplementedError

    @abstractmethod
    def _predict_reward(self, timestep: int, contexts: NDArray[float]) -> NDArray[float]:
        """Predict reward (i.e., auction outcome) to determine bidding price."""
        raise NotImplementedError

    @abstractmethod
    def _calc_ground_truth_reward(
        self, timestep: int, contexts: NDArray[float]
    ) -> NDArray[float]:
        """Calculate ground-truth reward (i.e., auction outcome) to determine bidding price."""
        raise NotImplementedError

    @abstractmethod
    def _map_idx_to_contexts(
        self, ad_ids: NDArray[int], user_ids: NDArray[int]
    ) -> NDArray[float]:
        """Map the ad and the user index into context vectors."""
        raise NotImplementedError

    @abstractmethod
    def _determine_bid_price(
        self, timestep: int, adjust_rate: int, contexts: NDArray[float]
    ) -> NDArray[int]:
        """Determine the bidding price using given adjust rate and the predicted/ground-truth rewards"""
        raise NotImplementedError

    @abstractmethod
    def _calc_and_sample_outcome(
        self, timestep: int, bid_prices: NDArray[int], contexts: NDArray[float]
    ) -> Tuple[NDArray[float]]:
        """Calculate pre-determined probabilities from contexts and stochastically sample the outcome."""
        raise NotImplementedError


@dataclass
class RTBSyntheticSimulator(BaseSimulator):
    """Simulate bidding auction in Real-Time Bidding (RTB) setting for display advertising.
    
    Note
    -------

    Parameters
    -------
    objective: str, default="conversion".
        Objective outcome (i.e., reward) of the auction.
        Choose either from "click" or "conversion".

    use_reward_predictor: bool, default=False.
        Whether to use predicted reward to determine bidding price or not.
        Otherwise, the ground-truth (expected) reward is used.

    reward_predictor: Optional[BaseEstimator], default=None.
        A machine learning prediction model to predict the reward.
        If use_reward_predictor=True, the reward predictor must be given.

    step_per_episode: int, default=24.
        Total timestep in a episode in reinforcement learning (RL) environment.

    n_ads: int, defalut=100.
        Number of ads used for fitting the reward predictor.

    n_users: int, default=100.
        Number of users used for fitting the reward predictor.

    ad_feature_dim: int, default=5.
        Dimentions of the ad feature vectors.

    user_feature_dim: int, default=5.
        Dimentions of the user feature vectors.

    standard_bid_price: int, default = 100.
        Bid price whose impression probability is expected to be 0.5.

    trend_interval: int, default=24.
        Length of the ctr/cvr trend cycle.
        Default number indicates that the trend cycles every 24h.

    n_dices: int, default=10.
        Number of the random_variables sampled to calculate second price.

    wf_alpha: float, default=2.0.
        Parameter (exponential coefficient) for WinningFunction used in the auction.

    random_state: int, default=12345.
        Random state.

    References
    -------

    """

    objective: str = "conversion"
    use_reward_predictor: bool = False
    reward_predictor: Optional[BaseEstimator] = None
    step_per_episode: int = 24
    n_ads: int = 100
    n_users: int = 100
    ad_feature_dim: int = 5
    user_feature_dim: int = 5
    standard_bid_price: int = 100
    trend_interval: int = 24
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
        if not (isinstance(self.n_ads, int) and self.n_ads > 0):
            raise ValueError(
                f"n_ads must be a positive interger, but {self.n_ads} is given"
            )
        if not (isinstance(self.step_per_episode, int) and self.step_per_episode > 0):
            raise ValueError(
                f"step_per_episode must be a positive interger, but {self.step_per_episode} is given"
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
        if not (isinstance(self.trend_interval, int) and self.trend_interval > 0):
            raise ValueError(
                f"trend_interval must be a positive interger, but {self.trend_interval} is given"
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
            self.trend_interval,
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
        self, timestep: int, adjust_rate: int, ad_ids: NDArray[int], user_ids: NDArray[int]
    ) -> Tuple[NDArray[int]]:
        """Simulate bidding auction for given queries and return outcome.
            
        Parameters
        -------
        timestep: int
            Timestep of the RL environment.
        
        adjust rate: int
            Adjust rate parameter for bidding price determination.
            Corresponds to the RL agent action.

        ad_ids: NDArray[int], shape (search_volume, )
            IDs of the ads used for the auction bidding.
            (search_volume is determined in RL environment.)

        user_ids: NDArray[int], shape (search_volume, )
            IDs of the users who receives the winning ads.
            (search_volume is determined in RL environment.)

        Returns
        -------
        auction_results: Tuple
            bid_prices: NDArray[int], shape (search_volume, )
                Bid price used for each auction.

            costs: NDArray[int], shape (search_volume, )
                Cost arised (i.e., second price) for each auction.

            impressions: NDArray[int], shape (search_volume, )
                Binary indicator of whether impression occured or not for each auction.

            clicks: NDArray[int], shape (search_volume, )
                Binary indicator of whether click occured or not for each auction.

            conversions: NDArray[int], shape (search_volume, )
                Binary indicator of whether conversion occured or not for each auction.

        """
        if not (isinstance(timestep, int) and 0 < timestep < self.step_per_episode):
            raise ValueError(
                f"timestep must be a positive interger and should not exceed {self.step_per_episode}, but {timestep} is given"
            )
        if not (isinstance(adjust_rate) and 0.1 <= adjust_rate <= 10):
            raise ValueError(
                f"adjust_rate must be float number in [0.1, 10], but {adjust_rate} is given"
            )
        if ad_ids.min() < 0 or self.ad_ids.max() >= self.n_ads:
            raise ValueError(
                "ad_ids must be chosen from integer values in [0, n_ads)"
            )
        if user_ids.min() < 0 or self.user_ids.max() >= self.n_users:
            raise ValueError(
                "user_ids must be chosen from integer values in [0, n_users)"
            )
        if len(ad_ids) != len(user_ids):
            raise ValueError("ad_ids and user_ids must have same length")

        contexts = self._map_idx_to_contexts(ad_ids, user_ids)
        wf_consts = self.wf_consts[ad_ids]
        bid_prices = self._determine_bid_price(timestep, adjust_rate, contexts)

        return self._calc_and_sample_outcome(timestep, wf_consts, bid_prices, contexts)

    def fit_reward_predictor(self, n_samples: int) -> None:
        """Fit reward predictor in advance to use prediction in bidding price determination.
        
        Note
        -------
        Intended only used when use_reward_predictor=True option.

        X and y of the prediction model is given as follows.
        X (feature_vectors): NDArray[float], shape (n_samples, action_feature_dim + user_feature_dim + 1)
            Concatenated vector of ad_feature_vector, user_feature_vector, and timestep.

        y (target values): NDArrray[int], shape (n_samples, )
            Reward (i.e., auction outcome) obtained in each auction.

        Parameters
        -------
        n_samples: int
            Number of samples to fit reward predictor.

        """
        if not self.use_reward_predictor:
            warnings.warn(
                "when initialized with use_reward_predictor=False option, fitting does not take place"
            )
            return
        
        if not (isinstance(n_samples, int) and n_samples > 0):
            raise ValueError(
                f"n_samples must be a positive interger, but {n_samples} is given"
            )

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

    def _predict_reward(self, timestep: int, contexts: NDArray[float]) -> NDArray[float]:
        """Predict reward (i.e., auction outcome) to determine bidding price.
        
        Note
        -------
        Intended only used when use_reward_predictor=True option.

        X and y of the prediction model is given as follows.
        X (feature_vectors): NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
            Concatenated vector of contexts (ad_feature_vector + user_feature_vector) and timestep.

        y (target values): NDArrray[int], shape (search_volume, )
            Reward (i.e., auction outcome) obtained in each auction.

        Parameters
        -------
        timestep: int
            Timestep of the (reinforcement learning (RL)) environment.

        contexts: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
            Context vector (contain both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        predicted_rewards: NDArray[float], shape (search_volume, )
            Predicted reward for each auction.

        """
        feature_vectors = np.concatenate(
            [contexts, np.full(len(contexts), timestep)], axis=1
        )
        
        return self.reward_predictor.predict(feature_vectors)

    def _calc_ground_truth_reward(
        self, timestep: int, contexts: NDArray[float]
    ) -> Array[float]:
        """Calculate ground-truth reward (i.e., auction outcome) to determine bidding price.

        Parameters
        -------
        timestep: int
            Timestep of the (reinforcement learning (RL)) environment.

        contexts: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
            Context vector (contain both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        expected_rewards: NDArray[float], shape(search_volume, )
            Ground-truth (expected) reward for each auction when impression occurs.

        """
        if self.objective == "click":
            extected_rewards = self.ctr.calc_prob(timesteps, contexts)

        else:  # "conversion"
            expected_rewawrds = self.ctr.calc_prob(timesteps, contexts) * self.cvr.calc_prob(
                timesteps, contexts
            )

        return expected_rewards

    def _map_idx_to_contexts(
        self, ad_ids: NDArray[int], user_ids: NDArray[int]
    ) -> NDArray[float]:
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
        contexts: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
            Context vector (contain both the ad and the user features) for each auction.

        """
        ad_features = self.ads[ad_ids]
        user_features = self.users[user_ids]
        contexts = np.concatenate([ad_features, user_features], axis=1)

        return contexts

    def _determine_bid_price(
        self, timestep: int, adjust_rate: int, contexts: NDArray[float]
    ) -> NDArray[int]:
        """Determine the bidding price using given adjust rate and the predicted/ground-truth rewards.
        
        Parameters
        -------
        timestep: int
            Timestep of the RL environment.
        
        adjust rate: int
            Adjust rate parameter for bidding price determination.
            Corresponds to the RL agent action.

        contexts: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
            Context vector (contain both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        bid_prices: NDArray[int], shape(search_volume, )
            Bid price for each auction.
            Calclated from given adjust rate and the predicted/ground-truth rewards.

        """
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
        wf_consts: NDArray[float],
        bid_prices: NDArray[int],
        contexts: NDArray[float],
    ) -> Tuple(NDArray[int]):
        """Calculate pre-determined probabilities from contexts and stochastically sample the outcome.

        Parameters
        -------
        timestep: int
            Timestep of the RL environment.

        wf_consts: NDArray[float], shape (search_volume, )
            Parameter for WinningFunction for each ad used in the auction.
            (search_volume is determined in RL environment.)

        bid_prices: NDArray[float], shape(search_volume, )
            Bid price for each action.
            (search_volume is determined in RL environment.)

        contexts: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
            Context vector (contain both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        auction_results: Tuple
            bid_prices: NDArray[int], shape (search_volume, )
                Bid price used for each auction.

            costs: NDArray[int], shape (search_volume, )
                Cost arised (i.e., second price) for each auction.

            impressions: NDArray[int], shape (search_volume, )
                Binary indicator of whether impression occured or not for each auction.

            clicks: NDArray[int], shape (search_volume, )
                Binary indicator of whether click occured or not for each auction.

            conversions: NDArray[int], shape (search_volume, )
                Binary indicator of whether conversion occured or not for each auction.

        """
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
