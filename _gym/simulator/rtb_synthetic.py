"""Synthetic Bidding Auction Simulation."""
from dataclasses import dataclass
from typing import Tuple, Union, Optional
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_X_y

from _gym.utils import NormalDistribution

from .base import BaseSimulator
from .function import WinningFunction, CTR, CVR


@dataclass
class RTBSyntheticSimulator(BaseSimulator):
    """Class to simulate bidding auction in Real-Time Bidding (RTB) setting for display advertising.

    Note
    -------
    Intended to be called and initialized from RTBEnv class in env.py.

    Simulate auctions as follows.
        1. Determine bid price.
            bid price = adjust rate * predicted/ground-truth reward ( * constant)

        2. Calculate outcome probability and stochastically determine auction result.
            auction results: (bid price,) cost (second price), impression, click, conversion

    Parameters
    -------
    objective: str, default="conversion"
        Objective outcome (i.e., reward) of the auction.
        Choose either from "click" or "conversion".

    use_reward_predictor: bool, default=False
        Whether to use predicted reward to determine bidding price or not.
        Otherwise, the ground-truth (expected) reward is used.

    reward_predictor: Optional[BaseEstimator], default=None
        A machine learning prediction model to predict the reward.
        Required only when using use_reward_predictor=True option.

    step_per_episode: int, default=24
        Total timestep in an episode in reinforcement learning (RL) environment.
        Note that we should use same value with RTBEnv class in env.py.

    n_ads: int, default=100
        Number of ads used for fitting the reward predictor.

    n_users: int, default=100
        Number of users used for fitting the reward predictor.

    ad_feature_dim: int, default=5
        Dimensions of the ad feature vectors.

    user_feature_dim: int, default=5
        Dimensions of the user feature vectors.

    standard_bid_price_distribution: NormalDistribution, default=NormalDistribution(mean=100, std=20)
        Distribution of the bid price whose average impression probability is expected to be 0.5.

    minimum_standard_bid_price: Optional[Union[int, float]], default=None
        Minimum value for standard bid price.
        If None, minimum_standard_bid_price is set to standard_bid_price_distribution / 2.

    trend_interval: Optional[int], default=None
        Length of the ctr/cvr trend cycle.
        For example, trend_interval=24 indicates that the trend cycles every 24h.
        If None, trend_interval is set to step_per_episode.

    random_state: int, default=12345
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
    standard_bid_price_distribution: NormalDistribution = NormalDistribution(
        mean=100, std=20
    )
    minimum_standard_bid_price: Optional[Union[int, float]] = None
    trend_interval: int = 24
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
        if not (isinstance(self.step_per_episode, int) and self.step_per_episode > 0):
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
            isinstance(self.standard_bid_price_distribution.mean, (int, float))
            and isinstance(self.standard_bid_price_distribution.std, (int, float))
        ):
            raise ValueError(
                "standard_bid_price_distribution must have single value for mean, std"
            )
        if self.standard_bid_price_distribution.mean <= 0:
            raise ValueError(
                f"standard_bid_price_distribution.mean must be a positive float value, but {self.standard_bid_price_distribution.mean} is given"
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

        # sample feature vectors for both ads and users
        self.ads = self.random_.normal(size=(self.n_ads, self.ad_feature_dim)) + 1
        self.users = self.random_.normal(size=(self.n_users, self.user_feature_dim)) + 1

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
        # average standard bid price
        self.standard_bid_price = self.standard_bid_price_distribution.mean

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

        # set trend_interval if None
        if self.trend_interval is None:
            self.trend_interval = self.step_per_episode

        # define click/imp and conversion/click rate function
        self.ctr = CTR(
            self.ad_feature_dim,
            self.user_feature_dim,
            self.trend_interval,
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
        self,
        timestep: int,
        adjust_rate: Union[int, float],
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Simulate bidding auction for given queries and return outcome.

        Note
        -------
        Simulation procedure is given as follows.
        1. Determine bid price.
            bid price = adjust rate * predicted/ground-truth reward ( * constant)

        2. Calculate outcome probability and stochastically determine auction result.
            auction results: (bid price,) cost (i.e., second price), impression, click, conversion

        Parameters
        -------
        timestep: int
            Timestep of the RL environment.

        adjust_rate: Union[int, float]
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
                Cost raised (i.e., second price) for each auction.

            impressions: NDArray[int], shape (search_volume, )
                Binary indicator of whether impression occurred or not for each auction.

            clicks: NDArray[int], shape (search_volume, )
                Binary indicator of whether click occurred or not for each auction.

            conversions: NDArray[int], shape (search_volume, )
                Binary indicator of whether conversion occurred or not for each auction.

        """
        if not (isinstance(timestep, int) and 0 <= timestep < self.step_per_episode):
            raise ValueError(
                f"timestep must be a interger within [0, {self.step_per_episode}), but {timestep} is given"
            )
        if not (isinstance(adjust_rate, (int, float)) and 0.1 <= adjust_rate <= 10):
            raise ValueError(
                f"adjust_rate must be a float number in [0.1, 10], but {adjust_rate} is given"
            )
        if not (isinstance(ad_ids, np.ndarray) and 0 <= ad_ids.all() < self.n_ads):
            raise ValueError("ad_ids must be chosen from integer within [0, n_ads)")
        if not (
            isinstance(user_ids, np.ndarray) and 0 <= user_ids.all() < self.n_users
        ):
            raise ValueError("user_ids must be chosen from integer within [0, n_users)")
        if len(ad_ids) != len(user_ids):
            raise ValueError("ad_ids and user_ids must have same length")

        ks, thetas = self.wf_ks[ad_ids], self.wf_thetas[ad_ids]
        contexts = self._map_idx_to_contexts(ad_ids, user_ids)
        bid_prices = self._determine_bid_price(timestep, adjust_rate, contexts)

        return self._calc_and_sample_outcome(timestep, ks, thetas, bid_prices, contexts)

    def fit_reward_predictor(self, n_samples: int = 100000) -> None:
        """Fit reward predictor in advance (pre-train) to use prediction in bidding price determination.

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
        n_samples: int, default=100000
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

        timesteps = self.random_.choice(self.step_per_episode, n_samples).reshape(
            (-1, 1)
        )
        feature_vectors = np.concatenate([contexts, timesteps], axis=1)

        if self.objective == "click":
            rewards = self.ctr.sample_outcome(timesteps, contexts)

        else:  # "conversion"
            rewards = self.ctr.sample_outcome(
                timesteps, contexts
            ) * self.cvr.sample_outcome(timesteps, contexts)

        X, y = check_X_y(feature_vectors, rewards)
        self.reward_predictor.fit(X, y)

    def _predict_reward(self, timestep: int, contexts: np.ndarray) -> np.ndarray:
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
            Timestep of the RL environment.

        contexts: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
            Context vector (contain both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        predicted_rewards: NDArray[float], shape (search_volume, )
            Predicted reward for each auction.

        """
        timesteps = np.full(len(contexts), timestep).reshape((-1, 1))
        feature_vectors = np.concatenate([contexts, timesteps], axis=1)

        return self.reward_predictor.predict_proba(feature_vectors)[:, 1]

    def _calc_ground_truth_reward(
        self, timestep: int, contexts: np.ndarray
    ) -> np.ndarray:
        """Calculate ground-truth reward (i.e., auction outcome) to determine bidding price.

        Parameters
        -------
        timestep: int
            Timestep of the RL environment.

        contexts: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
            Context vector (contain both the ad and the user features) for each auction.
            (search_volume is determined in RL environment.)

        Returns
        -------
        expected_rewards: NDArray[float], shape(search_volume, )
            Ground-truth (expected) reward for each auction when impression occurs.

        """
        if self.objective == "click":
            expected_rewards = self.ctr.calc_prob(timestep, contexts)

        else:  # "conversion"
            expected_rewards = self.ctr.calc_prob(
                timestep, contexts
            ) * self.cvr.calc_prob(timestep, contexts)

        return expected_rewards

    def _determine_bid_price(
        self, timestep: int, adjust_rate: float, contexts: np.ndarray
    ) -> np.ndarray:
        """Determine the bidding price using given adjust rate and the predicted/ground-truth rewards.

        Note
        -------
        Determine bid price as follows.
            bid price = adjust rate * predicted/ground-truth reward ( * constant)

        Parameters
        -------
        timestep: int
            Timestep of the RL environment.

        adjust rate: float
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
            )

        else:
            ground_truth_rewards = self._calc_ground_truth_reward(timestep, contexts)
            bid_prices = (
                adjust_rate
                * ground_truth_rewards
                * self.standard_bid_price
                * self.scaler
            )

        return bid_prices.astype(int)

    def _calc_and_sample_outcome(
        self,
        timestep: int,
        ks: np.ndarray,
        thetas: np.ndarray,
        bid_prices: np.ndarray,
        contexts: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Calculate pre-determined probabilities from contexts and stochastically sample the outcome.

        Note
        -------
        Calculate probabilities using following functions.
            - impression: WinningFunction
            - cost: SecondPrice
            - click per impression: CTR
            - conversion per click: CVR

        Parameters
        -------
        timestep: int.
            Timestep of the RL environment.

        ks: NDArray[float], shape (search_volume, )
            Shape parameter for WinningFunction for each ad used in the auction.
            (search_volume is determined in RL environment.)

        thetas: NDArray[float], shape (search_volume, )
            Scale parameter for WinningFunction for each ad used in the auction.
            (search_volume is determined in RL environment.)

        bid_prices: NDArray[int], shape(search_volume, )
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
                Cost raised (i.e., second price) for each auction.

            impressions: NDArray[int], shape (search_volume, )
                Binary indicator of whether impression occurred or not for each auction.

            clicks: NDArray[int], shape (search_volume, )
                Binary indicator of whether click occurred or not for each auction.

            conversions: NDArray[int], shape (search_volume, )
                Binary indicator of whether conversion occurred or not for each auction.

        """
        impressions, winning_prices = self.winning_function.sample_outcome(
            ks, thetas, bid_prices
        )
        clicks = self.ctr.sample_outcome(timestep, contexts) * impressions
        conversions = self.cvr.sample_outcome(timestep, contexts) * clicks
        costs = winning_prices * clicks

        return bid_prices, costs, impressions, clicks, conversions

    def _map_idx_to_contexts(
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
        ad_features = self.ads[ad_ids]
        user_features = self.users[user_ids]
        contexts = np.concatenate([ad_features, user_features], axis=1)

        return contexts
