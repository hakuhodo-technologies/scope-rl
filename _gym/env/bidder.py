"""Bid Price Calculation."""
from dataclasses import dataclass
from typing import Union, Optional
import warnings

import numpy as np
from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils import check_random_state, check_X_y

from .simulator.base import BaseSimulator


@dataclass
class Bidder:
    """Class to determine bid price.

    Note
    -------
    Intended to be called and initialized from RTBEnv class in env.py.

    Determine bid price by the following formula.
        :math:`bid_price_{t, i} = adjust_rate_{t} \\times predicted_reward_{t,i}/ground_truth_reward_{t, i} ( \\times const.)`

    Parameters
    -------
    simulator: BaseSimulator
        Auction simulator.

    objective: str, default="conversion"
        Objective outcome (i.e., reward) of the auction.
        Choose either from "click" or "conversion".

    reward_predictor: Optional[BaseEstimator], default=None
        A machine learning model to predict the reward to determine the bidding price.
        If None, the ground-truth (expected) reward is used instead of the predicted one.

    scaler: Optional[Union[int, float]], default=None
        Scaling factor (constant value) used for bid price determination.
        If None, one should call auto_fit_scaler().

    random_state: Optional[int], default=None
        Random state.

    References
    -------
    Di Wu, Xiujun Chen, Xun Yang, Hao Wang, Qing Tan, Xiaoxun Zhang, Jian Xu, and Kun Gai.
    "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising.", 2018.

    Jun Zhao, Guang Qiu, Ziyu Guan, Wei Zhao, and Xiaofei He.
    "Deep Reinforcement Learning for Sponsored Search Real-time Bidding.", 2018.

    """

    simulator: BaseSimulator
    objective: str = "conversion"
    reward_predictor: Optional[BaseEstimator] = None
    scaler: Optional[Union[int, float]] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.objective not in ["click", "conversion"]:
            raise ValueError(
                f'objective must be either "click" or "conversion", but {self.objective} is given'
            )
        if self.reward_predictor is not None and not isinstance(
            self.reward_predictor, BaseEstimator
        ):
            raise ValueError(
                "reward_predictor must be BaseEstimator or a child class of BaseEstimator"
            )
        if self.scaler is not None and not (
            isinstance(self.scaler, (int, float)) and self.scaler > 0
        ):
            raise ValueError(
                f"scaler must be a positive float value, but {self.scaler} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        self.use_reward_predictor = False if self.reward_predictor is None else True

    @property
    def standard_bid_price(self):
        return self.simulator.standard_bid_price

    def determine_bid_price(
        self,
        timestep: int,
        adjust_rate: float,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
    ) -> np.ndarray:
        """Determine the bidding price using given adjust rate and the predicted/ground-truth rewards.

        Note
        -------
        Determine bid price as follows.
            :math:`bid_price_{t, i} = adjust_rate_{t} \\times predicted_reward_{t,i}/ground_truth_reward_{t, i} ( \\times const.)`

        Parameters
        -------
        timestep: int
            Timestep of the RL environment.

        adjust_rate: float
            Adjust rate parameter for the bidding price.

        ad_ids: NDArray[int], shape (search_volume, )
            IDs of the ads.

        user_ids: NDArray[int], shape (search_volume, )
            IDs of the users.

        Returns
        -------
        bid_prices: NDArray[int], shape(search_volume, )
            Bid price for each auction.

        """
        if self.scaler is None:
            raise RuntimeError(
                "scalar should be given, please call .auto_fit_scaler() or .custom_set_scaler() before calling .determine_bid_price()"
            )

        if not (isinstance(timestep, int) and timestep >= 0):
            raise ValueError(
                f"timestep must be a non-negative interger, but {timestep} is given"
            )
        if not (
            isinstance(adjust_rate, (float, np.float, np.float32)) and adjust_rate >= 0
        ):
            print(adjust_rate, type(adjust_rate))
            raise ValueError(
                f"adjust_rate must be a non-negative float value, but {adjust_rate} is given"
            )

        ad_feature_vector, user_feature_vector = self.simulator.map_idx_to_features(
            ad_ids=ad_ids,
            user_ids=user_ids,
        )

        if self.use_reward_predictor:
            predicted_rewards = self._predict_reward(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timestep,
            )
            bid_prices = (
                adjust_rate * predicted_rewards * self.standard_bid_price * self.scaler
            )

        else:
            ground_truth_rewards = self._calc_ground_truth_reward(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timestep,
            )
            bid_prices = (
                adjust_rate
                * ground_truth_rewards
                * self.standard_bid_price
                * self.scaler
            )

        return bid_prices.astype(int)

    def custom_set_scaler(self, scaler: Union[int, float]) -> None:
        """Set scaling factor used for bid price calculation.

        Parameters
        -------
        scaler: Union[int, float]
            Scaling factor (constant value) used in bid price calculation.

        """
        if not (isinstance(scaler, (int, float)) and scaler > 0):
            raise ValueError(
                f"scaler must be a positive float value, but{scaler} is given"
            )
        warnings.warn(
            "autofit is recommended for scaler, please call .auto_fit_scaler() to autofit"
        )
        self.scaler = scaler

    def auto_fit_scaler(self, step_per_episode: int, n_samples: int = 100000) -> None:
        """Fit scaling factor used for bid price calculation.

        Note
        -------
        scaler is set to approximate reciprocal of the mean predicted/ground-truth rewards.
            scaler ~= 1 / mean of predicted/ground-truth rewards

        Parameters
        -------
        step_per_episode: int
            Number of timesteps in a episode.

        n_samples: int, default=100000
            Number of samples to fit bid_scaler.

        """
        if not (isinstance(step_per_episode, int) and step_per_episode >= 0):
            raise ValueError(
                f"step_per_episode must be a non-negative interger, but {step_per_episode} is given"
            )
        if not (isinstance(n_samples, int) and n_samples > 0):
            raise ValueError(
                f"n_samples must be a positive interger, but {n_samples} is given"
            )

        timesteps = self.random_.choice(step_per_episode, n_samples)
        ad_ids, user_ids = self.simulator.generate_auction(volume=n_samples)
        ad_feature_vector, user_feature_vector = self.simulator.map_idx_to_features(
            ad_ids=ad_ids,
            user_ids=user_ids,
        )

        if self.use_reward_predictor:
            predicted_rewards = self._predict_reward(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timesteps,
            )
            self.scaler = 1 / predicted_rewards.mean()

        else:
            ground_truth_rewards = self._calc_ground_truth_reward(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timesteps,
            )
            self.scaler = 1 / ground_truth_rewards.mean()

    def custom_set_reward_predictor(self, reward_predictor: BaseEstimator):
        """Set reward predictor used for bid price calculation.

        Parameters
        -------
        reward_predictor: Optional[BaseEstimator], default=None
            A machine learning model to predict the reward to determine the bidding price.
            If None, the ground-truth (expected) reward is used instead of the predicted one.

        """
        if reward_predictor is not None and not isinstance(
            reward_predictor, BaseEstimator
        ):
            raise ValueError(
                "reward_predictor must be BaseEstimator or a child class of BaseEstimator"
            )
        self.reward_predictor = reward_predictor
        self.use_reward_predictor = True

    def fit_reward_predictor(
        self, step_per_episode: int, n_samples: int = 100000
    ) -> None:
        """Fit reward predictor in advance (pre-train) to use prediction in bidding price determination.

        Note
        -------
        Intended only used when use_reward_predictor=True option.

        X and y of the prediction model is given as follows.
            X: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
                Concatenated vector of contexts (ad_feature_vector + user_feature_vector) and timestep.

            y: NDArrray[int], shape (search_volume, )
                Reward (i.e., auction outcome) obtained in each auction.

        Parameters
        -------
        step_per_episode: int
            Number of timesteps in a episode.

        n_samples: int, default=100000
            Number of samples to fit reward predictor.

        """
        if not self.use_reward_predictor:
            warnings.warn(
                "when reward_predictor is not given, fitting does not take place"
            )
            return

        if not (isinstance(step_per_episode, int) and step_per_episode >= 0):
            raise ValueError(
                f"step_per_episode must be a non-negative interger, but {step_per_episode} is given"
            )
        if not (isinstance(n_samples, int) and n_samples > 0):
            raise ValueError(
                f"n_samples must be a positive interger, but {n_samples} is given"
            )

        ad_ids, user_ids = self.simulator.generate_auction(n_samples)
        ad_feature_vector, user_feature_vector = self.simulator.map_idx_to_features(
            ad_ids, user_ids
        )
        contexts = np.concatenate([ad_feature_vector, user_feature_vector], axis=1)
        timesteps = self.random_.choice(step_per_episode, n_samples)
        feature_vectors = np.concatenate([contexts, timesteps.reshape((-1, 1))], axis=1)

        if self.objective == "click":
            rewards = self.simulator.ctr.sample_outcome(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timesteps,
            )
        else:  # "conversion"
            rewards = self.simulator.ctr.sample_outcome(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timesteps,
            ) * self.simulator.cvr.sample_outcome(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timesteps,
            )

        X, y = check_X_y(feature_vectors, rewards)
        self.reward_predictor.fit(X, y)

    def _predict_reward(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Predict reward (i.e., auction outcome) to determine bidding price.

        Note
        -------
        Intended only used when use_reward_predictor=True option.

        X and y of the prediction model is given as follows.
            X: NDArray[float], shape (search_volume, ad_feature_dim + user_feature_dim + 1)
                Concatenated vector of contexts (ad_feature_vector + user_feature_vector) and timestep.

            y: NDArrray[int], shape (search_volume, )
                Reward (i.e., auction outcome) obtained in each auction.

        Parameters
        -------
        ad_ids: NDArray[int], shape (search_volume, )
            IDs of the ads.

        user_ids: NDArray[int], shape (search_volume, )
            IDs of the users.

        ad_feature_vector: NDArray[float], shape (search_volume, ad_feature_dim)
            Feature vector of the ads.

        user_feature_vector: NDArray[float], shape (search_volume, user_feature_dim)
            Feature vector of the users.

        timestep: int
            Timestep in the RL environment.

        Returns
        -------
        predicted_rewards: NDArray[float], shape (search_volume, )
            Predicted reward for each auction.

        """
        if isinstance(timestep, int):
            timestep = np.full(len(ad_ids), timestep)
        timestep = timestep.reshape((-1, 1))
        contexts = np.concatenate([ad_feature_vector, user_feature_vector], axis=1)

        X = np.concatenate([contexts, timestep], axis=1)
        predicted_rewards = (
            self.reward_predictor.predict_proba(X)[:, 1]
            if is_classifier(self.reward_predictor)
            else self.reward_predictor.predict(X)
        )
        return predicted_rewards

    def _calc_ground_truth_reward(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Calculate ground-truth reward (i.e., auction outcome) to determine bidding price.

        Parameters
        -------
        ad_ids: NDArray[int], shape (search_volume, )
            IDs of the ads.

        user_ids: NDArray[int], shape (search_volume, )
            IDs of the users.

        ad_feature_vector: NDArray[float], shape (search_volume, ad_feature_dim)
            Feature vector of the ads.

        user_feature_vector: NDArray[float], shape (search_volume, user_feature_dim)
            Feature vector of the users.

        timestep: Union[int, np.ndarray]
            Timestep in the RL environment.

        Returns
        -------
        expected_rewards: NDArray[float], shape(search_volume, )
            Ground-truth (expected) reward for each auction when impression occurs.

        """
        if self.objective == "click":
            expected_rewards = self.simulator.ctr.calc_prob(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timestep,
            )
        else:  # "conversion"
            expected_rewards = self.simulator.ctr.calc_prob(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timestep,
            ) * self.simulator.cvr.calc_prob(
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
                timestep=timestep,
            )

        return expected_rewards
