# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Bid Price Calculation."""
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils import check_scalar, check_random_state, check_X_y

from .base import BaseSimulator
from ...utils import check_array
from ...types import Numeric


@dataclass
class Bidder:
    """Class to determine bid price.

    Imported as: :class:`rtbgym.envs.simulator.Bidder`

    Note
    -------
    Intended to be called and initialized from RTBEnv class in env.py.

    Determine bid price by the following formula.

    .. math::

        {bid price}_{t, i} = {adjust rate}_{t} \\times {predicted reward}_{t,i} ( \\times {const.})

    Parameters
    -------
    simulator: BaseSimulator
        Auction simulator.

    objective: {"click", "conversion"}, default="conversion"
        Objective outcome (i.e., reward) of the auction.

    reward_predictor: BaseEstimator, default=None
        A machine learning model to predict the reward to determine the bidding price.
        If `None`, the ground-truth (expected) reward is used instead of the predicted one.

    scaler: {int, float}, default=None (> 0)
        Scaling factor (constant value) used for bid price determination.
        If `None`, one should call auto_fit_scaler().

    random_state: int, default=None (>= 0)
        Random state.

    References
    -------
    Di Wu, Xiujun Chen, Xun Yang, Hao Wang, Qing Tan, Xiaoxun Zhang, Jian Xu, and Kun Gai.
    "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising." 2018.

    Jun Zhao, Guang Qiu, Ziyu Guan, Wei Zhao, and Xiaofei He.
    "Deep Reinforcement Learning for Sponsored Search Real-time Bidding." 2018.

    """

    simulator: BaseSimulator
    objective: str = "conversion"
    reward_predictor: Optional[BaseEstimator] = None
    scaler: Optional[Union[int, float]] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.simulator, BaseSimulator):
            raise ValueError("simulator must be a child class of BaseSimulator")
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
        if self.scaler is not None:
            check_scalar(
                self.scaler,
                name="scaler",
                target_type=(int, float),
                min_val=0,
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

        .. math::

            {bid price}_{t, i} = {adjust rate}_{t} \\times {predicted reward}_{t,i} ( \\times {const.})

        Parameters
        -------
        timestep: int (> 0)
            Timestep of the RL environment.

        adjust_rate: float (>= 0)
            Adjust rate parameter for the bidding price.

        ad_ids: array-like of shape (search_volume, )
            IDs of the ads.

        user_ids: array-like of shape (search_volume, )
            IDs of the users.

        Returns
        -------
        bid_prices: ndarray of shape(search_volume, )
            Bid price for each auction.

        """
        if self.scaler is None:
            raise RuntimeError(
                "scalar should be given, please call .auto_fit_scaler() or .custom_set_scaler() before calling .determine_bid_price()"
            )

        check_scalar(
            timestep,
            name="timestep",
            target_type=int,
            min_val=0,
        )
        check_scalar(
            adjust_rate,
            name="adjust_rate",
            target_type=Numeric,
            min_val=0,
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
        scaler: {int, float} (> 0)
            Scaling factor (constant value) used in bid price calculation.

        """
        check_scalar(
            scaler,
            name="scaler",
            target_type=(int, float),
            min_val=0,
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
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        n_samples: int, default=100000 (> 0)
            Number of samples to fit bid_scaler.

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            n_samples,
            name="n_samples",
            target_type=int,
            min_val=1,
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
        reward_predictor: BaseEstimator, default=None
            A machine learning model to predict the reward to determine the bidding price.
            If None, the ground-truth (expected) reward is used instead of the predicted one.

        """
        if reward_predictor is not None and not isinstance(
            reward_predictor, BaseEstimator
        ):
            raise ValueError("reward_predictor must be a child class of BaseEstimator")
        self.reward_predictor = reward_predictor
        self.use_reward_predictor = True

    def fit_reward_predictor(
        self, step_per_episode: int, n_samples: int = 100000
    ) -> None:
        """Fit reward predictor in advance (pre-train) to use prediction in bidding price determination.

        Note
        -------
        Intended to be used only when use_reward_predictor=True option.

        X and y of the prediction model is given as follows.
            X: array-like of shape (search_volume, ad_feature_dim + user_feature_dim + 1)
                Concatenated vector of contexts (ad_feature_vector + user_feature_vector) and timestep.

            y: array-like of shape (search_volume, )
                Reward (i.e., auction outcome) obtained in each auction.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        n_samples: int, default=100000 (> 0)
            Number of samples to fit reward predictor.

        """
        if not self.use_reward_predictor:
            raise RuntimeError(
                "Please set the attribute, reward_predictor, before calling .fit_reward_predictor()"
            )
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            n_samples,
            name="n_samples",
            target_type=int,
            min_val=1,
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
        """Predict the reward (i.e., auction outcome) to determine bidding price.

        Note
        -------
        Intended to be used only when use_reward_predictor=True option.

        X and y of the prediction model is given as follows.
            X: array-like of shape (search_volume, ad_feature_dim + user_feature_dim + 1)
                Concatenated vector of contexts (ad_feature_vector + user_feature_vector) and timestep.

            y: array-like of shape (search_volume, )
                Reward (i.e., auction outcome) obtained in each auction.

        Parameters
        -------
        ad_ids: array-like of shape (search_volume, )
            IDs of the ads.

        user_ids: array-like of shape (search_volume, )
            IDs of the users.

        ad_feature_vector: array-like of shape (search_volume, ad_feature_dim)
            Feature vector of the ads.

        user_feature_vector: array-like of shape (search_volume, user_feature_dim)
            Feature vector of the users.

        timestep: {int, array-like of shape (search_volume, )} (> 0)
            Timestep in the RL environment.

        Returns
        -------
        predicted_rewards: ndarray of shape (search_volume, )
            Predicted reward for each auction.

        """
        check_array(
            ad_ids,
            name="ad_ids",
            expected_dim=1,
        )
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
        contexts = np.concatenate([ad_feature_vector, user_feature_vector], axis=1)

        if isinstance(timestep, int):
            timestep = np.full(ad_ids.shape[0], timestep)
        check_array(timestep, name="timestep", expected_dim=1, min_val=0)
        timestep = timestep.reshape((-1, 1))

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
        """Calculate the ground-truth reward (i.e., auction outcome) to determine bidding price.

        Parameters
        -------
        ad_ids: array-like of shape (search_volume, )
            IDs of the ads.

        user_ids: array-like of shape (search_volume, )
            IDs of the users.

        ad_feature_vector: array-like of shape (search_volume, ad_feature_dim)
            Feature vector of the ads.

        user_feature_vector: array-like of shape (search_volume, user_feature_dim)
            Feature vector of the users.

        timestep: {int, array-like of shape (search_volume, )}
            Timestep in the RL environment.

        Returns
        -------
        expected_rewards: array-like of shape(search_volume, )
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
