# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Reinforcement Learning (RL) Environment for Real-Time Bidding (RTB)."""
from typing import Tuple, Optional, Any

import gym
from gym.spaces import Box
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .simulator.bidder import Bidder
from .simulator.rtb_synthetic import RTBSyntheticSimulator
from .simulator.base import (
    BaseWinningPriceDistribution,
    BaseClickAndConversionRate,
)
from .simulator.function import (
    WinningPriceDistribution,
    ClickThroughRate,
    ConversionRate,
)
from ..utils import NormalDistribution
from ..types import Action, Numeric


class RTBEnv(gym.Env):
    """Class for Real-Time Bidding (RTB) environment for reinforcement learning (RL) agent to interact.

    Bases: :class:`gym.Env`

    Imported as: :class:`rtbgym.RTBEnv`

    Note
    -------
    RTBGym works with OpenAI Gym and Gymnasium-like interface. See Examples below for the usage.
    This environment uses :class:`RTBSyntheticSimulator` to collect auction results.

    Constrained Markov Decision Process (CMDP) definition are given as follows:
        timestep: int (> 0)
            Set 24h a day or seven days per week for instance.
            We have (search volume, ) auctions during a timestep.
            Note that each single auction do NOT correspond to the timestep.

        state: array-like of shape (7, )
            Statistical feedbacks of auctions during the timestep, including following values.

            - timestep
            - remaining budget
            - impression level features at the previous timestep (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
            - adjust rate (i.e., RL agent action) at the previous timestep

        action: {int, float, array-like of shape (1, )} (>= 0)
            Adjust rate parameter used for determining the bid price as follows.
            (Bid price is individually determined for each auction.)

            .. math::

                {bid price}_{t, i} = {adjust rate}_{t} \\times {predicted reward}_{t,i} ( \\times {const.})

            Note that you can also use predicted reward instead of the ground-truth reward in the above equation.
            Please also refer to CustomizedRTBEnv Wrapper.

        reward: int (>= 0)
            Total clicks/conversions gained during the timestep.

        discount_rate: float
            Discount factor for cumulative reward calculation.
            Set discount_rate = 1 (i.e., no discount) in RTB.

        constraint: float (> 0)
            Total cost should not exceed the initial budget.

    Parameters
    -------

    objective: {"click", "conversion"}, default="conversion"
        Objective outcome (i.e., reward) of the auctions.

    cost_indicator: {"click", "conversion"}, default="click"
        Defines when the cost arises.

    step_per_episode: int, default=7 (> 0)
        Number of timesteps in an episode.

    initial_budget: int, default=3000 (> 0)
        Initial budget (i.e., constraint) for an episode.

    n_ads: int, default=100 (> 0)
        Number of (candidate) ads used for auction bidding.

    n_users: int, default=100 (> 0)
        Number of (candidate) users used for auction bidding.

    ad_feature_dim: int, default=5 (> 0)
        Dimension of the ad feature vectors.

    user_feature_dim: int, default=5 (> 0)
        Dimension of the user feature vectors.

    ad_feature_vector: ndarray of shape (n_ads, ad_feature_dim), default=None
        Feature vectors that characterizes each ad.

    user_feature_vector: ndarray of shape (n_users, user_feature_dim), default=None
        Feature vectors that characterizes each user.

    ad_sampling_rate: ndarray of shape (step_per_episode, n_ads), default=None
        Sampling probabilities to determine which ad (id) is used in each auction.

    user_sampling_rate: ndarray of shape (step_per_episode, n_users), default=None
        Sampling probabilities to determine which user (id) is used in each auction.

    WinningPriceDistribution: BaseWinningPriceDistribution
        Winning price distribution of auctions.
        Both class and instance are acceptable.

    ClickThroughRate: BaseClickAndConversionRate
        Click through rate (i.e., click / impression).
        Both class and instance are acceptable.

    ConversionRate: BaseClickAndConversionRate
        Conversion rate (i.e., conversion / click).
        Both class and instance are acceptable.

    standard_bid_price_distribution: NormalDistribution, default=None
        Distribution of the bid price whose average impression probability is expected to be 0.5.

    minimum_standard_bid_price: int, default=None (> 0)
        Minimum value for standard bid price.
        If `None`, minimum_standard_bid_price is set to :class:`standard_bid_price_distribution.mean / 2`.

    search_volume_distribution: NormalDistribution, default=None
        Search volume distribution for each timestep.

    minimum_search_volume: int, default = 10 (> 0)
        Minimum search volume at each timestep.

    random_state: int, default=None (>= 0)
        Random state.

    Examples
    -------

    Setup:

    .. code-block:: python

        # import necessary module from rtbgym
        from rtbgym import RTBEnv
        from scope_rl.policy import OnlineHead
        from scope_rl.ope.online import calc_on_policy_policy_value

        # import necessary module from other libraries
        from d3rlpy.algos import RandomPolicy
        from d3rlpy.preprocessing import MinMaxActionScaler

        # initialize environment
        env = RTBEnv(random_state=12345)

        # the following commands also work
        # import gym
        # env = gym.make("RTBEnv-continuous-v0")

        # define (RL) agent (i.e., policy)
        agent = OnlineHead(
            RandomPolicy(
                action_scaler=MinMaxActionScaler(
                    minimum=0.1,
                    maximum=10,
                )
            ),
            name="random",
        )
        agent.build_with_env(env)

    Interaction:

    .. code-block:: python

        # OpenAI Gym and Gymnasium-like interaction with agent
        for episode in range(1000):
            obs, info = env.reset()
            done = False

            while not done:
                action = agent.predict_online(obs)
                obs, reward, done, truncated, info = env.step(action)

    Online Evaluation:

    .. code-block:: python

        # calculate on-policy policy value
        on_policy_performance = calc_on_policy_policy_value(
            env,
            agent,
            n_trajectories=100,
            random_state=12345
        )

    Output:

    .. code-block:: python

        >>> on_policy_performance

        13.44

    References
    -------
    Di Wu, Xiujun Chen, Xun Yang, Hao Wang, Qing Tan, Xiaoxun Zhang, Jian Xu, and Kun Gai.
    "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising." 2018.

    Jun Zhao, Guang Qiu, Ziyu Guan, Wei Zhao, and Xiaofei He.
    "Deep Reinforcement Learning for Sponsored Search Real-time Bidding." 2018.

    Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba.
    "OpenAI Gym." 2016.

    """

    def __init__(
        self,
        objective: str = "conversion",  # "impression", "click"
        cost_indicator: str = "click",  # "impression", "conversion"
        step_per_episode: int = 7,
        initial_budget: int = 3000,
        n_ads: int = 100,
        n_users: int = 100,
        ad_feature_dim: int = 5,
        user_feature_dim: int = 5,
        ad_feature_vector: Optional[np.ndarray] = None,
        user_feature_vector: Optional[np.ndarray] = None,
        ad_sampling_rate: Optional[np.ndarray] = None,
        user_sampling_rate: Optional[np.ndarray] = None,
        WinningPriceDistribution: BaseWinningPriceDistribution = WinningPriceDistribution,
        ClickThroughRate: BaseClickAndConversionRate = ClickThroughRate,
        ConversionRate: BaseClickAndConversionRate = ConversionRate,
        standard_bid_price_distribution: Optional[NormalDistribution] = None,
        minimum_standard_bid_price: Optional[int] = None,
        search_volume_distribution: Optional[NormalDistribution] = None,
        minimum_search_volume: int = 10,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if objective not in ["click", "conversion"]:
            raise ValueError(
                f'objective must be either "click" or "conversion", but {objective} is given'
            )
        self.objective = objective

        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        self.step_per_episode = step_per_episode

        check_scalar(
            initial_budget,
            name="initial_budget",
            target_type=int,
            min_val=1,
        )
        self.initial_budget = initial_budget

        if random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(random_state)

        # initialize simulator and bidder
        self.simulator = RTBSyntheticSimulator(
            cost_indicator=cost_indicator,
            step_per_episode=step_per_episode,
            n_ads=n_ads,
            n_users=n_users,
            ad_feature_dim=ad_feature_dim,
            user_feature_dim=user_feature_dim,
            ad_feature_vector=ad_feature_vector,
            user_feature_vector=user_feature_vector,
            ad_sampling_rate=ad_sampling_rate,
            user_sampling_rate=user_sampling_rate,
            WinningPriceDistribution=WinningPriceDistribution,
            ClickThroughRate=ClickThroughRate,
            ConversionRate=ConversionRate,
            standard_bid_price_distribution=standard_bid_price_distribution,
            minimum_standard_bid_price=minimum_standard_bid_price,
            search_volume_distribution=search_volume_distribution,
            minimum_search_volume=minimum_search_volume,
            random_state=random_state,
        )
        self.bidder = Bidder(
            simulator=self.simulator,
            objective=self.objective,
            random_state=random_state,
        )
        self.bidder.auto_fit_scaler(step_per_episode=step_per_episode)

        # define observation space
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array(
                [step_per_episode, initial_budget, np.inf, np.inf, 1, np.inf, np.inf]
            ),
            dtype=float,
        )
        self.obs_keys = [
            "timestep",
            "remaining_budget",
            "budget_consumption_rate",
            "cost_per_mille_of_impression",
            "winning_rate",
            "reward",
            "adjust_rate",
        ]

        # define action space (adjust rate range)
        self.action_type = "continuous"
        self.action_dim = 1
        self.action_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=float)

        # define reward range
        self.reward_range = (0, np.inf)

    @property
    def standard_bid_price(self):
        return self.simulator.standard_bid_price

    def step(self, action: Action) -> Tuple[Any]:
        """Rollout auctions arise during the timestep and return feedbacks to the agent.

        Note
        -------
        The rollout procedure is given as follows.

        1. Sample ads and users for (search volume, ) auctions occur during the timestep. (in Simulator)

        2. Determine bid price. (In Bidder)

        3. Calculate outcome probability and stochastically determine auction result. (in Simulator) The auction results include cost (i.e., second price), impression, click, and conversion.

        4. Check if the cumulative cost during the timestep exceeds the remaining budget or not. (If exceeds, cancel the corresponding auction results.)

        5. Aggregate auction results and return feedbacks to the RL agent.

        Parameters
        -------
        action: {int, float, array-like of shape (1, )} (>= 0)
            RL agent action which corresponds to the adjust rate parameter used for bid price calculation.

        Returns
        -------
        feedbacks: Tuple
            obs: ndarray of shape (7, )
                Statistical feedbacks of auctions during the timestep.
                Corresponds to RL state, which include following components.
                    - timestep
                    - remaining budget
                    - impression level features at the previous timestep
                      (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
                    - adjust rate (i.e., agent action) at the previous timestep

            reward: int (>= 0)
                Total clicks/conversions gained during the timestep.

            done: bool
                Whether the episode end or not.

            info: dict
                Additional feedbacks (total impressions, clicks, and conversions) that may be useful for the package users.
                These are unavailable to the agent.

        """
        err = False
        if isinstance(action, Numeric):
            if not action >= 0:
                err = True
        elif isinstance(action, np.ndarray):
            if not action.shape == (1,):
                err = True
            else:
                action = action[0]
        else:
            err = True

        if err:
            raise ValueError(
                f"action must be a non-negative float value, but {action} is given"
            )

        adjust_rate = action

        # 1. sample ads and users for auctions occur in a timestep
        ad_ids, user_ids = self.simulator.generate_auction(timestep=self.t)

        # 2. determine bid price
        bid_prices = self.bidder.determine_bid_price(
            timestep=self.t, adjust_rate=adjust_rate, ad_ids=ad_ids, user_ids=user_ids
        )

        # 3. simulate an auctions and gain results
        (
            costs,
            impressions,
            clicks,
            conversions,
        ) = self.simulator.calc_and_sample_outcome(
            timestep=self.t, ad_ids=ad_ids, user_ids=user_ids, bid_prices=bid_prices
        )

        # 4. check if auction bidding is possible
        masks = np.cumsum(costs) < self.remaining_budget
        total_cost = np.sum(costs * masks)
        total_impression = np.sum(impressions * masks)
        total_click = np.sum(clicks * masks)
        total_conversion = np.sum(conversions * masks)

        self.remaining_budget -= total_cost

        # 5. prepare returns
        if self.objective == "click":
            reward = total_click
        elif self.objective == "conversion":
            reward = total_conversion

        done = self.t == self.step_per_episode - 1

        if done:
            obs, info = self.reset()

        else:
            # update timestep
            self.t += 1

            obs = {
                "timestep": self.t,
                "remaining_budget": self.remaining_budget,
                "budget_consumption_rate": (
                    self.prev_remaining_budget - self.remaining_budget
                )
                / self.prev_remaining_budget
                if self.prev_remaining_budget
                else 0,
                "cost_per_mille_of_impression": (total_cost * 1000) / total_impression
                if total_impression
                else 0,
                "winning_rate": total_impression / len(bid_prices),
                "reward": reward,
                "adjust_rate": adjust_rate,
            }
            obs = np.array(list(obs.values())).astype(float)

            # update logs
            self.prev_remaining_budget = self.remaining_budget

        # we use 'info' to obtain supplemental feedbacks beside rewards
        info = {
            "search_volume": len(bid_prices),
            "impression": total_impression,
            "click": total_click,
            "conversion": total_conversion,
            "average_bid_price": np.mean(bid_prices),
        }

        return obs, reward, done, False, info

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Initialize the environment.

        Note
        -------
        Remaining budget is initialized to the initial budget of an episode.

        Parameters
        -------
        seed: Optional[int], default=None
            Random state.

        Returns
        -------
        obs: ndarray of shape (7, )
            Statistical feedbacks of auctions during the timestep.
            Corresponds to RL state, which include following components.
                - timestep
                - remaining budget
                - impression level features at the previous timestep
                  (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
                - adjust rate (i.e., agent action) at the previous timestep

        info: (empty) dict
            Additional information that may be useful for the package users.
            This is unavailable to the RL agent.

        """
        if seed is not None:
            self.random_ = check_random_state(seed)
            self.simulator.random_ = check_random_state(seed)
            self.simulator.search_volume_distribution.random_ = check_random_state(seed)
            self.simulator.winning_price_distribution.random_ = check_random_state(seed)
            self.simulator.ctr.random_ = check_random_state(seed)
            self.simulator.cvr.random_ = check_random_state(
                seed + 1
            )  # to differentiate CVR from CTR

        # initialize internal env state
        self.t = 0
        self.prev_remaining_budget = self.remaining_budget = self.initial_budget

        # initialize obs
        random_variable_ = self.random_.uniform(size=3)
        reward_ = self.random_.randint(3)
        adjust_rate_ = self.action_space.sample()[0]
        obs = {
            "timestep": self.t,
            "remaining_budget": self.remaining_budget,
            "budget_consumption_rate": random_variable_[0],
            "cost_per_mille_of_impression": random_variable_[1],
            "winning_rate": random_variable_[2],
            "reward": reward_,
            "adjust_rate": adjust_rate_,
        }
        return np.array(list(obs.values())).astype(float), {}

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
