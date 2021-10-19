"""Reinforcement Learning (RL) Environment for Real-Time Bidding (RTB)."""
from typing import Tuple, Optional, Union, Any
from tqdm import tqdm
import warnings

import gym
from gym.spaces import Box
import numpy as np
from sklearn.utils import check_random_state

from _gym.utils import NormalDistribution
from _gym.types import Action

from .bidder import Bidder
from .simulator.rtb_synthetic import RTBSyntheticSimulator


class RTBEnv(gym.Env):
    """Class for Real-Time Bidding (RTB) environment for reinforcement learning (RL) agent to interact.

    Note
    -------
    Adopt OpenAI Gym like interface. See Examples below for the usage.
    Use RTBSyntheticSimulator/RTBSemiSyntheticSimulator in simulator.py to collect auction results.

    Constrained Markov Decision Process (CMDP) definition are given as follows:
        timestep: int
            Set 24h a day or seven days per week for instance.
            We have (search volume, ) auctions during a timestep.
            Note that each single auction do NOT correspond to the timestep.

        state: NDArray[float], shape (7, )
            Statistical feedbacks of auctions during the timestep, including following values.
                - timestep
                - remaining budget
                - impression level features at the previous timestep
                  (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
                - adjust rate (i.e., RL agent action) at previous timestep

        action: Union[int, float]
            Adjust rate parameter used for the bid price calculation as follows.
            (Bid price is individually determined for each auction.)
                bid price = adjust rate * ground-truth reward ( * constant)
            Note that, we can also use predicted reward instead of ground-truth reward in the above equation
            if we use CustomizedRTBEnv Wrapper.

            Acceptable action range is [0, np.infty).

        reward: int
            Total clicks/conversions gained during the timestep.

        discount_rate: int, 1
            Discount factor for cumulative reward calculation.
            Set discount_rate = 1 (i.e., no discount) in RTB.

        constraint: int
            Total cost should not exceed the initial budget.

    Parameters
    -------

    objective: str, default="conversion"
        Objective outcome (i.e., reward) of the auctions.
        Choose either from "click" or "conversion".

    step_per_episode: int, default=7
        Number of timesteps in an episode.

    initial_budget: int, default=3000
        Initial budget (i.e., constraint) for bidding during an episode.

    n_ads: int, default=100
        Number of ads used for auction bidding.

    n_users: int, default=100
        Number of users used for auction bidding.

    ad_feature_dim: int, default=5
        Parameter in RTBSyntheticSimulator class.
        Dimensions of the ad feature vectors.

    user_feature_dim: int, default=5
        Parameter in RTBSyntheticSimulator class.
        Dimensions of the user feature vectors.

    ad_sampling_rate: Optional[Union[NDArray[int], NDArray[float]]], shape (n_candidate_ads, ), default=None
        Parameter in RTBSyntheticSimulator class.
        Sampling probalities to determine which ad (id) is used in each auction.

    user_sampling_rate: Optional[Union[NDArray[int], NDArray[float]]], shape (n_candidate_users, ), default=None
        Parameter in RTBSyntheticSimulator class.
        Sampling probalities to determine which user (id) is used in each auction.

    standard_bid_price_distribution: NormalDistribution, default=NormalDistribution(mean=100, std=20)
        Parameter in RTBSyntheticSimulator class.
        Distribution of the bid price whose average impression probability is expected to be 0.5.

    minimum_standard_bid_price: Optional[int], default=None
        Parameter in RTBSyntheticSimulator class.
        Minimum value for standard bid price.
        If None, minimum_standard_bid_price is set to standard_bid_price_distribution.mean / 2.

    trend_interval: Optional[int], default=None
        Parameter in RTBSyntheticSimulator class.
        Length of the ctr/cvr trend cycle.
        If None, trend_interval is set to step_per_episode.

    search_volume_distribution: NormalDistribution, default=NormalDistribution(mean=30, std=10)
        Search volume distribution for each timestep.

    minimum_search_volume: int, default = 10
        Minimum search volume at each timestep.

    random_state: int, default=12345
        Random state.

    Examples
    -------

    .. codeblock:: python

        # import necessary module from _gym
        from _gym.env import RTBEnv
        from _gym.policy import RandomPolicy

        # initialize environment and define (RL) agent (i.e., policy)
        env = RTBEnv()
        agent = RandomPolicy(env)

        # OpenAI Gym like interaction with agent
        for episode in range(1000):
            obs = env.reset()
            done = False

            while not done:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)

        # calculate on-policy policy value
        performance = env.calc_on_policy_policy_value(
            evaluation_policy=agent,
            n_episodes=10000,
        )

    References
    -------

    """

    def __init__(
        self,
        objective: str = "conversion",  # "click"
        step_per_episode: int = 7,
        initial_budget: int = 3000,
        n_ads: int = 100,
        n_users: int = 100,
        ad_feature_dim: int = 5,
        user_feature_dim: int = 5,
        ad_sampling_rate: Optional[np.ndarray] = None,
        user_sampling_rate: Optional[np.ndarray] = None,
        standard_bid_price_distribution: NormalDistribution = NormalDistribution(
            mean=50, std=5
        ),
        minimum_standard_bid_price: Optional[int] = None,
        trend_interval: Optional[int] = None,
        search_volume_distribution: NormalDistribution = NormalDistribution(
            mean=200, std=20
        ),
        minimum_search_volume: int = 10,
        random_state: Optional[int]=None,
    ):
        super().__init__()
        if not (isinstance(objective, str) and objective in ["click", "conversion"]):
            raise ValueError(
                f'objective must be either "click" or "conversion", but {objective} is given'
            )
        if not (isinstance(step_per_episode, int) and step_per_episode > 0):
            raise ValueError(
                f"step_per_episode must be a positive interger, but {step_per_episode} is given"
            )
        if not (isinstance(initial_budget, int) and initial_budget > 0):
            raise ValueError(
                f"initial_budget must be a positive interger, but {initial_budget} is given"
            )
        if not (
            isinstance(search_volume_distribution.mean, (int, float))
            and search_volume_distribution.mean > 0
        ) and not (
            isinstance(search_volume_distribution.mean, np.ndarray)
            and search_volume_distribution.mean.ndim == 1
            and search_volume_distribution.mean.min() > 0
        ):
            raise ValueError(
                "search_volume_distribution.mean must be a positive float value or an NDArray of positive float values"
            )
        if not (
            isinstance(search_volume_distribution.mean, (int, float))
            or len(search_volume_distribution.mean) == step_per_episode
        ):
            raise ValueError(
                "length of search_volume_distribution must be equal to step_per_episode"
            )
        if not (isinstance(minimum_search_volume, int) and minimum_search_volume > 0):
            raise ValueError(
                f"minimum_search_volume must be a positive integer, but {minimum_search_volume} is given"
            )
        if random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(random_state)

        self.objective = objective

        if trend_interval is None:
            trend_interval = step_per_episode

        # initialize simulator and bidder
        self.simulator = RTBSyntheticSimulator(
            n_ads=n_ads,
            n_users=n_users,
            ad_feature_dim=ad_feature_dim,
            user_feature_dim=user_feature_dim,
            ad_sampling_rate=ad_sampling_rate,
            user_sampling_rate=user_sampling_rate,
            standard_bid_price_distribution=standard_bid_price_distribution,
            minimum_standard_bid_price=minimum_standard_bid_price,
            trend_interval=trend_interval,
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

        self.step_per_episode = step_per_episode
        self.initial_budget = initial_budget

        if isinstance(search_volume_distribution.mean, int):
            search_volume_distribution = NormalDistribution(
                mean=np.full(step_per_episode, search_volume_distribution.mean),
                std=np.full(step_per_episode, search_volume_distribution.std),
            )
        self.search_volumes = np.clip(
            search_volume_distribution.sample(size=100), minimum_search_volume, None
        ).astype(int)

        # idx of search_volumes to sample from
        self.T = 0

    @property
    def standard_bid_price(self):
        return self.simulator.standard_bid_price

    def step(self, action: Action) -> Tuple[Any]:
        """Rollout auctions arise during the timestep and return feedbacks to the agent.

        Note
        -------
        The rollout procedure is given as follows.
        1. Sample ads and users for (search volume, ) auctions occur during the timestep.

        2. Determine bid price. (In Bidder)
            bid price = adjust rate * predicted/ground-truth reward ( * constant)

        3. Calculate outcome probability and stochastically determine auction result. (in Simulator)
            auction results: cost (i.e., second price), impression, click, conversion

        4. Check if the cumulative cost during the timestep exceeds the remaining budget or not.
           (If exceeds, cancel the corresponding auction results.)

        5. Aggregate auction results and return feedbacks to the RL agent.

        Parameters
        -------
        action: Action (Union[int, float, np.integer, np.float, np.ndarray])
            RL agent action which corresponds to the adjust rate parameter used for bid price calculation.

        Returns
        -------
        feedbacks: Tuple
            obs: NDArray[float], shape (7, )
                Statistical feedbacks of auctions during the timestep.
                Corresponds to RL state, which include following components.
                    - timestep
                    - remaining budget
                    - impression level features at the previous timestep
                      (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
                    - adjust rate (i.e., agent action) at previous timestep

            reward: int
                Total clicks/conversions gained during the timestep.

            done: bool
                Wether the episode end or not.

            info: Dict[str, int]
                Additional feedbacks (total impressions, clicks, and conversions) for analysts.
                Note that those feedbacks are intended to be unobservable for the RL agent.

        """
        err = False
        if isinstance(action, (int, float, np.integer, np.float)):
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
        search_volume = self.search_volumes[self.T % 100][self.t - 1]
        ad_ids, user_ids = self.simulator.generate_auction(search_volume)

        # 2. determine bid price
        bid_prices = self.bidder.determine_bid_price(
            self.t, adjust_rate, ad_ids, user_ids
        )

        # 3. simulate auctions and gain results
        (
            costs,
            impressions,
            clicks,
            conversions,
        ) = self.simulator.calc_and_sample_outcome(self.t, ad_ids, user_ids, bid_prices)

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
        else:
            reward = total_conversion

        done = self.t == self.step_per_episode - 1

        if done:
            obs = self.reset()

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
            "impression": total_impression,
            "click": total_click,
            "conversion": total_conversion,
            "average_bid_price": np.mean(bid_prices),
        }

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        """Initialize the environment.

        Note
        -------
        Remaining budget is initialized to the initial budget of an episode.

        Returns
        -------
        obs: NDArray[float], shape (7, )
            Statistical feedbacks of auctions during the timestep.
            Corresponds to RL state, which include following components.
                - timestep
                - remaining budget
                - impression level features at the previous timestep
                  (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
                - adjust rate (i.e., agent action) at previous timestep

        """
        # initialize internal env state
        self.T += 1
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
        return np.array(list(obs.values())).astype(float)

    def render(self, mode: str = "human") -> None:
        pass

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int]=None) -> None:
        self.random_ = check_random_state(seed)
