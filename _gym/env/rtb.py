"""Reinforcement Learning (RL) Environment for Real-Time Bidding (RTB)."""
from typing import Tuple, Optional, Union, Any
from tqdm import tqdm
import warnings

import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.utils import check_random_state

from _gym.utils import NormalDistribution
from _gym.simulator import RTBSyntheticSimulator
from _gym.policy import BasePolicy


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
            Note that the following bid price is individually determined for each auction.
                bid price = adjust rate * predicted/ground-truth reward ( * constant)

            Both discrete and continuous actions are acceptable.
            Note that the value should be within [0.1, 10].

        reward: int
            Total clicks/conversions gained during the timestep.

        discount_rate: int, 1
            Discount factor for cumulative reward calculation.
            Set discount_rate = 1 (i.e., no discount) in RTB.

        constraint: int
            Total cost should not exceed the initial budget.

    Parameters
    -------
    semi_synthetic: bool, default=False
        Whether to use semi-synthetic environment (RTBSemiSyntheticSimulator) or not.
        Otherwise the RTBSyntheticSimulator is used.
        (Currently, only semi_synthetic=False option is available.)

        If semi_synthetic=True, we fit simulator (especially WinningFunction, SecondPrice, CTR, CVR inside)
        from the real-world dataset.

    objective: str, default="conversion"
        Objective outcome (i.e., reward) of the auctions.
        Choose either from "click" or "conversion".

    action_type: str, default="discrete"
        Action type of the RL agent.
        Choose either from "discrete" or "continuous".

    action_dim: int, default=10
        Dimensions of the discrete action.
        Used only when action_type="discrete" option.

    action_meaning: Optional[NDArray[float]], default=None
        Dictionary which maps discrete action index into specific actions.
        Used when only when using action_type="discrete" option.

        Note that if None, the action meaning values automatically set to [0.1, 10] log sampled values.
            np.logspace(-1, 1, action_dim)

    reward_predictor: Optional[BaseEstimator], default=None
        Parameter in RTBSyntheticSimulator class.
        A machine learning model to predict the reward to determine the bidding price.
        If None, the ground-truth (expected) reward is used instead of the predicted one.

    step_per_episode: int, default=24
        Number of timesteps in an episode.

    initial_budget: int, default=10000
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

    standard_bid_price_distribution: NormalDistribution, default=NormalDistribution(mean=100, std=20)
        Parameter in RTBSyntheticSimulator class.
        Distribution of the bid price whose average impression probability is expected to be 0.5.

    minimum_standard_bid_price: Optional[int], default=None
        Parameter in RTBSyntheticSimulator class.
        Minimum value for standard bid price.
        If None, minimum_standard_bid_price is set to standard_bid_price_distribution.mean / 2.

    bid_scaler: Optional[Union[int, float]], default=None
        Parameter in RTBSyntheticSimulator class.
        Scaling factor (constant value) used for bid price determination.
        If None, _set_bid_scaler() function is called to set bid_scaler.

    trend_interval: Optional[int], default=None
        Parameter in RTBSyntheticSimulator class.
        Length of the ctr/cvr trend cycle.
        If None, trend_interval is set to step_per_episode.

    ad_sampling_rate: Optional[Union[NDArray[int], NDArray[float]]], shape (n_candidate_ads, ), default=None
        Sampling probalities to determine which ad (id) is used in each auction.

    user_sampling_rate: Optional[Union[NDArray[int], NDArray[float]]], shape (n_candidate_users, ), default=None
        Sampling probalities to determine which user (id) is used in each auction.

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

        # import necessary module from other library
        from sklearn.linear_model import LogisticRegression

        # initialize environment and define (RL) agent (i.e., policy)
        env = RTBEnv(reward_predictor=LogisticRegression())
        agent = RandomPolicy()

        # when using use_reward_predictor=True option,
        # pretrain reward predictor used for bidding price determination
        env.fit_reward_predictor(n_samples=10000)

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
        semi_synthetic: bool = False,
        objective: str = "conversion",  # "click"
        step_per_episode: int = 24,
        initial_budget: int = 10000,
        n_ads: int = 100,
        n_users: int = 100,
        ad_feature_dim: int = 5,
        user_feature_dim: int = 5,
        standard_bid_price_distribution: NormalDistribution = NormalDistribution(
            mean=50, std=5
        ),
        minimum_standard_bid_price: Optional[int] = None,
        bid_scaler: Optional[Union[int, float]] = None,
        trend_interval: Optional[int] = None,
        ad_sampling_rate: Optional[np.ndarray] = None,
        user_sampling_rate: Optional[np.ndarray] = None,
        search_volume_distribution: NormalDistribution = NormalDistribution(
            mean=200, std=20
        ),
        minimum_search_volume: int = 10,
        random_state: int = 12345,
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
        if not (isinstance(n_ads, int) and n_ads > 0):
            raise ValueError(f"n_ads must be a positive interger, but {n_ads} is given")
        if not (isinstance(n_users, int) and n_users > 0):
            raise ValueError(
                f"n_users must be a positive interger, but {n_users} is given"
            )
        if not (
            ad_sampling_rate is None
            or (
                isinstance(ad_sampling_rate, np.ndarray)
                and ad_sampling_rate.ndim == 1
                and ad_sampling_rate.min() >= 0
                and ad_sampling_rate.max() > 0
            )
        ):
            raise ValueError(
                "ad_sampling_rate must be an 1-dimensional NDArray of non-negative float values"
            )
        if not (
            user_sampling_rate is None
            or (
                isinstance(user_sampling_rate, np.ndarray)
                and user_sampling_rate.ndim == 1
                and user_sampling_rate.min() >= 0
                and user_sampling_rate.max() > 0
            )
        ):
            raise ValueError(
                "user_sampling_rate must be an NDArray of non-negative float values"
            )
        if ad_sampling_rate is not None and n_ads != len(ad_sampling_rate):
            raise ValueError("length of ad_sampling_rate must be equal to n_ads")
        if user_sampling_rate is not None and n_users != len(user_sampling_rate):
            raise ValueError("length of user_sampling_rate must be equal to n_users")
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

        # initialize simulator
        if semi_synthetic:
            raise NotImplementedError()
        else:
            self.simulator = RTBSyntheticSimulator(
                objective=objective,
                step_per_episode=step_per_episode,
                n_ads=n_ads,
                n_users=n_users,
                ad_feature_dim=ad_feature_dim,
                user_feature_dim=user_feature_dim,
                standard_bid_price_distribution=standard_bid_price_distribution,
                minimum_standard_bid_price=minimum_standard_bid_price,
                trend_interval=trend_interval,
                random_state=random_state,
            )

        self.objective = objective
        self.use_reward_predictor = self.simulator.use_reward_predictor

        # define observation space
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0.1]),
            high=np.array(
                [step_per_episode, initial_budget, np.inf, np.inf, 1, np.inf, 10]
            ),
            dtype=float,
        )
        self.obs_keys = [
            "timestep",
            "remaining budget",
            "budget consumption rate",
            "cost per mille of impression",
            "winning rate",
            "reward",
            "adjust rate",
        ]

        # define action space (adjust rate range)
        self.action_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=float)

        # define reward range
        self.reward_range = (0, np.inf)

        self.step_per_episode = step_per_episode
        self.initial_budget = initial_budget

        self.ad_ids = np.arange(n_ads)
        self.user_ids = np.arange(n_users)

        # sample feature vectors for both ads and users
        self.ads = self.random_.normal(size=(n_ads, ad_feature_dim))
        self.users = self.random_.normal(size=(n_users, user_feature_dim))

        if ad_sampling_rate is None:
            self.ad_sampling_rate = np.full(n_ads, 1 / n_ads)
        else:
            self.ad_sampling_rate = ad_sampling_rate / np.sum(ad_sampling_rate)

        if user_sampling_rate is None:
            self.user_sampling_rate = np.full(n_users, 1 / n_users)
        else:
            self.user_sampling_rate = user_sampling_rate / np.sum(user_sampling_rate)

        if isinstance(search_volume_distribution.mean, int):
            search_volume_distribution = NormalDistribution(
                mean=np.full(step_per_episode, search_volume_distribution.mean),
                std=np.full(step_per_episode, search_volume_distribution.std),
            )
        self.search_volumes = np.clip(
            search_volume_distribution.sample(size=100), minimum_search_volume, None
        ).astype(int)

        # just for idx of search_volumes to sample from
        self.T = 0

    def step(self, action: Union[int, float]) -> Tuple[Any]:
        """Rollout auctions arise during the timestep and return feedbacks to the agent.

        Note
        -------
        The rollout procedure is given as follows.
        1. Sample ads and users for (search volume, ) auctions occur during the timestep.

        2. Determine bid price. (In Bidder)
                bid price = adjust rate * predicted/ground-truth reward ( * constant)

        3. Calculate outcome probability and stochastically determine auction result. (in )
                auction results: cost (i.e., second price), impression, click, conversion

        4. Check if the cumulative cost during the timestep exceeds the remaining budget or not.
           (If exceeds, cancel the corresponding auction results.)

        5. Aggregate auction results and return feedbacks to the RL agent.

        Parameters
        -------
        action: float
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
        if not isinstance(action, float):
            raise ValueError(f"action must be a float value, but {action} is given")
        adjust_rate = action

        # sample ads and users for auctions occur in a timestep
        ad_ids = self.random_.choice(
            self.ad_ids,
            size=self.search_volumes[self.T % 100][self.t - 1],
            p=self.ad_sampling_rate,
        )
        user_ids = self.random_.choice(
            self.user_ids,
            size=self.search_volumes[self.T % 100][self.t - 1],
            p=self.user_sampling_rate,
        )
        contexts = self._map_idx_to_contexts(ad_ids, user_ids)

        # determine bid price
        # bid_prices =  # TODO

        # simulate auctions and gain results
        (
            costs,
            impressions,
            clicks,
            conversions,
        ) = self.simulator.simulate_auction(self.t, adjust_rate, ad_ids, user_ids, contexts, bid_prices)

        # check if auction bidding is possible
        masks = np.cumsum(costs) < self.remaining_budget
        total_cost = np.sum(costs * masks)
        total_impression = np.sum(impressions * masks)
        total_click = np.sum(clicks * masks)
        total_conversion = np.sum(conversions * masks)

        self.remaining_budget -= total_cost

        # prepare returns
        if self.objective == "click":
            reward = total_click
        else:
            reward = total_conversion

        # update timestep
        self.t += 1

        obs = {
            "timestep": self.t,
            "remaining budget": self.remaining_budget,
            "budget consumption rate": (
                self.prev_remaining_budget - self.remaining_budget
            )
            / self.prev_remaining_budget
            if self.prev_remaining_budget
            else 0,
            "cost per mille of impression": (total_cost * 1000) / total_impression
            if total_impression
            else 0,
            "winning rate": total_impression / len(bid_prices),
            "reward": reward,
            "adjust rate": adjust_rate,
        }

        done = self.t == self.step_per_episode

        # we use 'info' to obtain supplemental feedbacks beside rewards
        info = {
            "impression": total_impression,
            "click": total_click,
            "conversion": total_conversion,
            "average bid price": np.mean(bid_prices),
        }

        # update logs
        self.prev_remaining_budget = self.remaining_budget

        return np.array(list(obs.values())).astype(float), reward, done, info

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
        adjust_rate_ = self.action_space.sample()
        obs = {
            "timestep": self.t,
            "remaining budget": self.remaining_budget,
            "budget consumption rate": random_variable_[0],
            "cost per mille of impression": random_variable_[1],
            "winning rate": random_variable_[2],
            "reward": reward_,
            "adjust rate": adjust_rate_,
        }
        return np.array(list(obs.values())).astype(float)

    def render(self, mode: str = "human") -> None:
        """Maybe add some plot later."""
        pass

    def close(self) -> None:
        warnings.warn(".close() is not implemented")
        pass

    def seed(self, seed: int = None) -> None:
        warnings.warn(
            ".seed() is not implemented, please reset seed by initializing the environment"
        )
        pass

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

    def calc_on_policy_policy_value(
        self, evaluation_policy: BasePolicy, n_episodes: int = 10000
    ) -> float:
        """Rollout the RL agent (i.e., policy) and calculate mean episodic reward.

        Parameters
        -------
        evaluation_policy: BasePolicy
            The RL agent (i.e., policy) to be evaluated.

        n_episodes: int, default=10000
            Number of episodes to rollout.

        Returns
        -------
        mean_reward: float
            Mean episode reward calculated through rollout.

        """
        if not (isinstance(n_episodes, int) and n_episodes > 0):
            raise ValueError(
                "n_episodes must be a positive integer, but {n_episodes} is given"
            )

        total_reward = 0.0
        for _ in tqdm(
            np.arange(n_episodes),
            desc="[calc_on_policy_policy_value]",
            total=n_episodes,
        ):
            state = self.reset()
            done = False

            while not done:
                action, _ = evaluation_policy.act(state)  # predict
                state, reward, done, _ = self.step(action)
                total_reward += reward

        return total_reward / n_episodes
