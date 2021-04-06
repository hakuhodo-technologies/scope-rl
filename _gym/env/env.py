from typing import Tuple, Dict, List
from typing import Optional, Union, Any
from nptyping import Array

import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from env.function import NormalDistribution
from env.simulator import RTBSyntheticSimulator
from policy.policy import BasePolicy


class RTBEnv(gym.Env):
    def __init__(
        self,
        semi_synthetic: bool = False,
        objective: str = "conversion",  # "click"
        action_type: str = "discrete",  # "continuous"
        action_dim: int = 10,
        action_meaning: Optional[
            Dict[int, float]
        ] = None,  # maps categorical actions to adjust rate
        use_reward_predictor: bool = False,
        reward_predictor: Optional[BaseEstimator] = None,
        step_per_episode: int = 24,
        initial_budget: int = 10000,
        n_ads: int = 100,
        n_users: int = 100,
        ad_feature_dim: int = 5,
        user_feature_dim: int = 5,
        standard_bid_price: int = 100,
        trend_interval: Optional[int] = None,
        n_dices: int = 10,
        wf_alpha: float = 2.0,
        candidate_ads: Array[int] = np.array([0]),  # ad idxes
        candidate_users: Array[int] = np.array(range(10)),  # user idxes
        candidate_ad_sampling_prob: Optional[Array[float]] = None,
        candidate_user_sampling_prob: Optional[Array[float]] = None,
        search_volume_distribution: Optional[List[NormalDistribution]] = None,
        random_state: int = 12345,
    ):
        super().__init__()

        # assertion
        if not (
            isinstance(action_type, str) and action_type in ["discrete", "continuous"]
        ):
            raise ValueError(
                f'action_type must be either "discrete" or "continuous", but {action_type} is given'
            )
        if not (isinstance(action_dim, int) and action_dim > 0):
            raise ValueError(
                f"action_dim must be a positive interger, but {action_dim} is given"
            )
        if action_meaning is not None:
            if len(self.action_meaning) != self.action_dim:
                raise ValueError("action_meaning must have same size with action_dim")
            if (
                min(self.action_meaning.values()) < 0.1
                or max(self.action_meaning.values()) > 10
            ):
                raise ValueError("the values of action_meaning must be in [0.1, 10]")
        if not (isinstance(step_per_episode, int) and step_per_episode > 0):
            raise ValueError(
                f"step_per_episode must be a positive interger, but {step_per_episode} is given"
            )
        if not (isinstance(initial_budget, int) and initial_budget > 0):
            raise ValueError(
                f"action_dim must be a positive interger, but {action_dim} is given"
            )
        if candidate_ad_sampling_prob is not None and len(candidate_ads) != len(
            candidate_ad_sampling_prob
        ):
            raise ValueError(
                f"candidate_ads and candidate_ad_sampling_prob must have same length"
            )
        if candidate_user_sampling_prob is not None and len(candidate_users) != len(
            candidate_user_sampling_prob
        ):
            raise ValueError(
                f"candidate_users and candidate_user_sampling_prob must have same length"
            )
        if self.candidate_ads.min() < 0 or self.candidate_ads.max() >= n_ads:
            raise ValueError(
                f"candidate_ads must be chosen from integer values in [0, n_ads)"
            )
        if self.candidate_users.min() < 0 or self.candidate_users.max() >= n_users:
            raise ValueError(
                f"candidate_users must be chosen from integer values in [0, n_users)"
            )
        if (
            search_volume_distribution is not None
            and len(search_volume_distribution) != self.step_per_episode
        ):
            raise ValueError(
                f"length of search_volume_distribution must be same with step_per_episode"
            )
        if random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        # initialize simulator
        if semi_synthetic:
            raise ValueError(
                "currently semi-synthetic env have not been implemented, please choose semi_synthetic=False option"
            )
        else:
            self.simulator = RTBSyntheticSimulator(
                objective=objective,
                use_reward_predictor=use_reward_predictor,
                reward_predictor=reward_predictor,
                n_ads=n_ads,
                n_users=n_users,
                ad_feature_dim=ad_feature_dim,
                user_feature_dim=user_feature_dim,
                standard_bid=standard_bid_price,
                trend_interval=trend_interval,
                n_dices=n_dices,
                wf_alpha=wf_alpha,
                random_state=random_state,
            )

        self.objective = objective

        # define observation space
        self.observation_space = Box(
            low=[0, 0, 0, 0, 0, 0],
            high=[step_per_episode, initial_budget, np.inf, np.inf, 1, np.inf]
            # observations = (timestep, remaining_budget, BCR, CPM, WR, reward)
        )

        # define action space
        if action_type == "discrete":
            self.action_space = Discrete(action_dim)

            if self.action_meaning is None:
                self.action_meaning = dict(
                    zip(range(action_dim), np.logspace(-1, 1, action_dim))
                )

        else:  # "continuous"
            self.action_space = Box(low=0.1, high=10)

        self.step_per_episode = step_per_episode
        self.initial_budget = initial_budget

        self.candidate_ads = candidate_ads
        self.candidate_users = candidate_users

        if candidate_ad_sampling_prob is None:
            candidate_ad_sampling_prob = (
                np.array([1 / len(candidate_ads)] * candidate_ads),
            )
        if candidate_user_sampling_prob is None:
            candidate_user_sampling_prob = (
                np.array([1 / len(candidate_users)] * candidate_users),
            )

        self.candidate_ad_sampling_prob = candidate_ad_sampling_prob / np.sum(
            candidate_ad_sampling_prob
        )
        self.candidate_user_sampling_prob = candidate_user_sampling_prob / np.sum(
            candidate_user_sampling_prob
        )

        if search_volume_distribution is None:
            search_volume_distribution = [NormalDistribution(mean=10, std=0.0)]
        *step_per_episode,

        self.search_volumes = np.zeros(step_per_episode, 100)
        for i in range(step_per_episode):
            self.search_volumes[i] = search_volume_distribution.sample(size=100).astype(
                int
            )

        # just for idx of search_volumes to sample from
        self.T = 0

    def step(self, action: Union[int, float]) -> Tuple[Any]:
        # sample ads and users for auctions occur in a timestep
        ad_ids = self.random_.choice(
            self.candidate_ads,
            size=self.search_volumes[self.t - 1][self.T % 100],
            p=self.candidate_ad_sampling_prob,
        )
        user_ids = self.random_.choice(
            self.candidate_users,
            size=self.search_volumes[self.t - 1][self.T % 100],
            p=self.candidate_user_sampling_prob,
        )

        # simulate auctions and gain results
        (
            bid_prices,
            costs,
            impressions,
            clicks,
            conversions,
        ) = self.simulator.simulate_auction(self.t, action, ad_ids, user_ids)

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

        obs = {
            "timestep": self.t,
            "remaining_budget": self.remaining_budget,
            "budget consumption rate": (
                self.prev_remaining_budget - self.remaining_budget
            )
            / self.prev_remaining_budget,
            "cost per mille of impression": (total_cost * 1000) / total_impression,
            "winning rate": total_impression / len(bid_prices),
            "reward": reward,
            "adjust rate": action,
        }

        done = self.t == self.step_per_episode

        # we use 'info' to obtain supplemental feedbacks beside rewards
        info = {
            "impression": total_impression,
            "click": total_click,
            "conversion": total_conversion,
        }

        # update logs
        self.t += 1
        self.prev_remaining_budget = self.remaining_budget

        return np.array(obs.values()).astype(float), reward, done, info

    def reset(self) -> Array[float]:
        # initialize internal env state
        self.T += 1
        self.t = 0
        self.prev_remaining_budget = self.remaining_budget = self.initial_budget

        # initialize obs
        obs = {
            "timestep": self.t,
            "remaining_budget": self.remaining_budget,
            "budget consumption rate": 0,
            "cost per mille of impression": 0,
            "winning rate": 0,
            "reward": 0,
            "adjust rate": 0,
        }

        return np.array(obs.values()).astype(float)

    def render(self, mode: str = "human") -> None:
        """maybe add some plot later"""
        pass

    def close(self) -> None:
        pass

    def fit_reward_predictor(self, n_samples: int = 10000) -> None:
        """pre-train reward regression used for calculating bids"""
        self.simulator.fit_reward_predictor(self.step_per_episode, n_samples)

    def calc_ground_truth_policy_value(
        self, evaluation_policy: BasePolicy, n_episodes: int = 10000
    ) -> None:
        """rollout policy and calculate mean episodic reward"""
        total_reward = 0.0
        for _ in range(n_episodes):
            state = self.reset()
            done = False

            while not done:
                action, _ = evaluation_policy.act(state)  # fix later
                state, reward, done, _ = self.step(action)
                total_reward += reward

        return total_reward / n_episodes
