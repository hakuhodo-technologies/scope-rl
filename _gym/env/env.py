"""Reinforcement Learning (RL) Environment for Real-Time Bidding (RTB)."""
from typing import Tuple, Dict, List
from typing import Optional, Union, Any
from nptyping import NDArray
import warnings

import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from env.function import NormalDistribution
from env.simulator import RTBSyntheticSimulator
from policy.policy import BasePolicy


class RTBEnv(gym.Env):
    """Class for Real-Time Bidding (RTB) environment for reinforcement learning (RL) agent to interact.

    Note
    -------
    Adopt OpenAI Gym like interface. See Examples below for the usage.
    Use RTBSyntheticSimulator/RTBSemiSyntheticSimulator in simulator.py to collect auction results.

    Constrained Markov Decision Process (CMDP) definition are given as follows:
        timestep: int.
            Set 24h a day or seven days per week for instance.
            We have (search volume, ) auctions during a timestep.
            Note that each single auction do NOT correspond to the timestep.

        state: NDArray[float], shape (7, ).
            Statistical feedbacks of auctions during the timestep, including following values.
                - timestep
                - remaining budget
                - impression level features at the previous timestep
                  (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
                - adjust rate (i.e., RL agent action) at previous timestep

        action: Unioun[int, float].
            Adjust rate parameter used for the bid price calculation as follows.
            Note that the following bid price is individually determined for each auction.
                bid price = adjust rate * predicted/ground-truth reward ( * constant)

            Both discrete and continuous actions are acceptable.
            Note that the value should be within [0.1, 10].

        reward: int.
            Total clicks/conversions gained during the timestep.

        discount_rate: int, 1.
            Discount factor for cululative reward calculation.
            Set discount_rate = 1 (i.e., no discount) in RTB.

        constraint: int.
            Total cost should not exceed the initial budget.

    Parameters
    -------
    semi_synthetic: bool, default=False.
        Whether to use semi-synthetic environment (RTBSemiSyntheticSimulator) or not.
        Otherwise the RTBSyntheticSimulator is used.
        (Currently, only semi_synthetic=False option is available.)

        If semi_eynthetic=True, we fit simulator (especially WinningFuction, SecondPrice, CTR, CVR inside)
        from the real-world dataset.

    objective: str, default="conversion".
        Objective outcome (i.e., reward) of the auctions.
        Choose either from "click" or "conversion".

    action_type: str, default="discrete".
        Action type of the RL agent.
        Choose either from "discrete" or "continuous".

    action_dim: Optional[int], default=10.
        Dimentions of the discrete action.
        Required and used only when using action_type="discrete" option.

    action_meaning: Optional[Dict[int, float]], default=None.
        Dictionary which maps discrete action index into specific actions.
        Used when only when using action_type="discrete" option.

        Note that if None, the action meaning values automatically set to [0.1, 10] log sampled values.
            np.logspace(-1, 1, action_dim)

    use_reward_predictor: bool, default=False.
        Parameter in RTBSyntheticSimulator class.
        Whether to use predicted reward to determine bidding price or not.
        Otherwise, the ground-truth (expected) reward is used.

    reward_predictor: Optional[BaseEstimator], default=None.
        Parameter in RTBSyntheticSimulator class.
        A machine learning prediction model to predict the reward.
        Required only when using use_reward_predictor=True option.

    step_per_episode: int, default=24.
        Total timesteps in an episode.

    initial_budget: int, default=10000.
        Initial budget (i.e., constraint) for bidding during an episode.

    n_ads: int, default=100.
        Parameter in RTBSyntheticSimulator class.
        Number of ads used for fitting the reward predictor.

    n_users: int, default=100.
        Parameter in RTBSyntheticSimulator class.
        Number of users used for fitting the reward predictor.

    ad_feature_dim: int, default=5.
        Parameter in RTBSyntheticSimulator class.
        Dimentions of the ad feature vectors.

    user_feature_dim: int, default=5.
        Parameter in RTBSyntheticSimulator class.
        Dimentions of the user feature vectors.

    standard_bid_price: int, default = 100.
        Parameter in RTBSyntheticSimulator class.
        Bid price whose impression probability is expected to be 0.5.

    trend_interval: int, default=24.
        Parameter in RTBSyntheticSimulator class.
        Length of the ctr/cvr trend cycle.

    n_dices: int, default=10.
        Parameter in RTBSyntheticSimulator class.
        Number of the random_variables sampled to calculate second price.

    wf_alpha: float, default=2.0.
        Parameter in RTBSyntheticSimulator class.
        Parameter (exponential coefficient) for WinningFunction used in the auction.

    candidate_ads: NDArray[int], shape (n_candidate_ads, ), default=np.arange(1).
        Ad ids used in auctions.

    candidate_users: NDArray[int], shape (n_candidate_users, ), default=np.arange(10).
        User ids used in auctions.

    candidate_ad_sampling_prob: Optional[NDArray[float]], shape (n_candidate_ads, ), default=None.
        Sampling probalities to determine which ad (id) is used in each auction.

    candidate_user_sampling_prob: Optional[NDArray[float]], shape (n_candidate_users, ), default=None.
        Sampling probalities to determine which user (id) is used in each auction.

    search_volume_distribution: Optional[List[NormalDistribution]], shape (step_per_episode, ), default=None.
        Search volume distribution for each timestep.

    random_state: int, default=12345.
        Random state.

    Examples
    -------

    .. codeblock:: python

        # import necessary module from _gym
        from env.env import RTBEnv
        from policy.base import RandomPolicy

        # import necessary module from other library
        from sklearn.linear_model import LogisticRegression

        # initialize environment and define (RL) agent (i.e., policy)
        env = RTBEnv(
            use_reward_predictor=True,
            reward_predictor=LogisticRegression(),
        )
        agent = RandomPolicy()

        # when using use_reward_predictor=True option,
        # pretrain reward predictor used for bidding price determination
        env.fit_reward_predictor(n_samples=10000)

        # openai gym like interaction with agent
        for episode in range(1000):
            obs = env.reset()
            done = False

            while not done:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)

        # calculate ground-truth policy value
        performance = env.calc_ground_truth_policy_value(
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
        action_type: str = "discrete",  # "continuous"
        action_dim: Optional[int] = 10,
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
        candidate_ads: NDArray[int] = np.arange(1),  # ad idxes
        candidate_users: NDArray[int] = np.arange(10),  # user idxes
        candidate_ad_sampling_prob: Optional[NDArray[float]] = None,
        candidate_user_sampling_prob: Optional[NDArray[float]] = None,
        search_volume_distribution: Optional[List[NormalDistribution]] = None,
        random_state: int = 12345,
    ):
        super().__init__()
        if not (isinstance(objective, str) and objective in ["click", "conversion"]):
            raise ValueError(
                f'objective must be either "click" or "conversion", but {self.objective} is given'
            )
        if not (
            isinstance(action_type, str) and action_type in ["discrete", "continuous"]
        ):
            raise ValueError(
                f'action_type must be either "discrete" or "continuous", but {action_type} is given'
            )
        if action_type == "discrete" and not (
            isinstance(action_dim, int) and action_dim > 0
        ):
            raise ValueError(
                f"action_dim must be a positive interger, but {action_dim} is given"
            )
        if action_type == "discrete" and action_meaning is not None:
            if len(action_meaning) != action_dim:
                raise ValueError(
                    "action_meaning must have the same size with action_dim"
                )
            if min(action_meaning.values()) < 0.1 or max(action_meaning.values()) > 10:
                raise ValueError(
                    "action_meaning must have float values within [0.1, 10]"
                )
        if not (isinstance(step_per_episode, int) and step_per_episode > 0):
            raise ValueError(
                f"step_per_episode must be a positive interger, but {step_per_episode} is given"
            )
        if not (isinstance(initial_budget, int) and initial_budget > 0):
            raise ValueError(
                f"action_dim must be a positive interger, but {action_dim} is given"
            )
        if not isinstance(candidate_ads, NDArray[int]):
            raise ValueError("candidate_ads must be an NDArray of integers")
        if not isinstance(candidate_users, NDArray[int]):
            raise ValueError("candidate_users must be an NDArray of integers")
        if candidate_ads.min() < 0 or candidate_ads.max() >= n_ads:
            raise ValueError(
                f"candidate_ads must be chosen from integer within [0, n_ads)"
            )
        if candidate_users.min() < 0 or candidate_users.max() >= n_users:
            raise ValueError(
                f"candidate_users must be chosen from integer within [0, n_users)"
            )
        if not (
            candidate_ad_sampling_prob is None
            or (
                isinstance(candidate_ad_sampling_prob, NDArray[float])
                and candidate_ad_sampling_prob.min() > 0
            )
        ):
            raise ValueError(
                "candidate_ad_sampling_prob must be an NDArray of positive float values"
            )
        if not (
            candidate_user_sampling_prob is None
            or (
                isinstance(candidate_user_sampling_prob, NDArray[float])
                and candidate_user_sampling_prob.min() > 0
            )
        ):
            raise ValueError(
                "candidate_user_sampling_prob must be an NDArray of float values"
            )
        if candidate_ad_sampling_prob is not None and len(candidate_ads) != len(
            candidate_ad_sampling_prob
        ):
            raise ValueError(
                f"candidate_ads and candidate_ad_sampling_prob must have the same length"
            )
        if candidate_user_sampling_prob is not None and len(candidate_users) != len(
            candidate_user_sampling_prob
        ):
            raise ValueError(
                f"candidate_users and candidate_user_sampling_prob must have the same length"
            )
        if not (
            search_volume_distribution is None
            or isinstance(search_volume_distribution[0], NormalDistribution)
        ):
            raise ValueError(
                "search_volume_distribution must be a list of NormalDistribution"
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
                step_per_episode=step_per_episode,
                n_ads=n_ads,
                n_users=n_users,
                ad_feature_dim=ad_feature_dim,
                user_feature_dim=user_feature_dim,
                standard_bid_price=standard_bid_price,
                trend_interval=trend_interval,
                n_dices=n_dices,
                wf_alpha=wf_alpha,
                random_state=random_state,
            )

        self.objective = objective

        # define observation space
        self.observation_space = Box(
            low=[0, 0, 0, 0, 0, 0, 0.1],
            high=[step_per_episode, initial_budget, np.inf, np.inf, 1, np.inf, 10]
            # observations = (timestep, remaining_budget, BCR, CPM, WR, reward, adjust_rate)
        )

        # define action space
        self.action_type = action_type

        if self.action_type == "discrete":
            self.action_space = Discrete(action_dim)

            if self.action_meaning is None:
                self.action_meaning = dict(
                    zip(range(action_dim), np.logspace(-1, 1, action_dim))
                )

        else:  # "continuous"
            self.action_space = Box(low=0.1, high=10, shape=(1,))

        # define reward range
        self.reward_range = (0, np.inf)

        self.step_per_episode = step_per_episode
        self.initial_budget = initial_budget

        self.candidate_ads = candidate_ads
        self.candidate_users = candidate_users

        if candidate_ad_sampling_prob is None:
            candidate_ad_sampling_prob = np.full(
                len(candidate_ads), 1 / len(candidate_ads)
            )
        else:
            self.candidate_ad_sampling_prob = candidate_ad_sampling_prob / np.sum(
                candidate_ad_sampling_prob
            )

        if candidate_user_sampling_prob is None:
            candidate_user_sampling_prob = np.full(
                len(candidate_users), 1 / len(candidate_users)
            )
        else:
            self.candidate_user_sampling_prob = candidate_user_sampling_prob / np.sum(
                candidate_user_sampling_prob
            )

        if search_volume_distribution is None:
            search_volume_distribution = [
                NormalDistribution(mean=10, std=0.0)
            ] * step_per_episode

        self.search_volumes = np.zeros(step_per_episode, 100)
        for i in range(step_per_episode):
            self.search_volumes[i] = search_volume_distribution.sample(size=100).astype(
                int
            )
        self.search_volumes = np.clip(self.search_volumes, 5, None)

        # just for idx of search_volumes to sample from
        self.T = 0

    def step(self, action: Union[int, float]) -> Tuple[Any]:
        """Rollout auctions arise during the timestep and return feedbacks to the agent.

        Note
        -------
        The rollout procedure is given as follows.
        1. Sample ads and users for (search volume, ) auctions occur during the timestep.

        2. Collect outcome for each auctions by submitting a query to the RTBSyntheticSimulator.

            (In RTBSyntheticSimulator)
            2-1. Determine bid price.
                bid price = adjust rate * predicted/ground-truth reward ( * constant)

            2-2. Calculate outcome probability and stochastically determine auction result.
                auction results: (bid price,) cost (i.e., second price), impression, click, conversion

        3. Check if the culumative cost during the timestep exceeds the remaining budget or not.
           (If exceeds, cancel the corresponding auction results.)

        4. Aggregate auction results and return feedbacks to the RL agent.

        Parameters
        -------
        action: Union[int, float].
            RL agent action which indicates adjust rate parameter used for bid price determination.
            Both discrete and continuos actions are acceptable.

        Returns
        -------
        feedbacks: Tuple.
            obs: NDArray[float], shape (7, ).
                Statistical feedbacks of auctions during the timestep.
                Corresponds to RL state, which include following components.
                    - timestep
                    - remaining budget
                    - impression level features at the previous timestep
                      (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
                    - adjust rate (i.e., agent action) at previous timestep

            reward: int.
                Total clicks/conversions gained during the timestep.

            done: bool.
                Wether the episode end or not.

            info: Dict[str, int].
                Additinal feedbacks (total impressions, clicks, and conversions) for analysts.
                Note that those feedbacks are intended to be unobservable for the RL agent.

        """
        if self.action_type == "discrete":
            if not (isinstance(action, int) and 0 <= action < self.action_space.n):
                raise ValueError(
                    f"action must be an integer within [0, {self.action_space.n})"
                )
        else:  # "continuous"
            if not (
                isinstance(action, float)
                and self.action_space.low[0] <= action <= self.action_space.high[0]
            ):
                raise ValueError(
                    f"action must be a float value within ({self.action_space.low}, {self.action_space.high})"
                )

        # map agent action into adjust rate
        adjust_rate = (
            action if self.action_type == "continuous" else self.action_meaning[action]
        )

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
        ) = self.simulator.simulate_auction(self.t, adjust_rate, ad_ids, user_ids)

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
        self.prev_remaining_budget = self.remaining_budget

        return np.array(obs.values()).astype(float), reward, done, info

    def reset(self) -> NDArray[float]:
        """Initialize the environment.

        Note
        -------
        Remaining budget is initialized to the initial budget of an episode.

        Returns
        -------
        obs: NDArray[float], shape (7, ).
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
        """Maybe add some plot later."""
        pass

    def close(self) -> None:
        warnings.warn(".close() is not implemented, nothing takes place")
        pass

    def seed(self, seed: int = None) -> None:
        warnings.warn(
            "nothing takes place since .seed() is not implemented, please reset seed by initializing the environment"
        )
        pass

    def fit_reward_predictor(self, n_samples: int = 10000) -> None:
        """Pre-train reward prediction model used in simulator to calculate bid price.

        Note
        -------
        Intended only used when use_reward_predictor=True option.

        Parameters
        -------
        n_samples: int, default=10000.
            Number of samples to fit reward predictor in RTBSyntheticSimulator.

        """
        self.simulator.fit_reward_predictor(n_samples)

    def calc_ground_truth_policy_value(
        self, evaluation_policy: BasePolicy, n_episodes: int = 10000
    ) -> float:
        """Rollout the RL agent (i.e., policy) and calculate mean episodic reward.

        Parameters
        -------
        evaluation_policy: BasePolicy.
            The RL agent (i.e., policy) to be evaluated.

        n_episodes: int, default=10000.
            Number of episodes to rollout.

        Returns
        -------
        mean_reward: float.
            Mean episode reward calculated through rollout.

        """
        if not isinstance(evaluation_policy, BasePolicy):
            raise ValueError(
                "evaluation_policy must be the BasePolicy or a child class of BasePolicy"
            )
        if not (isinstance(n_episodes, int) and n_episodes > 0):
            raise ValueError(
                "n_episodes must be a positive integer, but {n_episodes} is given"
            )

        total_reward = 0.0
        for _ in range(n_episodes):
            state = self.reset()
            done = False

            while not done:
                action, _ = evaluation_policy.act(state)  # fix later
                state, reward, done, _ = self.step(action)
                total_reward += reward

        return total_reward / n_episodes
