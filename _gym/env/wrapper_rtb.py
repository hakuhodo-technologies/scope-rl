"""Customization of RL setting by decision makers."""
from typing import Tuple, Optional, Union, Any
from tqdm import tqdm

import gym
from gym.spaces import Box, Discrete
from sklearn.base import BaseEstimator
import numpy as np

from _gym.env import RTBEnv
from _gym.policy import BasePolicy


class CustomizedRTBEnv(gym.Env):
    """Wrapper class for RTBEnv to customize RL action space and bidder by decision makers.

    Note
    -------
    We can customize three following decision making using CustomizedEnv.
        - reward_predictor in Bidder class
            We use predicted rewards to calculate bid price as follows.
                bid price = adjust rate * predicted reward ( * constant)
            If None, we use ground-truth reward instead of predicted reward.

        - scaler in Bidder class
            Scaler defines constant in the bid price calculation as follows.
                bid price = adjust rate * predicted reward ( * constant)
                constant = scaler * standard_bid_price
            where standard_bid_price indicates the average of standard_bid_price
            (bid price which has approximately 50% impression probability) over all ads.

        - action space for agent
            We transform continual adjust rate space [0, \infty) into agent action space.
            Both discrete and continuous actions are acceptable.

            Note that the action space should be within [0.1, 10].
            Instead, we can tune multiplication of adjust rate using scaler.

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
            Note that the value should be within [0.1, 10]. We can change adjust rate using scaler.

        reward: int
            Total clicks/conversions gained during the timestep.

        discount_rate: int, 1
            Discount factor for cumulative reward calculation.
            Set discount_rate = 1 (i.e., no discount) in RTB.

        constraint: int
            Total cost should not exceed the initial budget.

    Parameters
    -------
    original_env: RTBEnv
        Original RTB environment.

    reward_predictor: Optional[BaseEstimator], default=None
        Parameter in Bidder.
        A machine learning model to predict the reward to determine the bidding price.
        If None, the ground-truth (expected) reward is used instead of the predicted one.

    scaler: Optional[Union[int, float]], default=None
        Parameter in Bidder.
        Scaling factor (constant value) used for bid price determination.
        If None, scaler is autofitted by bidder.auto_fit_scaler().

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

    Examples
    -------

    .. codeblock:: python

        # import necessary module from _gym
        from _gym.env import RTBEnv, CustomizedRTBEnv
        from _gym.policy import RandomPolicy

        # import from other libraries
        from sklearn.linear_model import LogisticRegression

        # initialize and customize environment
        original_env = RTBEnv()
        env = CustomizedEnv(
            original_env,
            reward_predictor=LogisticRegression(),
            scaler=None,  # autofit
            action_type="discrete",
        )

        # define (RL) agent (i.e., policy)
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

    """

    def __init__(
        self,
        original_env: RTBEnv,
        reward_predictor: Optional[BaseEstimator] = None,
        scaler: Optional[Union[int, float]] = None,
        action_type: str = "discrete",  # "continuous"
        action_dim: int = 10,
        action_meaning: Optional[
            np.ndarray
        ] = None,  # maps categorical actions to adjust rate
    ):
        super().__init__()
        if not isinstance(original_env, RTBEnv):
            raise ValueError("original_env must be RTBEnv or a child class of RTBEnv")
        if not (
            isinstance(action_type, str) and action_type in ["discrete", "continuous"]
        ):
            raise ValueError(
                f'action_type must be either "discrete" or "continuous", but {action_type} is given'
            )
        if action_type == "discrete" and not (
            isinstance(action_dim, int) and action_dim > 1
        ):
            raise ValueError(
                f"action_dim must be a interger more than 1, but {action_dim} is given"
            )
        if action_type == "discrete" and action_meaning is not None:
            if len(action_meaning) != action_dim:
                raise ValueError(
                    "action_meaning must have the same size with action_dim"
                )
            if not (
                isinstance(action_meaning, np.ndarray)
                and action_meaning.ndim == 1
                and 0.1 <= action_meaning.min()
                and action_meaning.max() <= 10
            ):
                raise ValueError(
                    "action_meaning must be an 1-dimensional NDArray of float values within [0.1, 10]"
                )

        self.env = original_env

        # set reward predictor
        if reward_predictor is not None:
            self.env.bidder.custom_set_reward_predictor(
                reward_predictor=reward_predictor
            )
            self.env.bidder.fit_reward_predictor(
                step_per_episode=self.env.step_per_episode
            )

        # set scaler
        if scaler is None:
            self.env.bidder.auto_fit_scaler(step_per_episode=self.env.step_per_episode)
        else:
            self.env.bidder.custom_set_scaler(scaler)

        # define observation space
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0.1]),
            high=np.array(
                [
                    self.env.step_per_episode,
                    self.env.initial_budget,
                    np.inf,
                    np.inf,
                    1,
                    np.inf,
                    10,
                ]
            ),
            dtype=float,
        )

        # define action space
        self.action_type = action_type
        self.action_dim = action_dim
        self.action_meaning = action_meaning

        if self.action_type == "discrete":
            self.action_space = Discrete(action_dim)

            if self.action_meaning is None:
                self.action_meaning = np.logspace(-1, 1, self.action_dim)

        else:  # "continuous"
            self.action_space = Box(low=0.1, high=10, shape=(1,), dtype=float)

    @property
    def obs_keys(self):
        return self.env.obs_keys

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def step_per_episode(self):
        return self.env.step_per_episode

    @property
    def initial_budget(self):
        return self.env.initial_budget

    def step(self, action: Union[int, float]) -> Tuple[Any]:
        """Rollout auctions arise during the timestep and return feedbacks to the agent.

        Parameters
        -------
        action: Union[int, float]
            RL agent action which indicates adjust rate parameter used for bid price determination.
            Both discrete and continuos actions are acceptable.

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
        if not isinstance(action, (int, float, np.integer, np.floating)):
            raise ValueError(f"action must be a float number, but {action} is given")
        if self.action_type == "discrete":
            if not (
                isinstance(action, (int, np.integer))
                and 0 <= action < self.action_space.n
            ):
                raise ValueError(
                    f"action must be an integer within [0, {self.action_space.n})"
                )
        else:  # "continuous"
            if not self.action_space.contains(np.array([action])):
                raise ValueError(
                    f"action must be a float value within ({self.action_space.low}, {self.action_space.high})"
                )

        # map agent action into adjust rate
        adjust_rate = (
            action if self.action_type == "continuous" else self.action_meaning[action]
        )

        return self.env.step(action=adjust_rate)

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
        return self.env.reset()

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

    def render(self, mode: str = "human") -> None:
        self.env.render(mode)

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int = None) -> None:
        self.env.seed(seed)
