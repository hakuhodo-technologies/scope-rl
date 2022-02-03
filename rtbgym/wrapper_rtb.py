"""Customization of RL setting by decision makers."""
from typing import Tuple, Optional, Union, Any

import gym
from gym.spaces import Box, Discrete
from sklearn.base import BaseEstimator
from sklearn.utils import check_scalar
import numpy as np

from _gym.env.rtb import RTBEnv
from _gym.utils import check_array
from _gym.types import Action, Numeric


class CustomizedRTBEnv(gym.Env):
    """Wrapper class for RTBEnv to customize RL action space and bidder by decision makers.

    Note
    -------
    We can customize three following decision making using CustomizedEnv.
        - reward_predictor in Bidder class
            We use predicted rewards to calculate bid price as follows.
                :math:`bid_price_{t, i} = adjust_rate_{t} \\times predicted_reward_{t,i} ( \\times const.)`

            If None, we use ground-truth reward instead of predicted reward.

        - scaler in Bidder class
            Scaler defines const.in the bid price calculation as follows.
                :math:`const. = scaler \\times standard_bid_price`

            where standard_bid_price indicates the average of standard_bid_price
            (bid price which has approximately 50% impression probability) over all ads.

        - action space for agent
            We transform continual adjust rate space :math:`[0, \\infty)` into agent action space.
            Both discrete and continuous actions are acceptable.

            Note that we recommend you to set action space within [0.1, 10].
            Instead, you can tune multiplication of adjust rate using scaler.

    Constrained Markov Decision Process (CMDP) definition are given as follows:
        timestep: int (> 0)
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

        action: Union[int, float, NDArray] (> 0)
            Adjust rate parameter used for the bid price calculation as follows.
            Note that the following bid price is individually determined for each auction.

            .. math::
                {bid price}_{t, i} = {adjust rate}_{t} \\times {predicted reward}_{t,i} ( \\times {const.})

            Both discrete and continuous actions are acceptable.

        reward: int (> 0)
            Total clicks/conversions gained during the timestep.

        discount_rate: int (= 1)
            Discount factor for cumulative reward calculation.
            Set discount_rate = 1 (i.e., no discount) in RTB.

        constraint: int (> 0)
            Total cost should not exceed the initial budget.

    Parameters
    -------
    original_env: RTBEnv
        Original RTB environment.

    reward_predictor: Optional[BaseEstimator], default=None
        A machine learning model to predict the reward to determine the bidding price.
        If None, the ground-truth (expected) reward is used instead of the predicted one.

    scaler: Optional[Union[int, float]], default=None (> 0)
        Scaling factor (constant value) used for bid price determination.
        If None, scaler is autofitted by bidder.auto_fit_scaler().

    action_min: float, default=0.1 (> 0)
        Minimum value of action.

    action_max: float, default=10.0 (> 0)
        Maximum value of action.

    action_type: str, default="discrete"
        Action type of the RL agent.
        Choose either from "discrete" or "continuous".

    n_actions: int, default=10 (> 0)
        Number of the discrete actions.
        Used only when action_type="discrete" option.

    action_meaning: Optional[NDArray[float]], default=None
        Dictionary which maps discrete action index into specific actions.
        Used when only when using action_type="discrete" option.

        If None, the values are automatically set to [action_min, action_max] as follows.
            np.logspace(-1, 1, n_actions)

    Examples
    -------

    .. codeblock:: python

        # import necessary module from _gym
        from _gym.env import RTBEnv
        from _gym.policy import OnlineHead
        from _gym.ope.online import calc_on_policy_policy_value

        # import necessary module from other libraries
        from sklearn.linear_model import LogisticRegression
        from d3rlpy.algos import DiscreteRandomPolicy

        # initialize and customize environment
        env = RTBEnv(random_state=12345)
        env = CustomizedRTBEnv(
            original_env=env,
            reward_predictor=LogisticRegression(),
            action_type="discrete",
        )

        # define (RL) agent (i.e., policy)
        agent = OnlineHead(DiscreteRandomPolicy())
        agent.build_with_env(env)

        # OpenAI Gym like interaction with agent
        for episode in range(1000):
            obs = env.reset()
            done = False

            while not done:
                action = agent.predict_online(obs)
                obs, reward, done, info = env.step(action)

        # calculate on-policy policy value
        on_policy_performance = calc_on_policy_policy_value(
            env,
            agent,
            n_episodes=100,
            random_state=12345
        )
        on_policy_performance  # 11.75

    References
    -------
    Takuma Seno and Michita Imai.
    "d3rlpy: An Offline Deep Reinforcement Library.", 2021.

    Di Wu, Xiujun Chen, Xun Yang, Hao Wang, Qing Tan, Xiaoxun Zhang, Jian Xu, and Kun Gai.
    "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising.", 2018.

    Jun Zhao, Guang Qiu, Ziyu Guan, Wei Zhao, and Xiaofei He.
    "Deep Reinforcement Learning for Sponsored Search Real-time Bidding.", 2018.

    Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba.
    "OpenAI Gym.", 2016.

    """

    def __init__(
        self,
        original_env: RTBEnv,
        reward_predictor: Optional[BaseEstimator] = None,
        scaler: Optional[Union[int, float]] = None,
        action_min: float = 0.1,
        action_max: float = 10.0,
        action_type: str = "discrete",  # "continuous"
        n_actions: int = 10,
        action_meaning: Optional[
            np.ndarray
        ] = None,  # maps categorical actions to adjust rate
    ):
        super().__init__()
        if not isinstance(original_env, RTBEnv):
            raise ValueError("original_env must be RTBEnv or a child class of RTBEnv")
        self.env = original_env

        check_scalar(action_min, name="action_min", target_type=(int, float), min_val=0)
        check_scalar(action_max, name="action_max", target_type=(int, float), min_val=0)
        if action_min >= action_max:
            raise ValueError("action_min must be smaller than action_max")

        if action_type not in ["discrete", "continuous"]:
            raise ValueError(
                f'action_type must be either "discrete" or "continuous", but {action_type} is given'
            )
        if action_type == "discrete":
            check_scalar(n_actions, name="n_acitons", target_type=int, min_val=2)

            if action_meaning is None:
                action_meaning = np.logspace(
                    np.log10(action_min), np.log10(action_max), n_actions
                )

            check_array(
                action_meaning,
                name="action_meaning",
                expected_dim=1,
                min_val=action_min,
                max_val=action_max,
            )
            if action_meaning.shape[0] != n_actions:
                raise ValueError(
                    "Expected `action_meaning.shape[0] == n_actions`, but found False"
                )
            self.action_meaning = action_meaning

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
            low=np.array([0, 0, 0, 0, 0, 0, action_min]),
            high=np.array(
                [
                    self.env.step_per_episode,
                    self.env.initial_budget,
                    np.inf,
                    np.inf,
                    1,
                    np.inf,
                    action_max,
                ]
            ),
            dtype=float,
        )

        # define action space
        self.action_type = action_type
        self.n_actions = n_actions
        self.action_dim = 1
        self.action_meaning = action_meaning

        if self.action_type == "discrete":
            self.action_space = Discrete(n_actions)

        else:  # "continuous"
            self.action_space = Box(
                low=action_min, high=action_max, shape=(1,), dtype=float
            )

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

    def step(self, action: Action) -> Tuple[Any]:
        """Rollout auctions arise during the timestep and return feedbacks to the agent.

        Parameters
        -------
        action: Action (Union[int, float, np.integer, np.float, np.ndarray])
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

            reward: int (>= 0)
                Total clicks/conversions gained during the timestep.

            done: bool
                Wether the episode end or not.

            info: Dict[str, int]
                Additional feedbacks (total impressions, clicks, and conversions) for analysts.
                Note that those feedbacks are intended to be unobservable for the RL agent.

        """
        if self.action_type == "discrete":
            if not (
                isinstance(action, (int, np.integer))
                and 0 <= action < self.action_space.n
            ):
                raise ValueError(
                    f"action must be an integer within [0, {self.action_space.n}), but {action} is given"
                )
        else:  # "continuous"
            if isinstance(action, Numeric):
                action = np.array([action])
            if not self.action_space.contains(action):
                raise ValueError(
                    f"action must be a float value within ({self.action_space.low}, {self.action_space.high})"
                )

        # map agent action into meaningful value
        action = (
            action if self.action_type == "continuous" else self.action_meaning[action]
        )

        return self.env.step(action)

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

    def render(self, mode: str = "human") -> None:
        self.env.render(mode)

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int = None) -> None:
        self.env.seed(seed)
