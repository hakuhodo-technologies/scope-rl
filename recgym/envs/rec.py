# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Reinforcement Learning (RL) Environment for Recommender System (REC)."""
from typing import Tuple, Optional, Any

import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .simulator.base import BaseUserModel
from .simulator.function import UserModel

from ..utils import check_array
from ..types import Action


class RECEnv(gym.Env):
    """Class for a recommender system (REC) environment for reinforcement learning (RL) agent to interact.

    Bases: :class:`gym.Env`

    Imported as: :class:`recgym.RECEnv`

    Note
    -------
    RECGym works with OpenAI Gym and Gymnasium-like interface. See Examples below for the usage.

    (Partially Observable) Markov Decision Process ((PO)MDP) definition are given as follows:
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
            When the true state is unobservable, observation will be returned to the RL agent instead of state.

        action: int (>= 0)
            Indicating which item to present to the user.

        reward: bool or float
            User engagement signal as a reward. Either binary or continuous.

    Parameters
    -------
    step_per_episode: int, default=10 (> 0)
        Number of timesteps in an episode.

    n_items: int, default=100 (> 0)
        Number of items used in the recommender system.

    n_users: int, default=100 (> 0)
        Number of users used in the recommender system.

    item_feature_dim: int, default=5 (> 0)
        Dimension of the item feature vectors.

    user_feature_dim: int, default=5 (> 0)
        Dimension of the user feature vectors.

    item_feature_vector: array-like of shape (n_items, item_feature_dim), default=None
        Feature vectors that characterize each item.

    user_feature_vector: array-like of shape (n_users, user_feature_dim), default=None
        Feature vectors that characterize each user.

    reward_type: {"continuous", "binary"}, default="continuous"
        Reward type.

    reward_std: float, default=0.0 (>=0)
        Noise level of the reward. Applicable only when reward_type is "continuous".

    obs_std: float, default=0.0 (>=0)
        Noise level of the state observation.

    UserModel: BaseUserModel, default=UserModel
        User model that defines user_prefecture_dynamics (which simulates how the user preference changes through item interaction)
        and reward_function (which simulates how the user responds to the presented item).
        Both class and instance are acceptable.

    random_state: int, default=None (>= 0)
        Random state.

    Examples
    -------

    Setup:

    .. code-block:: python

        # import necessary module from recgym and scope_rl
        from recgym.rec import RECEnv
        from scope_rl.policy import OnlineHead
        from scope_rl.ope.online import calc_on_policy_policy_value

        # import necessary module from other libraries
        from d3rlpy.algos import DiscreteRandomPolicy

        # initialize environment and define (RL) agent (i.e., policy)
        env = RECEnv(random_state=12345)

        # the following commands also work
        # import gym
        # env = gym.make("RECEnv-v0")

        # define (RL) agent (i.e., policy)
        agent = OnlineHead(
            DiscreteRandomPolicy(),
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
                action = .sample_action_online(obs)
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

        -0.022

    References
    -------
    David Rohde, Stephen Bonner, Travis Dunlop, Flavian Vasile, Alexandros Karatzoglou.
    "RecoGym: A Reinforcement Learning Environment for the Problem of Product Recommendation in Online Advertising." 2018.

    Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba.
    "OpenAI Gym." 2016.

    """

    def __init__(
        self,
        step_per_episode: int = 10,
        n_items: int = 5,
        n_users: int = 100,
        item_feature_dim: int = 5,
        user_feature_dim: int = 5,
        item_feature_vector: Optional[np.ndarray] = None,
        user_feature_vector: Optional[np.ndarray] = None,
        reward_type: str = "continuous",  # "binary"
        reward_std: float = 0.0,
        obs_std: float = 0.0,
        UserModel: BaseUserModel = UserModel,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        self.step_per_episode = step_per_episode

        check_scalar(
            n_items,
            name="n_items",
            target_type=int,
            min_val=1,
        )
        self.n_items = n_items

        check_scalar(
            n_users,
            name="n_users",
            target_type=int,
            min_val=1,
        )
        self.n_users = n_users

        check_scalar(
            item_feature_dim,
            name="item_feature_dim",
            target_type=int,
            min_val=1,
        )
        self.item_feature_dim = item_feature_dim

        check_scalar(
            user_feature_dim,
            name="user_feature_dim",
            target_type=int,
            min_val=1,
        )
        self.user_feature_dim = user_feature_dim

        check_scalar(
            obs_std,
            name="obs_std",
            target_type=float,
            min_val=0.0,
        )
        self.obs_std = obs_std

        if random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(random_state)

        # initialize user_feature_vector
        if user_feature_vector is None:
            user_feature_vector = self.random_.uniform(
                low=-1.0, high=1.0, size=(n_users, user_feature_dim)
            )

        check_array(
            user_feature_vector,
            name="user_feature_vector",
            expected_dim=2,
        )
        if user_feature_vector.shape != (n_users, user_feature_dim):
            raise ValueError(
                f"The shape of user_feature_vector must be (n_users, user_feature_dim), but found {user_feature_vector.shape}."
            )
        self.user_feature_vector = user_feature_vector / np.linalg.norm(
            user_feature_vector, ord=2
        )

        # initialize item_feature_vector
        if item_feature_vector is None:
            item_feature_vector = self.random_.uniform(
                low=-1.0, high=1.0, size=(n_items, item_feature_dim)
            )
        check_array(
            item_feature_vector,
            name="item_feature_vector",
            expected_dim=2,
        )
        if item_feature_vector.shape != (n_items, item_feature_dim):
            raise ValueError(
                f"The shape of user_feature_vector must be (n_users, user_feature_dim), but found {user_feature_vector.shape}."
            )
        self.item_feature_vector = item_feature_vector / np.linalg.norm(
            item_feature_vector, ord=2
        )

        if isinstance(UserModel, BaseUserModel):
            self.user_model = UserModel
        elif issubclass(UserModel, BaseUserModel):
            self.user_model = UserModel(
                user_feature_dim=user_feature_dim,
                item_feature_dim=item_feature_dim,
                reward_type=reward_type,
                reward_std=reward_std,
                random_state=random_state,
            )
        else:
            raise ValueError("UserModel must be a child class of BaseUserModel")

        # define observation space
        self.observation_space = Box(
            low=np.full(self.user_feature_dim, -1),
            high=np.full(self.user_feature_dim, 1),
            dtype=float,
        )

        # define action space
        self.action_space = Discrete(self.n_items)

        # define reward range
        # define reward range
        if reward_type not in ["continuous", "binary"]:
            raise ValueError(
                f'reward_type must be either "continuous" or "binary", but {reward_type} is given'
            )
        if reward_type == "continuous":
            self.reward_range = (-np.inf, np.inf)
        else:
            self.reward_range = (0, 1)

        check_scalar(
            reward_std,
            name="reward_std",
            target_type=float,
            min_val=0.0,
        )
        self.reward_type = reward_type
        self.reward_std = reward_std

    def _observation(self, state: np.ndarray):
        """Add a observation noise."""
        obs = self.random_.normal(loc=state, scale=self.obs_std)
        obs = np.clip(obs, -1.0, 1.0)
        return obs

    def step(self, action: Action) -> Tuple[Any]:
        """Simulate a recommender interaction with a user.

        Note
        -------
        The simulation procedure is given as follows.

        1. Sample reward (i.e., feedback on user engagement) for the given item.

        2. Update user state with user_preference_dynamics.

        3. Return the user feedback to the RL agent.

        Parameters
        -------
        action: int or array-like of shape (1, )
            Indicating which item to present to the user.

        Returns
        -------
        feedbacks: Tuple
            obs: ndarray of shape (1,)
                A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
                When the true state is unobservable, the agent uses observations instead of the state.

            reward: float
                User engagement signal as a reward. Either binary or continuous.

            done: bool
                Whether the episode end or not.

            truncated: False
                For API consistency.

            info: dict
                Additional feedbacks (user_id, state) that may be useful for the package users.
                These are unavailable to the agent.

        """
        # 1. sample reward for the given item.
        reward = self.user_model.reward_function(
            self.state, action, item_feature_vector=self.item_feature_vector
        )

        # 2. update user state with user_preference_dynamics
        self.state = self.user_model.user_preference_dynamics(
            self.state, action, item_feature_vector=self.item_feature_vector
        )

        done = self.t == self.step_per_episode - 1

        if done:
            obs, info = self.reset()

        else:
            self.t += 1
            obs = self._observation(self.state)

        info = {"user_id": self.user_id, "state": self.state}

        return obs, reward, done, False, info

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Initialize the environment.

        Returns
        -------
        obs: ndarray of shape (1,)
            A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
            When the true state is unobservable, the agent uses observations instead of the state.

        info: dict
            Additional feedbacks (user_id, state) that may be useful for the package users.
            These are unavailable to the agent.

        """
        if seed is not None:
            self.random_ = check_random_state(seed)

        # initialize internal env state
        self.t = 0
        # select a user at random
        self.user_id = self.random_.randint(0, self.n_users)
        self.state = self.user_feature_vector[self.user_id]
        obs = self._observation(self.state)

        info = {"user_id": self.user_id, "state": self.state}

        return obs, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
