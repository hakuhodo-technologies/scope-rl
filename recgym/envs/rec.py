"""Reinforcement Learning (RL) Environment for recommend system (REC)."""
from typing import Tuple, Optional, Any

import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from recgym.envs.base import BaseUserModel
from recgym.envs.function import UserModel


class RECEnv(gym.Env):
    """Class for recommend system (REC) environment for reinforcement learning (RL) agent to interact.

    Bases: :class:`gym.Env`

    Imported as: :class:`recgym.RECEnv`

    Note
    -------
    RECGym works with OpenAI Gym and Gymnasium-like interface. See Examples below for the usage.

    (Partially Observable) Markov Decision Process ((PO)MDP) definition are given as follows:
        state: array-like of shape (user_feature_dim, )
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation instead of state.

        action: {int, array-like of shape (1, )} (>= 0)
            Index of an item to present to the user.

        reward: float
            User engagement signal. Either binary or continuous.

    Parameters
    -------

    UserModel: BaseUserModel
        User model which defines user_prefecture_dynamics and reward_model.
        Both class and instance are acceptable.

    reward_type: str = "continuous"
        Reward type (i.e., countinuous / binary).

    n_items: int, default=100 (> 0)
        Number of items used in the recommendation system.

    n_users: int, default=100 (> 0)
        Number of users used in the recommendation system.

    item_feature_dim: int, default=5 (> 0)
        Dimensions of the item feature vectors.

    user_feature_dim: int, default=5 (> 0)
        Dimensions of the user feature vectors.

    reward_std: float, default=0.0 (>=0)
        Standard deviation of the reward distribution. Applicable only when reward_type is "continuous".

    obs_std: float, default=0.0 (>=0)
        Standard deviation of the observation distribution.

    item_feature_vector: array-like of shape (n_items, item_feature_dim), default=None
        Feature vectors that characterize each item.

    user_feature_vector: array-like of shape (n_users, user_feature_dim), default=None
        Feature vectors that characterize each user.

    step_per_episode: int, default=10 (> 0)
        Number of timesteps in an episode.

    random_state: int, default=None (>= 0)
        Random state.

    Examples
    -------

    Setup:

    .. code-block:: python

        # import necessary module from recgym
        from recgym.rec import RECEnv
        from recgym import inner_reward_function, user_preference_dynamics

        # import necessary module from other libraries
        from d3rlpy.algos import DiscreteRandomPolicy

        # initialize environment and define (RL) agent (i.e., policy)
        env = RECEnv(random_state=random_state)

        # the following commands also work
        # import gym
        # env = gym.make("RECEnv-v0")

        # define (RL) agent (i.e., policy)
        agent = DiscreteEpsilonGreedyHead(
            base_policy = DiscreteRandomPolicy(),
            name = 'random',
            n_actions = env.n_items,
            epsilon = 1. ,
            random_state = random_state,
        )

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

        -0.11431378996045428

    References
    -------

    David Rohde, Stephen Bonner, Travis Dunlop, Flavian Vasile, Alexandros Karatzoglou.
    "RecoGym: A Reinforcement Learning Environment for the Problem of Product Recommendation in Online Advertising." 2018.

    Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba.
    "OpenAI Gym." 2016.

    """

    def __init__(
        self,
        UserModel: BaseUserModel = UserModel,
        reward_type: str = "continuous",  # "binary"
        n_items: int = 100,
        n_users: int = 100,
        item_feature_dim: int = 5,
        user_feature_dim: int = 5,
        reward_std: float = 0.0,
        obs_std: float = 0.0,
        item_feature_vector: Optional[np.ndarray] = None,
        user_feature_vector: Optional[np.ndarray] = None,
        step_per_episode=10,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.UserModel = UserModel
        self.n_items = n_items
        self.n_users = n_users
        self.item_feature_dim = item_feature_dim
        self.user_feature_dim = user_feature_dim
        self.reward_std = reward_std
        self.obs_std = obs_std

        if reward_type is "continuous":
            self.reward_std = reward_std
        elif reward_type is "binary":
            self.reward_std = 0.0
        else:
            raise ValueError(
                f'reward_type must be either "continuous" or "binary", but {reward_type} is given'
            )

        if random_state is None:
            raise ValueError("random_state must be given")
        self.random_state = random_state
        self.random_ = check_random_state(random_state)

        # initialize user_feature_vector
        if user_feature_vector is None:
            user_feature_vector = self.random_.uniform(
                low=-1.0, high=1.0, size=(self.n_users, self.user_feature_dim)
            )
        # initialize item_feature_vector
        if item_feature_vector is None:
            item_feature_vector = self.random_.uniform(
                low=-1.0, high=1.0, size=(self.n_items, self.item_feature_dim)
            )
        self.item_feature_vector = item_feature_vector
        self.user_feature_vector = user_feature_vector

        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        self.step_per_episode = step_per_episode

        # define observation space
        self.observation_space = Box(
            low=np.full(self.user_feature_dim, -1),
            high=np.full(self.user_feature_dim, 1),
            dtype=float,
        )

        # define action space
        self.action_type = "discrete"
        self.action_dim = 1
        self.action_space = Discrete(self.n_items)

        # define reward range
        self.reward_range = (-np.inf, np.inf)

    def _observation(self, state):
        # add noise to state
        obs = state + self.random_.normal(
            loc=0.0, scale=self.obs_std, size=self.user_feature_dim
        )
        # limit observation [-1, 1]
        obs = np.clip(obs, -1.0, 1.0)
        return obs

    def step(self, action: int) -> Tuple[Any]:
        """Simulate a recommender interaction with a user.

        Note
        -------
        The simulation procedure is given as follows.

        1. Sample reward (i.e., feedback on user engagement) for the given item.

        2. Update user state with user_preference_dynamics

        3. Return the user feedback to the RL agent.

        Parameters
        -------
        action: {int, array-like of shape (1, )} (>= 0)
            Indicating which item to present to the user.

        Returns
        -------
        feedbacks: Tuple
            obs: ndarray of shape (1,)
                A vector representing user preference with added noise
                A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
                When the true state is unobservable, you can gain observation as a state with added noise.

            reward: float
                User engagement signal. Either binary or continuous.

            done: bool
                Wether the episode end or not.

            truncated: False
                For API consistency.

            info: dict
                Additional feedbacks (user_id, state) for analysts.
                Note that those feedbacks are unobservable to the agent.

        """
        # 1. sample reward for the given item.
        usermodel = self.UserModel(
            self.state, action, self.item_feature_vector, self.random_state
        )
        reward = usermodel.reward_model() + self.random_.normal(
            loc=0.0, scale=self.reward_std
        )

        # 2. update user state with user_preference_dynamics
        self.state = usermodel.user_preference_dynamics()

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
            A vector representing user preference with added noise
            A vector representing user preference.  The preference changes over time in an episode by the actions presented by the RL agent.
            When the true state is unobservable, you can gain observation as a state with added noise.

        info: dict
            Additional feedbacks (user_id, state) for analysts.
            Note that those feedbacks are unobservable to the agent.

        """
        if seed is not None:
            self.random_ = check_random_state(seed)

        # initialize internal env state
        self.t = 0
        # select user at random
        self.user_id = self.random_.randint(0, self.n_users)
        # make state user_feature_vector of the selected user.
        state = self.user_feature_vector[self.user_id]
        self.state = state
        obs = self._observation(self.state)

        info = {"user_id": self.user_id, "state": self.state}

        return obs, info
