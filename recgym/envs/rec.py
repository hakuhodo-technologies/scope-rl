"""Reinforcement Learning (RL) Environment for recommend system (REC)."""
from typing import Optional

import gym
from gym.spaces import Box, Discrete
import numpy as np
import random


class RECEnv(gym.Env):
    """Class for recommend system (REC) environment for reinforcement learning (RL) agent to interact.

    Bases: :class:`gym.Env`

    Imported as: :class:`recgym.RECEnv`

    Note
    -------
    RECGym works with OpenAI Gym and Gymnasium-like interface. See Examples below for the usage.

    Markov Decision Process (MDP) definition are given as follows:
        state: array-like of shape (user_feature_dim, )
            make the initial state the user_feature of the chosen user.
            user_feature is amplified/attenuated by the recommended item_feature.

        action: {int, array-like of shape (1, )} (>= 0)
            selected an item to recommendation from n_items.

        reward: float
            user engagement/click gained.

    Parameters
    -------

    reward_function: default = inner_reward_function
        Functions that determine how reward is calculated.

    state_transition_function: default = user_preference_dynamics
        Functions that determine state update.

    n_items: int, default=100 (> 0)
        Number of items used for recommendation system.

    n_users: int, default=100 (> 0)
        Number of users used for recommendation system.

    item_feature_dim: int, default=5 (> 0)
        Dimensions of the item feature vectors.

    user_feature_dim: int, default=5 (> 0)
        Dimensions of the user feature vectors.

    item_feature_vector: ndarray of shape (n_items, item_feature_dim), default=None
        Feature vectors that characterizes each item.

    user_feature_vector: ndarray of shape (n_users, user_feature_dim), default=None
        Feature vectors that characterizes each user.

    noise_std: float, default = 0 (>=0)
        Amount of noise an observation has.

    step_per_episode: int, default=10 (> 0)
        Number of timesteps in an episode.

    Examples
    -------

    Setup:

    .. code-block:: python

        # import necessary module from recgym
        from recgym.rec import RECEnv
        from recgym.function import inner_reward_function
        from recgym.function import user_preference_dynamics

        # import necessary module from other libraries
        from d3rlpy.algos import DiscreteRandomPolicy

        # initialize environment and define (RL) agent (i.e., policy)
        env = RECEnv(
            reward_function = inner_reward_function,
            state_transition_function = user_preference_dynamics,
        )

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
            obs = env.reset()
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

        0.23269923657166344

    References
    -------
    Sarah Dean, Jamie Morgenstern.
    "Preference Dynamics Under Personalized Recommendations." 2022.

    Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba.
    "OpenAI Gym." 2016.

    """

    def __init__(
        self,
        reward_function,
        state_transition_function,
        n_items: int = 100,
        n_users: int = 100,
        item_feature_dim: int = 5,
        user_feature_dim: int = 5,
        item_feature_vector: Optional[np.ndarray] = None,
        user_feature_vector: Optional[np.ndarray] = None,
        noise_std: float = 0,
        step_per_episode=10,
    ):
        super().__init__()
        self.n_items = n_items
        self.n_users = n_users
        self.item_feature_dim = item_feature_dim
        self.user_feature_dim = user_feature_dim
        self.reward_function = reward_function
        self.state_transition_function = state_transition_function
        self.noise_std = noise_std
        self.step_per_episode = step_per_episode
        # initialize user_feature_vector
        if user_feature_vector is None:
            user_feature_vector = np.random.uniform(
                low=-1.0, high=1.0, size=(self.n_users, self.user_feature_dim)
            )
        # initialize item_feature_vector
        if item_feature_vector is None:
            item_feature_vector = np.random.uniform(
                low=-1.0, high=1.0, size=(self.n_items, self.item_feature_dim)
            )
        self.item_feature_vector = item_feature_vector
        self.user_feature_vector = user_feature_vector

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
        self.reward_range = (0, np.inf)

    def _observation(self, state):
        # add noise to state
        obs = state + np.random.normal(
            loc=0.0, scale=self.noise_std, size=self.user_feature_dim
        )
        # limit observation [-1, 1]
        obs = np.clip(obs, -1.0, 1.0)
        return obs

    def step(
        self,
        action: int,  # action: np.ndarray,  # selected from n_items
    ):
        """return feedbacks to the agent.

        Note
        -------
        The rollout procedure is given as follows.

        1. update state with state_transition_function.

        2. increase the number of step_per_episode.

        3. generate reward.

        4. return feedbacks to the RL agent.

        Parameters
        -------

        action: {int, array-like of shape (1, )} (>= 0)
            selected an item to recommendation from n_items.
        Returns
        -------
        feedbacks: Tuple
            obs: ndarray of shape (1,)
                        Statistical feedbacks of recommendation.
                            - add noise to state by _observation()
            reward: float
                user engagement/click gained.
            done: bool
                Wether the episode end or not.
            info: dict
                Additional feedbacks for analysts.
        """
        # update state with state_transition_function
        state = self.state_transition_function(
            self.state, action, self.item_feature_vector
        )

        done = self.t == self.step_per_episode - 1

        if done:
            obs, info = self.reset()

        else:
            # update timestep
            self.t += 1
            obs = self._observation(state)

        # generate reward
        reward = self.reward_function(state, action, self.item_feature_vector)

        info = {}

        return obs, reward, done, False, info

    def reset(self, seed: Optional[int] = None):
        """Initialize the environment.

        Note
        -------

        Returns
        -------
        obs: ndarray of shape (1,)
                    Statistical feedbacks of recommendation.
                        - add noise to state by _observation()
        info: dict
            Additional feedbacks for analysts.

        """
        # initialize internal env state
        self.t = 0
        # select user at random
        user_id = random.randint(0, self.n_users - 1)
        # make state user_feature_vector of the selected user.
        state = self.user_feature_vector[user_id]
        self.state = state
        obs = self._observation(self.state)

        info = {}

        return obs, info
