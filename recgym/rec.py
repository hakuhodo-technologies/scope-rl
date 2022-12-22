"""Reinforcement Learning (RL) Environment for recommend system (REC)."""
from typing import Tuple, Optional, Any

import random
import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.utils import check_scalar, check_random_state



class RECEnv(gym.Env):
    """Class for recommend system (REC) environment for reinforcement learning (RL) agent to interact.
    Note
    -------
    Adopt OpenAI Gym like interface. See Examples below for the usage.
    後で
    Markov Decision Process (MDP) definition are given as follows:
        state: array-like of shape (7, )
            Statistical feedbacks of auctions during the timestep, including following values.
                - timestep
                - remaining budget
                - impression level features at the previous timestep
                  (budget consumption rate, cost per mille of impressions, auction winning rate, and reward)
                - adjust rate (i.e., RL agent action) at the previous timestep
        action: {int, float, array-like of shape (1, )} (>= 0)
            Adjust rate parameter used for determining the bid price as follows.
            (Bid price is individually determined for each auction.)
            .. math::
                {bid price}_{t, i} = {adjust rate}_{t} \\times {predicted reward}_{t,i} ( \\times {const.})
            Note that, you can also use predicted reward instead of ground-truth reward in the above equation.
            Please also refer to CustomizedRTBEnv Wrapper.
        reward: int (>= 0)
            Total clicks/conversions gained during the timestep.

    Parameters
    -------
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
    reward_function: default = user_preference_dynamics
        select reward_function
    state_transition_function: default = inner_reward_function
        select state_transition_function
    noise_std: float, default = 0 (>=0)

    後で
    Examples
    -------
    .. codeblock:: python
        # import necessary module from rtbgym
        from rtbgym.env import RTBEnv
        from rtbgym.policy import OnlineHead
        from rtbgym.ope.online import calc_on_policy_policy_value
        # import necessary module from other libraries
        from d3rlpy.algos import RandomPolicy
        from d3rlpy.preprocessing import MinMaxActionScaler
        # initialize environment and define (RL) agent (i.e., policy)
        env = RTBEnv(random_state=12345)
        agent = OnlineHead(
            RandomPolicy(
                action_scaler=MinMaxActionScaler(
                    minimum=0.1,
                    maximum=10,
                )
            )
        )
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
        on_policy_performance  # 13.44
    References
    -------
    Takuma Seno and Michita Imai.
    "d3rlpy: An Offline Deep Reinforcement Library.", 2021.
    Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba.
    "OpenAI Gym.", 2016.
    """

    def __init__(
        self,
        n_items: int = 100,
        n_users: int = 100,
        item_feature_dim: int = 5,
        user_feature_dim: int = 5,
        item_feature_vector: Optional[np.ndarray] = None,
        user_feature_vector: Optional[np.ndarray] = None,
        reward_function = 'user_preference_dynamics',
        state_transition_function = 'inner_reward_function',
        noise_std: float = 0,
        step_per_episode = 10,
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

        #ユーザーの特徴ベクトルを初期化する
        user_feature_vector = np.random.uniform(low=-1.0, high=1.0, size=(self.n_users, self.user_feature_dim))
        #アイテムの特徴ベクトルを初期化する
        item_feature_vector = np.random.uniform(low=-1.0, high=1.0, size=(self.n_items, self.item_feature_dim))

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
        self.action_space = Discrete(self.n_items-1)

        # define reward range
        self.reward_range = (0, np.inf)

    
    def _observation(self, state):
        #stateにノイズを加えたものを観測とする
        obs = state + np.random.normal(loc=0.0, scale=self.noise_std, size=self.user_feature_dim)
        #-1.0から1.0に収まるように制限する
        obs = np.clip(obs, -1.0, 1.0)
        return obs


    def step(
      self,
      action: np.ndarray,  # selected item_feature_vector
    ):
        """Rollout auctions arise during the timestep and return feedbacks to the agent.

        Note
        -------
        The rollout procedure is given as follows.
        1. Sample ads and users for (search volume, ) auctions occur during the timestep. (in Simulator)
        2. Determine bid price. (In Bidder)
            :math:`bid_price_{t, i} = adjust_rate_{t} \\times predicted_reward_{t,i}/ground_truth_reward_{t, i} ( \\times const.)`
        3. Calculate outcome probability and stochastically determine auction result. (in Simulator)
            auction results: cost (i.e., second price), impression, click, conversion
        4. Check if the cumulative cost during the timestep exceeds the remaining budget or not.
           (If exceeds, cancel the corresponding auction results.)
        5. Aggregate auction results and return feedbacks to the RL agent.
        Parameters
        -------
        action: {int, float, array-like of shape (1, )} (>= 0)
            RL agent action which corresponds to the adjust rate parameter used for bid price calculation.
        Returns
        -------
        feedbacks: Tuple
            obs: ndarray of shape (1,)
                        Statistical feedbacks of recommendation during the timestep.
                        Corresponds to RL state, which include following components.
                            - state
            reward: int (>= 0)
                Total clicks/conversions gained during the timestep.
            done: bool
                Wether the episode end or not.
            info: dict
                Additional feedbacks for analysts.
                    #一旦空で返す
        """


        # state_transition_functionでstateを更新
        # state = self.state_transition_function(self.state, action)
        state =  eval("self."+ self.state_transition_function+"(self.state, action)")

        #step_per_episodeが決まっている場合の設定   
        done = self.t == self.step_per_episode - 1

        if done:
            obs = self.reset()

        else:
            # update timestep
            self.t += 1
            # obs = self._observation(state)
            # obs = [0]*self.user_feature_dim
            obs = np.ones(self.user_feature_dim)
            



        # 報酬を発生
        # reward = reward_function(state, action)
        reward = eval("self." + self.reward_function +"(self.state, action)")

        info = {}


        # return obs, reward, done, False, info
        return obs, reward, done, info


    def reset(self):
        """Initialize the environment.
        Note
        -------
        Returns
        -------
        obs: ndarray of shape (1,)
                    Statistical feedbacks of recommendation during the timestep.
                    Corresponds to RL state, which include following components.
                        - state
        """
        # initialize internal env state
        self.t = 0
        #userをランダムに選択
        user_id = random.randint(0, self.n_users-1)
        #user_feature_vectorから選ばれた人のuser_featureをstateにする
        state = self.user_feature_vector[user_id]
        self.state = state
        # obs = self._observation(self.state)
        # obs = np.array([0]*self.user_feature_dim)
        obs = np.ones(self.user_feature_dim)
        return obs



    #実際のuser_transitionの関数、今回はuser_preference_dynamicsを使う
    def user_preference_dynamics(self, state, action, alpha = 1):
        state = state + alpha * state @ self.item_feature_vector[action] * self.item_feature_vector[action] 
        state = state / np.linalg.norm(state, ord=2)
        return state


    #rewardの関数を決定する
    def inner_reward_function(self, state, action):
        reward = state @ self.item_feature_vector[action]
        return reward

    # def cos_similar_function(state, action):
    #     reward = 
