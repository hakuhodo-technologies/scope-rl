# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Basic Reinforcement Learning (RL) Environment."""
from typing import Tuple, Optional, Any

import gym
from gym.spaces import Box, Discrete
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .simulator.base import BaseStateTransitionFunction
from .simulator.function import StateTransitionFunction
from .simulator.base import BaseRewardFunction
from .simulator.function import RewardFunction

from ..utils import check_array
from ..types import Action


class BasicEnv(gym.Env):
    """Class for a basic environment for reinforcement learning (RL) agent to interact.

    Bases: :class:`gym.Env`

    Imported as: :class:`basicgym.BasicEnv`

    Note
    -------
    SyntheticGym works with OpenAI Gym and Gymnasium-like interface. See Examples below for the usage.

    Markov Decision Process (CMDP) definition are given as follows:
        timestep: int (> 0)

        state: array-like of shape (state_dim, )

        action: int, float, or array-like of shape (action_dim, )

        reward: bool or continuous

        discount_rate: float

    Parameters
    -------

    step_per_episode: int, default=10 (> 0)
        Number of timesteps in an episode.

    state_dim: int, default=5 (> 0)
        Dimension of the state.

    action_type: {"discrete", "continuous"}, default="continuous"
        Type of the action space.

    action_dim: int
        Dimension of the action (context).

    n_actions: int, default=10 (> 0)
        Number of actions in the discrete action case.

    action_context: array-like of shape (n_actions, action_dim), default=None
        Feature vectors that characterizes each action. Applicable only when action_type is "discrete".

    reward_type: {"continuous", "binary"}, default="continuous"
        Reward type.

    reward_std: float, default=0.0 (>=0)
        Noise level of the reward. Applicable only when reward_type is "continuous".

    obs_std: float, default=0.0 (>=0)
        Noise level of the state observation.

    StateTransitionFunction: BaseStateTransitionFunction, default=StateTransitionFunction
        State transition function. Both class and instance are acceptable.

    RewardFunction: BaseRewardFunction, default=RewardFunction
        Expected immediate reward function. Both class and instance are acceptable.

    random_state: int, default=None (>= 0)
        Random state.

    Examples
    -------

    Setup:

    .. code-block:: python

        # import necessary module from syntheticgym
        from syntheticgym import SyntheticEnv
        from scope_rl.policy import OnlineHead
        from scope_rl.ope.online import calc_on_policy_policy_value

        # import necessary module from other libraries
        from d3rlpy.algos import RandomPolicy
        from d3rlpy.preprocessing import MinMaxActionScaler

        # initialize environment
        env = SyntheticEnv(random_state=12345)

        # the following commands also work
        # import gym
        # env = gym.make("SyntheticEnv-continuous-v0")

        # define (RL) agent (i.e., policy)
        agent = OnlineHead(
            RandomPolicy(
                action_scaler=MinMaxActionScaler(
                    minimum=0.1,
                    maximum=10,
                )
            ),
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
                action = agent.predict_online(obs)
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

        27.59

    References
    -------
    Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba.
    "OpenAI Gym." 2016.

    """

    def __init__(
        self,
        step_per_episode: int = 10,
        state_dim: int = 5,
        action_type: str = "continuous",  # "discrete"
        n_actions: int = 10,  # Applicable only when action_type is "discrete"
        action_dim: int = 3,
        action_context: Optional[
            np.ndarray
        ] = None,  # Applicable only when action_type is "discrete"
        reward_type: str = "continuous",  # "binary"
        reward_std: float = 0.0,
        obs_std: float = 0.0,
        StateTransitionFunction: BaseStateTransitionFunction = StateTransitionFunction,
        RewardFunction: BaseRewardFunction = RewardFunction,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(random_state)

        check_scalar(
            state_dim,
            name="state_dim",
            target_type=int,
            min_val=1,
        )
        self.state_dim = state_dim

        check_scalar(
            n_actions,
            name="n_actions",
            target_type=int,
            min_val=1,
        )
        self.n_actions = n_actions

        check_scalar(
            action_dim,
            name="action_dim",
            target_type=int,
            min_val=1,
        )
        self.action_dim = action_dim

        check_scalar(
            obs_std,
            name="obs_std",
            target_type=float,
            min_val=0.0,
        )
        self.obs_std = obs_std

        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        self.step_per_episode = step_per_episode

        # define observation space
        self.observation_space = Box(
            low=np.full(state_dim, -1.0),
            high=np.full(state_dim, 1.0),
            dtype=float,
        )

        # define action space
        if action_type not in ["continuous", "discrete"]:
            raise ValueError(
                f'action_type must be either "continuous" or "discrete", but {action_type} is given'
            )

        if action_type == "continuous":
            self.action_type = "continuous"
            self.action_space = Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=float
            )

        elif action_type == "discrete":
            self.action_type = "discrete"
            self.action_space = Discrete(n_actions)

            if action_context is None:
                action_context = self.random_.normal(
                    loc=0.0, scale=1.0, size=(n_actions, action_dim)
                )

            check_array(
                action_context,
                name="action_context",
                expected_dim=2,
            )
            if action_context.shape != (n_actions, action_dim):
                raise ValueError(
                    f"The shape of action_context must be (n_actions, action_dim), but found {action_context.shape}."
                )
            self.action_context = action_context

        if isinstance(StateTransitionFunction, BaseStateTransitionFunction):
            self.state_transition_function = StateTransitionFunction
        elif issubclass(StateTransitionFunction, BaseStateTransitionFunction):
            self.state_transition_function = StateTransitionFunction(
                state_dim=state_dim,
                action_dim=action_dim,
                random_state=random_state,
            )
        else:
            raise ValueError(
                "StateTransitionFunction must be a child class of BaseStateTransitionFunction"
            )

        if isinstance(RewardFunction, BaseRewardFunction):
            self.reward_function = RewardFunction
        elif issubclass(RewardFunction, BaseRewardFunction):
            self.reward_function = RewardFunction(
                state_dim=state_dim,
                action_dim=action_dim,
                reward_type=reward_type,
                reward_std=reward_std,
                random_state=random_state,
            )
        else:
            raise ValueError(
                "RewardFunction must be a child class of BaseRewardFunction"
            )

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
        return obs

    def step(self, action: Action) -> Tuple[Any]:
        """Simulate a action interaction with a context.

        Note
        -------
        The simulation procedure is given as follows.

        1. Sample reward for the given state-action pair.

        2. Update state with state transition function.

        3. Return the feedback to the RL agent.

        Parameters
        -------
        action: {int, array-like of shape (action_dim, )} (>= 0)
            Indicating which action to present to the context.

        Returns
        -------
        feedbacks: Tuple
            obs: ndarray of shape (state_dim,)
                State observation, which possibly be noisy.

            reward: float
                Observed immediate rewards.

            done: bool
                Whether the episode end or not.

            truncated: False
                For API consistency.

            info: (empty) dict
                Additional information that may be useful for the package users.
            This is unavailable to the RL agent.

        """
        if self.action_type == "discrete":
            action = self.action_context[action]

        check_array(
            action,
            name="action",
            expected_dim=1,
        )
        if action.shape[0] != self.action_dim:
            raise ValueError(
                "Dimension of action must be equal to action_dim, but found False."
            )

        # 1. sample reward for the given action.
        reward = self.reward_function.sample_reward(self.state, action)

        # 2. update state with state_transition
        self.state = self.state_transition_function.step(self.state, action)

        done = self.t == self.step_per_episode - 1

        if done:
            obs, _ = self.reset()

        else:
            self.t += 1
            obs = self._observation(self.state)

        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Initialize the environment.

        Returns
        -------
        obs: ndarray of shape (state_dim,)
            State observation, which possibly be noisy.

        info: (empty) dict
            Additional information that may be useful for the package users.
            This is unavailable to the RL agent.

        """
        if seed is not None:
            self.random_ = check_random_state(seed)

        self.t = 0
        state = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim,))
        self.state = state / np.linalg.norm(state, ord=2)
        obs = self._observation(self.state)

        return obs, {}

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
