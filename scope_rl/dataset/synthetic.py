# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class to handle synthetic dataset generation."""
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Union
from tqdm.auto import tqdm

import gym
from gym.spaces import Discrete
import numpy as np
from sklearn.utils import check_scalar, check_random_state

from .base import BaseDataset
from ..policy.head import BaseHead
from ..utils import MultipleLoggedDataset
from ..types import LoggedDataset


@dataclass
class SyntheticDataset(BaseDataset):
    """Class for synthetic data generation.

    Bases: :class:`scope_rl.dataset.BaseDataset`

    Imported as: :class:`scope_rl.dataset.SyntheticDataset`

    Note
    -------
    Logged dataset is directly used for Off-Policy Evaluation (OPE).
    Moreover, it is also compatible with `d3rlpy <https://github.com/takuseno/d3rlpy>`_ (offline RL library) with the following command.

    .. code-block:: python

        d3rlpy_dataset = MDPDataset(
            observations=logged_datasets["state"],
            actions=logged_datasets["action"],
            rewards=logged_datasets["reward"],
            terminals=logged_datasets["done"],
            episode_terminals=logged_datasets["terminal"],
            discrete_action=(logged_datasets["action_type"]=="discrete"),
        )

    .. seealso::

        (external) `d3rlpy's documentation about MDPDataset <https://d3rlpy.readthedocs.io/en/latest/references/dataset.html>`_

    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    max_episode_steps: int, default=None (> 0)
        Maximum number of timesteps in an episode.

    action_meaning: dict
        Dictionary to map discrete action index to a specific action.
        If action_type is "continuous", `None` is recorded.

    action_keys: list of str
        Name of each dimension in the action space.
        If action_type is "discrete", `None` is recorded.

    state_keys: list of str
        Name of each dimension of the state space.

    info_keys: Dict[str, type]
        Dictionary containing the key and type of info components.

    Examples
    -------

    Preparation:

    .. code-block:: python

        # import necessary module from SCOPE-RL
        from scope_rl.dataset import SyntheticDataset
        from scope_rl.policy import EpsilonGreedyHead

        # import necessary module from other libraries
        import gym
        import rtbgym
        from d3rlpy.algos import DoubleDQN
        from d3rlpy.online.buffers import ReplayBuffer
        from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        env = gym.make("RTBEnv-discrete-v0")

        # for api compatibility to d3rlpy
        from scope_rl.utils import OldGymAPIWrapper
        env_ = OldGymAPIWrapper(env)

        # define (RL) agent (i.e., policy) and train on the environment
        ddqn = DoubleDQN()
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env_,
        )
        explorer = ConstantEpsilonGreedy(
            epsilon=0.3,
        )
        ddqn.fit_online(
            env=env_,
            buffer=buffer,
            explorer=explorer,
            n_steps=10000,
            n_steps_per_epoch=1000,
        )

        # convert ddqn policy to stochastic data collection policy
        behavior_policy = EpsilonGreedyHead(
            ddqn,
            n_actions=env.action_space.n,
            epsilon=0.3,
            name="ddqn_epsilon_0.3",
            random_state=12345,
        )

    **Synthetic Dataset Generation**:

    .. code-block:: python

        # initialize dataset class
        dataset = SyntheticDataset(
            env=env,
            max_episode_steps=env.step_per_episode,
            action_meaning=env.action_meaning,
            state_keys=env.obs_keys,
            info_keys={
                "search_volume": int,
                "impression": int,
                "click": int,
                "conversion": int,
                "average_bid_price": float,
            },
        )

        # data collection
        logged_datasets = dataset.obtain_episodes(
            behavior_policies=behavior_policy,
            n_trajectories=100,
            obtain_info=True,
            random_state=12345,
        )

    **Output**:

    .. code-block:: python

        >>> logged_datasets

        {'size': 700,
        'n_trajectories': 100,
        'step_per_trajectory': 7,
        'action_type': 'discrete',
        'action_dim': 10,
        'action_keys': None,
        'action_meaning': array([ 0.1       ,  0.16681005,  0.27825594,  0.46415888,  0.77426368,
                1.29154967,  2.15443469,  3.59381366,  5.9948425 , 10.        ]),
        'state_dim': 7,
        'state_keys': ['timestep',
        'remaining_budget',
        'budget_consumption_rate',
        'cost_per_mille_of_impression',
        'winning_rate',
        'reward',
        'adjust_rate'],
        'state': array([[0.00000000e+00, 3.00000000e+03, 9.29616093e-01, ...,
             1.83918812e-01, 2.00000000e+00, 4.71334329e-01],
            [1.00000000e+00, 1.91000000e+03, 3.63333333e-01, ...,
            1.00000000e+00, 6.00000000e+00, 1.00000000e+01],
            [2.00000000e+00, 1.91000000e+03, 0.00000000e+00, ...,
            0.00000000e+00, 0.00000000e+00, 1.66810054e-01],
            ...,
            [4.00000000e+00, 9.54000000e+02, 5.40904716e-01, ...,
            1.00000000e+00, 2.00000000e+00, 3.59381366e+00],
            [5.00000000e+00, 6.10000000e+01, 9.36058700e-01, ...,
            9.90049751e-01, 7.00000000e+00, 3.59381366e+00],
            [6.00000000e+00, 6.10000000e+01, 0.00000000e+00, ...,
            0.00000000e+00, 0.00000000e+00, 1.00000000e-01]]),
        'action': array([9., 1., 9., ..., 7., 0., 9.]),
        'reward': array([ 6.,  0.,  1., ..., 7.,  0.,  0.]),
        'done': array([0., 0., 0., ..., 0., 0., 1.]),
        'terminal': array([0., 0., 0., ..., 0., 0., 1.]),
        'info': {'search_volume': array([201.,   205.,  217., ..., 201.,   191., 186.]),
        'impression': array([201.,   0.,  217., ..., 199.,   0.,   8.]),
        'click': array([21.,  0.,  24., ...,  18.,  0.,  0.]),
        'conversion': array([ 6.,  0.,  1., ..., 7.,  0.,  0.]),
        'average_bid_price': array([544.55223881,   8.24390244, 523.24423963, ..., 172.58706468,
                   4.2565445 , 458.76344086])},
        'pscore': array([0.73, 0.73, 0.73, ..., 0.73, 0.03, 0.73]),
        'behavior_policy': 'ddqn_epsilon_0.3',
        'dataset_id': 0}

    .. seealso::

        * :doc:`Quickstart </documentation/quickstart>`

    """

    env: gym.Env
    max_episode_steps: Optional[int] = None
    action_meaning: Optional[Dict[int, Any]] = None
    action_keys: Optional[List[str]] = None
    state_keys: Optional[List[str]] = None
    info_keys: Optional[Dict[str, type]] = None

    def __post_init__(self):
        if not isinstance(self.env, gym.Env):
            raise ValueError(
                "env must be a child class of gym.Env",
            )

        self.state_dim = self.env.observation_space.shape[0]

        if isinstance(self.env.action_space, Discrete):
            self.action_type = "discrete"
            self.n_actions = self.env.action_space.n
            self.action_dim = None
        else:
            self.action_type = "continuous"
            self.n_actions = None
            self.action_dim = self.env.action_space.shape[0]
            self.action_min = self.env.action_space.low + 1e-10
            self.action_max = self.env.action_space.high - 1e-10

        if self.max_episode_steps is None:
            if self.env.spec.max_episode_steps is None:
                raise ValueError(
                    "when env.spec.max_episode_steps is None, max_episode_steps must be given."
                )
            else:
                self.max_episode_steps = self.env.spec.max_episode_steps

        check_scalar(
            self.max_episode_steps,
            name="maximum_episode_steps",
            target_type=int,
            min_val=1,
        )

        self.random_ = check_random_state(0)

    def _obtain_episodes(
        self,
        behavior_policy: BaseHead,
        dataset_id: int = 0,
        n_trajectories: int = 10000,
        step_per_trajectory: Optional[int] = None,
        obtain_info: bool = False,
        record_unclipped_action: bool = False,
        random_state: Optional[int] = None,
    ) -> LoggedDataset:
        """Rollout the behavior policy and obtain episodes.

        Note
        -------
        This function is intended to be used for the environment which has a fixed length of episodes (episodic setting).

        For non-episodic, stationary setting (such as cartpole or taxi as used in (Liu et al., 2018) and (Uehara et al., 2020)),
        please also consider using :class:`.obtain_steps()` to generate a logged dataset.

        **References**

        Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
        "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

        Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
        "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

        Parameters
        -------
        behavior_policy: BaseHead
            RL policy to generate a logged dataset.

        dataset_id: int, default=0 (>= 0)
            Id of the logged dataset.

        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to generate by rolling out the behavior policy.

        step_per_trajectory: int, default=None (> 0)
            Number of timesteps in an trajectory.

        obtain_info: bool, default=False
            Whether to gain info from the environment or not.

        record_unclipped_action: bool, default=False
            Whether to record unclipped action values in the logged dataset. Only applicable when action_type is continuous.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        logged_dataset: list of dict
            Dictionary containing environmental settings and trajectories generated by the behavior policy.

            .. code-block:: python

                key: [
                    size,
                    n_trajectories,
                    step_per_trajectory,
                    action_type,
                    n_actions,
                    action_dim,
                    action_keys,
                    action_meaning,
                    state_dim,
                    state_keys,
                    state,
                    action,
                    reward,
                    done,
                    terminal,
                    info,
                    pscore,
                    behavior_policy,
                    dataset_id,
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Type of the action space.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the action space.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of each dimension in the action space.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary to map discrete action index to a specific action.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the state space.

            state_keys: list of str
                Name of each dimension of the state space.

            state: ndarray of shape (size, state_dim)
                State observed by the behavior policy.

            action: ndarray of shape (size, ) or (size, action_dim)
                Action chosen by the behavior policy.

            reward: ndarray of shape (size, )
                Reward observed for each (state, action) pair.

            done: ndarray of shape (size, )
                Whether an episode ends or not.

            terminal: ndarray of shape (size, )
                Whether an episode reaches the pre-defined maximum steps.

            info: dict
                Additional feedbacks from the environment.

            pscore: ndarray of shape (size, )
                Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

            behavior_policy: str
                Name of the behavior policy.

            dataset_id: int
                Id of the logged dataset.

        """
        if not isinstance(behavior_policy, BaseHead):
            raise ValueError("behavior_policy must be a child class of BaseHead")

        if step_per_trajectory is None:
            step_per_trajectory = self.max_episode_steps

        check_scalar(
            n_trajectories,
            name="n_espisodes",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            step_per_trajectory,
            name="step_per_trajectory",
            target_type=int,
            min_val=1,
        )

        if random_state is not None:
            self.env.reset(seed=random_state)

        states = np.zeros(
            (n_trajectories * step_per_trajectory, self.env.observation_space.shape[0])
        )
        if self.action_type == "discrete":
            actions = np.zeros(n_trajectories * step_per_trajectory, dtype=int)
            action_probs = np.zeros(n_trajectories * step_per_trajectory)
        else:
            actions = np.zeros((n_trajectories * step_per_trajectory, self.action_dim))
            action_probs = np.zeros(
                (n_trajectories * step_per_trajectory, self.action_dim)
            )

        rewards = np.zeros(n_trajectories * step_per_trajectory)
        dones = np.zeros(n_trajectories * step_per_trajectory)
        terminals = np.zeros(n_trajectories * step_per_trajectory)
        info = {}

        idx = 0
        for _ in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_trajectories]",
            total=n_trajectories,
        ):
            state, info_ = self.env.reset()
            terminal = False

            for t in range(step_per_trajectory):
                (
                    action,
                    action_prob,
                ) = behavior_policy.sample_action_and_output_pscore_online(state)

                if self.action_type == "continuous":
                    if record_unclipped_action:
                        val_action = np.clip(action, self.action_min, self.action_max)
                    else:
                        action = np.clip(action, self.action_min, self.action_max)
                        val_action = action
                else:
                    val_action = action

                next_state, reward, done, truncated, info_ = self.env.step(val_action)

                if (idx + 1) % step_per_trajectory == 0:
                    done = terminal = True

                states[idx] = state
                actions[idx] = action
                action_probs[idx] = action_prob
                rewards[idx] = reward
                dones[idx] = done
                terminals[idx] = terminal

                if obtain_info:
                    if idx == 0:
                        for key, type_ in self.info_keys.items():
                            if type_ in [int, float]:
                                info[key] = np.zeros(
                                    n_trajectories * step_per_trajectory, dtype=type_
                                )
                            else:
                                info[key] = []

                    for key, type_ in self.info_keys.items():
                        if type_ in [int, float]:
                            info[key][idx] = info_[key]
                        else:
                            info[key].append(info_[key])

                state = next_state
                idx += 1

        logged_dataset = {
            "size": n_trajectories * step_per_trajectory,
            "n_trajectories": n_trajectories,
            "step_per_trajectory": step_per_trajectory,
            "action_type": self.action_type,
            "n_actions": self.n_actions,
            "action_dim": self.action_dim,
            "action_meaning": self.action_meaning,
            "action_keys": self.action_keys,
            "state_dim": self.state_dim,
            "state_keys": self.state_keys,
            "state": states,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "terminal": terminals,
            "info": info,
            "pscore": action_probs,
            "behavior_policy": behavior_policy.name,
            "dataset_id": dataset_id,
        }
        return logged_dataset

    def _obtain_steps(
        self,
        behavior_policy: BaseHead,
        dataset_id: int = 0,
        n_trajectories: int = 10000,
        step_per_trajectory: int = 10,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        obtain_info: bool = False,
        obtain_trajectories_from_single_interaction: bool = False,
        record_unclipped_action: bool = False,
        random_state: Optional[int] = None,
    ) -> LoggedDataset:
        """Rollout the behavior policy and obtain steps.

        Note
        -------
        This function is intended to be used for the environment which has a stationary state distribution
        (such as cartpole or taxi as used in (Liu et al., 2018) and (Uehara et al., 2020)).

        For the (standard) episodic RL setting, please also consider using :class:`.obtain_episodes()`.

        **References**

        Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
        "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

        Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
        "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

        Parameters
        -------
        behavior_policy: BaseHead
            RL policy to generate a logged dataset.

        dataset_id: int, default=0 (>= 0)
            Id of the logged dataset.

        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to generate by rolling out the behavior policy.

        step_per_trajectory: int, default=100 (> 0)
            Number of timesteps in an trajectory.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout by the behavior policy before generating the logged dataset
            when working on the infinite horizon setting.
            This argument is irrelevant when working on the finite horizon setting.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout by the behavior policy before generating the logged dataset
            when working on the infinite horizon setting.
            This argument is irrelevant when working on the finite horizon setting.

        obtain_info: bool, default=False
            Whether to gain info from the environment or not.

        obtain_trajectories_from_single_interaction: bool, default=False
            Whether to collect whole data from a single trajectory.
            If `True`, the initial state of trajectory i is the next state of the trajectory (i-1)'s last state.
            If `False`, the initial state will be sampled by rolling out the behavior policy after resetting the environment.

        record_unclipped_action: bool, default=False
            Whether to record unclipped action values in the logged dataset. Only applicable when action_type is continuous.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        logged_dataset: dict
            Dictionary containing environmental settings and trajectories generated by the behavior policy.

            .. code-block:: python

                key: [
                    size,
                    n_trajectories,
                    step_per_trajectory,
                    action_type,
                    n_actions,
                    action_dim,
                    action_keys,
                    action_meaning,
                    state_dim,
                    state_keys,
                    state,
                    action,
                    reward,
                    done,
                    terminal,
                    info,
                    pscore,
                    behavior_policy,
                    dataset_id,
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Type of the action space.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the action space.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of each dimension in the action space.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary to map discrete action index to a specific action.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the state space.

            state_keys: list of str
                Name of each dimension of the state space.

            state: ndarray of shape (size, state_dim)
                State observed by the behavior policy.

            action: ndarray of shape (size, ) or (size, action_dim)
                Action chosen by the behavior policy.

            reward: ndarray of shape (size, )
                Reward observed for each (state, action) pair.

            done: ndarray of shape (size, )
                Whether an episode ends or not.

            terminal: ndarray of shape (size, )
                Whether an episode reaches the pre-defined maximum steps.

            info: dict
                Additional feedbacks from the environment.

            pscore: ndarray of shape (size, )
                Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

            behavior_policy: str
                Name of the behavior policy.

            dataset_id: int
                Id of the logged dataset.

        """
        if not isinstance(behavior_policy, BaseHead):
            raise ValueError("behavior_policy must be a child class of BaseHead")

        check_scalar(n_trajectories, name="n_trajectories", target_type=int, min_val=1)
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_scalar(
            minimum_rollout_length,
            name=minimum_rollout_length,
            target_type=int,
            min_val=0,
        )
        check_scalar(
            maximum_rollout_length,
            name=maximum_rollout_length,
            target_type=int,
            min_val=0,
        )
        if maximum_rollout_length < minimum_rollout_length:
            raise ValueError(
                "maximum_rollout_length must be larger than minimum_rollout_length, but found False."
            )

        if random_state is not None:
            self.env.reset(seed=random_state)
            self.random_ = check_random_state(random_state)

        states = np.zeros(
            (n_trajectories * step_per_trajectory, self.env.observation_space.shape[0])
        )
        if self.action_type == "discrete":
            actions = np.zeros(n_trajectories * step_per_trajectory, dtype=int)
            action_probs = np.zeros(n_trajectories * step_per_trajectory, dtype=int)
        else:
            actions = np.zeros(n_trajectories * step_per_trajectory, self.action_dim)
            action_probs = np.zeros(
                n_trajectories * step_per_trajectory, self.action_dim
            )

        rewards = np.zeros(n_trajectories * step_per_trajectory)
        dones = np.zeros(n_trajectories * step_per_trajectory)
        terminals = np.zeros(n_trajectories * step_per_trajectory)
        info = {}

        rollout_lengths = self.random_.choice(
            np.arange(minimum_rollout_length, maximum_rollout_length),
            size=n_trajectories,
        )

        idx, step = 0, 0
        done = False
        state, info_ = self.env.reset()

        for i in tqdm(
            np.arange(n_trajectories),
            desc="[obtain_trajectories]",
            total=n_trajectories,
        ):
            state = next_state

            if not obtain_trajectories_from_single_interaction:
                done = True

                for rollout_step in rollout_lengths[i]:
                    if done:
                        state, info_ = self.env.reset()
                        step = 0

                    action = behavior_policy.sample_action_online(state)

                    if self.action_type == "continuous":
                        if record_unclipped_action:
                            val_action = np.clip(
                                action, self.action_min, self.action_max
                            )
                        else:
                            action = np.clip(action, self.action_min, self.action_max)
                            val_action = action
                    else:
                        val_action = action

                    state, reward, done, truncated, info_ = self.env.step(val_action)
                    step += 1

            for t in range(step_per_trajectory):
                if done:
                    state, info_ = self.env.reset()
                    done = False
                    step = 0

                (
                    action,
                    action_prob,
                ) = self.behavior_policy.sample_action_and_output_pscore_online(state)
                next_state, reward, done, truncated, info_ = self.env.step(action)

                states[idx] = state
                actions[idx] = action
                action_probs[idx] = action_prob
                rewards[idx] = reward
                dones[idx] = done
                terminals[idx] = step + 1 == self.max_episode_steps

                if obtain_info:
                    if idx == 0:
                        for key, type_ in self.info_keys.items():
                            if type_ in [int, float]:
                                info[key] = np.zeros(
                                    n_trajectories * step_per_trajectory, dtype=type_
                                )
                            else:
                                info[key] = []

                    for key, type_ in self.info_keys.items():
                        if type_ in [int, float]:
                            info[key][idx] = info_[key]
                        else:
                            info[key].append(info_[key])

                state = next_state
                idx += 1
                step += 1

        logged_dataset = {
            "size": n_trajectories * step_per_trajectory,
            "n_trajectories": n_trajectories,
            "step_per_trajectory": step_per_trajectory,
            "action_type": self.action_type,
            "n_actions": self.n_actions,
            "action_dim": self.action_dim,
            "action_meaning": self.action_meaning,
            "action_keys": self.action_keys,
            "state_dim": self.state_dim,
            "state_keys": self.state_keys,
            "state": states,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "terminal": terminals,
            "info": info,
            "pscore": action_probs,
            "behavior_policy": behavior_policy.name,
            "dataset_id": dataset_id,
        }
        return logged_dataset

    def obtain_episodes(
        self,
        behavior_policies: Union[BaseHead, List[BaseHead]],
        n_datasets: int = 1,
        n_trajectories: int = 10000,
        step_per_trajectory: Optional[int] = None,
        obtain_info: bool = False,
        record_unclipped_action: bool = False,
        path: str = "logged_dataset/",
        save_relative_path: bool = False,
        random_state: Optional[int] = None,
    ) -> LoggedDataset:
        """Rollout the behavior policy and obtain episodes.

        Note
        -------
        This function calls :class:`obtain_episodes` and save multiple logged dataset in :class:`MultipleLoggedDataset`.

        Note
        -------
        This function is intended to be used for the environment which has a fixed length of episodes (episodic setting).

        For non-episodic, stationary setting (such as cartpole or taxi as used in (Liu et al., 2018) and (Uehara et al., 2020)),
        please also consider using :class:`.obtain_steps()` to generate a logged dataset.

        **References**

        Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
        "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

        Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
        "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

        Parameters
        -------
        behavior_policies: list of BaseHead or BaseHead
            List of RL policies that generate logged data.

        n_datasets: int, default=1 (> 0)
            Number of generated (independent) datasets.
            If the value is more than 1, the method returns :class:`MultipleLoggedDataset` instead of :class:`LoggedDataset`.

        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to generate by rolling out the behavior policy.

        step_per_trajectory: int, default=None (> 0)
            Number of timesteps in an trajectory.

        obtain_info: bool, default=False
            Whether to gain info from the environment or not.

        record_unclipped_action: bool, default=False
            Whether to record unclipped action values in the logged dataset. Only applicable when action_type is continuous.

        path: str
            Path to the directory. Either absolute or relative path is acceptable.

        save_relative_path: bool, default=False.
            Whether to save a relative path.
            If `True`, a path relative to the scope-rl directory will be saved.
            If `False`, the absolute path will be saved.

            Note that this option was added in order to run examples in the documentation properly.
            Otherwise, the default setting (`False`) is recommended.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        logged_dataset(s): LoggedDataset or MultipleLoggedDataset
            MultipleLoggedDataset is an instance containing (multiple) logged datasets.

            Each logged dataset is accessible by the following command.

            .. code-block:: python

                logged_dataset_0 = logged_datasets.get(behavior_policy.name, 0)

            Each logged dataset consists of the following.

            .. code-block:: python

                key: [
                    size,
                    n_trajectories,
                    step_per_trajectory,
                    action_type,
                    n_actions,
                    action_dim,
                    action_keys,
                    action_meaning,
                    state_dim,
                    state_keys,
                    state,
                    action,
                    reward,
                    done,
                    terminal,
                    info,
                    pscore,
                    behavior_policy,
                    dataset_id,
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Type of the action space.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the action space.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of each dimension in the action space.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary to map discrete action index to a specific action.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the state space.

            state_keys: list of str
                Name of each dimension of the state space.

            state: ndarray of shape (size, state_dim)
                State observed by the behavior policy.

            action: ndarray of shape (size, ) or (size, action_dim)
                Action chosen by the behavior policy.

            reward: ndarray of shape (size, )
                Reward observed for each (state, action) pair.

            done: ndarray of shape (size, )
                Whether an episode ends or not.

            terminal: ndarray of shape (size, )
                Whether an episode reaches the pre-defined maximum steps.

            info: dict
                Additional feedbacks from the environment.

            pscore: ndarray of shape (size, )
                Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

            behavior_policy: str
                Name of the behavior policy.

            dataset_id: int
                Id of the logged dataset.

        """
        if isinstance(behavior_policies, BaseHead):
            if n_datasets == 1:
                logged_dataset = self._obtain_episodes(
                    behavior_policy=behavior_policies,
                    n_trajectories=n_trajectories,
                    step_per_trajectory=step_per_trajectory,
                    obtain_info=obtain_info,
                    record_unclipped_action=record_unclipped_action,
                    random_state=random_state,
                )

            else:
                logged_dataset = MultipleLoggedDataset(
                    action_type=self.action_type,
                    path=path,
                    save_relative_path=save_relative_path,
                )
                for i in tqdm(
                    np.arange(n_datasets),
                    desc="[obtain_datasets: dataset_id]",
                    total=n_datasets,
                ):
                    random_state_ = (
                        random_state if random_state is not None and i == 0 else None
                    )
                    logged_dataset_ = self._obtain_episodes(
                        behavior_policy=behavior_policies,
                        n_trajectories=n_trajectories,
                        step_per_trajectory=step_per_trajectory,
                        obtain_info=obtain_info,
                        record_unclipped_action=record_unclipped_action,
                        random_state=random_state_,
                    )
                    logged_dataset.add(
                        logged_dataset_,
                        behavior_policy_name=behavior_policies.name,
                    )

        else:
            logged_dataset = MultipleLoggedDataset(
                action_type=self.action_type,
                path=path,
                save_relative_path=save_relative_path,
            )

            for j in tqdm(
                np.arange(len(behavior_policies)),
                desc="[obtain_datasets: behavior_policy]",
                total=len(behavior_policies),
            ):
                if n_datasets == 1:
                    logged_dataset = self._obtain_episodes(
                        behavior_policy=behavior_policies[j],
                        n_trajectories=n_trajectories,
                        step_per_trajectory=step_per_trajectory,
                        obtain_info=obtain_info,
                        record_unclipped_action=record_unclipped_action,
                        random_state=random_state,
                    )
                    logged_dataset.add(
                        logged_dataset_, behavior_policy_name=behavior_policies[j].name
                    )

                else:
                    for i in tqdm(
                        np.arange(n_datasets),
                        desc="[obtain_datasets: dataset_id]",
                        total=n_datasets,
                    ):
                        random_state_ = (
                            random_state
                            if random_state is not None and i == 0
                            else None
                        )
                        logged_dataset_ = self._obtain_episodes(
                            behavior_policy=behavior_policies[j],
                            n_trajectories=n_trajectories,
                            step_per_trajectory=step_per_trajectory,
                            obtain_info=obtain_info,
                            record_unclipped_action=record_unclipped_action,
                            random_state=random_state_,
                        )
                        logged_dataset.add(
                            logged_dataset_,
                            behavior_policy_name=behavior_policies[j].name,
                        )

        return logged_dataset

    def obtain_steps(
        self,
        behavior_policies: Union[BaseHead, List[BaseHead]],
        n_datasets: int = 1,
        n_trajectories: int = 10000,
        step_per_trajectory: int = 10,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        obtain_info: bool = False,
        obtain_trajectories_from_single_interaction: bool = False,
        record_unclipped_action: bool = False,
        path: str = "logged_dataset/",
        save_relative_path: bool = False,
        random_state: Optional[int] = None,
    ) -> LoggedDataset:
        """Rollout the behavior policy and obtain steps.

        Note
        -------
        This function is intended to be used for the environment which has a stationary state distribution
        (such as cartpole or taxi as used in (Liu et al., 2018) and (Uehara et al., 2020)).

        For the (standard) episodic RL setting, please also consider using :class:`.obtain_episodes()`.

        **References**

        Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
        "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

        Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
        "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

        Parameters
        -------
        behavior_policies: list of BaseHead or BaseHead
            List of RL policies that generate logged data.

        n_datasets: int, default=1 (> 0)
            Number of generated (independent) datasets.
            If the value is more than 1, the method returns :class:`MultiplLoggedeDataset` instead of :class:`LoggedDataset`.

        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to generate by rolling out the behavior policy.

        step_per_trajectory: int, default=100 (> 0)
            Number of timesteps in an trajectory.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout by the behavior policy before generating the logged dataset
            when working on the infinite horizon setting.
            This argument is irrelevant when working on the finite horizon setting.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout by the behavior policy before generating the logged dataset
            when working on the infinite horizon setting.
            This argument is irrelevant when working on the finite horizon setting.

        obtain_info: bool, default=False
            Whether to gain info from the environment or not.

        obtain_trajectories_from_single_interaction: bool, default=False
            Whether to collect whole data from a single trajectory.
            If `True`, the initial state of trajectory i is the next state of the trajectory (i-1)'s last state.
            If `False`, the initial state will be sampled by rolling out the behavior policy after resetting the environment.

        record_unclipped_action: bool, default=False
            Whether to record unclipped action values in the logged dataset. Only applicable when action_type is continuous.

        seed_env: bool, default=False
            Whether to set seed on environment or not.

        path: str
            Path to the directory. Either absolute or relative path is acceptable.

        save_relative_path: bool, default=False.
            Whether to save a relative path.
            If `True`, a path relative to the scope-rl directory will be saved.
            If `False`, the absolute path will be saved.

            Note that this option was added in order to run examples in the documentation properly.
            Otherwise, the default setting (`False`) is recommended.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        logged_dataset(s): LoggedDataset or MultipleLoggedDataset
            MultipleLoggedDataset is an instance containing (multiple) logged datasets.

            By calling the following command, we can access each logged dataset as follows.

            .. code-block:: python

                logged_dataset_0 = logged_datasets.get(behavior_policy.name, 0)

            Each logged dataset consists the following.

            .. code-block:: python

                key: [
                    size,
                    n_trajectories,
                    step_per_trajectory,
                    action_type,
                    n_actions,
                    action_dim,
                    action_keys,
                    action_meaning,
                    state_dim,
                    state_keys,
                    state,
                    action,
                    reward,
                    done,
                    terminal,
                    info,
                    pscore,
                    behavior_policy,
                    dataset_id,
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Type of the action space.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the action space.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of each dimension in the action space.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary to map discrete action index to a specific action.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the state space.

            state_keys: list of str
                Name of each dimension of the state space.

            state: ndarray of shape (size, state_dim)
                State observed by the behavior policy.

            action: ndarray of shape (size, ) or (size, action_dim)
                Action chosen by the behavior policy.

            reward: ndarray of shape (size, )
                Reward observed for each (state, action) pair.

            done: ndarray of shape (size, )
                Whether an episode ends or not.

            terminal: ndarray of shape (size, )
                Whether an episode reaches the pre-defined maximum steps.

            info: dict
                Additional feedbacks from the environment.

            pscore: ndarray of shape (size, )
                Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

            behavior_policy: str
                Name of the behavior policy.

            dataset_id: int
                Id of the logged dataset.

        """
        if isinstance(behavior_policies, BaseHead):
            if n_datasets == 1:
                logged_dataset = self._obtain_steps(
                    behavior_policy=behavior_policies,
                    n_trajectories=n_trajectories,
                    step_per_trajectory=step_per_trajectory,
                    minimum_rollout_length=minimum_rollout_length,
                    maximum_rollout_length=maximum_rollout_length,
                    obtain_info=obtain_info,
                    obtain_trajectories_from_single_interaction=obtain_trajectories_from_single_interaction,
                    record_unclipped_action=record_unclipped_action,
                    random_state=random_state,
                )

            else:
                behavior_policies = [behavior_policies]

        else:
            logged_dataset = MultipleLoggedDataset(
                action_type=self.action_type,
                path=path,
                save_relative_path=save_relative_path,
            )

            for j in tqdm(
                np.arange(len(behavior_policies)),
                desc="[obtain_datasets: behavior_policy]",
                total=len(behavior_policies),
            ):
                if n_datasets == 1:
                    logged_dataset = self._obtain_steps(
                        behavior_policy=behavior_policies[j],
                        n_trajectories=n_trajectories,
                        step_per_trajectory=step_per_trajectory,
                        minimum_rollout_length=minimum_rollout_length,
                        maximum_rollout_length=maximum_rollout_length,
                        obtain_info=obtain_info,
                        obtain_trajectories_from_single_interaction=obtain_trajectories_from_single_interaction,
                        record_unclipped_action=record_unclipped_action,
                        random_state=random_state,
                    )
                    logged_dataset.add(
                        logged_dataset_, behavior_policy_name=behavior_policies[j].name
                    )

                else:
                    for i in tqdm(
                        np.arange(n_datasets),
                        desc="[obtain_datasets: dataset_id]",
                        total=n_datasets,
                    ):
                        random_state_ = (
                            random_state
                            if random_state is not None and i == 0
                            else None
                        )
                        logged_dataset_ = self._obtain_steps(
                            behavior_policy=behavior_policies[j],
                            n_trajectories=n_trajectories,
                            step_per_trajectory=step_per_trajectory,
                            minimum_rollout_length=minimum_rollout_length,
                            maximum_rollout_length=maximum_rollout_length,
                            obtain_info=obtain_info,
                            obtain_trajectories_from_single_interaction=obtain_trajectories_from_single_interaction,
                            record_unclipped_action=record_unclipped_action,
                            random_state=random_state_,
                        )
                        logged_dataset.add(
                            logged_dataset_,
                            behavior_policy_name=behavior_policies[j].name,
                        )

        return logged_dataset
