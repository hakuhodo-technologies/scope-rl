"""Class to handle synthetic dataset generation."""
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
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

    Bases: :class:`ofrl.dataset.BaseDataset`

    Imported as: :class:`ofrl.dataset.SyntheticDataset`

    Note
    -------
    Logged dataset is directly used for Off-Policy Evaluation (OPE).
    Moreover, it is also compatible to `d3rlpy <https://github.com/takuseno/d3rlpy>`_ (offline RL library) with the following command.

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

    behavior_policy: AlgoBase
        RL policy that collects the logged data.

    max_episode_steps: int, default=None (> 0)
        Maximum number of timesteps in an episode.

    action_meaning: dict
        Dictionary which maps discrete action index into specific actions.
        If action_type is "continuous", `None` is recorded.

    action_keys: list of str
        Name of the action variable at each dimension.
        If action_type is "discrete", `None` is recorded.

    state_keys: list of str
        Name of the state variable at each dimension.

    info_keys: Dict[str, type]
        Dictionary containing of key and the type of info components.

    random_state: int, default=None (>= 0)
        Random state.

    Examples
    -------

    Preparation:

    .. code-block:: python

        # import necessary module from OFRL
        from ofrl.dataset import SyntheticDataset
        from ofrl.policy import DiscreteEpsilonGreedyHead

        # import necessary module from other libraries
        import gym
        import rtbgym
        from d3rlpy.algos import DoubleDQN
        from d3rlpy.online.buffers import ReplayBuffer
        from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        env = gym.make("RTBEnv-discrete-v0")

        # define (RL) agent (i.e., policy) and train on the environment
        ddqn = DoubleDQN()
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env,
        )
        explorer = ConstantEpsilonGreedy(
            epsilon=0.3,
        )
        ddqn.fit_online(
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=10000,
            n_steps_per_epoch=1000,
        )

        # convert ddqn policy to stochastic data collection policy
        behavior_policy = DiscreteEpsilonGreedyHead(
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
            behavior_policy=behavior_policy,
            action_meaning=env.action_meaning,
            state_keys=env.obs_keys,
            info_keys={
                "search_volume": int,
                "impression": int,
                "click": int,
                "conversion": int,
                "average_bid_price": float,
            },
            random_state=12345,
        )

        # data collection
        logged_datasets = dataset.obtain_trajectories(n_trajectories=100, obtain_info=True)

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
        'pscore': array([0.73, 0.73, 0.73, ..., 0.73, 0.03, 0.73])}

    .. seealso::

        * :doc:`Quickstart </documentation/quickstart>`
        * :doc:`Related tutorials </documentation/_autogallery/ofrl_others/index>`

    """

    env: gym.Env
    behavior_policy: BaseHead
    max_episode_steps: Optional[int] = None
    action_meaning: Optional[Dict[int, Any]] = None
    action_keys: Optional[List[str]] = None
    state_keys: Optional[List[str]] = None
    info_keys: Optional[Dict[str, type]] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.env, gym.Env):
            raise ValueError(
                "env must be a child class of gym.Env",
            )
        if not isinstance(self.behavior_policy, BaseHead):
            raise ValueError("behavior_policy must be a child class of BaseHead")

        self.state_dim = self.env.observation_space.shape[0]

        if isinstance(self.env.action_space, Discrete):
            self.action_type = "discrete"
            self.n_actions = self.env.action_space.n
            self.action_dim = None
        else:
            self.action_type = "continuous"
            self.n_actions = None
            self.action_dim = self.env.action_space.shape[0]

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

        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def obtain_episodes(
        self,
        n_trajectories: int = 10000,
        step_per_trajectory: Optional[int] = None,
        obtain_info: bool = False,
        seed_env: bool = False,
    ) -> LoggedDataset:
        """Rollout the behavior policy and obtain episodes.

        Note
        -------
        This function is intended to be used for the environment which has a fixed length of episodes (episodic setting).

        For non-episodic, stationary setting (such as cartpole or taxi as used in (Liu et al., 2018) and (Uehara et al., 2020)),
        please also consider using :class:`.obtain_steps()` to collect logged dataset.

        **References**

        Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
        "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

        Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
        "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

        Parameters
        -------
        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to rollout the behavior policy and collect data.

        step_per_trajectory: int, default=None (> 0)
            Number of timesteps in an trajectory.

        obtain_info: bool, default=False
            Whether to gain info from the environment or not.

        seed_env: bool, default=False
            Whether to set seed on environment or not.

        Returns
        -------
        logged_dataset: list of dict
            Dictionary containing the environmental settings and trajectories obtained by the behavior policy.

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
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of discrete actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the actions.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of the action variable at each dimension.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary which maps discrete action index into specific actions.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the states.

            state_keys: list of str
                Name of the state variable at each dimension.

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
                Action choice probability of the behavior policy for the chosen action.

        """
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

        if seed_env:
            self.env.reset(self.random_state)

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
                ) = self.behavior_policy.stochastic_action_with_pscore_online(state)
                next_state, reward, done, truncated, info_ = self.env.step(action)

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
        }
        return logged_dataset

    def obtain_steps(
        self,
        n_trajectories: int = 10000,
        step_per_trajectory: int = 10,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        obtain_info: bool = False,
        obtain_trajectories_from_single_interaction: bool = False,
        seed_env: bool = False,
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
        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to rollout the behavior policy and collect data.

        step_per_trajectory: int, default=100 (> 0)
            Number of timesteps in an trajectory.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout before collecting dataset.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout before collecting dataset.

        obtain_info: bool, default=False
            Whether to gain info from the environment or not.

        obtain_trajectories_from_single_interaction: bool, default=False
            Whether to collect whole data from a single trajectory.
            If `True`, the initial state of trajectory i is the next state of the trajectory (i-1)'s last state.
            If `False`, the initial state will be sampled by rolling out the behavior policy after resetting the environment.

        seed_env: bool, default=False
            Whether to set seed on environment or not.

        Returns
        -------
        logged_dataset: dict
            Dictionary containing the environmental settings and trajectories obtained by the behavior policy.

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
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of discrete actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the actions.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of the action variable at each dimension.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary which maps discrete action index into specific actions.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the states.

            state_keys: list of str
                Name of the state variable at each dimension.

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
                Action choice probability of the behavior policy for the chosen action.

        """
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

        if seed_env:
            self.env.reset(self.random_state)

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

                    action = self.behavior_policy.sample_action_online(state)
                    state, reward, done, truncated, info_ = self.env.step(action)
                    step += 1

            for t in range(step_per_trajectory):

                if done:
                    state, info_ = self.env.reset()
                    done = False
                    step = 0

                (
                    action,
                    action_prob,
                ) = self.behavior_policy.stochastic_action_with_pscore_online(state)
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
        }
        return logged_dataset

    def obtain_multiple_datasets_with_episodes(
        self,
        n_datasets: int = 5,
        n_trajectories: int = 10000,
        step_per_trajectory: Optional[int] = None,
        obtain_info: bool = False,
        seed_env: bool = False,
        path: str = "logged_dataset/",
        save_relative_path: bool = False,
    ) -> LoggedDataset:
        """Rollout the behavior policy and obtain episodes.

        Note
        -------
        This function calls :class:`obtain_episodes` and save multiple logged dataset in :class:`MultipleDatasetClass`.

        Note
        -------
        This function is intended to be used for the environment which has a fixed length of episodes (episodic setting).

        For non-episodic, stationary setting (such as cartpole or taxi as used in (Liu et al., 2018) and (Uehara et al., 2020)),
        please also consider using :class:`.obtain_steps()` to collect logged dataset.

        **References**

        Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
        "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

        Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou.
        "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." 2018

        Parameters
        -------
        n_datasets: int, default=5
            Number of (independent) dataset.

        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to rollout the behavior policy and collect data.

        step_per_trajectory: int, default=None (> 0)
            Number of timesteps in an trajectory.

        obtain_info: bool, default=False
            Whether to gain info from the environment or not.

        seed_env: bool, default=False
            Whether to set seed on environment or not. (Only applicable to the first dataset.)

        path: str
            Path to the directory. Either absolute and relative path is acceptable.

        save_relative_path: bool, default=False.
            Whether to save a relative path.
            If `True`, a path relative to the ofrl directory will be saved.
            If `False`, the absolute path will be saved.

            Note that, this option was added in order to run examples in the documentation properly.
            Otherwise, the default setting (`False`) is recommended.

        Returns
        -------
        logged_datasets: MultipleLoggedDataset
            Instance containing multiple logged datasets.

            Each logged dataset is accessible by the following command.

            .. code-block:: python

                logged_dataset_0 = logged_datasets.get(0)

            Each logged dataset contains the following.

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
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of discrete actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the actions.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of the action variable at each dimension.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary which maps discrete action index into specific actions.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the states.

            state_keys: list of str
                Name of the state variable at each dimension.

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
                Action choice probability of the behavior policy for the chosen action.

        """
        multiple_logged_datasets = MultipleLoggedDataset(
            path=path, save_relative_path=save_relative_path
        )

        for i in range(n_datasets):
            logged_dataset = self.obtain_episodes(
                n_trajectories=n_trajectories,
                step_per_trajectory=step_per_trajectory,
                obtain_info=obtain_info,
                seed_env=(seed_env and i == 0),
            )
            multiple_logged_datasets.add(logged_dataset)

        return multiple_logged_datasets

    def obtain_multiple_logged_dataset_with_steps(
        self,
        n_datasets: int = 5,
        n_trajectories: int = 10000,
        step_per_trajectory: int = 10,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        obtain_info: bool = False,
        obtain_trajectories_from_single_interaction: bool = False,
        seed_env: bool = False,
        path: str = "logged_dataset/",
        save_relative_path: bool = False,
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
        n_datasets: int, default=5
            Number of (independent) dataset.

        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to rollout the behavior policy and collect data.

        step_per_trajectory: int, default=100 (> 0)
            Number of timesteps in an trajectory.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout before collecting dataset.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout before collecting dataset.

        obtain_info: bool, default=False
            Whether to gain info from the environment or not.

        obtain_trajectories_from_single_interaction: bool, default=False
            Whether to collect whole data from a single trajectory.
            If `True`, the initial state of trajectory i is the next state of the trajectory (i-1)'s last state.
            If `False`, the initial state will be sampled by rolling out the behavior policy after resetting the environment.

        seed_env: bool, default=False
            Whether to set seed on environment or not.

        path: str
            Path to the directory. Either absolute and relative path is acceptable.

        save_relative_path: bool, default=False.
            Whether to save a relative path.
            If `True`, a path relative to the ofrl directory will be saved.
            If `False`, the absolute path will be saved.

            Note that, this option was added in order to run examples in the documentation properly.
            Otherwise, the default setting (`False`) is recommended.

        Returns
        -------
        logged_dataset: MultipleLoggedDataset
            Instance containing multiple logged datasets.

            By calling the following command, we can access each logged dataset as follows.

            .. code-block:: python

                logged_dataset_0 = logged_datasets.get(0)

            Each logged dataset contains the following.

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
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of discrete actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the actions.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of the action variable at each dimension.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary which maps discrete action index into specific actions.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the states.

            state_keys: list of str
                Name of the state variable at each dimension.

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
                Action choice probability of the behavior policy for the chosen action.

        """
        multiple_logged_datasets = MultipleLoggedDataset(
            path=path, save_relative_path=save_relative_path
        )
        for i in range(n_datasets):
            logged_dataset = self.obtain_step(
                n_trajectories=n_trajectories,
                step_per_trajectory=step_per_trajectory,
                minimum_rollout_length=minimum_rollout_length,
                maximum_rollout_length=maximum_rollout_length,
                obtain_info=obtain_info,
                obtain_trajectories_from_single_interaction=obtain_trajectories_from_single_interaction,
                seed_env=(seed_env and i == 0),
            )
            multiple_logged_datasets.add(logged_dataset)

        return multiple_logged_datasets
