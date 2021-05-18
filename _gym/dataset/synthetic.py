"""Synthetic Dataset Generation."""
from dataclasses import dataclass

import gym
import numpy as np
from sklearn.utils import check_random_state

from .base import BaseDataset
from _gym.types import LoggedDataset
from _gym.policy import BasePolicy


@dataclass
class SyntheticDataset(BaseDataset):
    """Class for synthetic data generation.

    Note
    -------
    Generate dataset for Offline reinforcement learning (RL) and off-policy evaluation (OPE).


    Parameters
    -------
    env: gym.Env
        Reinforcement learning (RL) environment.

    behavior_policy: BasePolicy
        RL policy for data collection.

    n_samples_pretrain_reward_predictor: int, default=10000
        Numbers of samples used to pretrain reward predictor.
        Note that pretrain only takes place when env uses reward predictor.

    random_state: int, default=12345
        Random state.

    Examples
    -------

    .. ::code-block:: python

        # import necessary module from _gym
        >>> from _gym.env import RTBEnv
        >>> from _gym.dataset import SyntheticDataset
        >>> from _gym.policy import DQN
        # import necessary module from other library
        >>> from sklearn.linear_model import LogisticRegression

        # initialize environment and define (RL) agent (i.e., policy)
        >>> env = RTBEnv(
                use_reward_predictor=True,
                reward_predictor=LogisticRegression(),
            )
        >>> dqn = DQN()

        # initialize dataset class
        >>> dataset = SyntheticDataset(
                env=env,
                behavior_policy=dqn,
            )

        # pretrain behavior policy before data collection
        >>> dataset.pretrain_behavior_policy()

        # data collection
        >>> logged_dataset = dataset.obtain_trajectories(n_episodes=100)
        >>> logged_dataset
        {'size': 2400,
        'n_episodes': 100,
        'step_per_episode': 24,
        'action_type': 'discrete',
        'action_dim': 10,
        'state_keys': ['timestep',
          'remaining budget',
          'budget consumption rate',
          'cost per mille of impression',
          'winning rate',
          'reward',
          'adjust rate'],
        'state': array([[0.00000000e+00, 1.00000000e+04, 0.00000000e+00, ...,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                [1.00000000e+00, 9.32100000e+03, 6.79000000e-02, ...,
                5.60975610e-01, 3.00000000e+00, 4.00000000e+00],
                [2.00000000e+00, 9.32100000e+03, 0.00000000e+00, ...,
                2.94117647e-02, 0.00000000e+00, 1.00000000e+00],
                ...,
                [2.10000000e+01, 6.30000000e+01, 7.84982935e-01, ...,
                2.00000000e-01, 1.00000000e+00, 8.00000000e+00],
                [2.20000000e+01, 6.30000000e+01, 0.00000000e+00, ...,
                4.16666667e-02, 0.00000000e+00, 8.00000000e+00],
                [2.30000000e+01, 6.30000000e+01, 0.00000000e+00, ...,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]),
        'action': array([4., 1., 4., ..., 8., 0., 7.]),
        'reward': array([3., 0., 2., ..., 0., 0., 0.]),
        'done': array([0., 0., 0., ..., 0., 0., 1.]),
        'info': {'impression': array([23.,  1., 12., ...,  1.,  0.,  1.]),
          'click': array([7., 0., 4., ..., 0., 0., 0.]),
          'conversion': array([3., 0., 2., ..., 0., 0., 0.]),
          'average bid price': array([116.29268293,  24.88235294,  94.5       , ..., 696.33333333,
                13.61290323, 473.75      ])},
        'pscore': array([0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1])}

        # ground-truth policy value of behavior policy
        >>> on_policy_policy_value = dataset.calculate_on_policy_policy_value(
                n_episodes=10000
            )
        >>> on_policy_policy_value
        ...

    """

    env: gym.Env
    behavior_policy: BasePolicy
    n_samples_pretrain_reward_predictor: int = 10000
    random_state: int = 12345

    def __post_init__(self):
        if not (
            isinstance(self.n_samples_pretrain_reward_predictor, int)
            and self.n_samples_pretrain_reward_predictor > 0
        ):
            raise ValueError(
                f"n_samples_pretrain_reward_predictor must be an positive integer, but {self.n_samples_pretrain_reward_predictor} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

        if self.env.use_reward_predictor:
            self.env.fit_reward_predictor(self.n_samples_pretrain_reward_predictor)

    def obtain_trajectories(self, n_episodes: int = 10000) -> LoggedDataset:
        """Rollout behavior policy and obtain trajectories.

        Parameters
        -------
        n_episodes: int, default=10000
            Number of trajectories to rollout behavior policy and collect data.

        Returns
        -------
        logged_dataset: Dict
            size: int
                Total steps the dataset records.

            n_episodes: int
                Total episodes the dataset records.

            step_per_episode: int
                Total timesteps in an episode.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            action_dim: int
                Action dimensions of discrete actions.
                If action_type is "continuous", None is recorded.

            state: NDArray[float], shape (size, 7)
                State of the RL environment.

            action: Union[NDArray[int], NDArray[float]], shape (size, )
                Action chosen by the behavior policy.

            reward: NDArray[int], shape (size, )
                Reward observed for each (state, action) pair.

            done: NDArray[int], shape (size, )
                Whether an episode ends or not.

            info: Dict[str, NDArray[int]], shape (size, )
                Additional feedback information from RL environment.

            pscore: NDArray[float], shape (size, )
                Action choice probability of the behavior policy for the chosen action.

        """
        if not (isinstance(n_episodes, int) and n_episodes > 0):
            raise ValueError(
                f"n_episodes must be a positive integer, but {n_episodes} is given"
            )
        states = np.empty(
            (
                n_episodes * self.env.step_per_episode,
                self.env.observation_space.shape[0],
            )
        )
        actions = np.empty(n_episodes * self.env.step_per_episode)
        action_probs = np.empty(n_episodes * self.env.step_per_episode)
        rewards = np.empty(n_episodes * self.env.step_per_episode)
        dones = np.empty(n_episodes * self.env.step_per_episode)
        infos = {}

        idx = 0
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action, action_prob = self.behavior_policy.act(state)  # fix later
                next_state, reward, done, info = self.env.step(action)

                if idx == 0:
                    info_keys = info.keys()
                    for key in info_keys:
                        infos[key] = np.empty(n_episodes * self.env.step_per_episode)

                states[idx] = state
                actions[idx] = action
                action_probs[idx] = action_prob
                rewards[idx] = reward
                dones[idx] = done

                for key, value in info.items():
                    infos[key][idx] = value

                idx += 1
                state = next_state

        logged_dataset = {
            "size": n_episodes * self.env.step_per_episode,
            "n_episodes": n_episodes,
            "step_per_episode": self.env.step_per_episode,
            "action_type": self.env.action_type,
            "action_dim": self.env.action_dim
            if self.env.action_type == "discrete"
            else None,
            "action_meaning": self.env.action_meaning
            if self.env.action_type == "discrete"
            else None,
            "state_keys": self.env.obs_keys,
            "state": states,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "info": infos,
            "pscore": action_probs,
        }
        return logged_dataset

    def calc_on_policy_policy_value(self, n_episodes: int = 10000) -> float:
        """Calculate on-policy policy value of the behavior policy by rollout.

        Parameters
        -------
        n_episodes: int, default=10000
            Number of episodes to train behavior policy.

        Returns
        -------
        mean_reward: float
            Mean episode reward calculated through rollout.

        """
        return self.env.calc_on_policy_policy_value(self.behavior_policy, n_episodes)

    def pretrain_behavior_policy(self, n_episodes: int = 10000) -> None:
        """Pre-train behavior policy by interacting with the environment.

        Parameters
        -------
        n_episodes: int, default=10000
            Number of episodes to train behavior policy.

        """
        if not (isinstance(n_episodes, int) and n_episodes > 0):
            raise ValueError(
                f"n_episodes must be a positive integer, but {n_episodes} is given"
            )
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action, action_prob = self.behavior_policy.act(state)  # fix later
                next_state, reward, done, _ = self.env.step(action)

                self.behavior_policy.update(
                    state, action, next_state, reward, done
                )  # fix later
                state = next_state
