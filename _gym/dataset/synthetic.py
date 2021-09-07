"""Synthetic Dataset Generation."""
from dataclasses import dataclass
from tqdm import tqdm

import gym
import numpy as np
from sklearn.utils import check_random_state

from _gym.dataset.base import BaseDataset
from _gym.types import LoggedDataset
from _gym.policy import BaseHead


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

    behavior_policy: AlgoBase
        RL policy for data collection.

    random_state: int, default=12345
        Random state.

    Examples
    -------

    .. ::code-block:: python

        # import necessary module from _gym
        >>> from _gym.env import RTBEnv, CustomizedRTBEnv
        >>> from _gym.dataset import SyntheticDataset
        >>> from _gym.policy import DQN
        # import necessary module from other library
        >>> from sklearn.linear_model import LogisticRegression

        # initialize environment and define (RL) agent (i.e., policy)
        >>> env = RTBEnv()
        >>> dqn = DQN()

        # customize environment from the decision makers' perspective
        >>> env = CustomizedRTBEnv(
                original_env=env,
                reward_predictor=LogisticRegression(),
                action_type="discrete",
            )

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
        'action_meaning': array([ 0.1       ,  0.16681005,  0.27825594,  0.46415888,  0.77426368,
                1.29154967,  2.15443469,  3.59381366,  5.9948425 , 10.        ]),
        'state_keys': ['timestep',
        'remaining_budget',
        'budget_consumption_rate',
        'cost_per_mille_of_impression',
        'winning_rate',
        'reward',
        'adjust_rate'],
        'state': array([[0.00000000e+00, 1.00000000e+04, 2.15246222e-01, ...,
                1.88961951e-01, 2.00000000e+00, 2.98123434e-01],
                [1.00000000e+00, 9.00300000e+03, 9.97000000e-02, ...,
                9.07079646e-01, 0.00000000e+00, 1.29154967e+00],
                [2.00000000e+00, 9.00300000e+03, 0.00000000e+00, ...,
                0.00000000e+00, 0.00000000e+00, 1.00000000e-01],
                ...,
                [2.10000000e+01, 3.00000000e+00, 0.00000000e+00, ...,
                1.32158590e-02, 0.00000000e+00, 4.64158883e-01],
                [2.20000000e+01, 3.00000000e+00, 0.00000000e+00, ...,
                8.46560847e-02, 0.00000000e+00, 5.99484250e+00],
                [2.30000000e+01, 3.00000000e+00, 0.00000000e+00, ...,
                0.00000000e+00, 0.00000000e+00, 5.99484250e+00]]),
        'action': array([5., 0., 9., ..., 8., 8., 5.]),
        'reward': array([0., 0., 6., ..., 0., 0., 0.]),
        'done': array([0., 0., 0., ..., 0., 0., 1.]),
        'info': {'impression': array([205.,   0., 226., ...,  16.,   0.,  15.]),
        'click': array([20.,  0., 20., ...,  0.,  0.,  0.]),
        'conversion': array([0., 0., 6., ..., 0., 0., 0.]),
        'average_bid_price': array([ 68.64159292,   4.93777778, 487.30088496, ..., 348.17460317,
                302.79534884,  58.60294118])},
        'pscore': array([0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1])}

        # on-policy policy value of behavior policy
        >>> on_policy_policy_value = dataset.calc_on_policy_policy_value(
                n_episodes=10000
            )
        >>> on_policy_policy_value
        48.7

    """

    env: gym.Env
    behavior_policy: BaseHead
    random_state: int = 12345

    def __post_init__(self):
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

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
        info = {}

        idx = 0
        for _ in tqdm(
            np.arange(n_episodes),
            desc="[obtain_trajectories]",
            total=n_episodes,
        ):
            state = self.env.reset()
            done = False

            while not done:
                (
                    action,
                    action_prob,
                ) = self.behavior_policy.stochastic_action_with_pscore(state)
                next_state, reward, done, info_ = self.env.step(action)

                if idx == 0:
                    info_keys = info_.keys()
                    for key in info_keys:
                        info[key] = np.empty(n_episodes * self.env.step_per_episode)

                states[idx] = state
                actions[idx] = action
                action_probs[idx] = action_prob
                rewards[idx] = reward
                dones[idx] = done

                for key, value in info_.items():
                    info[key][idx] = value

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
            "action": actions
            if self.env.action_type == "discrete"
            else actions.reshape((-1, 1)),
            "reward": rewards,
            "done": dones,
            "info": info,
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
        raise NotImplementedError()

    def pretrain_behavior_policy(self, n_episodes: int = 10000) -> None:
        """Pre-train behavior policy by interacting with the environment. (fix later)

        Parameters
        -------
        n_episodes: int, default=10000
            Number of episodes to train behavior policy.

        """
        if not (isinstance(n_episodes, int) and n_episodes > 0):
            raise ValueError(
                f"n_episodes must be a positive integer, but {n_episodes} is given"
            )
        raise NotImplementedError()
