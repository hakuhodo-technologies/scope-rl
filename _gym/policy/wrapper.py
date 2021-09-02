"""Wrapper class to convert greedy policy into stochastic."""
from abc import abstractmethod
import warnings
from typing import List
from typing import Sequence, Optional, Union, Any
from dataclasses import dataclass

import numpy as np

import gym
from sklearn.utils import check_random_state

from d3rlpy.algos import AlgoBase
from d3rlpy.dataset import MDPDataset, Transition


@dataclass
class BaseHead(metaclass=ABCMeta):
    """Base class to convert greedy policy into stochastic."""

    base_algo: AlgoBase

    @abstractmethod
    def stochastic_action_with_pscore(self, x: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def calculate_action_choice_propability(self, x: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        raise NotImplementedError()

    def predict(self, x: np.ndarray):
        return self.base_algo.predict(x)

    def predict_value(self, x: np.ndarray, action: np.ndarray, with_std: bool = False):
        return self.base_algo.predict_value(x, action, with_std)

    def sample_action(self, x: np.ndarray):
        return self.base_algo.sample_action(x)

    def build_with_dataset(self, dataset: MDPDataset):
        return self.base_algo.build_with_dataset(dataset)

    def build_with_env(self, env: gym.env):
        return self.base_algo.build_with_env(env)

    def copy_policy_from(self, algo: AlgoBase):
        return self.base_algo.copy_policy_from(algo)

    def copy_q_function_from(self, algo: AlgoBase):
        return self.base_algo.copy_q_function_from(algo)

    def fit(self, dataset: MDPDataset, **kwargs):
        return self.base_algo.fit(dataset, **kwargs)

    def fit_batch_online(self, env: gym.Env, **kwargs):
        return self.base_algo.fit_batch_online(env, **kwargs)

    def fit_online(self, env: gym.Env, **kwargs):
        return self.base_algo.fit_online(env, **kwargs)

    def fitter(self, env: gym.Env, **kwargs):
        return self.base_algo.fitter(env, **kwargs)

    def generate_new_data(self, transition: Transition):
        return self.base_algo.generate_new_data(transition)

    def collect(self, env: gym.env, **kwargs):
        return self.base_algo.collect(env, **kwargs)

    def update(self, batch: TransitionMiniBatch):
        return self.base_algo.update(batch)

    def create_impl(self, observation_shape: Sequence[int], action_size: int):
        return self.base_algo.create_impl(observation_shape, action_size)

    def get_action_type(self):
        return self.base_algo.get_action_type()

    def get_params(self, **kwargs):
        return self.base_algo.get_parames()

    def load_model(self, fname: str):
        return self.base_algo.load_model(fname)

    def from_json(self, fname: str, **kwargs):
        return self.base_algo.from_json(fname, **kwargs)

    def save_model(self, fname: str):
        return self.base_algo.save_model(fname)

    def save_params(self, logger: D3RLPyLogger):
        return self.base_algo.save_model(logger)

    def save_policy(self, fname: str, **kwargs):
        return self.base_algo.save_policy(fname, **kwargs)

    def set_active_logger(self, logger: D3RLPyLogger):
        return self.base_algo.set_active_logger(logger)

    def set_grad_step(self, grad_step: int):
        return self.base_algo.set_grad_step(grad_step)

    def set_params(self, **params):
        return self.base_algo.set_params(**params)

    @property
    def scaler(self):
        return self.base_algo.scaler

    @property
    def action_scalar(self):
        return self.base_algo.action_scaler

    @property
    def reward_scaler(self):
        return self.base_algo.reward_scaler

    @property
    def observation_space(self):
        return self.base_algo.observation_space

    @property
    def action_size(self):
        return self.base_algo.action_size

    @property
    def gamma(self):
        return self.base_algo.gamma

    @property
    def batch_size(self):
        return self.base_algo.batch_size

    @property
    def grad_step(self):
        return self.base_algo.grad_step

    @property
    def n_frames(self):
        return self.base_algo.n_frames

    @property
    def n_steps(self):
        return self.base_algo.n_steps

    @property
    def impl(self):
        return self.base_algo.impl

    @property
    def action_logger(self):
        return self.base_algo.action_logger


@dataclass
class EpsilonGreedyHead(BaseHead):
    """Class to convert greedy policy into e-greedy."""

    base_algo: AlgoBase
    n_actions: int
    epsilon: float
    random_state: int = 12345

    def __post_init__(self):
        self.action_matrix = np.eye(self.n_actions)
        self.random_ = check_random_state(self.random_state)

    def stochastic_action_with_pscore(self, x: np.ndarrray):
        greedy_action = self.base_algo.predict(x)
        random_action = self.random_.randint(self.n_actions, size=len(x))
        greedy_mask = self.random_.rand(len(x)) > self.epsilon
        action = greedy_action * greedy_mask + random_action * (1 - greedy_mask)
        pscore = (1 - self.epsilon) * greedy_mask + (self.epsilon / self.n_actions) * (
            1 - greedy_mask
        )
        return action, pscore

    def calculate_pscore(self, x: np.ndarray):
        greedy_action = self.base_algo.predict(x)
        greedy_action_matrix = self.action_matrix[greedy_action]
        uniform_matrix = np.ones_like(greedy_action_matrix, dtype=float)
        pscore = (1 - self.epsilon) * greedy_action_matrix + (
            self.epsilon / self.n_actions
        ) * uniform_matrix
        return pscore  # shape (n_steps, n_actions)

    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        greedy_action = self.base_algo.predict(x)
        greedy_mask = greedy_action == action
        pscore = (1 - self.epsilon + self.epsilon / self.n_actions) * greedy_mask + (
            self.epsilon / self.n_actions
        ) * (1 - greedy_mask)
        return pscore

    def sample_action(self, x: np.ndarray):
        greedy_action = self.base_algo.predict(x)
        random_action = self.random_.randint(self.n_actions, size=len(x))
        greedy_mask = self.random_.rand(len(x)) > self.epsilon
        action = greedy_action * greedy_mask + random_action * (1 - greedy_mask)
        return action


@dataclass
class SoftmaxHead(BaseHead):
    """Class to convert policy values into softmax policy."""

    base_algo: AlgoBase
    n_actions: int
    tau: float = 1.0
    random_state = 12345

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)

    def _softmax(self, x: np.ndarray):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def stochastic_action_with_pscore(self, x: np.ndarray):
        prob = self.calculate_pscore(x)

        action = []
        for i in range(len(prob)):
            action.append(self.random_.choice(self.n_actions, p=prob[i]))
        action = np.array(action)

        action_id = np.array(
            [action[i] + i * self.n_actions for i in range(len(action))]
        ).flatten()
        pscore = prob.flatten()[action_id]

        return action, pscore

    def calculate_pscore(self, x: np.ndarray):
        # duplicate x
        # (n_samples, dim) -> (n_samples * n_actions, dim)
        x_ = []
        for i in range(x.shape[0]):
            x_.append(np.tile(x[i], (self.n_actions, 1)))
        x_ = np.array(x_).reshape((-1, x.shape[1]))

        a_ = np.tile(np.arange(self.n_actions), x.shape[0])
        predicted_value = self.base_algo.predict_value(x_, a_)

        return self._softmax(predicted_value)

    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        prob = self.calculate_pscore(x)
        actions_id = np.array(
            [action[i] + i * self.n_actions for i in range(len(action))]
        ).flatten()
        return prob.flatten()[actions_id]

    def sample_action(self, x: np.ndarray):
        prob = self.calculate_pscore(x)

        action = []
        for i in range(len(prob)):
            action.append(self.random_.choice(self.n_actions, p=prob[i]))

        return np.array(action)


@dataclass
class GaussianHead:
    base_algo: AlgoBase
    sigma: np.ndarray
    random_state = 12345

    def __post_init__(self):
        if not (
            isinstance(self.sigma, np.ndarray)
            and self.sigma.shape == (self.base_algo.action_size,)
        ):
            raise ValueError("sigma must have the same size with env.action_space")
        self.random_ = check_random_state(self.random_state)

    def _calc_pscore(self, greedy_action: np.ndarray, action: np.ndarray):
        prob = np.exp(
            -((greedy_action - action) ** 2) / (2 * self.sigma ** 2)
        ) / np.sqrt(2 * np.pi * self.sigma ** 2)
        return np.prod(prob, axis=1)

    def stochastic_action_with_pscore(self, x: np.ndarray):
        greedy_action = self.base_algo.predict(x)
        action = self.random_.normal(loc=greedy_action, scale=self.sigma)
        pscore = self._calc_pscore(greedy_action, action)
        return action, pscore

    def calculate_pscore_given_action(self, x: np.ndrray, action: np.ndarray):
        greedy_action = self.base_algo.predict(x)
        return self._calc_pscore(greedy_action, action)

    def sample_action(self, x: np.ndarray):
        greedy_action = self.base_algo.predict(x)
        return self.random_.normal(loc=greedy_action, scale=self.sigma)
