# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Wrapper class to convert greedy policy into stochastic."""
from abc import abstractmethod
from typing import Union, Optional, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, truncnorm
from sklearn.utils import check_scalar, check_random_state

import gym
from d3rlpy.algos import AlgoBase
from d3rlpy.dataset import MDPDataset, Transition, TransitionMiniBatch
from d3rlpy.logger import D3RLPyLogger

from ..utils import check_array


@dataclass
class BaseHead(AlgoBase):
    """Base class to convert a greedy policy into a stochastic policy.

    Bases: :class:`d3rlpy.algos.AlgoBase`

    Imported as: :class:`scope_rl.policy.BaseHead`

    Note
    -------
    To ensure API compatibility with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, :class:`BaseHead` inherits :class:`d3rlpy.algos.AlgoBase`.
    This base class also has additional methods including :class:`fit`, :class:`predict`, and :class:`predict_value`. Please also refer to the following documentation for the methods that are not described in this API reference.

    .. seealso::

        (external) `d3rlpy's documentation about AlgoBase <https://d3rlpy.readthedocs.io/en/latest/references/algos.html>`_

    """

    @abstractmethod
    def sample_action_and_output_pscore(self, x: np.ndarray):
        """Sample an action stochastically with its pscore."""
        raise NotImplementedError()

    @abstractmethod
    def calc_action_choice_probability(self, x: np.ndarray):
        """Calculate the action choice probabilities."""
        raise NotImplementedError()

    @abstractmethod
    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate the pscore of the given action."""
        raise NotImplementedError()

    def predict_online(self, x: np.ndarray):
        """Predict the best action in an online environment."""
        return self.predict(x.reshape((1, -1)))[0]

    def predict_value_online(
        self,
        x: np.ndarray,
        action: Union[np.ndarray, int, float],
        with_std: bool = False,
    ):
        """Predict the state action value in an online environment."""
        if isinstance(action, (int, float)):
            action = np.array([[action]])
        else:
            action.reshape((1, -1))
        return self.predict_value(x.reshape((1, -1)), action, with_std=with_std)[0]

    def sample_action_online(self, x: np.ndarray):
        """Sample an action in an online environment."""
        return self.sample_action(x.reshape(1, -1))[0]

    def sample_action_and_output_pscore_online(self, x: np.ndarray):
        """Sample an action and calculate its pscore in an online environment."""
        action, pscore = self.sample_action_and_output_pscore(x.reshape(1, -1))
        return action[0], pscore[0]

    def predict(self, x: np.ndarray):
        return self.base_policy.predict(x)

    def predict_value(self, x: np.ndarray, action: np.ndarray, with_std: bool = False):
        return self.base_policy.predict_value(x, action, with_std)

    def sample_action(self, x: np.ndarray):
        return self.base_policy.sample_action(x)

    def build_with_dataset(self, dataset: MDPDataset):
        return self.base_policy.build_with_dataset(dataset)

    def build_with_env(self, env: gym.Env):
        return self.base_policy.build_with_env(env)

    def copy_policy_from(self, algo: AlgoBase):
        return self.base_policy.copy_policy_from(algo)

    def copy_q_function_from(self, algo: AlgoBase):
        return self.base_policy.copy_q_function_from(algo)

    def fit(self, dataset: MDPDataset, **kwargs):
        return self.base_policy.fit(dataset, **kwargs)

    def fit_batch_online(self, env: gym.Env, **kwargs):
        return self.base_policy.fit_batch_online(env, **kwargs)

    def fit_online(self, env: gym.Env, **kwargs):
        return self.base_policy.fit_online(env, **kwargs)

    def fitter(self, env: gym.Env, **kwargs):
        return self.base_policy.fitter(env, **kwargs)

    def generate_new_data(self, transition: Transition):
        return self.base_policy.generate_new_data(transition)

    def collect(self, env: gym.Env, **kwargs):
        return self.base_policy.collect(env, **kwargs)

    def update(self, batch: TransitionMiniBatch):
        return self.base_policy.update(batch)

    def create_impl(self, observation_shape: Sequence[int], action_size: int):
        return self.base_policy.create_impl(observation_shape, action_size)

    def get_action_type(self):
        return self.base_policy.get_action_type()

    def get_params(self, **kwargs):
        return self.base_policy.get_params()

    def load_model(self, fname: str):
        return self.base_policy.load_model(fname)

    def from_json(self, fname: str, **kwargs):
        return self.base_policy.from_json(fname, **kwargs)

    def save_model(self, fname: str):
        return self.base_policy.save_model(fname)

    def save_params(self, logger: D3RLPyLogger):
        return self.base_policy.save_model(logger)

    def save_policy(self, fname: str, **kwargs):
        return self.base_policy.save_policy(fname, **kwargs)

    def set_active_logger(self, logger: D3RLPyLogger):
        return self.base_policy.set_active_logger(logger)

    def set_grad_step(self, grad_step: int):
        return self.base_policy.set_grad_step(grad_step)

    def set_params(self, **params):
        return self.base_policy.set_params(**params)

    @property
    def scaler(self):
        return self.base_policy.scaler

    @property
    def action_scalar(self):
        return self.base_policy.action_scaler

    @property
    def reward_scaler(self):
        return self.base_policy.reward_scaler

    @property
    def observation_space(self):
        return self.base_policy.observation_space

    @property
    def action_size(self):
        return self.base_policy.action_size

    @property
    def gamma(self):
        return self.base_policy.gamma

    @property
    def batch_size(self):
        return self.base_policy.batch_size

    @property
    def grad_step(self):
        return self.base_policy.grad_step

    @property
    def n_frames(self):
        return self.base_policy.n_frames

    @property
    def n_steps(self):
        return self.base_policy.n_steps

    @property
    def impl(self):
        return self.base_policy.impl

    @property
    def action_logger(self):
        return self.base_policy.action_logger


@dataclass
class OnlineHead(BaseHead):
    """Class to enable online interaction.

    Bases: :class:`scope_rl.policy.BaseHead`

    Imported as: :class:`scope_rl.policy.OnlineHead`

    Note
    -------
    This class aims to make a d3rlpy's policy an instance of :class:`BaseHead`.

    Note
    -------
    To ensure API compatibility with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, :class:`BaseHead` inherits :class:`d3rlpy.algos.AlgoBase`.
    This base class also has additional methods including :class:`fit`, :class:`predict`, and :class:`predict_value`. Please also refer to the following documentation for the methods that are not described in this API reference.

    .. seealso::

        (external) `d3rlpy's documentation about AlgoBase <https://d3rlpy.readthedocs.io/en/latest/references/algos.html>`_

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    """

    base_policy: AlgoBase
    name: str

    def sample_action_and_output_pscore(self, x: np.ndarray):
        """Only for API consistency."""
        pass

    def calc_action_choice_probability(self, x: np.ndarray):
        """Only for API consistency."""
        pass

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Only for API consistency."""
        pass


@dataclass
class EpsilonGreedyHead(BaseHead):
    """Class to convert a deterministic policy into an epsilon-greedy policy (applicable to discrete action case).

    Bases: :class:`scope_rl.policy.BaseHead`

    Imported as: :class:`scope_rl.policy.EpsilonGreedyHead`

    Note
    -------
    Epsilon-greedy policy stochastically chooses actions (i.e., :math:`a \\in \\mathcal{A}`) given state :math:`s` as follows.

    .. math::

        \\pi(a \\mid s) := (1 - \\epsilon) * \\mathbb{I}(a = a*)) + \\epsilon / |\\mathcal{A}|

    where :math:`\\epsilon` is the probability of taking random actions and :math:`a*` is the greedy action.
    :math:`\\mathbb{I}(\\cdot)` denotes the indicator function.

    Note
    -------
    To ensure API compatibility with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, :class:`BaseHead` inherits :class:`d3rlpy.algos.AlgoBase`.
    This base class also has additional methods including :class:`fit`, :class:`predict`, and :class:`predict_value`. Please also refer to the following documentation for the methods that are not described in this API reference.

    .. seealso::

        (external) `d3rlpy's documentation about AlgoBase <https://d3rlpy.readthedocs.io/en/latest/references/algos.html>`_

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    n_actions: int (> 0)
        Number of actions.

    epsilon: float
        Probability of exploration. The value should be within `[0, 1]`.

    random_state: int, default=None (>= 0)
        Random state.

    """

    base_policy: AlgoBase
    name: str
    n_actions: int
    epsilon: float
    random_state: Optional[int] = None

    def __post_init__(self):
        "Initialize class."
        self.action_type = "discrete"

        if not isinstance(self.base_policy, AlgoBase):
            raise ValueError("base_policy must be a child class of AlgoBase")

        check_scalar(self.n_actions, name="n_actions", target_type=int, min_val=2)
        self.action_matrix = np.eye(self.n_actions)

        check_scalar(
            self.epsilon, name="epsilon", target_type=float, min_val=0.0, max_val=1.0
        )

        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def sample_action_and_output_pscore(self, x: np.ndarray):
        """Sample an action stochastically based on the pscore.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, )
            Sampled action.

        pscore: ndarray of shape (n_samples, )
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        action = self.sample_action(x)
        pscore = self.calc_pscore_given_action(x, action)
        return action, pscore

    def calc_action_choice_probability(self, x: np.ndarray):
        """Calculate action choice probabilities.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        pscore: ndarray of shape (n_samples, n_actions)
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        greedy_action = self.base_policy.predict(x)
        greedy_action_matrix = self.action_matrix[greedy_action]
        uniform_matrix = np.ones_like(greedy_action_matrix, dtype=float)
        pscore = (1 - self.epsilon) * greedy_action_matrix + (
            self.epsilon / self.n_actions
        ) * uniform_matrix
        return pscore  # shape (n_samples, n_actions)

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate the pscore of a given action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        action: array-like of shape (n_samples, )
            Action.

        Return
        -------
        pscore: ndarray of shape (n_samples, )
            Pscore of the given state and action.

        """
        greedy_action = self.base_policy.predict(x)
        greedy_mask = greedy_action == action
        pscore = (1 - self.epsilon + self.epsilon / self.n_actions) * greedy_mask + (
            self.epsilon / self.n_actions
        ) * (1 - greedy_mask)
        return pscore

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, )
            Sampled action for each state.

        """
        greedy_action = self.base_policy.predict(x)
        random_action = self.random_.randint(self.n_actions, size=len(x))
        greedy_mask = self.random_.rand(len(x)) > self.epsilon
        action = greedy_action * greedy_mask + random_action * (1 - greedy_mask)
        return action


@dataclass
class SoftmaxHead(BaseHead):
    """Class to convert a Q-learning based policy into a softmax policy (applicable to discrete action space).

    Bases: :class:`scope_rl.policy.BaseHead`

    Imported as: :class:`scope_rl.policy.SoftmaxHead`

    Note
    -------
    A softmax policy stochastically chooses an action (i.e., :math:`a \\in \\mathcal{A}`) given state :math:`s` as follows.

    .. math::

        \\pi(a \\mid s) := \\frac{\\exp(Q(s, a) / \\tau)}{\\sum_{a' \\in \\mathcal{A}} \\exp(Q(s, a') / \\tau)}

    where :math:`\\tau` is the temperature parameter of the softmax function.
    :math:`Q(s, a)` is the predicted value for the given :math:`(s, a)` pair.

    Note
    -------
    To ensure API compatibility with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, :class:`BaseHead` inherits :class:`d3rlpy.algos.AlgoBase`.
    This base class also has additional methods including :class:`fit`, :class:`predict`, and :class:`predict_value`. Please also refer to the following documentation for the methods that are not described in this API reference.

    .. seealso::

        (external) `d3rlpy's documentation about AlgoBase <https://d3rlpy.readthedocs.io/en/latest/references/algos.html>`_

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    n_actions: int (> 0)
        Number of actions.

    tau: float, default=1.0
        Temperature parameter. The value should not be zero.
        A negative value leads to a sub-optimal policy.

    random_state: int, default=None (>= 0)
        Random state.

    """

    base_policy: AlgoBase
    name: str
    n_actions: int
    tau: float = 1.0
    random_state: Optional[int] = None

    def __post_init__(self):
        """Initialize class."""
        self.action_type = "discrete"

        if not isinstance(self.base_policy, AlgoBase):
            raise ValueError("base_policy must be a child class of AlgoBase")

        check_scalar(self.n_actions, name="actions", target_type=int, min_val=2)
        check_scalar(self.tau, name="tau", target_type=float)
        if self.tau == 0:
            self.tau += 1e-10

        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def _softmax(self, x: np.ndarray):
        """Softmax function."""
        x = x - np.tile(np.max(x, axis=1), (x.shape[1], 1)).T  # to avoid overflow
        return np.exp(x / self.tau) / (
            np.sum(np.exp(x / self.tau), axis=1, keepdims=True)
        )

    def _gumble_max_trick(self, x: np.ndarray):
        """Gumble max trick to sample action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, )
            Sampled action.

        """
        gumble_variable = self.random_.gumbel(size=(len(x), self.n_actions))
        return np.argmax(x / self.tau + gumble_variable, axis=1).astype(int)

    def _predict_value(self, x: np.ndarray):
        """Predict state action value for all possible actions.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        state_action_value: ndarray of shape (n_samples, n_actions)
            State action values for all observed states and possible actions.

        """
        x_ = []
        for i in range(x.shape[0]):
            x_.append(np.tile(x[i], (self.n_actions, 1)))
        x_ = np.array(x_).reshape((-1, x.shape[1]))
        a_ = np.tile(np.arange(self.n_actions), x.shape[0])
        return self.base_policy.predict_value(x_, a_).reshape(
            (-1, self.n_actions)
        )  # (n_samples, n_actions)

    def sample_action_and_output_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, )
            Sampled action.

        pscore: ndarray of shape (n_samples, )
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        predicted_value = self._predict_value(x)
        prob = self._softmax(predicted_value)
        action = self._gumble_max_trick(predicted_value)

        action_id = (
            np.array([action[i] + i * self.n_actions for i in range(len(action))])
            .flatten()
            .astype(int)
        )
        pscore = prob.flatten()[action_id]

        return action, pscore

    def calc_action_choice_probability(self, x: np.ndarray):
        """Calculate action choice probabilities.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        pscore: ndarray of shape (n_samples, n_actions)
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        predicted_value = self._predict_value(x)
        return self._softmax(predicted_value)  # (n_samples, n_actions)

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate the pscore of a given action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        action: array-like of shape (n_samples, )
            Action.

        Return
        -------
        pscore: ndarray of shape (n_samples, )
            Pscore of the given state and action.

        """
        prob = self.calc_action_choice_probability(x)
        action_id = (
            np.array([action[i] + i * self.n_actions for i in range(len(action))])
            .flatten()
            .astype(int)
        )
        return prob.flatten()[action_id]

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, )
            Sampled action for each state.

        """
        predicted_value = self._predict_value(x)
        return self._gumble_max_trick(predicted_value)


@dataclass
class GaussianHead(BaseHead):
    """Class to sample action from Gaussian distribution (applicable to continuous action case).

    Bases: :class:`scope_rl.policy.BaseHead`

    Imported as: :class:`scope_rl.policy.GaussianHead`

    Note
    -------
    This class should be used when action_space is not clipped.
    Otherwise, please use :class:`TruncatedGaussianHead` instead.

    Given a deterministic policy, a gaussian policy samples an action :math:`a \\in \\mathcal{A}` given state :math:`s` as follows.

    .. math::

        a \\sim Normal(\\pi(s), \\sigma)

    where :math:`\\sigma` is the standard deviation of the normal distribution.
    :math:`\\pi(s)` is the action chosen by the deterministic policy.

    Note
    -------
    To ensure API compatibility with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, :class:`BaseHead` inherits :class:`d3rlpy.algos.AlgoBase`.
    This base class also has additional methods including :class:`fit`, :class:`predict`, and :class:`predict_value`. Please also refer to the following documentation for the methods that are not described in this API reference.

    .. seealso::

        (external) `d3rlpy's documentation about AlgoBase <https://d3rlpy.readthedocs.io/en/latest/references/algos.html>`_

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    sigma: array-like of shape (action_dim, )
        Standard deviation of Gaussian distribution.

    random_state: int, default=None (>= 0)
        Random state.

    """

    base_policy: AlgoBase
    name: str
    sigma: np.ndarray
    random_state: Optional[int] = None

    def __post_init__(self):
        """Initialize class."""
        self.action_type = "continuous"

        if not isinstance(self.base_policy, AlgoBase):
            raise ValueError("base_policy must be a child class of AlgoBase")

        check_array(self.sigma, name="sigma", expected_dim=1, min_val=0.0)

        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def _calc_pscore(self, greedy_action: np.ndarray, action: np.ndarray):
        """Calculate pscore.

        Parameters
        -------
        greedy_action: array-like of shape (n_samples, action_dim)
            Greedy action.

        action: array-like of shape (n_samples, action_dim)
            Sampled Action.

        Return
        -------
        pscore: ndarray of shape (n_samples, )
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        prob = norm.pdf(
            action,
            loc=greedy_action,
            scale=self.sigma,
        )
        return np.prod(prob, axis=1)

    def sample_action_and_output_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, action_dim)
            Sampled action.

        pscore: ndarray of shape (n_samples, )
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        greedy_action = self.base_policy.predict(x)
        action = self.sample_action(x)
        pscore = self._calc_pscore(greedy_action, action)
        return action, pscore

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate the pscore of a given action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        action: array-like of shape (n_samples, action_dim)
            Action.

        Return
        -------
        pscore: ndarray of shape (n_samples, )
            Pscore of the given state and action.

        """
        greedy_action = self.base_policy.predict(x)
        return self._calc_pscore(greedy_action, action)

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, action_dim)
            Sampled action for each state.

        """
        greedy_action = self.base_policy.predict(x)
        action = norm.rvs(
            loc=greedy_action,
            scale=self.sigma,
        ).reshape(greedy_action.shape)
        return action


@dataclass
class TruncatedGaussianHead(BaseHead):
    """Class to sample continuous actions from Truncated Gaussian distribution (applicable to continuous action space).

    Bases: :class:`scope_rl.policy.BaseHead`

    Imported as: :class:`scope_rl.policy.TruncatedGaussianHead`

    Note
    -------
    Given a deterministic policy, a truncated gaussian policy samples an action :math:`a \\in \\mathcal{A}` given state :math:`s` as follows.

    .. math::

        a \\sim TruncNorm(\\pi(s), \\sigma)

    where :math:`\\sigma` is the standard deviation of the truncated normal distribution.
    :math:`\\pi(s)` is the action chosen by the deterministic policy.

    Note
    -------
    To ensure API compatibility with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, :class:`BaseHead` inherits :class:`d3rlpy.algos.AlgoBase`.
    This base class also has additional methods including :class:`fit`, :class:`predict`, and :class:`predict_value`. Please also refer to the following documentation for the methods that are not described in this API reference.

    .. seealso::

        (external) `d3rlpy's documentation about AlgoBase <https://d3rlpy.readthedocs.io/en/latest/references/algos.html>`_

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    sigma: array-like of shape (action_dim, )
        Standard deviation of Gaussian distribution.

    minimum: array-like of shape (action_dim, )
        Minimum value of action vector.

    maximum: array-like of shape (action_dim, )
        Maximum value of action vector.

    random_state: int, default=None (>= 0)
        Random state.

    """

    base_policy: AlgoBase
    name: str
    sigma: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    random_state: Optional[int] = None

    def __post_init__(self):
        """Initialize class."""
        self.action_type = "continuous"

        if not isinstance(self.base_policy, AlgoBase):
            raise ValueError("base_policy must be a child class of AlgoBase")

        check_array(self.sigma, name="sigma", expected_dim=1, min_val=0.0)
        check_array(self.minimum, name="minimum", expected_dim=1)
        check_array(self.maximum, name="maximum", expected_dim=1)

        if np.any(self.minimum >= self.maximum):
            raise ValueError("minimum must be smaller than maximum")
        self.uniform_pscore = 1 / (self.maximum - self.minimum)

        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def _calc_pscore(self, greedy_action: np.ndarray, action: np.ndarray):
        """Calculate pscore.

        Parameters
        -------
        greedy_action: array-like of shape (n_samples, action_dim)
            Greedy action.

        action: array-like of shape (n_samples, action_dim)
            Sampled Action.

        Return
        -------
        pscore: ndarray of shape (n_samples, )
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        prob = truncnorm.pdf(
            action,
            a=(self.minimum - greedy_action) / self.sigma,
            b=(self.maximum - greedy_action) / self.sigma,
            loc=greedy_action,
            scale=self.sigma,
        )
        return np.prod(prob, axis=1)

    def sample_action_and_output_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, action_dim)
            Sampled action.

        pscore: ndarray of shape (n_samples, )
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        greedy_action = self.base_policy.predict(x)
        action = self.sample_action(x)
        pscore = self._calc_pscore(greedy_action, action)
        return action, pscore

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate the pscore of a given action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        action: array-like of shape (n_samples, action_dim)
            Action.

        Return
        -------
        pscore: ndarray of shape (n_samples, )
            Pscore of the given state and action.

        """
        greedy_action = self.base_policy.predict(x)
        return self._calc_pscore(greedy_action, action)

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, action_dim)
            Sampled action for each state.

        """
        greedy_action = self.base_policy.predict(x)
        action = truncnorm.rvs(
            a=(self.minimum - greedy_action) / self.sigma,
            b=(self.maximum - greedy_action) / self.sigma,
            loc=greedy_action,
            scale=self.sigma,
        ).reshape(greedy_action.shape)
        return action


@dataclass
class ContinuousEvalHead(BaseHead):
    """Class to transform the base policy into a deterministic evaluation policy.

    Bases: :class:`scope_rl.policy.BaseHead`

    Imported as: :class:`scope_rl.policy.ContinuousEvalHead`

    Note
    -------
    To ensure API compatibility with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, :class:`BaseHead` inherits :class:`d3rlpy.algos.AlgoBase`.
    This base class also has additional methods including :class:`fit`, :class:`predict`, and :class:`predict_value`. Please also refer to the following documentation for the methods that are not described in this API reference.

    .. seealso::

        (external) `d3rlpy's documentation about AlgoBase <https://d3rlpy.readthedocs.io/en/latest/references/algos.html>`_

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    random_state: int, default=None (>= 0)
        Random state. (This is for API consistency.)

    """

    base_policy: AlgoBase
    name: str
    random_state: Optional[int] = None

    def __post_init__(self):
        """Initialize class."""
        self.action_type = "continuous"
        if not isinstance(self.base_policy, AlgoBase):
            raise ValueError("base_policy must be a child class of AlgoBase")

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: array-like of shape (n_samples, state_dim)
            State (we will follow the implementation of d3rlpy and thus use 'x' rather than 's').

        Return
        -------
        action: ndarray of shape (n_samples, action_dim)
            Sampled action for each state.

        """
        return self.base_policy.predict(x)  # greedy-action
