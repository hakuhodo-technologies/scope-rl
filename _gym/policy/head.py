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

from _gym.utils import check_array


@dataclass
class BaseHead(AlgoBase):
    """Base class to convert greedy policy into stochastic."""

    @abstractmethod
    def stochastic_action_with_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore."""
        raise NotImplementedError()

    @abstractmethod
    def calc_action_choice_probability(self, x: np.ndarray):
        """Calcullate action choice probabilities."""
        raise NotImplementedError()

    @abstractmethod
    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate pscore given action."""
        raise NotImplementedError()

    def predict_online(self, x: np.ndarray):
        return self.predict(x.reshape((1, -1)))[0]

    def predict_value_online(
        self,
        x: np.ndarray,
        action: Union[np.ndarray, int, float],
        with_std: bool = False,
    ):
        if isinstance(action, (int, float)):
            action = np.array([[action]])
        else:
            action.reshape((1, -1))
        return self.predict_value(x.reshape((1, -1)), action, with_std=with_std)[0]

    def sample_action_online(self, x: np.ndarray):
        return self.sample_action(x.reshape(1, -1))[0]

    def stochastic_action_with_pscore_online(self, x: np.ndarray):
        action, pscore = self.stochastic_action_with_pscore(x.reshape(1, -1))
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
    """Class to enable OpenAI Gym like online interaction.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    """

    base_policy: AlgoBase

    def stochastic_action_with_pscore(self, x: np.ndarray):
        pass

    def calc_action_choice_probability(self, x: np.ndarray):
        pass

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        pass


@dataclass
class DiscreteEpsilonGreedyHead(BaseHead):
    """Class to convert a deterministic policy into an epsilon-greedy policy.

    Note
    -------
    Epsilon-greedy policy stochastically chooses actions (i.e., :math:`a \\in \\mathcal{A}`) given state :math:`s` as follows.

    .. math::
        \\pi(a \\mid s) := (1 - \\epsilon) * \\mathbb{I}(a = a*)) + \\epsilon / |\\mathcal{A}|

    where :math:`\\epsilon` is the probability of taking random actions and :math:`a*` is the greedy action.
    :math:`\\mathbb{I}(\\cdot)` denotes indicator function.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    n_actions: int (> 0)
        Numbers of actions.

    epsilon: float [0, 1]
        Probability of exploration.

    random_state: Optional[int], default=None (>= 0)
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

    def stochastic_action_with_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, )
            Sampled action.

        pscore: NDArray, shape (n_samples, )
            Pscore of the sampled action.

        """
        action = self.sample_action(x)
        pscore = self.calc_pscore_given_action(x, action)
        return action, pscore

    def calc_action_choice_probability(self, x: np.ndarray):
        """Calcullate action choice probabilities.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        pscore: NDArray, shape (n_samples, n_actions)
            Pscore of the sample action.

        """
        greedy_action = self.base_policy.predict(x)
        greedy_action_matrix = self.action_matrix[greedy_action]
        uniform_matrix = np.ones_like(greedy_action_matrix, dtype=float)
        pscore = (1 - self.epsilon) * greedy_action_matrix + (
            self.epsilon / self.n_actions
        ) * uniform_matrix
        return pscore  # shape (n_samples, n_actions)

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate pscore given action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        action: NDArray, shape (n_samples, )
            Action.

        Return
        -------
        pscore: NDArray, shape (n_samples, )
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
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, )
            Sampled action for each state.

        """
        greedy_action = self.base_policy.predict(x)
        random_action = self.random_.randint(self.n_actions, size=len(x))
        greedy_mask = self.random_.rand(len(x)) > self.epsilon
        action = greedy_action * greedy_mask + random_action * (1 - greedy_mask)
        return action


@dataclass
class DiscreteSoftmaxHead(BaseHead):
    """Class to convert a Q-learning based policy into a softmax policy.

    Note
    -------
    Softmax policy stochastically chooses actions (i.e., :math:`a \\in \\mathcal{A}`) given state :math:`s` as follows.

    .. math::
        \\pi(a \\mid s) := \\frac{\\exp(Q(s, a) / \\tau)}{\\sum_{a' \\in A} \\exp(Q(s, a') / \\tau)}

    where :math:`\\tau` is the temperature parameter of the softmax function.
    :math:`Q(s, a)` is the predicted value for the given :math:`(s, a)` pair.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    n_actions: int (> 0)
        Numbers of actions.

    tau: float, default=1.0 (:math:`\\in (- \\infty, \\infty)`)
        Temperature parameter.
        A negative value leads to a sub-optimal policy.

    random_state: Optional[int], default=None (>= 0)
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
        return np.exp(x / self.tau) / np.sum(
            np.exp(x / self.tau), axis=1, keepdims=True
        )

    def _gumble_max_trick(self, x: np.ndarray):
        """Gumble max trick to sample action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, )
            Sampled action.

        """
        gumble_variable = -np.log(-np.log(self.random_.rand(len(x), self.n_actions)))
        return np.argmax(x / self.tau + gumble_variable, axis=1)

    def _predict_value(self, x: np.ndarray):
        """Predict state action value for all possible actions.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        counterfactual_state_action_value: NDArray, shape (n_samples, n_actions)
            State action values for all observed state and possible action.

        """
        x_ = []
        for i in range(x.shape[0]):
            x_.append(np.tile(x[i], (self.n_actions, 1)))
        x_ = np.array(x_).reshape((-1, x.shape[1]))
        a_ = np.tile(np.arange(self.n_actions), x.shape[0])
        return self.base_policy.predict_value(x_, a_).reshape(
            (-1, self.n_actions)
        )  # (n_samples, n_actions)

    def stochastic_action_with_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, )
            Sampled action.

        pscore: NDArray, shape (n_samples, )
            Pscore of the sampled action.

        """
        predicted_value = self._predict_value(x)
        prob = self._softmax(predicted_value)
        action = self._gumble_max_trick(predicted_value)

        action_id = np.array(
            [action[i] + i * self.n_actions for i in range(len(action))]
        ).flatten()
        pscore = prob.flatten()[action_id]

        return action, pscore

    def calc_action_choice_probability(self, x: np.ndarray):
        """Calcullate action choice probabilities.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        pscore: NDArray, shape (n_samples, n_actions)
            Pscore of the sample action.

        """
        predicted_value = self._predict_counterfactual_state_action_value(x)
        return self._softmax(predicted_value)  # (n_samples, n_actions)

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate pscore given action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        action: NDArray, shape (n_samples, )
            Action.

        Return
        -------
        pscore: NDArray, shape (n_samples, )
            Pscore of the given state and action.

        """
        prob = self.calc_pscore(x)
        actions_id = np.array(
            [action[i] + i * self.n_actions for i in range(len(action))]
        ).flatten()
        return prob.flatten()[actions_id]

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, )
            Sampled action for each state.

        """
        predicted_value = self._predict_value(x)
        return self._gumble_max_trick(predicted_value)


@dataclass
class ContinuousGaussianHead(BaseHead):
    """Class to sample action from Gaussian distribution.

    Note
    -------
    This class should be used when action_space is not clipped.
    Otherwise, please use ContinuousTruncatedGaussianHead instead.

    Given a deterministic policy, gaussian policy samples action :math:`a \\in \\mathcal{A}` given state :math:`s` as follows.

    .. math::
        a \\sim Normal(\\pi(s), \\sigma)

    where :math:`\\sigma` is the standard deviation of the normal distribution.
    :math:`\\pi(s)` is the action chosen by the deterministic policy.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    sigma: NDArray, shape (action_dim, )
        Standard deviation of Gaussian distribution.

    random_state: Optional[int], default=None (>= 0)
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
        greedy_action: NDArray, (n_samples, action_dim)
            Greedy action.

        action: NDArray, (n_samples, action_dim)
            Sampled Action.

        Return
        -------
        pscore: NDArray, (n_samples, )
            Pscore of the sampled action.

        """
        prob = norm.pdf(
            action,
            loc=greedy_action,
            scale=self.sigma,
        )
        return np.prod(prob, axis=1)

    def stochastic_action_with_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, action_dim)
            Sampled action.

        pscore: NDArray, shape (n_samples, )
            Pscore of the sampled action.

        """
        greedy_action = self.base_policy.predict(x)
        action = self.sample_action(x)
        pscore = self._calc_pscore(greedy_action, action)
        return action, pscore

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate pscore given action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        action: NDArray, shape (n_samples, action_dim)
            Action.

        Return
        -------
        pscore: NDArray, shape (n_samples, )
            Pscore of the given state and action.

        """
        greedy_action = self.base_policy.predict(x)
        return self._calc_pscore(greedy_action, action)

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, action_dim)
            Sampled action for each state.

        """
        greedy_action = self.base_policy.predict(x)
        action = norm.rvs(
            loc=greedy_action,
            scale=self.sigma,
        ).reshape((-1, 1))
        return action


@dataclass
class ContinuousTruncatedGaussianHead(BaseHead):
    """Class to sample action from Truncated Gaussian distribution.

    Note
    -------
    Given a deterministic policy, truncated gaussian policy samples action :math:`a \\in \\mathcal{A}` given state :math:`s` as follows.

    .. math::
        a \\sim TruncNorm(\\pi(s), \\sigma)

    where :math:`\\sigma` is the standard deviation of the truncated normal distribution.
    :math:`\\pi(s)` is the action chosen by the deterministic policy.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    sigma: NDArray, shape (action_dim, )
        Standard deviation of Gaussian distribution.

    minimum: NDArray, shape (action_dim, )
        Minimum value of action vector.

    maximum: NDArray, shape (action_dim, )
        Maximum value of action vector.

    random_state: Optional[int], default=None (>= 0)
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
        greedy_action: NDArray, (n_samples, action_dim)
            Greedy action.

        action: NDArray, (n_samples, action_dim)
            Sampled Action.

        Return
        -------
        pscore: NDArray, (n_samples, )
            Pscore of the sampled action.

        """
        prob = truncnorm.pdf(
            action,
            a=(self.minimum - greedy_action) / self.sigma,
            b=(self.maximum - greedy_action) / self.sigma,
            loc=greedy_action,
            scale=self.sigma,
        )
        return np.prod(prob, axis=1)

    def stochastic_action_with_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, action_dim)
            Sampled action.

        pscore: NDArray, shape (n_samples, )
            Pscore of the sampled action.

        """
        greedy_action = self.base_policy.predict(x)
        action = self.sample_action(x)
        pscore = self._calc_pscore(greedy_action, action)
        return action, pscore

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        """Calculate pscore given action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        action: NDArray, shape (n_samples, action_dim)
            Action.

        Return
        -------
        pscore: NDArray, shape (n_samples, )
            Pscore of the given state and action.

        """
        greedy_action = self.base_policy.predict(x)
        return self._calc_pscore(greedy_action, action)

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, action_dim)
            Sampled action for each state.

        """
        greedy_action = self.base_policy.predict(x)
        action = truncnorm.rvs(
            a=(self.minimum - greedy_action) / self.sigma,
            b=(self.maximum - greedy_action) / self.sigma,
            loc=greedy_action,
            scale=self.sigma,
        ).reshape((-1, 1))
        return action


@dataclass
class ContinuousEvalHead(BaseHead):
    """Class to transform into a deterministic evaluation policy.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    """

    base_policy: AlgoBase
    name: str

    def __post_init__(self):
        """Initialize class."""
        self.action_type = "continuous"
        if not isinstance(self.base_policy, AlgoBase):
            raise ValueError("base_policy must be a child class of AlgoBase")

    def sample_action(self, x: np.ndarray):
        """Sample action.

        Parameters
        -------
        x: NDArray, shape (n_samples, state_dim)
            State.

        Return
        -------
        action: NDArray, shape (n_samples, action_dim)
            Sampled action for each state.

        """
        return self.base_policy.predict(x)  # greedy-action
