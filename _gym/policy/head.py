"""Wrapper class to convert greedy policy into stochastic."""
from abc import abstractmethod
from typing import Union, Optional
from dataclasses import dataclass

import numpy as np
from scipy.stats import truncnorm
from sklearn.utils import check_scalar, check_random_state
from d3rlpy.algos import AlgoBase


@dataclass
class BaseHead(AlgoBase):
    """Base class to convert greedy policy into stochastic."""

    @abstractmethod
    def stochastic_action_with_pscore(self, x: np.ndarray):
        """Sample stochastic action with its pscore."""
        raise NotImplementedError()

    @abstractmethod
    def calculate_action_choice_probability(self, x: np.ndarray):
        """Calcullate action choice probabilities."""
        raise NotImplementedError()

    @abstractmethod
    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
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

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self.base_policy, key)
        except:
            raise AttributeError()


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

    def calculate_action_choice_probability(self, x: np.ndarray):
        pass

    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
        pass


@dataclass
class DiscreteEpsilonGreedyHead(BaseHead):
    """Class to convert greedy policy into e-greedy.

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

        check_scalar(self.n_actions, name="actions", target_type=int, min_val=2)
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
        pscore = self.calculate_pscore_given_action(x, action)
        return action, pscore

    def calculate_action_choice_probability(self, x: np.ndarray):
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

    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
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
    """Class to convert policy values into softmax policy.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    n_actions: int (> 0)
        Numbers of actions.

    tau: float, default=1.0 (>= 0)
        Temperature parameter.

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
        check_scalar(self.tau, name="tau", target_type=float, min_val=0.0)
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

    def _predict_counterfactual_state_action_value(self, x: np.ndarray):
        """Predict counterfactual state action value.

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
        return self.base_policy.predict_value(x_, a_)  # (n_samples, n_actions)

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

    def calculate_action_choice_probability(self, x: np.ndarray):
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

    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
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
        prob = self.calculate_pscore(x)
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
class ContinuousEpsilonGreedyHead(BaseHead):
    """Class to convert greedy policy into e-greedy.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    epsilon: float [0, 1]
        Probability of exploration.

    minimum: NDArray (action_dim, )
        Minimum value of action vector.

    maximum: NDArray (action_dim, )
        Maximum value of action vector.

    random_state: Optional[int], default=None (>= 0)
        Random state.

    """

    base_policy: AlgoBase
    name: str
    epsilon: float
    minimum: np.ndarray
    maximum: np.ndarray
    random_state: Optional[int] = None

    def __post_init__(self):
        """Initialize class."""
        self.action_type = "continuous"

        if not isinstance(self.base_policy, AlgoBase):
            raise ValueError("base_policy must be a child class of AlgoBase")

        check_scalar(
            self.epsilon, name="epsilon", target_type=float, min_val=0.0, max_val=1.0
        )

        check_scalar(self.minimum, name="minimum", target_type=float)
        check_scalar(self.maximum, name="maximum", target_type=float)
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be smaller than maximum")
        self.uniform_pscore = 1 / (self.maximum - self.minimum)

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
        action: NDArray, shape (n_samples, action_dim)
            Sampled action.

        pscore: NDArray, shape (n_samples, )
            Pscore of the sampled action.

        """
        action = self.sample_action(x)
        pscore = self.calculate_pscore_given_action(x, action)
        return action, pscore

    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
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
        greedy_mask = greedy_action == action
        pscore = (
            1 - self.epsilon + self.epsilon * self.uniform_pscore
        ) * greedy_mask + (self.epsilon * self.uniform_pscore) * (1 - greedy_mask)
        return pscore

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
        random_action = self.random_.uniform(
            low=self.minimum, high=self.maximum, size=greedy_action.shape
        )
        greedy_mask = self.random_.rand(len(x)) > self.epsilon
        greedy_mask = np.tile(greedy_mask, (greedy_action.shape[1], 1)).T
        action = greedy_action * greedy_mask + random_action * (1 - greedy_mask)
        return action


@dataclass
class ContinuousTruncatedGaussianHead(BaseHead):
    """Class to sample action from Gaussian distribution.

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

        check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)

        check_scalar(self.minimum, name="minimum", target_type=float)
        check_scalar(self.maximum, name="maximum", target_type=float)
        if self.minimum >= self.maximum:
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

    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
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
class ContinuousMixtureHead(BaseHead):
    """Class to sample action using both Gaussian distribution and e-greedy.

    Parameters
    -------
    base_policy: AlgoBase
        Reinforcement learning (RL) policy.

    name: str
        Name of the policy.

    epsilon: float [0, 1]
        Probability of exploration.

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
    epsilon: float
    sigma: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    random_state: Optional[int] = None

    def __post_init__(self):
        """Initialize dataset."""
        self.action_type = "continuous"

        if not isinstance(self.base_policy, AlgoBase):
            raise ValueError("base_policy must be a child class of AlgoBase")

        check_scalar(
            self.epsilon, name="epsilon", target_type=float, min_val=0.0, max_val=1.0
        )
        check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)

        check_scalar(self.minimum, name="minimum", target_type=float)
        check_scalar(self.maximum, name="maximum", target_type=float)
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be smaller than maximum")
        self.uniform_pscore = 1 / (self.maximum - self.minimum)

        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def truncnorm_pscore(self, greedy_action: np.ndarray, action: np.ndarray):
        """Calculate pscore under Gaussian distribution.

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
        action = self.sample_action(x)
        pscore = self.calculate_pscore_given_action(x, action)
        return action, pscore

    def calculate_pscore_given_action(self, x: np.ndarray, action: np.ndarray):
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
        pscore = (1 - self.epsilon) * self.truncnorm_pscore(
            greedy_action, action
        ) + self.epsilon * self.uniform_pscore
        return pscore

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
        gaussian_action = truncnorm.rvs(
            a=(self.minimum - greedy_action) / self.sigma,
            b=(self.maximum - greedy_action) / self.sigma,
            loc=greedy_action,
            scale=self.sigma,
        ).reshape((-1, 1))
        random_action = self.random_.uniform(
            low=self.minimum, high=self.maximum, size=greedy_action.shape
        )
        greedy_mask = self.random_.rand(len(x)) > self.epsilon
        greedy_mask = np.tile(greedy_mask, (greedy_action.shape[1], 1)).T
        action = gaussian_action * greedy_mask + random_action * (1 - greedy_mask)
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
