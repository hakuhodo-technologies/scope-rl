from dataclasses import dataclass
from typing import Optional, Union
from tqdm import tqdm

import torch
from torch import optim
from torch.nn import functional as F

import numpy as np
from sklearn.utils import check_scalar

from .base import BaseWeightValueLearner
from .function import (
    ContinuousStateActionWeightFunction,
    StateWeightFunction,
)
from ...utils import check_array, gaussian_kernel


@dataclass
class ContinuousMinimaxStateActionWeightLearning(BaseWeightValueLearner):
    """Minimax Weight Learning for marginal OPE estimators (for continuous action space).

    Note
    -------
    Minimax Weight Learning uses that the following holds true about Q-function.

    .. math::

        \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \\sim d^{\\pi_0}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1})} [w(s_t, a_t) (Q(s_t, a_t) - \\gamma Q(s_{t+1}, a_{t+1}))]
        = \\mathbb{E}_{s_0 \\sim d^{\\pi_0}, a_0 \\sim \\pi(a_0 | s_0)} [Q(s_0, a_0)]

    where :math:`Q(s_t, a_t)` is the Q-function, :math:`w(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_0}(s_t, a_t)` is the state-action marginal importance weight.

    Then, it adversarially minimize the difference between RHS and LHS (which we denote :math:`L_w(w, Q)`) to the worst case in terms of :math:`Q(\\cdot)`
    using a discriminator defined in reproducing kernel Hilbert space (RKHS) as follows.

    .. math::

        \\max_w L_w^2(w, Q) = \\mathbb{E}_{(s_t, a_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_0}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1}), \\tilde{a}_{t+1} \\sim \\pi(\\tilde{a}_{t+1} | \\tilde{s}_{t+1})}[
            w(s_t, a_t) w(\\tilde{s}_t, \\tilde{a}_t) ( K((s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t)) + K((s_{t+1}, a_{t+1}), (\\tilde{s}_{t+1}, \\tilde{a}_{t+1})) - \\gamma ( K((s_t, a_t), (\\tilde{s}_{t+1}, \\tilde{a}_{t+1})) + K((s_{t+1}, a_{t+1}), (\\tilde{s}_t, \\tilde{a}_t)) ))
        ] + \\gamma (1 - \\gamma) \\mathbb{E}_{(s_t, a_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_0}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1}), \\tilde{a}_{t+1} \\sim \\pi(\\tilde{a}_{t+1} | \\tilde{s}_{t+1}), s_0 \\sim d(s_0), \\tilde{s}_0 \\sim d(\\tilde{s}_0), a_0 \\sim \\pi(a_0 | s_0), \\tilde{a}_0 \\sim \\pi(\\tilde{a}_0 | \\tilde{s}_0)}[
            w(s_t, a_t) K((s_{t+1}, a_{t+1}), (\\tilde{s}_0, \\tilde{a}_0)) + w(\\tilde{s}_t, \\tilde{a}_t) K((\\tilde{s}_{t+1}, \\tilde{a}_{t+1}), (s_0, a_0))
        ] - (1 - \\gamma) \\mathbb{E}_{(s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t) \\sim d^{\\pi_0}, s_0 \\sim d(s_0), \\tilde{s}_0 \\sim d(\\tilde{s}_0), a_0 \\sim \\pi(a_0 | s_0), \\tilde{a}_0 \\sim \\pi(\\tilde{a}_0 | \\tilde{s}_0)}[
            w(s_t, a_t) K((s_t, a_t), (\\tilde{s}_0, \\tilde{a}_0)) + w(\\tilde{s}_t, \\tilde{a}_t) K((\\tilde{s}_t, \\tilde{a}_t), (s_0, a_0))
        ]

    where :math:`K(\\cdot, \\cdot)` is a kernel function.

    Parameters
    -------
    w_function: ContinuousStateActionWeightFunction
        Weight function model.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    """

    w_function: ContinuousStateActionWeightFunction
    device: str = "cuda:0"

    def __post_init__(self):
        self.w_function.to(self.device)

    def _gaussian_kernel(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        """Gaussian kernel for all input pairs."""
        with torch.no_grad():
            input = torch.cat((state, action / action_scaler[None, :]), dim=1)
            # (x - x') ** 2 = x ** 2 + x' ** 2 - 2 x x'
            x_2 = (input ** 2).sum(dim=1)
            x_y = input @ input.T
            distance = x_2[:, None] + x_2[None, :] - 2 * x_y
            kernel = torch.exp(-distance / sigma)

        return kernel  # shape (n_episodes, n_episodes)

    def _first_term(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        importance_weight = importance_weight @ importance_weight.T
        positive_term = self._gaussian_kernel(
            state,
            action,
            state,
            action,
            sigma=sigma,
            action_scaler=action_scaler,
        ) + self._gaussian_kernel(
            next_state,
            next_action,
            next_state,
            next_action,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        negative_term = self._gaussian_kernel(
            state,
            action,
            next_state,
            next_action,
            sigma=sigma,
            action_scaler=action_scaler,
        ) + self._gaussian_kernel(
            next_state,
            next_action,
            state,
            action,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        return (importance_weight * (positive_term - gamma * negative_term)).mean()

    def _second_term(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        base_term = importance_weight[:, None] * self._gaussian_kernel(
            next_state,
            next_action,
            initial_state,
            initial_action,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        return gamma * (1 - gamma) * (base_term @ base_term.T).mean()

    def _third_term(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        base_term = importance_weight[:, None] * self._gaussian_kernel(
            state,
            action,
            initial_state,
            initial_action,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        return gamma * (1 - gamma) * (base_term @ base_term.T).mean()

    def _objective_function(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        gamma: float,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        """Objective function of Minimax Weight Learning.

        Parameters
        -------
        initial_state: Tensor of shape (n_episodes, state_dim)
            Initial state of a trajectory (or states sampled from a stationary distribution).

        initial_action: Tensor of shape (n_episodes, action_dim)
            Initial action chosen by the evaluation policy.

        state: array-like of shape (n_episodes, state_dim)
            State observed by the behavior policy.

        action: Tensor of shape (n_episodes, action_dim)
            Action chosen by the behavior policy.

        next_state: Tensor of shape (n_episodes, state_dim)
            Next state observed for each (state, action) pair.

        next_action: Tensor of shape (n_episodes, action_dim)
            Next action chosen by the evaluation policy.

        gamma: float
            Discount factor. The value should be within `(0, 1]`.

        sigma: float
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: Tensor of shape (action_dim, )
            Scaling factor of action.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of MWL.

        """
        importance_weight = self.w_function(state, action)

        first_term = self._first_term(
            state=state,
            action=action,
            next_state=next_state,
            next_action=next_action,
            importance_weight=importance_weight,
            gamma=gamma,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        second_term = self._second_term(
            initial_state=initial_state,
            initial_action=initial_action,
            next_state=next_state,
            next_action=next_action,
            importance_weight=importance_weight,
            gamma=gamma,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        third_term = self._third_term(
            initial_state=initial_state,
            initial_action=initial_action,
            state=state,
            action=action,
            importance_weight=importance_weight,
            gamma=gamma,
            sigma=sigma,
            action_scaler=action_scaler,
        )

        return first_term + second_term - third_term

    def fit(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        gamma: float = 1.0,
        sigma: float = 1.0,
        lr: float = 1e-3,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Fit weight function.

        Parameters
        -------
        state: array-like of shape (n_episodes, step_per_episode, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes, step_per_episode)
            Reward observed for each (state, action) pair.

        evaluation_policy_action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        n_epochs: int, default=100
            Number of epochs to train.

        n_steps_per_epoch: int, default=100
            Number of gradient steps in a epoch.

        batch_size: int, default=32
            Batch size.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        lr: float, default=1e-3
            Learning rate.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None
            Random state.

        """
        check_array(state, name="state", expected_dim=3)
        check_array(action, name="action", expected_dim=3)
        check_array(reward, name="reward", expected_dim=2)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=3,
        )
        if not (
            state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == reward.shape[0] == evaluation_policy_next_action_dist.shape[0]`, but found False"
            )
        if not (
            state.shape[1]
            == action.shape[1]
            == reward.shape[1]
            == evaluation_policy_action.shape[1]
        ):
            raise ValueError(
                "Expected `state.shape[1] == action.shape[1] == reward.shape[1] == evaluation_policy_next_action_dist.shape[1]`, but found False"
            )
        if action.shape[2] != evaluation_policy_action.shape[2]:
            raise ValueError(
                "Expected `action.shape[2] == evaluation_policy_action.shape[2]`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)
        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )
        check_scalar(batch_size, name="batch_size", target_type=int, min_val=1)
        check_scalar(lr, name="lr", target_type=float, min_val=0.0)

        action_dim = action.shape[2]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[2]`, but found False"
            )

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        n_episodes, step_per_episode, state_dim = state.shape
        state = torch.FloatTensor(state, device=self.device)
        action = torch.FloatTensor(action, device=self.device)
        reward = torch.FloatTensor(reward, device=self.device)
        evaluation_policy_action = torch.FloatTensor(
            evaluation_policy_action, device=self.device
        )

        optimizer = optim.SGD(self.w_function.parameters(), lr=lr, momentum=0.9)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc=["fitting_weight_and_value_functions"],
            total=n_epochs,
        ):
            for grad_step in range(n_steps_per_epoch):
                idx_ = torch.randint(n_episodes, size=(batch_size,))
                t_ = torch.randint(step_per_episode, size=(batch_size,))

                objective_loss = self._objective_function(
                    initial_state=state[idx_, 0],
                    initial_action=action[idx_, 0],
                    state=state[idx_, t_],
                    action=action[idx_, t_],
                    reward=reward[idx_, t_],
                    next_state=state[idx_, t_ + 1],
                    next_action=action[idx_, t_ + 1],
                    gamma=gamma,
                    sigma=sigma,
                    action_scaler=action_scaler,
                )

                optimizer.zero_grad()
                objective_loss.backward()
                optimizer.step()

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict function.

        Parameters
        -------
        state: array-like of shape (n_episodes, step_per_episode, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the behavior policy.

        Return
        -------
        importance_weight: ndarray of shape (n_episodes, step_per_episode)
            Estimated state-action marginal importance weight.

        """
        check_array(state, name="state", expected_dim=3)
        check_array(action, name="action", expected_dim=3)
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0]`, but found False"
            )
        if state.shape[1] != action.shape[1]:
            raise ValueError(
                "Expected `state.shape[1] == action.shape[1]`, but found False"
            )

        n_episodes, step_per_episode, state_dim = state.shape
        action_dim = action.shape[2]
        state = torch.FloatTensor(state.reshape((-1, state_dim)), device=self.device)
        action = torch.FloatTensor(action.reshape((-1, action_dim)), device=self.device)

        with torch.no_grad():
            importance_weight = (
                self.w_function(state, action).to("cpu").detach().numpy()
            )

        return importance_weight.reshape((n_episodes, step_per_episode))

    def fit_predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        gamma: float = 1.0,
        sigma: float = 1.0,
        lr: float = 1e-3,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Fit and predict weight function.

        Parameters
        -------
        state: array-like of shape (n_episodes, step_per_episode, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes, step_per_episode)
            Reward observed for each (state, action) pair.

        evaluation_policy_action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Next action chosen by the evaluation policy.

        n_epochs: int, default=100
            Number of epochs to train.

        n_steps_per_epoch: int, default=100
            Number of gradient steps in a epoch.

        batch_size: int, default=32
            Batch size.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        lr: float, default=1e-3
            Learning rate.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None
            Random state.

        Return
        -------
        importance_weight: ndarray of shape (n_episodes, )
            Estimated state-action marginal importance weight.

        """
        self.fit(
            state=state,
            action=action,
            reward=reward,
            evaluation_policy_action=evaluation_policy_action,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size,
            gamma=gamma,
            sigma=sigma,
            lr=lr,
            action_scaler=action_scaler,
            random_state=random_state,
        )
        return self.predict(state=state, action=action)


@dataclass
class ContinuousMinimaxStateWeightLearning(BaseWeightValueLearner):
    """Minimax Weight Learning for marginal OPE estimators (for continuous action space).

    Note
    -------
    Minimax Weight Learning uses that the following holds true about Q-function.

    .. math::

        \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \\sim d^{\\pi_0}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1})} [w(s_t, a_t) (Q(s_t, a_t) - \\gamma Q(s_{t+1}, a_{t+1}))]
        = \\mathbb{E}_{s_0 \\sim d^{\\pi_0}, a_0 \\sim \\pi(a_0 | s_0)} [Q(s_0, a_0)]

    where :math:`Q(s_t, a_t)` is the Q-function, :math:`w(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_0}(s_t, a_t) = d^{\\pi}(s_t) \\pi(a_t | s_t) / d^{\\pi_0}(s_t) \\pi_0(a_t | s_t)`
    is the state-action marginal importance weight.

    Then, it adversarially minimize the difference between RHS and LHS (which we denote :math:`L_w(w, Q)`) to the worst case in terms of :math:`Q(\\cdot)`
    using a discriminator defined in reproducing kernel Hilbert space (RKHS) as follows.

    .. math::

        \\max_w L_w^2(w, Q) = \\mathbb{E}_{(s_t, a_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_0}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1}), \\tilde{a}_{t+1} \\sim \\pi(\\tilde{a}_{t+1} | \\tilde{s}_{t+1})}[
            w_s(s_t) w_a(s_t, a_t) w_s(\\tilde{s}_t) w_a(\\tilde{s}_t, \\tilde{a}_t) ( K((s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t)) + K((s_{t+1}, a_{t+1}), (\\tilde{s}_{t+1}, \\tilde{a}_{t+1})) - \\gamma ( K((s_t, a_t), (\\tilde{s}_{t+1}, \\tilde{a}_{t+1})) + K((s_{t+1}, a_{t+1}), (\\tilde{s}_t, \\tilde{a}_t)) ))
        ] + \\gamma (1 - \\gamma) \\mathbb{E}_{(s_t, a_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_0}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1}), \\tilde{a}_{t+1} \\sim \\pi(\\tilde{a}_{t+1} | \\tilde{s}_{t+1}), s_0 \\sim d(s_0), \\tilde{s}_0 \\sim d(\\tilde{s}_0), a_0 \\sim \\pi(a_0 | s_0), \\tilde{a}_0 \\sim \\pi(\\tilde{a}_0 | \\tilde{s}_0)}[
            w_s(s_t) w_a(s_t, a_t) K((s_{t+1}, a_{t+1}), (\\tilde{s}_0, \\tilde{a}_0)) + w_s(\\tilde{s}_t) w_a(\\tilde{s}_t, \\tilde{a}_t) K((\\tilde{s}_{t+1}, \\tilde{a}_{t+1}), (s_0, a_0))
        ] - (1 - \\gamma) \\mathbb{E}_{(s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t) \\sim d^{\\pi_0}, s_0 \\sim d(s_0), \\tilde{s}_0 \\sim d(\\tilde{s}_0), a_0 \\sim \\pi(a_0 | s_0), \\tilde{a}_0 \\sim \\pi(\\tilde{a}_0 | \\tilde{s}_0)}[
            w_s(s_t) w_a(s_t, a_t) K((s_t, a_t), (\\tilde{s}_0, \\tilde{a}_0)) + w_s(\\tilde{s}_t) w_a(\\tilde{s}_t, \\tilde{a}_t) K((\\tilde{s}_t, \\tilde{a}_t), (s_0, a_0))
        ]

    where :math:`K(\\cdot, \\cdot)` is a kernel function, :math:`w_s(s_t) \\approx d^{\\pi}(s_t) / d^{\\pi_0}(s_t)` is the state-marginal importance weight,
    and :math:`w_a(s_t, a_t) := \\pi(a_t | s_t) / \\pi_0(a_t | s_t)` is the immediate importance weight.

    Parameters
    -------
    w_function: StateWeightFunction
        Weight function model.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    """

    w_function: StateWeightFunction
    device: str = "cuda:0"

    def __post_init__(self):
        self.w_function.to(self.device)

    def _gaussian_kernel(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        """Gaussian kernel for all input pairs."""
        with torch.no_grad():
            input = torch.cat((state, action / action_scaler[None, :]), dim=1)
            # (x - x') ** 2 = x ** 2 + x' ** 2 - 2 x x'
            x_2 = (input ** 2).sum(dim=1)
            x_y = input @ input.T
            distance = x_2[:, None] + x_2[None, :] - 2 * x_y
            kernel = torch.exp(-distance / sigma)

        return kernel  # shape (n_episodes, n_episodes)

    def _first_term(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        importance_weight = importance_weight @ importance_weight.T
        positive_term = self._gaussian_kernel(
            state,
            action,
            state,
            action,
            sigma=sigma,
            action_scaler=action_scaler,
        ) + self._gaussian_kernel(
            next_state,
            next_action,
            next_state,
            next_action,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        negative_term = self._gaussian_kernel(
            state,
            action,
            next_state,
            next_action,
            sigma=sigma,
            action_scaler=action_scaler,
        ) + self._gaussian_kernel(
            next_state,
            next_action,
            state,
            action,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        return (importance_weight * (positive_term - gamma * negative_term)).mean()

    def _second_term(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        base_term = importance_weight[:, None] * self._gaussian_kernel(
            next_state,
            next_action,
            initial_state,
            initial_action,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        return gamma * (1 - gamma) * (base_term @ base_term.T).mean()

    def _third_term(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        base_term = importance_weight[:, None] * self._gaussian_kernel(
            state,
            action,
            initial_state,
            initial_action,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        return gamma * (1 - gamma) * (base_term @ base_term.T).mean()

    def _objective_function(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        action_scaler: torch.Tensor,
    ):
        """Objective function of Minimax Weight Learning.

        Parameters
        -------
        initial_state: Tensor of shape (n_episodes, state_dim)
            Initial state of a trajectory (or states sampled from a stationary distribution).

        initial_action: Tensor of shape (n_episodes, )
            Initial action chosen by the evaluation policy.

        state: array-like of shape (n_episodes, state_dim)
            State observed by the behavior policy.

        action: Tensor of shape (n_episodes, )
            Action chosen by the behavior policy.

        next_state: Tensor of shape (n_episodes, state_dim)
            Next state observed for each (state, action) pair.

        next_action: Tensor of shape (n_episodes, )
            Next action chosen by the evaluation policy.

        importance_weight: Tensor of shape (n_episodes, )
            Immediate importance weight of the given (state, action) pair,
            i.e., :math:`\\pi(a_t | s_t) / \\pi_0(a_t | s_t)`.

        gamma: float
            Discount factor. The value should be within `(0, 1]`.

        sigma: float
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: Tensor of shape (action_dim, )
            Scaling factor of action.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of MWL.

        """
        importance_weight = self.w_function(state) * importance_weight

        first_term = self._first_term(
            state=state,
            action=action,
            next_state=next_state,
            next_action=next_action,
            importance_weight=importance_weight,
            gamma=gamma,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        second_term = self._second_term(
            initial_state=initial_state,
            initial_action=initial_action,
            next_state=next_state,
            next_action=next_action,
            importance_weight=importance_weight,
            gamma=gamma,
            sigma=sigma,
            action_scaler=action_scaler,
        )
        third_term = self._third_term(
            initial_state=initial_state,
            initial_action=initial_action,
            state=state,
            action=action,
            importance_weight=importance_weight,
            gamma=gamma,
            sigma=sigma,
            action_scaler=action_scaler,
        )

        return first_term + second_term - third_term

    def fit(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        gamma: float = 1.0,
        sigma: float = 1.0,
        lr: float = 1e-3,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Fit weight function.

        Parameters
        -------
        state: array-like of shape (n_episodes, step_per_episode, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes, step_per_episode)
            Reward observed for each (state, action) pair.

        pscore: array-like of shape (n_episodes, step_per_episode)
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        n_epochs: int, default=100
            Number of epochs to train.

        n_steps_per_epoch: int, default=100
            Number of gradient steps in a epoch.

        batch_size: int, default=32
            Batch size.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        lr: float, default=1e-3
            Learning rate.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None
            Random state.

        """
        check_array(state, name="state", expected_dim=3)
        check_array(action, name="action", expected_dim=3)
        check_array(reward, name="reward", expected_dim=2)
        check_array(pscore, name="pscore", expected_dim=2, min_val=0.0, max_val=1.0)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=3,
        )
        if not (
            state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action.shape[0]`, but found False"
            )
        if not (
            state.shape[1]
            == action.shape[1]
            == reward.shape[1]
            == pscore.shape[1]
            == evaluation_policy_action.shape[1]
        ):
            raise ValueError(
                "Expected `state.shape[1] == action.shape[1] == reward.shape[1] == pscore.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if action.shape[2] != evaluation_policy_action.shape[2]:
            raise ValueError(
                "Expected `action.shape[2] == evaluation_policy_action_dist.shape[2]`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)
        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )
        check_scalar(batch_size, name="batch_size", target_type=int, min_val=1)
        check_scalar(lr, name="lr", target_type=float, min_val=0.0)

        action_dim = action.shape[2]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[2]`, but found False"
            )

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        n_episodes, step_per_episode, state_dim = state.shape
        state = torch.FloatTensor(state, device=self.device)
        action = torch.FloatTensor(action, device=self.device)
        reward = torch.FloatTensor(reward, device=self.device)

        similarity_weight = gaussian_kernel(
            evaluation_policy_action.reshape((-1, action_dim)) / action_scaler[None, :],
            action.reshape((-1, action_dim)) / action_scaler[None, :],
            sigma=sigma,
        ).reshape((n_episodes, step_per_episode))
        importance_weight = torch.FloatTensor(similarity_weight / pscore)

        evaluation_policy_next_action = torch.FloatTensor(
            evaluation_policy_next_action, device=self.device
        )

        optimizer = optim.SGD(self.w_function.parameters(), lr=lr, momentum=0.9)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc=["fitting_weight_and_value_functions"],
            total=n_epochs,
        ):
            for grad_step in range(n_steps_per_epoch):
                idx_ = torch.randint(n_episodes, size=(batch_size,))
                t_ = torch.randint(step_per_episode - 1, size=(batch_size,))

                objective_loss = self._objective_function(
                    initial_state=state[idx_, 0],
                    initial_action=action[idx_, 0],
                    state=state[idx_, t_],
                    action=action[idx_, t_],
                    reward=reward[idx_, t_],
                    next_state=state[idx_, t_ + 1],
                    next_action=evaluation_policy_action[idx_, t_ + 1],
                    importance_weight=importance_weight[idx_, t_],
                    gamma=gamma,
                    sigma=sigma,
                    action_scaler=action_scaler,
                )

                optimizer.zero_grad()
                objective_loss.backward()
                optimizer.step()

    def predict_state_marginal_importance_weight(
        self,
        state: np.ndarray,
    ):
        """Predict state marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_episodes, step_per_episode, state_dim)
            State observed by the behavior policy.

        Return
        -------
        importance_weight: ndarray of shape (n_episodes, step_per_episode)
            Estimated state marginal importance weight.

        """
        check_array(state, name="state", expected_dim=3)
        n_episodes, step_per_episode, state_dim = state.shape
        state = torch.FloatTensor(state.reshape((-1, state_dim)), device=self.device)

        with torch.no_grad():
            importance_weight = self.w_function(state).to("cpu").detach().numpy()

        return importance_weight.reshape((n_episodes, step_per_episode))

    def predict_state_action_marginal_importance_weight(
        self,
        state: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        sigma: float,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
    ):
        """Predict state-action marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_episodes, step_per_episode, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the behavior policy.

        pscore: array-like of shape (n_episodes, step_per_episode)
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        sigma: float, default=1.0
            Bandwidth hyperparameter of gaussian kernel.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        Return
        -------
        importance_weight: ndarray of shape (n_episodes, step_per_episode)
            Estimated state-action marginal importance weight.

        """
        check_array(state, name="state", expected_dim=3)
        check_array(action, name="action", expected_dim=3)
        check_array(pscore, name="pscore", expected_dim=2, min_val=0.0, max_val=1.0)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=3,
        )
        if not (
            state.shape[0]
            == action.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == pscore.shape[0] == evaluation_policy_action.shape[0]`, but found False"
            )
        if not (
            state.shape[1]
            == action.shape[1]
            == pscore.shape[1]
            == evaluation_policy_action.shape[1]
        ):
            raise ValueError(
                "Expected `state.shape[1] == action.shape[1] == pscore.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )
        if action.shape[2] != evaluation_policy_action.shape[2]:
            raise ValueError(
                "Expected `action.shape[2] == evaluation_policy_action.shape[2]`, but found False"
            )

        n_episodes, step_per_episode, action_dim = action.shape
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[2]`, but found False"
            )

        similarity_weight = gaussian_kernel(
            evaluation_policy_action.reshape((-1, action_dim)) / action_scaler[None, :],
            action.reshape((-1, action_dim)) / action_scaler[None, :],
            sigma=sigma,
        ).reshape((n_episodes, step_per_episode))

        state_marginal_importance_weight = (
            self.predict_state_marginal_importance_weight(state)
        )

        return state_marginal_importance_weight * similarity_weight / pscore

    def predict(
        self,
        state: np.ndarray,
    ):
        """Predict state marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_episodes, step_per_episode, state_dim)
            State observed by the behavior policy.

        Return
        -------
        importance_weight: ndarray of shape (n_episodes, step_per_episode)
            Estimated state marginal importance weight.

        """
        return self.predict_state_marginal_importance_weight(state)

    def fit_predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        gamma: float = 1.0,
        sigma: float = 1.0,
        lr: float = 1e-3,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Fit and predict weight function.

        Parameters
        -------
        state: array-like of shape (n_episodes, step_per_episode, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes, step_per_episode)
            Reward observed for each (state, action) pair.

        pscore: array-like of shape (n_episodes, step_per_episode)
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_episodes, step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        n_epochs: int, default=100
            Number of epochs to train.

        n_steps_per_epoch: int, default=100
            Number of gradient steps in a epoch.

        batch_size: int, default=32
            Batch size.

        lr: float, default=1e-3
            Learning rate.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None
            Random state.

        Return
        -------
        importance_weight: ndarray of shape (n_episodes, step_per_episode)
            Estimated state-action marginal importance weight.

        """
        self.fit(
            state=state,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size,
            gamma=gamma,
            sigma=sigma,
            lr=lr,
            action_scaler=action_scaler,
            random_state=random_state,
        )
        return self.predict(state=state)
