"""Minimax value function learning (continuous action cases)."""
from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm
from pathlib import Path

import torch
from torch import optim

import numpy as np
from sklearn.utils import check_scalar

from d3rlpy.preprocessing import Scaler, ActionScaler

from .base import BaseWeightValueLearner
from .function import (
    ContinuousQFunction,
    VFunction,
)
from ...utils import check_array, gaussian_kernel


@dataclass
class ContinuousMinimaxStateActionValueLearning(BaseWeightValueLearner):
    """Minimax Q Learning for marginal OPE estimators (for continuous action space).

    Bases: :class:`ofrl.ope.weight_value_learning.BaseWeightValueLearner`

    Imported as: :class:`ofrl.ope.weight_value_learning.ContinuousMinimaxStateActionValueLearning`

    Note
    -------
    Minimax Q Learning uses that the following holds true about Q-function.

    .. math::

        \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \\sim d^{\\pi_0}} [w(s_t, a_t) (r_t + \\gamma Q(s_{t+1}, \\pi(s_{t+1})))]
        = \\mathbb{E}_{(s_t, a_t) \\sim d^{\\pi_0}} [Q(s_t, a_t)]

    where :math:`Q(s_t, a_t)` is the Q-function, :math:`w(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_0}(s_t, a_t)` is the state-action marginal importance weight.

    Then, it adversarially minimize the difference between RHS and LHS (which we denote :math:`L_Q(w, Q)`) to the worst case in terms of :math:`w(\\cdot)`
    using a discriminator defined in reproducing kernel Hilbert space (RKHS) as follows.

    .. math::

        \\max_Q L_Q^2(w, Q) = \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{r}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_0}}[
            (r_t + \\gamma Q(s_{t+1}, \\pi(s_{t+1})) - Q(s_t, a_t)) K((s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t)) (\\tilde{r}_t + \\gamma Q(\\tilde{s}_{t+1}, \\pi(\\tilde{s}_{t+1})) - Q(\\tilde{s}_t, \\tilde{a}_t))
        ]

    where :math:`K(\\cdot, \\cdot)` is a kernel function.

    Parameters
    -------
    q_function: ContinuousQFunction
        Q function model.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    sigma: float, default=1.0 (> 0)
        Bandwidth hyperparameter of gaussian kernel.

    state_scaler: d3rlpy.preprocessing.Scaler, default=None
        Scaling factor of state.

    action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
        Scaling factor of action.

    batch_size: int, default=32 (> 0)
        Batch size.

    alpha: float, default=1e-3 (> 0)
        Regularization of Q-function.

    lambda_: float, default=1e-3 (>= 0)
        Stabilizer to obtain a fast convergence rate.

    lr: float, default=1e-3 (> 0)
        Learning rate.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    """

    q_function: ContinuousQFunction
    gamma: float = 1.0
    sigma: float = 1.0
    state_scaler: Optional[Scaler] = None
    action_scaler: Optional[ActionScaler] = None
    batch_size: int = 32
    alpha: float = 1e-3
    lambda_: float = 1e-3
    lr: float = 1e-3
    device: str = "cuda:0"

    def __post_init__(self):
        self.q_function.to(self.device)

        check_scalar(
            self.gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0
        )
        check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)
        if self.state_scaler is not None and not isinstance(self.state_scaler, Scaler):
            raise ValueError(
                "state_scaler must be an instance of d3rlpy.preprocessing.Scaler, but found False"
            )
        if self.action_scaler is not None and not isinstance(
            self.action_scaler, ActionScaler
        ):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        check_scalar(self.batch_size, name="batch_size", target_type=int, min_val=1)
        check_scalar(self.alpha, name="alpha", target_type=float, min_val=0.0)
        check_scalar(self.lambda_, name="lambda_", target_type=float, min_val=0.0)
        check_scalar(self.lr, name="lr", target_type=float, min_val=0.0)

    def load(self, path: Path):
        self.q_function.load_state_dict(torch.load(path))

    def save(self, path: Path):
        torch.save(self.q_function.state_dict(), path)

    def _inverse(
        self,
        symmetric_matrix: torch.Tensor,
    ):
        """Calculate inverse of a symmetric matrix."""
        return torch.linalg.pinv(symmetric_matrix, hermitian=True)

    def _sqrt(
        self,
        symmetric_matrix: torch.Tensor,
    ):
        """Calculate sqrt of a symmetric matrix."""
        v, w = torch.linalg.eigh(symmetric_matrix)
        return w @ torch.diag_embed(torch.sqrt(v))

    def _gaussian_kernel(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        """Gaussian kernel for all input pairs."""
        with torch.no_grad():
            input = torch.cat((state, action), dim=1)
            # (x - x') ** 2 = x ** 2 + x' ** 2 - 2 x x'
            x_2 = (input**2).sum(dim=1)
            x_y = input @ input.T
            distance = 2 * (x_2[:, None] - x_y)
            kernel = torch.exp(-distance / self.sigma)

        return kernel  # shape (n_trajectories, n_trajectories)

    def _kernel_term(
        self,
        kernel: torch.Tensor,
    ):
        """Kernel term in the objective function."""
        n_trajectories = len(kernel)
        sqrt_kernel = self._sqrt(kernel)
        inverse_kernel = self._inverse(
            self.alpha * torch.eye(n_trajectories) + self.lambda_ * kernel
        )
        return (
            sqrt_kernel @ inverse_kernel @ sqrt_kernel
        )  # shape (n_trajectories, n_trajectories)

    def _objective_function(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        """Objective function of Minimax Q Learning.

        Parameters
        -------
        state: array-like of shape (n_trajectories, state_dim)
            State observed by the behavior policy.

        action: Tensor of shape (n_trajectories, action_dim)
            Action chosen by the behavior policy.

        reward: Tensor of shape (n_trajectories, )
            Reward observed for each (state, action) pair.

        next_state: Tensor of shape (n_trajectories, state_dim)
            Next state observed for each (state, action) pair.

        next_action: Tensor of shape (n_trajectories, action_dim)
            Next action chosen by the evaluation policy.

        importance_weight: Tensor of shape (n_trajectories, )
            Immediate importance weight of the given (state, action) pair,
            i.e., :math:`\\pi(a_t | s_t) / \\pi_0(a_t | s_t)`.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of MQL.

        """
        kernel = self._gaussian_kernel(state, action)
        current_q = self.q_function(state, action)
        next_q = self.q_function(next_state, next_action)

        td_error = importance_weight * (reward + self.gamma * next_q) - current_q
        return td_error.T @ self._kernel_term(kernel) @ td_error

    def fit(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit Q-function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory)
            Reward observed for each (state, action) pair.

        pscore: array-like of shape (n_trajectories * step_per_trajectory)
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        n_epochs: int, default=100 (> 0)
            Number of epochs to train.

        n_steps_per_epoch: int, default=100 (> 0)
            Number of gradient steps in a epoch.

        random_state: int, default=None (>= 0)
            Random state.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=2)
        check_array(reward, name="reward", expected_dim=1)
        check_array(pscore, name="pscore", expected_dim=2, min_val=0.0, max_val=1.0)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if not (
            state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == reward.shape[0] == == pscore.shape[0] == evaluation_policy_action.shape[0]`, but found False"
            )
        if state.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state.shape[0] % step_per_trajectory == 0`, but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
            )

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        state_dim = state.shape[1]
        action_dim = action.shape[1]
        state = state.reshape((-1, step_per_trajectory, state_dim))
        action = action.reshape((-1, step_per_trajectory, action_dim))
        reward = reward.reshape((-1, step_per_trajectory))
        pscore = pscore.prod(axis=1).reshape((-1, step_per_trajectory))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_trajectory, action_dim)
        )

        if self.action_scaler is not None:
            evaluation_policy_action_ = self.action_scaler.transform_numpy(
                evaluation_policy_action.reshape((-1, action_dim))
            )
            action_ = self.action_scaler.transform_numpy(
                action.reshape((-1, action_dim))
            )
        else:
            evaluation_policy_action_ = evaluation_policy_action
            action_ = action

        similarity_weight = gaussian_kernel(
            evaluation_policy_action_,
            action_,
            sigma=self.sigma,
        ).reshape((-1, step_per_trajectory))

        n_trajectories, step_per_trajectory, _ = state.shape
        state = torch.FloatTensor(state, device=self.device)
        action = torch.LongTensor(action, device=self.device)
        reward = torch.FloatTensor(reward, device=self.device)
        evaluation_policy_action = torch.FloatTensor(
            evaluation_policy_action, device=self.device
        )
        importance_weight = torch.FloatTensor(similarity_weight / pscore)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)
        if self.action_scaler is not None:
            action = self.action_scaler.transform(action)

        optimizer = optim.SGD(self.q_function.parameters(), lr=self.lr, momentum=0.9)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc="[fitting_value_function]",
            total=n_epochs,
        ):
            for grad_step in range(n_steps_per_epoch):
                idx_ = torch.randint(n_trajectories, size=(self.batch_size,))
                t_ = torch.randint(step_per_trajectory - 1, size=(self.batch_size,))

                objective_loss = self._objective_function(
                    state=state[idx_, t_],
                    action=action[idx_, t_],
                    reward=reward[idx_, t_],
                    next_state=state[idx_, t_ + 1],
                    next_action=evaluation_policy_action[idx_, t_ + 1],
                    importance_weight=importance_weight[idx_, t_],
                )

                optimizer.zero_grad()
                objective_loss.backward()
                optimizer.step()

    def predict_q_function(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict Q function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior/evaluation policy.

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory)
            Q value of each (state, action) pair.

        """
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=2)
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0]`, but found False"
            )

        state = torch.FloatTensor(state, device=self.device)
        action = torch.FloatTensor(action, device=self.device)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)
        if self.action_scaler is not None:
            action = self.action_scaler.transform(action)

        with torch.no_grad():
            q_value = self.q_function(state, action).to("cpu").detach().numpy()

        return q_value

    def predict_v_function(
        self,
        state: np.ndarray,
        evaluation_policy_action: np.ndarray,
    ):
        """Predict V function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        Return
        -------
        v_function: ndarray of shape (n_trajectories * step_per_trajectory)
            State value.

        """
        return self.predict_q_function(state, evaluation_policy_action)

    def predict_value(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory)
            Q value of each (state, action) pair.

        """
        return self.predict_q_function(state, action)

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory)
            Q value of each (state, action) pair.

        """
        return self.predict_q_function(state, action)

    def fit_predict(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit and predict Q-function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory)
            Reward observed for each (state, action) pair.

        pscore: array-like of shape (n_trajectories * step_per_trajectory)
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        n_epochs: int, default=100 (> 0)
            Number of epochs to train.

        n_steps_per_epoch: int, default=100 (> 0)
            Number of gradient steps in a epoch.

        random_state: int, default=None (>= 0)
            Random state.

        """
        self.fit(
            step_per_trajectory=step_per_trajectory,
            state=state,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            random_state=random_state,
        )
        return self.predict_value(state, action)


@dataclass
class ContinuousMinimaxStateValueLearning(BaseWeightValueLearner):
    """Minimax V Learning for marginal OPE estimators (for continuous action space).

    Bases: :class:`ofrl.ope.weight_value_learning.BaseWeightValueLearner`

    Imported as: :class:`ofrl.ope.weight_value_learning.ContinuousMinimaxStateValueLearning`

    Note
    -------
    Minimax V Learning uses that the following holds true about V-function.

    .. math::

        \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \\sim d^{\\pi_0}} [w(s_t, a_t) (r_t + \\gamma V(s_{t+1}))]
        = \\mathbb{E}_{s_t \\sim d^{\\pi_0}} [V(s_t)]

    where :math:`V(s_t)` is the Q-function, :math:`w(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_0}(s_t, a_t)` is the state-action marginal importance weight.

    Then, it adversarially minimize the difference between RHS and LHS (which we denote :math:`L_V(w, V)`) to the worst case in terms of :math:`w(\\cdot)`
    using a discriminator defined in reproducing kernel Hilbert space (RKHS) as follows.

    .. math::

        \\max_V L_V^2(w, V) = \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{r}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_0}}[
            (r_t + \\gamma V(s_{t+1}) - V(s_t)) K(s_t, \\tilde{s_t}) (\\tilde{r}_t + \\gamma V(\\tilde{s}_{t+1}) - V(\\tilde{s}_t)
        ]

    where :math:`K(\\cdot, \\cdot)` is a kernel function.

    Parameters
    -------
    v_function: DiscreteQFunction
        V function model.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    sigma: float, default=1.0 (> 0)
        Bandwidth hyperparameter of gaussian kernel.

    state_scaler: d3rlpy.preprocessing.Scaler, default=None
        Scaling factor of state.

    action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
        Scaling factor of action.

    batch_size: int, default=32 (> 0)
        Batch size.

    alpha: float, default=1e-3 (> 0)
        Regularization of V-function.

    lambda_: float, default=1e-3 (>= 0)
        Stabilizer to obtain a fast convergence rate.

    lr: float, default=1e-3 (> 0)
        Learning rate.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    """

    v_function: VFunction
    gamma: float = 1.0
    sigma: float = 1.0
    state_scaler: Optional[Scaler] = None
    action_scaler: Optional[ActionScaler] = None
    batch_size: int = 32
    alpha: float = 1e-3
    lambda_: float = 1e-3
    lr: float = 1e-3
    device: str = "cuda:0"

    def __post_init__(self):
        self.v_function.to(self.device)

        check_scalar(
            self.gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0
        )
        check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)
        if self.state_scaler is not None and not isinstance(self.state_scaler, Scaler):
            raise ValueError(
                "state_scaler must be an instance of d3rlpy.preprocessing.Scaler, but found False"
            )
        if self.action_scaler is not None and not isinstance(
            self.action_scaler, ActionScaler
        ):
            raise ValueError(
                "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
            )

        check_scalar(self.batch_size, name="batch_size", target_type=int, min_val=1)
        check_scalar(self.alpha, name="alpha", target_type=float, min_val=0.0)
        check_scalar(self.lambda_, name="lambda_", target_type=float, min_val=0.0)
        check_scalar(self.lr, name="lr", target_type=float, min_val=0.0)

    def load(self, path: Path):
        self.v_function.load_state_dict(torch.load(path))

    def save(self, path: Path):
        torch.save(self.v_function.state_dict(), path)

    def _inverse(
        self,
        symmetric_matrix: torch.Tensor,
    ):
        """Calculate inverse of a symmetric matrix."""
        return torch.linalg.pinv(symmetric_matrix, hermitian=True)

    def _sqrt(
        self,
        symmetric_matrix: torch.Tensor,
    ):
        """Calculate sqrt of a symmetric matrix."""
        v, w = torch.linalg.eigh(symmetric_matrix)
        return w @ torch.diag_embed(torch.sqrt(v))

    def _gaussian_kernel(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        """Gaussian kernel for all input pairs."""
        with torch.no_grad():
            input = torch.cat((state, action), dim=1)
            # (x - x') ** 2 = x ** 2 + x' ** 2 - 2 x x'
            x_2 = (input**2).sum(dim=1)
            x_y = input @ input.T
            distance = x_2[:, None] + x_2[None, :] - 2 * x_y
            kernel = torch.exp(-distance / self.sigma)

        return kernel  # shape (n_trajectories, n_trajectories)

    def _kernel_term(self, kernel: torch.Tensor):
        """Kernel term in the objective function."""
        n_trajectories = len(kernel)
        sqrt_kernel = self._sqrt(kernel)
        inverse_kernel = self._inverse(
            self.alpha * torch.eye(n_trajectories) + self.lambda_ * kernel
        )
        return (
            sqrt_kernel @ inverse_kernel @ sqrt_kernel
        )  # shape (n_trajectories, n_trajectories)

    def _objective_function(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        """Objective function of Minimax V Learning.

        Parameters
        -------
        state: array-like of shape (n_trajectories, state_dim)
            State observed by the behavior policy.

        action: Tensor of shape (n_trajectories, action_dim)
            Action chosen by the behavior policy.

        reward: Tensor of shape (n_trajectories, )
            Reward observed for each (state, action) pair.

        next_state: Tensor of shape (n_trajectories, state_dim)
            Next state observed for each (state, action) pair.

        importance_weight: Tensor of shape (n_trajectories, )
            Immediate importance weight of the given (state, action) pair,
            i.e., :math:`\\pi(a_t | s_t) / \\pi_0(a_t | s_t)`.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of MVL.

        """
        kernel = self._gaussian_kernel(state, action)
        current_v = self.v_function(state)
        next_v = self.v_function(next_state)

        td_error = importance_weight * (reward + self.gamma * next_v) - current_v
        return td_error.T @ self._kernel_term(kernel) @ td_error

    def fit(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit Q-function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory)
            Reward observed for each (state, action) pair.

        pscore: array-like of shape (n_trajectories * step_per_trajectory)
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        n_epochs: int, default=100 (> 0)
            Number of epochs to train.

        n_steps_per_epoch: int, default=100 (> 0)
            Number of gradient steps in a epoch.

        random_state: int, default=None (>= 0)
            Random state.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=2)
        check_array(reward, name="reward", expected_dim=1)
        check_array(pscore, name="pscore", expected_dim=2, min_val=0.0, max_val=1.0)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
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
        if state.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state.shape[0] % step_per_trajectory == 0`, but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
            )

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        state_dim = state.shape[1]
        action_dim = action.shape[1]
        if self.action_scaler is not None:
            evaluation_policy_action_ = self.action_scaler.transform_numpy(
                evaluation_policy_action.reshape((-1, action_dim))
            )
            action_ = self.action_scaler.transform_numpy(
                action.reshape((-1, action_dim))
            )
        else:
            evaluation_policy_action_ = evaluation_policy_action
            action_ = action

        similarity_weight = gaussian_kernel(
            evaluation_policy_action_,
            action_,
            sigma=self.sigma,
        ).reshape((-1, step_per_trajectory))

        state = state.reshape((-1, step_per_trajectory, state_dim))
        action = action.reshape((-1, step_per_trajectory, action_dim))
        reward = reward.reshape((-1, step_per_trajectory))
        pscore = pscore.prod(axis=1).reshape((-1, step_per_trajectory))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_trajectory, action_dim)
        )

        n_trajectories, step_per_trajectory, _ = state.shape
        state = torch.FloatTensor(state, device=self.device)
        action = torch.FloatTensor(action, device=self.device)
        reward = torch.FloatTensor(reward, device=self.device)
        importance_weight = torch.FloatTensor(similarity_weight / pscore)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)
        if self.action_scaler is not None:
            action = self.action_scaler.transform(action)

        optimizer = optim.SGD(self.v_function.parameters(), lr=self.lr, momentum=0.9)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc="[fitting_value_function]",
            total=n_epochs,
        ):
            for grad_step in range(n_steps_per_epoch):
                idx_ = torch.randint(n_trajectories, size=(self.batch_size,))
                t_ = torch.randint(step_per_trajectory - 1, size=(self.batch_size,))

                objective_loss = self._objective_function(
                    state=state[idx_, t_],
                    action=action[idx_, t_],
                    reward=reward[idx_, t_],
                    next_state=state[idx_, t_ + 1],
                    importance_weight=importance_weight[idx_, t_],
                )

                optimizer.zero_grad()
                objective_loss.backward()
                optimizer.step()

    def predict_v_function(
        self,
        state: np.ndarray,
    ):
        """Predict V function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        Return
        -------
        v_function: ndarray of shape (n_trajectories * step_per_trajectory)
            State value.

        """
        check_array(state, name="state", expected_dim=2)
        state = torch.FloatTensor(state, device=self.device)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        with torch.no_grad():
            v_value = self.v_function(state).to("cpu").detach().numpy()

        return v_value

    def predict_value(
        self,
        state: np.ndarray,
    ):
        """Predict V function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        Return
        -------
        v_function: ndarray of shape (n_trajectories * step_per_trajectory)
            State value.

        """
        return self.predict_v_function(state)

    def predict(
        self,
        state: np.ndarray,
    ):
        """Predict V function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        Return
        -------
        v_function: ndarray of shape (n_trajectories * step_per_trajectory)
            State value.

        """
        return self.predict_v_function(state)

    def fit_predict(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit and predict Q-function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory, )
            Reward observed for each (state, action) pair.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        n_epochs: int, default=100 (> 0)
            Number of epochs to train.

        n_steps_per_epoch: int, default=100 (> 0)
            Number of gradient steps in a epoch.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None (>= 0)
            Random state.

        """
        self.fit(
            step_per_trajectory=step_per_trajectory,
            state=state,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            random_state=random_state,
        )
        return self.predict_value(state)
