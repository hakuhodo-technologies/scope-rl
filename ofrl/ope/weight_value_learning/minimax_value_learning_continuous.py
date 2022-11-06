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
    ContinuousQFunction,
    VFunction,
)
from ...utils import check_array, gaussian_kernel


@dataclass
class DiscreteMinimaxStateActionValueLearning(BaseWeightValueLearner):
    """Minimax Q Learning for marginal OPE estimators (for continuous action space).

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

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    """

    q_function: ContinuousQFunction
    device: str = "cuda:0"

    def __post_init__(self):
        self.q_function.to(self.device)

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

        return kernel  # shape (n_samples, n_samples)

    def _kernel_term(
        self,
        kernel: torch.Tensor,
        alpha: float,
        lambda_: float,
    ):
        """Kernel term in the objective function."""
        n_samples = len(kernel)
        sqrt_kernel = self._sqrt(kernel)
        inverse_kernel = self._inverse(alpha * torch.eye(n_samples) + lambda_ * kernel)
        return (
            sqrt_kernel @ inverse_kernel @ sqrt_kernel
        )  # shape (n_samples, n_samples)

    def _objective_function(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        alpha: float,
        lambda_: float,
        action_scaler: torch.Tensor,
    ):
        """Objective function of Minimax Q Learning.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        action: Tensor of shape (n_samples, action_dim)
            Action chosen by the behavior policy.

        reward: Tensor of shape (n_samples, )
            Reward observed for each (state, action) pair.

        next_state: Tensor of shape (n_samples, state_dim)
            Next state observed for each (state, action) pair.

        next_action: Tensor of shape (n_samples, action_dim)
            Next action chosen by the evaluation policy.

        importance_weight: Tensor of shape (n_samples, )
            Immediate importance weight of the given (state, action) pair,
            i.e., :math:`\\pi(a_t | s_t) / \\pi_0(a_t | s_t)`.

        gamma: float
            Discount factor. The value should be within `(0, 1]`.

        sigma: float
            Bandwidth hyperparameter of gaussian kernel.

        alpha: float

        lambda_: float

        action_scaler: Tensor of shape (action_dim, )
            Scaling factor of action.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of MQL.

        """
        kernel = self._gaussian_kernel(state, action, sigma=sigma)
        current_q = self.q_function(state, action)
        next_q = self.q_function(next_state, next_action)

        td_error = importance_weight * (reward + gamma * next_q) - current_q
        return (
            td_error.T
            @ self._kernel_term(
                kernel, alpha=alpha, lambda_=lambda_, action_scaler=action_scaler
            )
            @ td_error
        )

    def fit(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        evaluation_policy_next_action: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        gamma: float = 1.0,
        sigma: float = 1.0,
        alpha: float = 1e-3,
        lambda_: float = 1e-3,
        lr: float = 1e-3,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Fit Q-function.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_samples, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_samples, )
            Reward observed for each (state, action) pair.

        next_state: array-like of shape (n_samples, state_dim)
            Next state observed for each (state, action) pair.

        evaluation_policy_next_action: array-like of shape (n_samples, action_dim)
            Next action chosen by the evaluation policy.

        pscore: array-like of shape (n_samples, )
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_samples, action_dim)
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

        alpha: float, default=1e-3

        lambda_: float, default=1e-3

        lr: float, default=1e-3
            Learning rate.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None
            Random state.

        """
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=2)
        check_array(reward, name="reward", expected_dim=1)
        check_array(next_state, name="next_state", expected_dim=2)
        check_array(
            evaluation_policy_next_action_dist,
            name="evaluation_policy_next_action",
            expected_dim=2,
        )
        check_array(pscore, name="pscore", expected_dim=1, min_val=0.0, max_val=1.0)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if not (
            state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == next_state.shape[0]
            == evaluation_policy_next_action.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == reward.shape[0] == next_state.shape[0] == evaluation_policy_next_action.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0]`, but found False"
            )
        if state.shape[1] != next_state.shape[1]:
            raise ValueError(
                "Expected `state.shape[1] == next_state.shape[1]`, but found False"
            )
        if not (
            action.shape[1]
            == evaluation_policy_next_action.shape[1]
            == evaluation_policy_action.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_next_action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)
        check_scalar(alpha, name="alpha", min_val=0.0)
        check_scalar(lambda_, name="lambda_", min_val=0.0)
        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )
        check_scalar(batch_size, name="batch_size", target_type=int, min_val=1)
        check_scalar(lr, name="lr", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        n_samples = len(state)
        state = torch.FloatTensor(state, device=self.device)
        action = torch.LongTensor(action, device=self.device)
        reward = torch.FloatTensor(reward, device=self.device)
        next_state = torch.FloatTensor(next_state, device=self.device)
        evaluation_policy_next_action_dist = torch.FloatTensor(
            evaluation_policy_next_action_dist, device=self.device
        )

        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action_scaler[None, :],
            action / action_scaler[None, :],
            sigma=sigma,
        )
        importance_weight = torch.FloatTensor(similarity_weight / pscore)

        optimizer = optim.SGD(self.q_function.parameters(), lr=lr, momentum=0.9)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc=["fitting_weight_and_value_functions"],
            total=n_epochs,
        ):
            for grad_step in range(n_steps_per_epoch):
                idx_ = torch.randint(n_samples, size=(batch_size,))
                objective_loss = self._objective_function(
                    state=state[idx_],
                    action=action[idx_],
                    reward=reward[idx_],
                    next_state=next_state[idx_],
                    next_action=evaluation_policy_next_action[idx_],
                    importance_weight=importance_weight[idx_],
                    gamma=gamma,
                    sigma=sigma,
                    alpha=alpha,
                    lambda_=lambda_,
                )

                optimizer.zero_grad()
                objective_loss.backward()
                optimizer.step()

    def predict_q_function_for_all_actions(
        self,
        state: np.ndarray,
    ):
        """Predict Q function for all actions.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        Return
        -------
        q_value: ndarray of shape (n_samples, n_actions)
            Q value of each (state, action) pair.

        """
        check_array(state, name="state", expected_dim=2)
        state = torch.FloatTensor(state, device=self.device)

        with torch.no_grad():
            q_value = self.q_function.all(state).to("cpu").detach().numpy()
        return q_value

    def predict_q_function(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict Q function.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_samples, action_dim)
            Action chosen by the behavior/evaluation policy.

        Return
        -------
        q_value: ndarray of shape (n_samples, )
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
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        evaluation_policy_action: array-like of shape (n_samples, action_dim)
            Action chosen by the evaluation policy.

        Return
        -------
        v_function: ndarray of shape (n_samples, )
            State value.

        """
        return self.predict_q_function(state=state, action=evaluation_policy_action)

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict function.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_samples, action_dim)
            Action chosen by the behavior policy.

        Return
        -------
        q_value: ndarray of shape (n_samples, )
            Q value of each (state, action) pair.

        """
        return self.predict_q_function(state=state, action=action)

    def fit_predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        evaluation_policy_next_action: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        gamma: float = 1.0,
        sigma: float = 1.0,
        alpha: float = 1e-3,
        lambda_: float = 1e-3,
        lr: float = 1e-3,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Fit and predict Q-function.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_samples, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_samples, )
            Reward observed for each (state, action) pair.

        next_state: array-like of shape (n_samples, state_dim)
            Next state observed for each (state, action) pair.

        evaluation_policy_next_action: array-like of shape (n_samples, action_dim)
            Next action chosen by the evaluation policy.

        pscore: array-like of shape (n_samples, )
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_samples, action_dim)
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

        alpha: float, default=1e-3

        lambda_: float, default=1e-3

        lr: float, default=1e-3
            Learning rate.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None
            Random state.

        """
        self.fit(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            evaluation_policy_next_action=evaluation_policy_next_action,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size,
            gamma=gamma,
            sigma=sigma,
            alpha=alpha,
            lambda_=lambda_,
            lr=lr,
            action_scaler=action_scaler,
            random_state=random_state,
        )
        return self.predict(state=state, action=action)


@dataclass
class ContinuousMinimaxStateValueLearning(BaseWeightValueLearner):
    """Minimax V Learning for marginal OPE estimators (for continuous action space).

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
    q_function: DiscreteQFunction
        Q function model.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    """

    v_function: VFunction
    device: str = "cuda:0"

    def __post_init__(self):
        self.v_function.to(self.device)

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

        return kernel  # shape (n_samples, n_samples)

    def _kernel_term(
        self,
        kernel: torch.Tensor,
        alpha: float,
        lambda_: float,
    ):
        """Kernel term in the objective function."""
        n_samples = len(kernel)
        sqrt_kernel = self._sqrt(kernel)
        inverse_kernel = self._inverse(alpha * torch.eye(n_samples) + lambda_ * kernel)
        return (
            sqrt_kernel @ inverse_kernel @ sqrt_kernel
        )  # shape (n_samples, n_samples)

    def _objective_function(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        importance_weight: torch.Tensor,
        gamma: float,
        sigma: float,
        alpha: float,
        lambda_: float,
        action_scaler: torch.Tensor,
    ):
        """Objective function of Minimax V Learning.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        action: Tensor of shape (n_samples, action_dim)
            Action chosen by the behavior policy.

        reward: Tensor of shape (n_samples, )
            Reward observed for each (state, action) pair.

        next_state: Tensor of shape (n_samples, state_dim)
            Next state observed for each (state, action) pair.

        importance_weight: Tensor of shape (n_samples, )
            Immediate importance weight of the given (state, action) pair,
            i.e., :math:`\\pi(a_t | s_t) / \\pi_0(a_t | s_t)`.

        gamma: float
            Discount factor. The value should be within `(0, 1]`.

        sigma: float
            Bandwidth hyperparameter of gaussian kernel.

        alpha: float

        lambda_: float

        action_scaler: Tensor of shape (action_dim, )
            Scaling factor of action.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of MVL.

        """
        kernel = self._gaussian_kernel(
            state, action, sigma=sigma, action_scaler=action_scaler
        )
        current_v = self.v_function(state)
        next_v = self.v_function(next_state)

        td_error = importance_weight * (reward + gamma * next_v) - current_v
        return (
            td_error.T
            @ self._kernel_term(kernel, alpha=alpha, lambda_=lambda_)
            @ td_error
        )

    def fit(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        gamma: float = 1.0,
        sigma: float = 1.0,
        alpha: float = 1e-3,
        lambda_: float = 1e-3,
        lr: float = 1e-3,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Fit Q-function.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_samples, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_samples, )
            Reward observed for each (state, action) pair.

        next_state: array-like of shape (n_samples, state_dim)
            Next state observed for each (state, action) pair.

        pscore: array-like of shape (n_samples, )
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_samples, action_dim)
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

        alpha: float, default=1e-3

        lambda_: float, default=1e-3

        lr: float, default=1e-3
            Learning rate.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None
            Random state.

        """
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=2)
        check_array(reward, name="reward", expected_dim=1)
        check_array(next_state, name="next_state", expected_dim=2)
        check_array(pscore, name="pscore", expected_dim=1, min_val=0.0, max_val=1.0)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if not (
            state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == next_state.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == reward.shape[0] == next_state.shape[0] "
                "== pscore.shape[0] == evaluation_policy_action.shape[0]`, but found False"
            )
        if state.shape[1] != next_state.shape[1]:
            raise ValueError(
                "Expected `state.shape[1] == next_state.shape[1]`, but found False"
            )
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(sigma, name="sigma", target_type=float, min_val=0.0)
        check_scalar(alpha, name="alpha", min_val=0.0)
        check_scalar(lambda_, name="lambda_", min_val=0.0)
        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )
        check_scalar(batch_size, name="batch_size", target_type=int, min_val=1)
        check_scalar(lr, name="lr", target_type=float, min_val=0.0)

        action_dim = action.shape[1]
        if action_scaler is None:
            action_scaler = np.ones(action_dim)
        elif isinstance(action_scaler, float):
            action_scaler = np.full(action_dim, action_scaler)

        check_array(action_scaler, name="action_scaler", expected_dim=1, min_val=0.0)
        if action_scaler.shape[0] != action_dim:
            raise ValueError(
                "Expected `action_scaler.shape[0] == action.shape[1]`, but found False"
            )

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        n_samples = len(state)
        state = torch.FloatTensor(state, device=self.device)
        action = torch.FloatTensor(action, device=self.device)
        reward = torch.FloatTensor(reward, device=self.device)
        next_state = torch.FloatTensor(next_state, device=self.device)
        evaluation_policy_next_action = torch.FloatTensor(
            evaluation_policy_next_action, device=self.device
        )

        similarity_weight = gaussian_kernel(
            evaluation_policy_action / action_scaler[None, :],
            action / action_scaler[None, :],
            sigma=sigma,
        )
        importance_weight = torch.FloatTensor(similarity_weight / pscore)

        optimizer = optim.SGD(self.v_function.parameters(), lr=lr, momentum=0.9)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc=["fitting_V_function"],
            total=n_epochs,
        ):
            for grad_step in range(n_steps_per_epoch):
                idx_ = torch.randint(n_samples, size=(batch_size,))
                objective_loss = self._objective_function(
                    state=state[idx_],
                    action=action[idx_],
                    reward=reward[idx_],
                    next_state=next_state[idx_],
                    importance_weight=importance_weight[idx_],
                    gamma=gamma,
                    sigma=sigma,
                    alpha=alpha,
                    lambda_=lambda_,
                    action_scaler=action_scaler,
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
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        Return
        -------
        v_function: ndarray of shape (n_samples, )
            State value.

        """
        check_array(state, name="state", expected_dim=2)
        state = torch.FloatTensor(state, device=self.device)

        with torch.no_grad():
            v_value = self.v_function(state).to("cpu").detach().numpy()
        return v_value

    def predict(
        self,
        state: np.ndarray,
    ):
        """Predict V function.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        Return
        -------
        v_function: ndarray of shape (n_samples, )
            State value.

        """
        return self.predict_v_function(state)

    def fit_predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        gamma: float = 1.0,
        sigma: float = 1.0,
        alpha: float = 1e-3,
        lambda_: float = 1e-3,
        lr: float = 1e-3,
        action_scaler: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Fit and predict Q-function.

        Parameters
        -------
        state: array-like of shape (n_samples, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_samples, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_samples, )
            Reward observed for each (state, action) pair.

        next_state: array-like of shape (n_samples, state_dim)
            Next state observed for each (state, action) pair.

        pscore: array-like of shape (n_samples, )
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action: array-like of shape (n_samples, action_dim)
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

        alpha: float, default=1e-3

        lambda_: float, default=1e-3

        lr: float, default=1e-3
            Learning rate.

        action_scaler: {float, array-like of shape (action_dim, )}, default=None
            Scaling factor of action.

        random_state: int, default=None
            Random state.

        """
        self.fit(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size,
            gamma=gamma,
            sigma=sigma,
            alpha=alpha,
            lambda_=lambda_,
            lr=lr,
            action_scaler=action_scaler,
            random_state=random_state,
        )
        return self.predict(state=state)
