# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Minimax weight function learning (continuous action cases)."""
from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm
from pathlib import Path

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
from sklearn.utils import check_scalar

from d3rlpy.preprocessing import Scaler, ActionScaler

from .base import BaseWeightValueLearner
from .function import (
    ContinuousStateActionWeightFunction,
    StateWeightFunction,
)
from ...utils import check_array, gaussian_kernel


@dataclass
class ContinuousMinimaxStateActionWeightLearning(BaseWeightValueLearner):
    """Minimax Weight Learning for marginal OPE estimators (for continuous action space).

    Bases: :class:`scope_rl.ope.weight_value_learning.BaseWeightValueLearner`

    Imported as: :class:`scope_rl.ope.weight_value_learning.ContinuousMinimaxStateActionWightLearning`

    Note
    -------
    Minimax Weight Learning uses that the following holds true about Q-function.

    .. math::

        \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \\sim d^{\\pi_b}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1})} [w(s_t, a_t) (Q(s_t, a_t) - \\gamma Q(s_{t+1}, a_{t+1}))]
        = \\mathbb{E}_{s_0 \\sim d^{\\pi_b}, a_0 \\sim \\pi(a_0 | s_0)} [Q(s_0, a_0)]

    where :math:`Q(s_t, a_t)` is the Q-function, :math:`w(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_b}(s_t, a_t)` is the state-action marginal importance weight.

    Then, it adversarially minimize the difference between RHS and LHS (which we denote :math:`L_w(w, Q)`) to the worst case in terms of :math:`Q(\\cdot)`
    using a discriminator defined in reproducing kernel Hilbert space (RKHS) as follows.

    .. math::

        \\max_w L_w^2(w, Q)
        &= \\mathbb{E}_{(s_t, a_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_b}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1}), \\tilde{a}_{t+1} \\sim \\pi(\\tilde{a}_{t+1} | \\tilde{s}_{t+1})}[
            w(s_t, a_t) w(\\tilde{s}_t, \\tilde{a}_t) ( K((s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t)) + K((s_{t+1}, a_{t+1}), (\\tilde{s}_{t+1}, \\tilde{a}_{t+1})) - \\gamma ( K((s_t, a_t), (\\tilde{s}_{t+1}, \\tilde{a}_{t+1})) + K((s_{t+1}, a_{t+1}), (\\tilde{s}_t, \\tilde{a}_t)) ))
        ] \\\\
        & \\quad \\quad + \\gamma (1 - \\gamma) \\mathbb{E}_{(s_t, a_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_b}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1}), \\tilde{a}_{t+1} \\sim \\pi(\\tilde{a}_{t+1} | \\tilde{s}_{t+1}), s_0 \\sim d(s_0), \\tilde{s}_0 \\sim d(\\tilde{s}_0), a_0 \\sim \\pi(a_0 | s_0), \\tilde{a}_0 \\sim \\pi(\\tilde{a}_0 | \\tilde{s}_0)}[
            w(s_t, a_t) K((s_{t+1}, a_{t+1}), (\\tilde{s}_0, \\tilde{a}_0)) + w(\\tilde{s}_t, \\tilde{a}_t) K((\\tilde{s}_{t+1}, \\tilde{a}_{t+1}), (s_0, a_0))
        ] \\\\
        & \\quad \\quad - (1 - \\gamma) \\mathbb{E}_{(s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t) \\sim d^{\\pi_b}, s_0 \\sim d(s_0), \\tilde{s}_0 \\sim d(\\tilde{s}_0), a_0 \\sim \\pi(a_0 | s_0), \\tilde{a}_0 \\sim \\pi(\\tilde{a}_0 | \\tilde{s}_0)}[
            w(s_t, a_t) K((s_t, a_t), (\\tilde{s}_0, \\tilde{a}_0)) + w(\\tilde{s}_t, \\tilde{a}_t) K((\\tilde{s}_t, \\tilde{a}_t), (s_0, a_0))
        ]

    where :math:`K(\\cdot, \\cdot)` is a kernel function.

    Parameters
    -------
    w_function: ContinuousStateActionWeightFunction
        Weight function model.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    bandwidth: float, default=1.0 (> 0)
        Bandwidth hyperparameter of the Gaussian kernel.

    state_scaler: d3rlpy.preprocessing.Scaler, default=None
        Scaling factor of state.

    action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
        Scaling factor of action.

    batch_size: int, default=128 (> 0)
        Batch size.

    lr: float, default=1e-4 (> 0)
        Learning rate.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    """

    w_function: ContinuousStateActionWeightFunction
    gamma: float = 1.0
    bandwidth: float = 1.0
    state_scaler: Optional[Scaler] = None
    action_scaler: Optional[ActionScaler] = None
    batch_size: int = 128
    lr: float = 1e-4
    device: str = "cuda:0"

    def __post_init__(self):
        self.w_function.to(self.device)

        check_scalar(
            self.gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0
        )
        check_scalar(self.bandwidth, name="bandwidth", target_type=float, min_val=0.0)
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
        check_scalar(self.lr, name="lr", target_type=float, min_val=0.0)

    def load(self, path: Path):
        self.w_function.load_state_dict(torch.load(path))

    def save(self, path: Path):
        torch.save(self.w_function.state_dict(), path)

    def _gaussian_kernel(
        self,
        state_1: torch.Tensor,
        action_1: torch.Tensor,
        state_2: torch.Tensor,
        action_2: torch.Tensor,
    ):
        """Gaussian kernel for all input pairs."""
        with torch.no_grad():
            x1 = torch.cat((state_1, action_1), dim=1)
            x2 = torch.cat((state_2, action_2), dim=1)
            # (x - x') ** 2 = x ** 2 + x' ** 2 - 2 x x'
            x1_2 = (x1**2).sum(dim=1)
            x2_2 = (x2**2).sum(dim=1)
            x_y = x1 @ x2.T
            distance = x1_2[:, None] + x2_2[None, :] - 2 * x_y
            kernel = torch.exp(-distance / (2 * self.bandwidth**2)) / np.sqrt(
                2 * np.pi * self.bandwidth**2
            )

        return kernel  # shape (n_trajectories, n_trajectories)

    def _first_term(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        importance_weight = importance_weight @ importance_weight.T
        positive_term = self._gaussian_kernel(
            state,
            action,
            state,
            action,
        ) + self._gaussian_kernel(
            next_state,
            next_action,
            next_state,
            next_action,
        )
        base_term = self._gaussian_kernel(
            state,
            action,
            next_state,
            next_action,
        )
        return (
            importance_weight * (positive_term - self.gamma * (base_term + base_term.T))
        ).mean()

    def _second_term(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        base_term = importance_weight[:, None] * self._gaussian_kernel(
            next_state,
            next_action,
            initial_state,
            initial_action,
        )
        return self.gamma * (1 - self.gamma) * (base_term + base_term.T).mean()

    def _third_term(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        base_term = importance_weight[:, None] * self._gaussian_kernel(
            state,
            action,
            initial_state,
            initial_action,
        )
        return (1 - self.gamma) * (base_term + base_term.T).mean()

    def _objective_function(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
    ):
        """Objective function of Minimax Weight Learning.

        Parameters
        -------
        initial_state: Tensor of shape (n_trajectories, state_dim)
            Initial state of a trajectory (or states sampled from a stationary distribution).

        initial_action: Tensor of shape (n_trajectories, action_dim)
            Initial action chosen by the evaluation policy.

        state: array-like of shape (n_trajectories, state_dim)
            State observed by the behavior policy.

        action: Tensor of shape (n_trajectories, action_dim)
            Action chosen by the behavior policy.

        next_state: Tensor of shape (n_trajectories, state_dim)
            Next state observed for each (state, action) pair.

        next_action: Tensor of shape (n_trajectories, action_dim)
            Next action chosen by the evaluation policy.

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
        )
        second_term = self._second_term(
            initial_state=initial_state,
            initial_action=initial_action,
            next_state=next_state,
            next_action=next_action,
            importance_weight=importance_weight,
        )
        third_term = self._third_term(
            initial_state=initial_state,
            initial_action=initial_action,
            state=state,
            action=action,
            importance_weight=importance_weight,
        )
        objective_loss = first_term + second_term - third_term

        # constraint to keep the expectation of the importance weight to be one
        regularization_loss = torch.pow(importance_weight.mean() - 1.0, 2)

        return objective_loss, regularization_loss

    def fit(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_steps: int = 10000,
        n_steps_per_epoch: int = 10000,
        regularization_weight: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit weight function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of gradient steps in a epoch.

        regularization_weight: float, default=1.0 (> 0)
            Scaling factor of the regularization weight.

        random_state: int, default=None (>= 0)
            Random state.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=2)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if not (state.shape[0] == action.shape[0] == evaluation_policy_action.shape[0]):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == evaluation_policy_next_action_dist.shape[0]`, but found False"
            )
        if state.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state.shape[0] % step_per_trajectory == 0`, but found False"
            )
        if action.shape[1] != evaluation_policy_action.shape[1]:
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`, but found False"
            )

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )
        n_epochs = (n_steps - 1) // n_steps_per_epoch + 1

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        state_dim = state.shape[1]
        action_dim = action.shape[1]
        state = state.reshape((-1, step_per_trajectory, state_dim))
        action = action.reshape((-1, step_per_trajectory, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_trajectory, action_dim)
        )

        n_trajectories, step_per_trajectory, _ = state.shape
        state = torch.FloatTensor(state, device=self.device)
        action = torch.FloatTensor(action, device=self.device)
        evaluation_policy_action = torch.FloatTensor(
            evaluation_policy_action, device=self.device
        )

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)
        if self.action_scaler is not None:
            action = self.action_scaler.transform(action)

        optimizer = optim.Adam(self.w_function.parameters(), lr=self.lr)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc="[fitting_weight_function]",
            total=n_epochs,
        ):
            for grad_step in tqdm(
                np.arange(n_steps_per_epoch),
                desc=f"[epoch: {epoch: >4}]",
                total=n_steps_per_epoch,
            ):
                idx_ = torch.randint(n_trajectories, size=(self.batch_size,))
                t_ = torch.randint(step_per_trajectory - 2, size=(self.batch_size,))

                objective_loss_, regularization_loss_ = self._objective_function(
                    initial_state=state[idx_, 0],
                    initial_action=action[idx_, 0],
                    state=state[idx_, t_],
                    action=action[idx_, t_],
                    next_state=state[idx_, t_ + 1],
                    next_action=action[idx_, t_ + 1],
                )
                loss_ = objective_loss_ - regularization_weight * regularization_loss_

                optimizer.zero_grad()
                loss_.backward()
                clip_grad_norm_(self.w_function.parameters(), max_norm=0.01)
                optimizer.step()

                # if grad_step % 10 == 0:
                #     print(objective_loss_.item(), regularization_loss_.item())

            print(
                f"epoch={epoch: >4}, "
                f"objective_loss={objective_loss_.item():.3f}, "
                f"regularization_loss={regularization_loss_.item():.3f}, "
            )

    def predict_weight(
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
        importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
            Estimated state-action marginal importance weight.

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
            importance_weight = (
                self.w_function(state, action).to("cpu").detach().numpy()
            )

        return importance_weight

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
        importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
            Estimated state-action marginal importance weight.

        """
        return self.predict_weight(state, action)

    def fit_predict(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_steps: int = 10000,
        n_steps_per_epoch: int = 10000,
        regularization_weight: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit and predict weight function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Next action chosen by the evaluation policy.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of gradient steps in a epoch.

        regularization_weight: float, default=1.0 (> 0)
            Scaling factor of the regularization weight.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        importance_weight: ndarray of shape (n_trajectories, )
            Estimated state-action marginal importance weight.

        """
        self.fit(
            step_per_trajectory=step_per_trajectory,
            state=state,
            action=action,
            evaluation_policy_action=evaluation_policy_action,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            regularization_weight=regularization_weight,
            random_state=random_state,
        )
        return self.predict_weight(state, action)


@dataclass
class ContinuousMinimaxStateWeightLearning(BaseWeightValueLearner):
    """Minimax Weight Learning for marginal OPE estimators (for continuous action space).

    Bases: :class:`scope_rl.ope.weight_value_learning.BaseWeightValueLearner`

    Imported as: :class:`scope_rl.ope.weight_value_learning.ContinuousMinimaxStateWightLearning`

    Note
    -------
    Minimax Weight Learning uses that the following holds true about Q-function.

    .. math::

        \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \\sim d^{\\pi_b}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1})} [w(s_t, a_t) (Q(s_t, a_t) - \\gamma Q(s_{t+1}, a_{t+1}))]
        = \\mathbb{E}_{s_0 \\sim d^{\\pi_b}, a_0 \\sim \\pi(a_0 | s_0)} [Q(s_0, a_0)]

    where :math:`Q(s_t, a_t)` is the Q-function, :math:`w(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_b}(s_t, a_t) = d^{\\pi}(s_t) \\pi(a_t | s_t) / d^{\\pi_b}(s_t) \\pi_0(a_t | s_t)`
    is the state-action marginal importance weight.

    Then, it adversarially minimize the difference between RHS and LHS (which we denote :math:`L_w(w, Q)`) to the worst case in terms of :math:`Q(\\cdot)`
    using a discriminator defined in reproducing kernel Hilbert space (RKHS) as follows.

    .. math::

        \\max_w L_w^2(w, Q)
        &= \\mathbb{E}_{(s_t, a_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_b}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1}), \\tilde{a}_{t+1} \\sim \\pi(\\tilde{a}_{t+1} | \\tilde{s}_{t+1})}[
            w_s(s_t) w_a(s_t, a_t) w_s(\\tilde{s}_t) w_a(\\tilde{s}_t, \\tilde{a}_t) ( K((s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t)) + K((s_{t+1}, a_{t+1}), (\\tilde{s}_{t+1}, \\tilde{a}_{t+1})) - \\gamma ( K((s_t, a_t), (\\tilde{s}_{t+1}, \\tilde{a}_{t+1})) + K((s_{t+1}, a_{t+1}), (\\tilde{s}_t, \\tilde{a}_t)) ))
        ] \\\\
        & \\quad \\quad + \\gamma (1 - \\gamma) \\mathbb{E}_{(s_t, a_t, s_{t+1}), (\\tilde{s}_t, \\tilde{a}_t, \\tilde{s}_{t+1}) \\sim d^{\\pi_b}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1}), \\tilde{a}_{t+1} \\sim \\pi(\\tilde{a}_{t+1} | \\tilde{s}_{t+1}), s_0 \\sim d(s_0), \\tilde{s}_0 \\sim d(\\tilde{s}_0), a_0 \\sim \\pi(a_0 | s_0), \\tilde{a}_0 \\sim \\pi(\\tilde{a}_0 | \\tilde{s}_0)}[
            w_s(s_t) w_a(s_t, a_t) K((s_{t+1}, a_{t+1}), (\\tilde{s}_0, \\tilde{a}_0)) + w_s(\\tilde{s}_t) w_a(\\tilde{s}_t, \\tilde{a}_t) K((\\tilde{s}_{t+1}, \\tilde{a}_{t+1}), (s_0, a_0))
        ] \\\\
        & \\quad \\quad - (1 - \\gamma) \\mathbb{E}_{(s_t, a_t), (\\tilde{s}_t, \\tilde{a}_t) \\sim d^{\\pi_b}, s_0 \\sim d(s_0), \\tilde{s}_0 \\sim d(\\tilde{s}_0), a_0 \\sim \\pi(a_0 | s_0), \\tilde{a}_0 \\sim \\pi(\\tilde{a}_0 | \\tilde{s}_0)}[
            w_s(s_t) w_a(s_t, a_t) K((s_t, a_t), (\\tilde{s}_0, \\tilde{a}_0)) + w_s(\\tilde{s}_t) w_a(\\tilde{s}_t, \\tilde{a}_t) K((\\tilde{s}_t, \\tilde{a}_t), (s_0, a_0))
        ]

    where :math:`K(\\cdot, \\cdot)` is a kernel function, :math:`w_s(s_t) \\approx d^{\\pi}(s_t) / d^{\\pi_b}(s_t)` is the state-marginal importance weight,
    and :math:`w_a(s_t, a_t) := \\pi(a_t | s_t) / \\pi_0(a_t | s_t)` is the immediate importance weight.

    Parameters
    -------
    w_function: StateWeightFunction
        Weight function model.

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    bandwidth: float, default=1.0 (> 0)
        Bandwidth hyperparameter of the Gaussian kernel.

    state_scaler: d3rlpy.preprocessing.Scaler, default=None
        Scaling factor of state.

    action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
        Scaling factor of action.

    batch_size: int, default=128 (> 0)
        Batch size.

    lr: float, default=1e-4 (> 0)
        Learning rate.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation." 2020.

    """

    w_function: StateWeightFunction
    gamma: float = 1.0
    bandwidth: float = 1.0
    state_scaler: Optional[Scaler] = None
    action_scaler: Optional[ActionScaler] = None
    batch_size: int = 128
    lr: float = 1e-4
    device: str = "cuda:0"

    def __post_init__(self):
        self.w_function.to(self.device)

        check_scalar(
            self.gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0
        )
        check_scalar(self.bandwidth, name="bandwidth", target_type=float, min_val=0.0)
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
        check_scalar(self.lr, name="lr", target_type=float, min_val=0.0)

    def load(self, path: Path):
        self.w_function.load_state_dict(torch.load(path))

    def save(self, path: Path):
        torch.save(self.w_function.state_dict(), path)

    def _gaussian_kernel(
        self,
        state_1: torch.Tensor,
        state_2: torch.Tensor,
    ):
        """Gaussian kernel for all input pairs."""
        with torch.no_grad():
            # (x - x') ** 2 = x ** 2 + x' ** 2 - 2 x x'
            x_2 = (state_1**2).sum(dim=1)
            y_2 = (state_2**2).sum(dim=1)
            x_y = state_1 @ state_2.T
            distance = x_2[:, None] + y_2[None, :] - 2 * x_y

            kernel = torch.exp(-distance / (2 * self.bandwidth**2)) / np.sqrt(
                2 * np.pi * self.bandwidth**2
            )

        return kernel  # shape (n_trajectories, n_trajectories)

    def _first_term(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        importance_weight = importance_weight @ importance_weight.T
        positive_term = self._gaussian_kernel(
            state,
            state,
        ) + self._gaussian_kernel(
            next_state,
            next_state,
        )
        base_term = self._gaussian_kernel(
            state,
            next_state,
        )
        return (
            importance_weight * (positive_term - self.gamma * (base_term + base_term.T))
        ).mean()

    def _second_term(
        self,
        initial_state: torch.Tensor,
        next_state: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        base_term = importance_weight[:, None] * self._gaussian_kernel(
            next_state,
            initial_state,
        )
        return self.gamma * (1 - self.gamma) * (base_term + base_term.T).mean()

    def _third_term(
        self,
        initial_state: torch.Tensor,
        state: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        base_term = importance_weight[:, None] * self._gaussian_kernel(
            state,
            initial_state,
        )
        return (1 - self.gamma) * (base_term + base_term.T).mean()

    def _objective_function(
        self,
        initial_state: torch.Tensor,
        state: torch.Tensor,
        next_state: torch.Tensor,
        importance_weight: torch.Tensor,
    ):
        """Objective function of Minimax Weight Learning.

        Parameters
        -------
        initial_state: Tensor of shape (n_trajectories, state_dim)
            Initial state of a trajectory (or states sampled from a stationary distribution).

        state: array-like of shape (n_trajectories, state_dim)
            State observed by the behavior policy.

        next_state: Tensor of shape (n_trajectories, state_dim)
            Next state observed for each (state, action) pair.

        importance_weight: Tensor of shape (n_trajectories, )
            Immediate importance weight of the given (state, action) pair,
            i.e., :math:`\\pi(a_t | s_t) / \\pi_0(a_t | s_t)`.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of MWL.

        """
        marginal_importance_weight = self.w_function(state)
        importance_weight = marginal_importance_weight * importance_weight

        first_term = self._first_term(
            state=state,
            next_state=next_state,
            importance_weight=importance_weight,
        )
        second_term = self._second_term(
            initial_state=initial_state,
            next_state=next_state,
            importance_weight=importance_weight,
        )
        third_term = self._third_term(
            initial_state=initial_state,
            state=state,
            importance_weight=importance_weight,
        )
        objective_loss = first_term + second_term - third_term

        # constraint to keep the expectation of the importance weight to be one
        regularization_loss = torch.pow(marginal_importance_weight.mean() - 1.0, 2)

        return objective_loss, regularization_loss

    def fit(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_steps: int = 10000,
        n_steps_per_epoch: int = 10000,
        regularization_weight: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit weight function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        pscore: array-like of shape (n_trajectories * step_per_trajectory)
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of gradient steps in a epoch.

        regularization_weight: float, default=1.0 (> 0)
            Scaling factor of the regularization weight.

        random_state: int, default=None (>= 0)
            Random state.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=2)
        check_array(pscore, name="pscore", expected_dim=2, min_val=0.0)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
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
        if state.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state.shape[0] % step_per_trajectory == 0`, but found False"
            )
        if not (
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action_dist.shape[1] == pscore.shape[1]`, but found False"
            )

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )
        n_epochs = (n_steps - 1) // n_steps_per_epoch + 1

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        state_dim = state.shape[1]
        action_dim = action.shape[1]
        state = state.reshape((-1, step_per_trajectory, state_dim))
        action = action.reshape((-1, step_per_trajectory, action_dim))
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
            bandwidth=self.bandwidth,
        ).reshape((-1, step_per_trajectory))

        n_trajectories, step_per_trajectory, _ = state.shape
        state = torch.FloatTensor(state, device=self.device)
        importance_weight = torch.FloatTensor(similarity_weight / pscore)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        optimizer = optim.Adam(self.w_function.parameters(), lr=self.lr)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc="[fitting_weight_function]",
            total=n_epochs,
        ):
            for grad_step in tqdm(
                np.arange(n_steps_per_epoch),
                desc=f"[epoch: {epoch: >4}]",
                total=n_steps_per_epoch,
            ):
                idx_ = torch.randint(n_trajectories, size=(self.batch_size,))
                t_ = torch.randint(step_per_trajectory - 2, size=(self.batch_size,))

                objective_loss_, regularization_loss_ = self._objective_function(
                    initial_state=state[idx_, 0],
                    state=state[idx_, t_],
                    next_state=state[idx_, t_ + 1],
                    importance_weight=importance_weight[idx_, t_],
                )
                loss_ = objective_loss_ - regularization_weight * regularization_loss_

                optimizer.zero_grad()
                loss_.backward()
                clip_grad_norm_(self.w_function.parameters(), max_norm=0.01)
                optimizer.step()

            print(
                f"epoch={epoch: >4}, "
                f"objective_loss={objective_loss_.item():.3f}, "
                f"regularization_loss={regularization_loss_.item():.3f}, "
            )

    def predict_state_marginal_importance_weight(
        self,
        state: np.ndarray,
    ):
        """Predict state marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        Return
        -------
        importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
            Estimated state marginal importance weight.

        """
        check_array(state, name="state", expected_dim=2)
        state = torch.FloatTensor(state, device=self.device)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        with torch.no_grad():
            importance_weight = self.w_function(state).to("cpu").detach().numpy()

        return importance_weight

    def predict_state_action_marginal_importance_weight(
        self,
        state: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
    ):
        """Predict state-action marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        pscore: array-like of shape (n_trajectories * step_per_trajectory)
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        Return
        -------
        importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
            Estimated state-action marginal importance weight.

        """
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=2)
        check_array(pscore, name="pscore", expected_dim=2, min_val=0.0)
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
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
            action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]
        ):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1] == pscore.shape[1]`, but found False"
            )

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
            bandwidth=self.bandwidth,
        )

        state_marginal_importance_weight = (
            self.predict_state_marginal_importance_weight(state)
        )
        state_action_marginal_importance_weight = (
            state_marginal_importance_weight * similarity_weight / pscore.prod(axis=1)
        )

        return state_action_marginal_importance_weight

    def predict_weight(
        self,
        state: np.ndarray,
    ):
        """Predict state marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        Return
        -------
        importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
            Estimated state marginal importance weight.

        """
        return self.predict_state_marginal_importance_weight(state)

    def predict(
        self,
        state: np.ndarray,
    ):
        """Predict state marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        Return
        -------
        importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
            Estimated state marginal importance weight.

        """
        return self.predict_state_marginal_importance_weight(state)

    def fit_predict(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        n_steps: int = 10000,
        n_steps_per_epoch: int = 10000,
        regularization_weight: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit and predict weight function.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        pscore: array-like of shape (n_trajectories * step_per_trajectory)
            Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of gradient steps in a epoch.

        regularization_weight: float, default=1.0 (> 0)
            Scaling factor of the regularization weight.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
            Estimated state-action marginal importance weight.

        """
        self.fit(
            step_per_trajectory=step_per_trajectory,
            state=state,
            action=action,
            pscore=pscore,
            evaluation_policy_action=evaluation_policy_action,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            regularization_weight=regularization_weight,
            random_state=random_state,
        )
        return self.predict_weight(state)
