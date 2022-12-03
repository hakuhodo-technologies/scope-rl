from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from pathlib import Path
from warnings import warn

import torch
from torch import optim

import numpy as np
from sklearn.utils import check_scalar

from d3rlpy.preprocessing import Scaler

from .base import BaseWeightValueLearner
from .function import (
    VFunction,
    StateWeightFunction,
    DiscreteQFunction,
    DiscreteStateActionWeightFunction,
)
from ...utils import check_array


@dataclass
class DiscreteAugmentedLagrangianStateActionWightValueLearning(BaseWeightValueLearner):
    """Augmented Lagrangian method for weight/value function of marginal OPE estimators (for discrete action space).

    Note
    -------
    Augmented Lagrangian method simultaneously learns the weight and value functions using the Lagrangian relaxation
    of the primal dual problem of weight/value learning as follows (See (Yang et al., 2020) for the theories behind):

    .. math::

        \\max_{w \\leq 0} \\min_{Q, \\lambda} L(w, Q, \\lambda)

    where

    .. math::

        L(w, Q, \\lambda) := (1 - \\gamma) \\mathbb{E}_{s_0 \\sim d(s_0), a_0 \\sim \\pi(s_0)} [Q(s_0, a_0)] + \\lambda
                            + \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \\sim d^{\\pi_0}, a_{t+1} \\sim \\pi(a_{t+1} | s_{t+1})} [w(s_t, a_t) (\\alpha_r r_t + \\gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) - \\lambda)]
                            + \\alpha_Q \\mathbb{E}_{(s_t, a_t) \\sim d^{\\pi_0}} [Q^2(s_t, a_t)] - \\alpha_w \\mathbb{E}_{(s_t, a_t) \\sim d^{\\pi_0}} [w^2(s_t, a_t)]

    where :math:`Q(s_t, a_t)` is the Q-function, :math:`w(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_0}(s_t, a_t)` is the state-action marginal importance weight.

    This estimator corresponds to the following estimators in its special cases.
    - DualDICE (Nachum et al., 2019): :math:`\\alpha_Q = 1, \\alpha_w = 0, \\alpha_r = 0, \\lambda = 0`.
    - GenDICE (Zhang et al., 2020), GradientDICE (Zhang et al., 2020): :math:`\\alpha_Q = 1, \\alpha_w = 0, \\alpha_r = 0`
    - AlgaeDICE (Nachum et al., 2019): :math:`\\alpha_Q = 0, \\alpha_w = 1, \\alpha_r = 1, \\lambda = 0`
    - BestDICE (Yang et al., 2020): :math:`\\alpha_Q = 0, \\alpha_w = 1, \\alpha_r = 1`
    - Minimax Q Learning (MQL) (Uehara and Jiang, 2019): :math:`alpha_Q = 0, \\alpha_w = 0, \\alpha_r = 1, \\lambda = 0`
    - Minimax Weight Learning (MWL) (Uehara and Jiang, 2019): :math:`alpha_Q = 0, \\alpha_w = 0, \\alpha_r = 0, \\lambda = 0`

    Note that, the positivity constraint :math:`w \\leq 0` should be imposed in the function approximation model.

    Parameters
    -------
    q_function: DiscreteQFunction
        Q function model.

    w_function: DiscreteStateActionWeightFunction
        Weight function model.

    gamma: float, default=1.0
        Discount factor. The value should be within `(0, 1]`.

    sigma: float, default=1.0 (> 0.0)
        Bandwidth hyperparameter of gaussian kernel. (This is for API consistency)

    state_scaler: d3rlpy.preprocessing.Scaler, default=None
        Scaling factor of state.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Shangtong Zhang, Bo Liu, and Shimon Whiteson.
    "GradientDICE: Rethinking Generalized Offline Estimation of Stationary Values.", 2020.

    Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans.
    "GenDICE: Generalized Offline Estimation of Stationary Values.", 2020.

    Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans.
    "AlgaeDICE: Policy Gradient from Arbitrary Experience.", 2019.

    Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li.
    "DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections.", 2019.

    """

    method: str
    q_function: DiscreteQFunction
    w_function: DiscreteStateActionWeightFunction
    gamma: float = 1.0
    sigma: float = 1.0
    state_scaler: Optional[Scaler] = None
    device: str = "cuda:0"

    def __post_init__(self):
        self.q_function.to(self.device)
        self.w_function.to(self.device)

        check_scalar(self.gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)
        if self.state_scaler is not None and not isinstance(self.state_scaler, Scaler):
            raise ValueError(
                "state_scaler must be an instance of d3rlpy.preprocessing.Scaler, but found False"
            )

    def load(self, path_q: Path, path_w: Path):
        self.q_function.load_state_dict(torch.load(path_q))
        self.w_function.load_state_dict(torch.load(path_w))

    def save(self, path_q: Path, path_w: Path):
        torch.save(self.q_function.state_dict(), path_q)
        torch.save(self.w_function.state_dict(), path_w)

    def _objective_function(
        self,
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        lambda_: torch.Tensor,
        alpha_q: float,
        alpha_w: float,
        alpha_r: bool,
    ):
        """Objective function of Augmented Lagrangian method.

        Parameters
        -------
        initial_state: Tensor of shape (n_trajectories, state_dim)
            Initial state of a trajectory (or states sampled from a stationary distribution).

        initial_action: Tensor of shape (n_trajectories, )
            Initial action chosen by the evaluation policy.

        state: array-like of shape (n_trajectories, state_dim)
            State observed by the behavior policy.

        action: Tensor of shape (n_trajectories, )
            Action chosen by the behavior policy.

        reward: Tensor of shape (n_trajectories, )
            Reward observed for each (state, action) pair.

        next_state: Tensor of shape (n_trajectories, state_dim)
            Next state observed for each (state, action) pair.

        next_action: Tensor of shape (n_trajectories, )
            Next action chosen by the evaluation policy.

        lambda_: Tensor of shape (1, )
            lambda_ hyperparameter to stabilize the optimization.

        gamma: float
            Discount factor. The value should be within `(0, 1]`.

        alpha_q: float
            Regularization coefficient of the Q-function.

        alpha_w: float
            Regularization coefficient of the weight function.

        alpha_r: bool
            Wether to consider the reward observation.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of the Augmented Lagrangian method.

        """
        initial_value = (1 - self.gamma) * self.q_function(
            initial_state, initial_action
        ).mean() + lambda_
        td_value = (
            self.w_function(state, action)
            * (
                alpha_r * reward
                + self.gamma * self.q_function(next_state, next_action)
                - self.q_function(state, action)
                - lambda_
            )
        ).mean()
        q_regularization = alpha_q * (self.q_function(state, action) ** 2).mean()
        w_regularization = alpha_w * (self.w_function(state, action) ** 2).mean()
        return initial_value + td_value + q_regularization - w_regularization

    def fit(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        method: str = "best_dice",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        q_lr: float = 1e-3,
        w_lr: float = 1e-3,
        lambda_lr: float = 1e-3,
        alpha_q: Optional[float] = None,
        alpha_w: Optional[float] = None,
        alpha_r: Optional[bool] = None,
        enable_lambda: Optional[bool] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit value and weight functions.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory)
            Reward observed for each (state, action) pair.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_actions)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        method: {"dual_dice", "gen_dice", "algae_dice", "best_dice", "mql", "mwl", "custom"}, default="best_dice"
            Indicates which parameter set should be used. When, "custom" users can specify their own parameter.

        n_epochs: int, default=100
            Number of epochs to train.

        n_steps_per_epoch: int, default=100
            Number of gradient steps in a epoch.

        batch_size: int, default=32
            Batch size.

        q_lr: float, default=1e-3
            Learning rate of q_function.

        w_lr: float, default=1e-3
            Learning rate of w_function.

        lambda_lr: float, default=1e-3
            Learning rate of lambda_.

        alpha_q: float, default=None
            Regularization coefficient of the Q-function.
            A value should be given when `method == "custom"`.

        alpha_w: float, default=None
            Regularization coefficient of the weight function.
            A value should be given when `method == "custom"`.

        alpha_r: bool, default=None
            Wether to consider the reward observation.
            A value should be given when `method == "custom"`.

        enable_lambda: bool, default=None
            Whether to optimize :math:`\\lambda`. If False, :math:`\\lambda` is automatically set to zero.
            A boolean value should be given when `method == "custom"`.

        random_state: int, default=None
            Random state.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=1)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        if not (
            state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == reward.shape[0] == evaluation_policy_action_dist.shape[0]`, but found False"
            )
        if state.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state.shape[0] % step_per_trajectory == 0`, but found False"
            )
        if not np.allclose(
            np.ones(evaluation_policy_action_dist.shape[0]),
            evaluation_policy_action_dist.sum(axis=1),
        ):
            raise ValueError(
                "evaluation_policy_action_dist must sums up to one in axis=1, but found False"
            )

        if method not in [
            "dual_dice",
            "gen_dice",
            "algae_dice",
            "best_dice",
            "mql",
            "mwl",
            "custom",
        ]:
            raise ValueError(
                f"method must be one of {'dual_dice', 'gen_dice', 'algae_dice', 'best_dice', 'mql', 'mwl', 'custom'}, but {method} is given"
            )
        if method == "custom":
            if alpha_q is None:
                raise ValueError("alpha_q must be given when `method == 'custom'`")
            if alpha_w is None:
                raise ValueError("alpha_w must be given when `method == 'custom'`")
            if alpha_r is None:
                raise ValueError("alpha_r must be given when `method == 'custom'`")
            if enable_lambda is None:
                raise ValueError(
                    "enable_lambda must be given when `method == 'custom'`"
                )
        else:
            if alpha_q is not None:
                warn(
                    f"alpha_q is given, but alpha_q will be initialized with by the setting of {method}."
                    "To customize, `method` should be set to 'custom'."
                )
            if alpha_w is not None:
                warn(
                    f"alpha_w is given, but alpha_w will be initialized with by the setting of {method}."
                    "To customize, `method` should be set to 'custom'."
                )
            if alpha_r is not None:
                warn(
                    f"alpha_r is given, but alpha_r will be initialized with by the setting of {method}."
                    "To customize, `method` should be set to 'custom'."
                )

            alpha_q = method in ["dual_dice", "gen_dice"]
            alpha_w = method in ["algae_dice", "best_dice"]
            alpha_r = method in ["algae_dice", "best_dice", "mql"]
            enable_lambda = method in ["gen_dice", "best_dice"]
            self.lambda_ = torch.zeros(size=(1,), device=self.device)

        check_scalar(alpha_q, name="alpha_q", target_type=float)
        check_scalar(alpha_w, name="alpha_w", target_type=float)
        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )
        check_scalar(batch_size, name="batch_size", target_type=int, min_val=1)
        check_scalar(q_lr, name="q_lr", target_type=float, min_val=0.0)
        check_scalar(w_lr, name="w_lr", target_type=float, min_val=0.0)
        check_scalar(lambda_lr, name="lambda_lr", target_type=float, min_val=0.0)

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        state_dim = state.shape[1]
        n_actions = evaluation_policy_action_dist.shape[1]
        state = state.reshape((-1, step_per_trajectory, state_dim))
        action = action.reshape((-1, step_per_trajectory))
        reward = reward.reshape((-1, step_per_trajectory))
        evaluation_policy_action_dist = evaluation_policy_action_dist.reshape(
            (-1, step_per_trajectory, n_actions)
        )

        n_trajectories, step_per_trajectory, _ = state.shape
        state = torch.FloatTensor(state, device=self.device)
        action = torch.LongTensor(action, device=self.device)
        reward = torch.FloatTensor(reward, device=self.device)
        evaluation_policy_action_dist = torch.FloatTensor(
            evaluation_policy_action_dist, device=self.device
        )

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        q_optimizer = optim.SGD(self.q_function.parameters(), lr=q_lr, momentum=0.9)
        w_optimizer = optim.SGD(self.w_function.parameters(), lr=w_lr, momentum=0.9)

        if enable_lambda:
            self.lambda_ = torch.ones(size=(1,), device=self.device, requires_grad=True)
            lambda_optimizer = optim.SGD(self.lambda_, lr=lambda_lr, momentum=0.9)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc=["fitting_weight_and_value_functions"],
            total=n_epochs,
        ):
            for grad_step in range(n_steps_per_epoch):
                idx_ = torch.randint(n_trajectories, size=(batch_size,))
                t_ = torch.randint(step_per_trajectory - 1, size=(batch_size,))

                initial_action = torch.multinomial(
                    evaluation_policy_action_dist[idx_, 0], num_samples=1
                ).flatten()
                next_action = torch.multinomial(
                    evaluation_policy_action_dist[idx_, t_ + 1], num_samples=1
                ).flatten()

                objective_loss = self._objective_function(
                    initial_state=state[idx_, 0],
                    initial_action=initial_action,
                    state=state[idx_, t_],
                    action=action[idx_, t_],
                    reward=reward[idx_, t_],
                    next_state=state[idx_, t_ + 1],
                    next_action=next_action,
                    lambda_=self.lambda_,
                    alpha_q=alpha_q,
                    alpha_w=alpha_w,
                    alpha_r=alpha_r,
                )

                q_optimizer.zero_grad()
                w_optimizer.zero_grad()
                if enable_lambda:
                    lambda_optimizer.zero_grad()

                objective_loss.backward()

                q_optimizer.step()
                w_optimizer.step()
                if enable_lambda:
                    lambda_optimizer.step()

    def predict_q_function_for_all_actions(
        self,
        state: np.ndarray,
    ):
        """Predict Q function for all actions.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory, n_actions)
            Q value of each (state, action) pair.

        """
        check_array(state, name="state", expected_dim=2)
        state = torch.FloatTensor(state, device=self.device)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        with torch.no_grad():
            q_value = self.q_function.all(state).to("cpu").detach().numpy()

        return q_value + self.lambda_.item()

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

        action: array-like of shape (n_trajectories * step_per_trajectory)
            Action chosen by the behavior policy.

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory)
            Q value of each (state, action) pair.

        """
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=1)
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0]`, but found False"
            )

        state = torch.FloatTensor(state, device=self.device)
        action = torch.LongTensor(action, device=self.device)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        with torch.no_grad():
            q_value = self.q_function(state, action).to("cpu").detach().numpy()

        return q_value + self.lambda_.item()

    def predict_v_function(
        self,
        state: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        """Predict V function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_actions)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory)
            Q value of each (state, action) pair.

        """
        check_array(state, name="state", expected_dim=2)
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        if state.shape[0] != evaluation_policy_action_dist.shape[0]:
            raise ValueError(
                "Expected `state.shape[0] == evaluation_policy_action_dist.shape[0]`, but found False"
            )
        if not np.allclose(
            np.ones(evaluation_policy_action_dist.shape[:2]),
            evaluation_policy_action_dist.sum(axis=2),
        ):
            raise ValueError(
                "evaluation_policy_action_dist must sums up to one in axis=2, but found False"
            )

        state = torch.FloatTensor(state, device=self.device)
        evaluation_policy_action_dist = torch.FloatTensor(
            evaluation_policy_action_dist, device=self.device
        )

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        with torch.no_grad():
            v_value = (
                self.q_function.expectation(state, evaluation_policy_action_dist)
                .to("cpu")
                .detach()
                .numpy()
            )
        return v_value + self.lambda_.item()

    def predict_value(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict Q function.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory)
            Action chosen by the behavior policy.

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory)
            Q value of each (state, action) pair.

        """
        return self.predict_q_function(state, action)

    def predict_weight(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict state-action marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory)
            Action chosen by the behavior policy.

        Return
        -------
        w_hat: ndarray of shape (n_trajectories * step_per_trajectory)
            Estimated state-action marginal importance weight.

        """
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=1)
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0]`, but found False"
            )

        state = torch.FloatTensor(state, device=self.device)
        action = torch.LongTensor(action, device=self.device)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        with torch.no_grad():
            w_hat = self.w_function(state, action).to("cpu").detach().numpy()

        return w_hat

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ):
        """Predict Q value and state-action marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory)
            Action chosen by the behavior policy.

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory)
            Q value of each (state, action) pair.

        w_hat: ndarray of shape (n_trajectories * step_per_trajectory)
            Estimated state-action marginal importance weight.

        """
        q_value = self.predict_value(state, action)
        w_hat = self.predict_weight(state, action)
        return q_value, w_hat

    def fit_predict(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        method: str = "best_dice",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        q_lr: float = 1e-3,
        w_lr: float = 1e-3,
        lambda_lr: float = 1e-3,
        alpha_q: Optional[float] = None,
        alpha_w: Optional[float] = None,
        alpha_r: Optional[bool] = None,
        enable_lambda: Optional[bool] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit and predict value/weight functions.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories, )
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories, )
            Reward observed for each (state, action) pair.

        evaluation_policy_action_dist: array-like of shape (n_trajectories, step_per_trajectory, n_actions)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        method: {"dual_dice", "gen_dice", "algae_dice", "best_dice", "mql", "mwl", "custom"}, default="best_dice"
            Indicates which parameter set should be used. When, "custom" users can specify their own parameter.

        n_epochs: int, default=100
            Number of epochs to train.

        n_steps_per_epoch: int, default=100
            Number of gradient steps in a epoch.

        batch_size: int, default=32
            Batch size.

        q_lr: float, default=1e-3
            Learning rate of q_function.

        w_lr: float, default=1e-3
            Learning rate of w_function.

        lambda_lr: float, default=1e-3
            Learning rate of lambda_.

        alpha_q: float, default=None
            Regularization coefficient of the Q-function.
            A value should be given when `method == "custom"`.

        alpha_w: float, default=None
            Regularization coefficient of the weight function.
            A value should be given when `method == "custom"`.

        alpha_r: bool, default=None
            Wether to consider the reward observation.
            A value should be given when `method == "custom"`.

        enable_lambda: bool, default=None
            Whether to optimize :math:`\\lambda`. If False, :math:`\\lambda` is automatically set to zero.
            A boolean value should be given when `method == "custom"`.

        random_state: int, default=None
            Random state.

        Return
        -------
        q_value: ndarray of shape (n_trajectories, step_per_trajectory)
            Q value of each (state, action) pair.

        w_hat: ndarray of shape (n_trajectories, step_per_trajectory)
            Estimated state-action marginal importance weight.

        """
        self.fit(
            step_per_trajectory=step_per_trajectory,
            state=state,
            action=action,
            reward=reward,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            method=method,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size,
            q_lr=q_lr,
            w_lr=w_lr,
            lambda_lr=lambda_lr,
            alpha_q=alpha_q,
            alpha_w=alpha_w,
            alpha_r=alpha_r,
            enable_lambda=enable_lambda,
            random_state=random_state,
        )
        return self.predict(state, action)


@dataclass
class DiscreteAugmentedLagrangianStateWightValueLearning(BaseWeightValueLearner):
    """Augmented Lagrangian method for weight/value function of marginal OPE estimators (for discrete action space).

    Note
    -------
    Augmented Lagrangian method simultaneously learns the weight and value functions using the Lagrangian relaxation
    of the primal dual problem of weight/value learning (See (Yang et al., 2020) for the theories behind).

    This class aims to learn V-function and state-marginal importance weight rather than estimating Q-function and
    state-action marginal importance weight.

    .. math::

        \\max_{w \\leq 0} \\min_{V, \\lambda} L(w, V, \\lambda)

    where

    .. math::

        L(w, V, \\lambda) := (1 - \\gamma) \\mathbb{E}_{s_0 \\sim d(s_0)} [V(s_0)] + \\lambda
                            + \\mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \\sim d^{\\pi_0}} [w_s(s_t) w_a(s_t, a_t) (\\alpha_r r_t + \\gamma V(s_{t+1}) - V(s_t) - \\lambda)]
                            + \\alpha_V \\mathbb{E}_{s_t \\sim d^{\\pi_0}} [V^2(s_t)] - \\alpha_w \\mathbb{E}_{s_t \\sim d^{\\pi_0}} [w_s^2(s_t)]

    where :math:`V(s_t)` is the V-function, :math:`w_s(s_t) \\approx d^{\\pi}(s_t) / d^{\\pi_0}(s_t)` is the state marginal importance weight.
    :math:`w_a(s_t, a_t) = \\pi(a_t | s_t) / \\pi_0(a_t | s_t)` is the immediate importance weight.

    This estimator is analogous to the following estimators in its special cases (although the following uses Q-function and state-action marginal importance weight).
    - DualDICE (Nachum et al., 2019): :math:`\\alpha_Q = 1, \\alpha_w = 0, \\alpha_r = 0, \\lambda = 0`.
    - GenDICE (Zhang et al., 2020), GradientDICE (Zhang et al., 2020): :math:`\\alpha_Q = 1, \\alpha_w = 0, \\alpha_r = 0`
    - AlgaeDICE (Nachum et al., 2019): :math:`\\alpha_Q = 0, \\alpha_w = 1, \\alpha_r = 1, \\lambda = 0`
    - BestDICE (Yang et al., 2020): :math:`\\alpha_Q = 0, \\alpha_w = 1, \\alpha_r = 1`
    - Minimax Value Learning (MVL) (Uehara and Jiang, 2019): :math:`alpha_Q = 0, \\alpha_w = 0, \\alpha_r = 1, \\lambda = 0`
    - Minimax Weight Learning (MWL) (Uehara and Jiang, 2019): :math:`alpha_Q = 0, \\alpha_w = 0, \\alpha_r = 0, \\lambda = 0`

    Note that, the positivity constraint :math:`w \\leq 0` should be imposed in the function approximation model.

    Parameters
    -------
    v_function: VFunction
        V function model.

    w_function: StateWeightFunction
        Weight function model.

    gamma: float, default=1.0
        Discount factor. The value should be within `(0, 1]`.

    sigma: float, default=1.0 (> 0.0)
        Bandwidth hyperparameter of gaussian kernel. (This is for API consistency)

    state_scaler: d3rlpy.preprocessing.Scaler, default=None
        Scaling factor of state.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Shangtong Zhang, Bo Liu, and Shimon Whiteson.
    "GradientDICE: Rethinking Generalized Offline Estimation of Stationary Values.", 2020.

    Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans.
    "GenDICE: Generalized Offline Estimation of Stationary Values.", 2020.

    Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans.
    "AlgaeDICE: Policy Gradient from Arbitrary Experience.", 2019.

    Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li.
    "DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections.", 2019.

    """

    v_function: VFunction
    w_function: StateWeightFunction
    gamma: float = 1.0
    sigma: float = 1.0
    state_scaler: Optional[Scaler] = None
    device: str = "cuda:0"

    def __post_init__(self):
        self.v_function.to(self.device)
        self.w_function.to(self.device)

        check_scalar(self.gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)
        if self.state_scaler is not None and not isinstance(self.state_scaler, Scaler):
            raise ValueError(
                "state_scaler must be an instance of d3rlpy.preprocessing.Scaler, but found False"
            )

    def load(self, path_v: Path, path_w: Path):
        self.v_function.load_state_dict(torch.load(path_v))
        self.w_function.load_state_dict(torch.load(path_w))

    def save(self, path_v: Path, path_w: Path):
        torch.save(self.v_function.state_dict(), path_v)
        torch.save(self.w_function.state_dict(), path_w)

    def _objective_function(
        self,
        initial_state: torch.Tensor,
        state: torch.Tensor,
        importance_weight: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        lambda_: torch.Tensor,
        alpha_v: float,
        alpha_w: float,
        alpha_r: bool,
    ):
        """Objective function of Augmented Lagrangian method.

        Parameters
        -------
        initial_state: Tensor of shape (n_trajectories, state_dim)
            Initial state of a trajectory (or states sampled from a stationary distribution).

        state: array-like of shape (n_trajectories, state_dim)
            State observed by the behavior policy.

        reward: Tensor of shape (n_trajectories, )
            Reward observed for each (state, action) pair.

        next_state: Tensor of shape (n_trajectories, state_dim)
            Next state observed for each (state, action) pair.

        importance_weight: Tensor of shape (n_trajectories, )
            Immediate importance weight at each step, i.e., :math:`\\pi(a_t | s_t) / \\pi_0(a_t | s_t)`.

        lambda_: Tensor of shape (1, )
            lambda_ hyperparameter to stabilize the optimization.

        alpha_v: float
            Regularization coefficient of the V-function.

        alpha_w: float
            Regularization coefficient of the weight function.

        alpha_r: bool
            Wether to consider the reward observation.

        Return
        -------
        objective_function: Tensor of shape (1, )
            Objective function of the Augmented Lagrangian method.

        """
        initial_value = (1 - self.gamma) * self.v_function(initial_state).mean() + lambda_
        td_value = (
            self.w_function(state)
            * importance_weight
            * (
                alpha_r * reward
                + self.gamma * self.v_function(next_state)
                - self.v_function(state)
                - lambda_
            )
        ).mean()
        q_regularization = alpha_v * (self.v_function(state) ** 2).mean()
        w_regularization = alpha_w * (self.w_function(state) ** 2).mean()
        return initial_value + td_value + q_regularization - w_regularization

    def fit(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        method: str = "best_dice",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        v_lr: float = 1e-3,
        w_lr: float = 1e-3,
        lambda_lr: float = 1e-3,
        alpha_v: Optional[float] = None,
        alpha_w: Optional[float] = None,
        alpha_r: Optional[bool] = None,
        enable_lambda: Optional[bool] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit value and weight functions.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory)
            Reward observed for each (state, action) pair.

        pscore: array-like of shape (n_trajectories * step_per_trajectory)
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_actions)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        method: {"dual_dice", "gen_dice", "algae_dice", "best_dice", "mvl", "mwl", "custom"}, default="best_dice"
            Indicates which parameter set should be used. When, "custom" users can specify their own parameter.

        n_epochs: int, default=100
            Number of epochs to train.

        n_steps_per_epoch: int, default=100
            Number of gradient steps in a epoch.

        batch_size: int, default=32
            Batch size.

        v_lr: float, default=1e-3
            Learning rate of v_function.

        w_lr: float, default=1e-3
            Learning rate of w_function.

        lambda_lr: float, default=1e-3
            Learning rate of lambda_.

        alpha_v: float, default=None
            Regularization coefficient of the V-function.
            A value should be given when `method == "custom"`.

        alpha_w: float, default=None
            Regularization coefficient of the weight function.
            A value should be given when `method == "custom"`.

        alpha_r: bool, default=None
            Wether to consider the reward observation.
            A value should be given when `method == "custom"`.

        enable_lambda: bool, default=None
            Whether to optimize :math:`\\lambda`. If False, :math:`\\lambda` is automatically set to zero.
            A boolean value should be given when `method == "custom"`.

        random_state: int, default=None
            Random state.

        """
        check_scalar(
            step_per_trajectory, name="step_per_trajectory", target_type=int, min_val=1
        )
        check_array(state, name="state", expected_dim=2)
        check_array(action, name="action", expected_dim=1)
        check_array(reward, name="reward", expected_dim=2)
        check_array(pscore, name="pscore", expected_dim=1, min_val=0.0, max_val=1.0)
        check_array(
            evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        if not (
            state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == pscore.shape[0]
            == evaluation_policy_action_dist.shape[0]
        ):
            raise ValueError(
                "Expected `state.shape[0] == action.shape[0] == reward.shape[0] == pscore.shape[0] == evaluation_policy_action_dist.shape[0]`, but found False"
            )
        if state.shape[0] % step_per_trajectory:
            raise ValueError(
                "Expected `state.shape[0] % step_per_trajectory == 0`, but found False"
            )
        if not np.allclose(
            np.ones(evaluation_policy_action_dist.shape[0]),
            evaluation_policy_action_dist.sum(axis=1),
        ):
            raise ValueError(
                "evaluation_policy_action_dist must sums up to one in axis=1, but found False"
            )

        if method not in [
            "dual_dice",
            "gen_dice",
            "algae_dice",
            "best_dice",
            "mvl",
            "mwl",
            "custom",
        ]:
            raise ValueError(
                f"method must be one of {'dual_dice', 'gen_dice', 'algae_dice', 'best_dice', 'mql', 'mwl', 'custom'}, but {method} is given"
            )
        if method == "custom":
            if alpha_v is None:
                raise ValueError("alpha_v must be given when `method == 'custom'`")
            if alpha_w is None:
                raise ValueError("alpha_w must be given when `method == 'custom'`")
            if alpha_r is None:
                raise ValueError("alpha_r must be given when `method == 'custom'`")
            if enable_lambda is None:
                raise ValueError(
                    "enable_lambda must be given when `method == 'custom'`"
                )
        else:
            if alpha_v is not None:
                warn(
                    f"alpha_v is given, but alpha_v will be initialized with by the setting of {method}."
                    "To customize, `method` should be set to 'custom'."
                )
            if alpha_w is not None:
                warn(
                    f"alpha_w is given, but alpha_w will be initialized with by the setting of {method}."
                    "To customize, `method` should be set to 'custom'."
                )
            if alpha_r is not None:
                warn(
                    f"alpha_r is given, but alpha_r will be initialized with by the setting of {method}."
                    "To customize, `method` should be set to 'custom'."
                )

            alpha_q = method in ["dual_dice", "gen_dice"]
            alpha_w = method in ["algae_dice", "best_dice"]
            alpha_r = method in ["algae_dice", "best_dice", "mvl"]
            enable_lambda = method in ["gen_dice", "best_dice"]
            self.lambda_ = torch.zeros(size=(1,), device=self.device)

        check_scalar(alpha_q, name="alpha_q", target_type=float)
        check_scalar(alpha_w, name="alpha_w", target_type=float)
        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )
        check_scalar(batch_size, name="batch_size", target_type=int, min_val=1)
        check_scalar(v_lr, name="v_lr", target_type=float, min_val=0.0)
        check_scalar(w_lr, name="w_lr", target_type=float, min_val=0.0)
        check_scalar(lambda_lr, name="lambda_lr", target_type=float, min_val=0.0)

        if random_state is None:
            raise ValueError("Random state mush be given.")
        torch.manual_seed(random_state)

        state_dim = state.shape[1]
        n_actions = evaluation_policy_action_dist.shape[1]
        state = state.reshape((-1, step_per_trajectory, state_dim))
        action = action.reshape((-1, step_per_trajectory))
        reward = reward.reshape((-1, step_per_trajectory))
        pscore = pscore.reshape((-1, step_per_trajectory))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_trajectory, n_actions)
        )

        n_trajectories, step_per_trajectory, _ = state.shape
        state = torch.FloatTensor(state, device=self.device)
        reward = torch.FloatTensor(reward, device=self.device)
        importance_weight = torch.FloatTensor(
            evaluation_policy_action_dist.flatten()[
                np.arange(n_trajectories * step_per_trajectory), action.flatten()
            ]
            / pscore.flatten()
        ).reshape((n_trajectories, step_per_trajectory))

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        v_optimizer = optim.SGD(self.v_function.parameters(), lr=v_lr, momentum=0.9)
        w_optimizer = optim.SGD(self.w_function.parameters(), lr=w_lr, momentum=0.9)

        if enable_lambda:
            self.lambda_ = torch.ones(size=(1,), device=self.device, requires_grad=True)
            lambda_optimizer = optim.SGD(self.lambda_, lr=lambda_lr, momentum=0.9)

        for epoch in tqdm(
            np.arange(n_epochs),
            desc=["fitting_weight_and_value_functions"],
            total=n_epochs,
        ):
            for grad_step in range(n_steps_per_epoch):
                idx_ = torch.randint(n_trajectories, size=(batch_size,))
                t_ = torch.randint(step_per_trajectory, size=(batch_size,))

                objective_loss = self._objective_function(
                    initial_state=state[idx_, 0],
                    state=state[idx_, t_],
                    reward=reward[idx_, t_],
                    next_state=state[idx_, t_ + 1],
                    importance_weight=importance_weight[idx_, t_],
                    lambda_=self.lambda_,
                    alpha_v=alpha_v,
                    alpha_w=alpha_w,
                    alpha_r=alpha_r,
                )

                v_optimizer.zero_grad()
                w_optimizer.zero_grad()
                if enable_lambda:
                    lambda_optimizer.zero_grad()

                objective_loss.backward()

                v_optimizer.step()
                w_optimizer.step()
                if enable_lambda:
                    lambda_optimizer.step()

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
        check_array(state, name="state", expected_dim=2)
        state = torch.FloatTensor(state, device=self.device)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        with torch.no_grad():
            v_function = self.v_function(state).to("cpu").detach().numpy()

        return v_function + self.lambda_.item()

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
        w_hat: ndarray of shape (n_trajectories * step_per_trajectory)
            Estimated state marginal importance weight.

        """
        check_array(state, name="state", expected_dim=2)
        state = torch.FloatTensor(state, device=self.device)

        if self.state_scaler is not None:
            state = self.state_scaler.transform(state)

        with torch.no_grad():
            w_hat = self.w_function(state).to("cpu").detach().numpy()

        return w_hat

    def predict(
        self,
        state: np.ndarray,
    ):
        """Predict V function and state-action marginal importance weight.

        Parameters
        -------
        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        Return
        -------
        v_function: ndarray of shape (n_trajectories * step_per_trajectory)
            V function of each (state, action) pair.

        w_hat: ndarray of shape (n_trajectories * step_per_trajectory)
            Estimated state-action marginal importance weight.

        """
        v_function = self.predict_value(state)
        w_hat = self.predict_weight(state)
        return v_function, w_hat

    def fit_predict(
        self,
        step_per_trajectory: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        method: str = "best_dice",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        q_lr: float = 1e-3,
        w_lr: float = 1e-3,
        lambda_lr: float = 1e-3,
        alpha_v: Optional[float] = None,
        alpha_w: Optional[float] = None,
        alpha_r: Optional[bool] = None,
        enable_lambda: Optional[bool] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Fit and predict value/weight functions.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state: array-like of shape (n_trajectories * step_per_trajectory, state_dim)
            State observed by the behavior policy.

        action: array-like of shape (n_trajectories * step_per_trajectory)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_trajectories * step_per_trajectory)
            Reward observed for each (state, action) pair.

        pscore: array-like of shape (n_trajectories * step_per_trajectory)
            Action choice probability of the behavior policy for the chosen action.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_actions)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        method: {"dual_dice", "gen_dice", "algae_dice", "best_dice", "mvl", "mwl", "custom"}, default="best_dice"
            Indicates which parameter set should be used. When, "custom" users can specify their own parameter.

        n_epochs: int, default=100
            Number of epochs to train.

        n_steps_per_epoch: int, default=100
            Number of gradient steps in a epoch.

        batch_size: int, default=32
            Batch size.

        q_lr: float, default=1e-3
            Learning rate of q_function.

        w_lr: float, default=1e-3
            Learning rate of w_function.

        lambda_lr: float, default=1e-3
            Learning rate of lambda_.

        alpha_v: float, default=None
            Regularization coefficient of the V-function.
            A value should be given when `method == "custom"`.

        alpha_w: float, default=None
            Regularization coefficient of the weight function.
            A value should be given when `method == "custom"`.

        alpha_r: bool, default=None
            Wether to consider the reward observation.
            A value should be given when `method == "custom"`.

        enable_lambda: bool, default=None
            Whether to optimize :math:`\\lambda`. If False, :math:`\\lambda` is automatically set to zero.
            A boolean value should be given when `method == "custom"`.

        random_state: int, default=None
            Random state.

        Return
        -------
        q_value: ndarray of shape (n_trajectories * step_per_trajectory)
            Q value of each (state, action) pair.

        w_hat: ndarray of shape (n_trajectories * step_per_trajectory)
            Estimated state-action marginal importance weight.

        """
        self.fit(
            step_per_trajectory=step_per_trajectory,
            state=state,
            action=action,
            reward=reward,
            pscore=pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            method=method,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size,
            q_lr=q_lr,
            w_lr=w_lr,
            lambda_lr=lambda_lr,
            alpha_v=alpha_v,
            alpha_w=alpha_w,
            alpha_r=alpha_r,
            enable_lambda=enable_lambda,
            random_state=random_state,
        )
        return self.predict(state, action)
