"""Distributional Off-Policy Estimators for Discrete action."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import check_scalar

from offlinegym.ope.estimators_base import (
    BaseDistributionallyRobustOffPolicyEstimator,
)
from offlinegym.utils import check_array


@dataclass
class DiscreteDistributionallyRobustImportanceSampling(
    BaseDistributionallyRobustOffPolicyEstimator
):
    """Importance Sampling (IS) for estimating the worst case performance in a distributionally robust manner.

     Note
    -------
    IS estimates the worst case policy value using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    estimator_name: str, default="dr_is"
        Name of the estimator.

    References
    -------
    Nathan Kallus, Xiaojie Mao, Kaiwen Wang, and Zhengyuan Zhou.
    "Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning.", 2022.

    Nian Si, Fan Zhang, Zhengyuan Zhou, and Jose Blanchet.
    "Distributional Robust Batch Contextual Bandits.", 2020.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "dr_is"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_policy_value_momentum_given_alpha(
        self,
        trajectory_wise_reward: np.ndarray,
        trajectory_wise_importance_weight: np.ndarray,
        alpha: float,
        momentum: int,
    ):
        """Calculate exponential policy value given alpha.

        Parameters
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        alpha: float
            Temperature parameter of the exponential function.

        momentum: int
            When :math:`n` is given, :math:`W_n` is calculated.
            See Kallus et al. (2022) for the details.

        Return
        -------
        estimated_policy_value_momentum: float
            Estimated exponential policy value (:math:`W_n`).

        """
        trajectory_wise_reward_momentum = trajectory_wise_reward ** momentum * np.exp(
            -trajectory_wise_reward / alpha
        )
        return (
            trajectory_wise_importance_weight * trajectory_wise_reward_momentum
        ).mean()

    def estimate_worst_case_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        delta: float = 0.05,
        alpha_prior: float = 1.0,
        max_steps: int = 100,
        epsilon: float = 0.01,
        **kwargs,
    ) -> float:
        """Estimate the worst case policy value in a distributionally robust manner.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        delta: float, default=0.05 (> 0)
            Allowance of the distributional shift.

        alpha_prior: float, default=1.0 (> 0)
            Initial temperature parameter of the exponential function.

        max_steps: int, default=100 (> 0)
            Maximum steps in turning alpha.

        epsilon: float, default=0.01
            Convergence criterion of alpha.

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(delta, name="delta", target_type=float, min_val=0.0)
        check_scalar(alpha_prior, name="alpha_prior", target_type=float, min_val=0.0)
        check_scalar(max_steps, name="max_steps", target_type=int, min_val=1)
        check_scalar(epsilon, name="epsilon", target_type=float, min_val=0.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_trajectory_wise_pscore,
            name="evaluation_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_trajectory_wise_pscore.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, but found False"
            )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )

        n = trajectory_wise_reward.shape[0]

        alpha = alpha_prior
        for _ in range(max_steps):
            W_0 = self._estimate_policy_value_momentum_given_alpha(
                trajectory_wise_reward=trajectory_wise_reward,
                trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                alpha=alpha,
                momentum=0,
            )
            W_1 = self._estimate_policy_value_momentum_given_alpha(
                trajectory_wise_reward=trajectory_wise_reward,
                trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                alpha=alpha,
                momentum=1,
            )
            W_2 = self._estimate_policy_value_momentum_given_alpha(
                trajectory_wise_reward=trajectory_wise_reward,
                trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                alpha=alpha,
                momentum=2,
            )
            objective = -alpha * (np.log(W_0) + delta)
            first_order_derivative = -W_1 / (alpha * n * W_0) - np.log(W_0) - delta
            second_order_derivative = W_1 ** 2 / (
                alpha ** 3 * n ** 2 * W_0 ** 2
            ) - W_2 / (alpha ** 3 * n * W_0)

            alpha_prior = alpha
            alpha = np.clip(
                alpha_prior - first_order_derivative / second_order_derivative,
                0,
                1 / delta,
            )

            if np.abs(alpha - alpha_prior) < epsilon:
                break

        return objective


@dataclass
class DiscreteDistributionallyRobustSelfNormalizedImportanceSampling(
    BaseDistributionallyRobustOffPolicyEstimator
):
    """Self Normalized Importance Sampling (SNIS) for estimating the worst case performance in a distributionally robust manner.

     Note
    -------
    SNIS estimates the worst case policy value using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    estimator_name: str, default="dr_snis"
        Name of the estimator.

    References
    -------
    Nathan Kallus, Xiaojie Mao, Kaiwen Wang, and Zhengyuan Zhou.
    "Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning.", 2022.

    Nian Si, Fan Zhang, Zhengyuan Zhou, and Jose Blanchet.
    "Distributional Robust Batch Contextual Bandits.", 2020.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "dr_snis"

    def __post_init__(self):
        self.action_type = "discrete"

    def _estimate_policy_value_momentum_given_alpha(
        self,
        trajectory_wise_reward: np.ndarray,
        trajectory_wise_importance_weight: np.ndarray,
        alpha: float,
        momentum: int,
    ):
        """Calculate exponential policy value given alpha.

        Parameters
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        alpha: float
            Temperature parameter of the exponential function.

        momentum: int
            When :math:`n` is given, :math:`W_n` is calculated.
            See Kallus et al. (2022) for the details.

        Return
        -------
        estimated_policy_value_momentum: float
            Estimated exponential policy value (:math:`W_n`).

        """
        trajectory_wise_reward_momentum = trajectory_wise_reward ** momentum * np.exp(
            -trajectory_wise_reward / alpha
        )
        return (
            trajectory_wise_importance_weight * trajectory_wise_reward_momentum
        ).sum() / trajectory_wise_importance_weight.sum()

    def estimate_worst_case_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
        delta: float = 0.05,
        alpha_prior: float = 1.0,
        max_steps: int = 100,
        epsilon: float = 0.01,
        **kwargs,
    ) -> float:
        """Estimate the worst case policy value in a distributionally robust manner.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        delta: float, default=0.05 (> 0)
            Allowance of the distributional shift.

        alpha_prior: float, default=1.0 (> 0)
            Initial temperature parameter of the exponential function.

        max_steps: int, default=100 (> 0)
            Maximum steps in turning alpha.

        epsilon: float, default=0.01
            Convergence criterion of alpha.

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(delta, name="delta", target_type=float, min_val=0.0)
        check_scalar(alpha_prior, name="alpha_prior", target_type=float, min_val=0.0)
        check_scalar(max_steps, name="max_steps", target_type=int, min_val=1)
        check_scalar(epsilon, name="epsilon", target_type=float, min_val=0.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_trajectory_wise_pscore,
            name="evaluation_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_trajectory_wise_pscore.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, but found False"
            )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )

        weight_sum = trajectory_wise_importance_weight.sum()

        alpha = alpha_prior
        for _ in range(max_steps):
            W_0 = self._estimate_policy_value_momentum_given_alpha(
                trajectory_wise_reward=trajectory_wise_reward,
                trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                alpha=alpha,
                momentum=0,
            )
            W_1 = self._estimate_policy_value_momentum_given_alpha(
                trajectory_wise_reward=trajectory_wise_reward,
                trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                alpha=alpha,
                momentum=1,
            )
            W_2 = self._estimate_policy_value_momentum_given_alpha(
                trajectory_wise_reward=trajectory_wise_reward,
                trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                alpha=alpha,
                momentum=2,
            )
            objective = -alpha * (np.log(W_0) + delta)
            first_order_derivative = (
                -W_1 / (alpha * weight_sum * W_0) - np.log(W_0) - delta
            )
            second_order_derivative = W_1 ** 2 / (
                alpha ** 3 * weight_sum ** 2 * W_0 ** 2
            ) - W_2 / (alpha ** 3 * weight_sum * W_0)

            alpha_prior = alpha
            alpha = np.clip(
                alpha_prior - first_order_derivative / second_order_derivative,
                0,
                1 / delta,
            )

            if np.abs(alpha - alpha_prior) < epsilon:
                break

        return objective


@dataclass
class DiscreteDistributionallyRobustDoublyRobust(
    BaseDistributionallyRobustOffPolicyEstimator
):
    """Doubly Robust (DR) for estimating the worst case performance in a distributionally robust manner.

     Note
    -------
    DR estimates the worst case policy value using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    baseline_estimator: BaseEstimator
        Baseline Estimator :math:`\hat{f}(\\cdot)`.

    alpha_prior: float, default=1.0 (> 0)
        Initial temperature parameter of the exponential function.

    max_steps: int, default=100 (> 0)
        Maximum steps in turning alpha.

    epsilon: float, default=0.01
        Convergence criterion of alpha.

    n_folds: int, default=3
        Number of folds in cross-fitting.

    estimator_name: str, default="dr_dr"
        Name of the estimator.

    References
    -------
    Nathan Kallus, Xiaojie Mao, Kaiwen Wang, and Zhengyuan Zhou.
    "Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning.", 2022.

    Nian Si, Fan Zhang, Zhengyuan Zhou, and Jose Blanchet.
    "Distributional Robust Batch Contextual Bandits.", 2020.

    Nathan Kallus, Xiaojie Mao, Masatoshi Uehara.
    "Localized Debiased Machine Learning: Efficient Inference on Quantile Treatment Effects and Beyond.", 2019.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    baseline_estimator: BaseEstimator
    n_folds: int = 3
    estimator_name: str = "dr_dr"

    def __post_init__(self):
        self.action_type = "discrete"
        check_scalar(self.n_folds, name="n_folds", target_type=int, min_val=1)

        if not isinstance(self.baseline_estimator, BaseEstimator):
            raise ValueError(
                "baseline_estimator must be a child class of BaseEstimator"
            )

    def _estimate_policy_value_momentum_by_snis(
        self,
        trajectory_wise_reward: np.ndarray,
        trajectory_wise_importance_weight: np.ndarray,
        alpha: float,
        momentum: int,
    ):
        """Calculate exponential policy value given alpha.

        Parameters
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        alpha: float
            Temperature parameter of the exponential function.

        momentum: int
            When :math:`n` is given, :math:`W_n` is calculated.
            See Kallus et al. (2022) for the details.

        Return
        -------
        estimated_policy_value_momentum: float
            Estimated exponential policy value (:math:`W_n`).

        """
        trajectory_wise_reward_momentum = trajectory_wise_reward ** momentum * np.exp(
            -trajectory_wise_reward / alpha
        )
        return (
            trajectory_wise_importance_weight * trajectory_wise_reward_momentum
        ).sum() / trajectory_wise_importance_weight.sum()

    def _initialize_alpha(
        self,
        trajectory_wise_reward: np.ndarray,
        trajectory_wise_importance_weight: np.ndarray,
        fit_episodes: np.ndarray,
        delta: float = 0.05,
        alpha_prior: float = 1.0,
        max_steps: int = 100,
        epsilon: float = 0.01,
    ):
        """Initialize alpha for each fold.

        Parameters
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        fit_episodes: NDArray, shape (n_folds, n_episodes // 2)
            Episodes used for fitting alpha.

        delta: float, default=0.05 (> 0)
            Allowance of the distributional shift.

        alpha_prior: float, default=1.0 (> 0)
            Initial temperature parameter of the exponential function.

        max_steps: int, default=100 (> 0)
            Maximum steps in turning alpha.

        epsilon: float, default=0.01
            Convergence criterion of alpha.

        Return
        -------
        Initial_alpha: NDArray, shape (n_folds, )
            Initial alpha for each fold.

        """
        alpha = np.zeros(self.n_folds)
        for k in self.n_folds:
            weight_sum_ = trajectory_wise_importance_weight[fit_episodes[k]].sum()

            alpha_ = alpha_prior
            for _ in range(max_steps):
                W_0 = self._estimate_policy_value_momentum_by_snis(
                    trajectory_wise_reward=trajectory_wise_reward[fit_episodes[k]],
                    trajectory_wise_importance_weight=trajectory_wise_importance_weight[
                        fit_episodes[k]
                    ],
                    alpha=alpha_,
                    momentum=0,
                )
                W_1 = self._estimate_policy_value_momentum_by_snis(
                    trajectory_wise_reward=trajectory_wise_reward[fit_episodes[k]],
                    trajectory_wise_importance_weight=trajectory_wise_importance_weight[
                        fit_episodes[k]
                    ],
                    alpha=alpha_,
                    momentum=1,
                )
                W_2 = self._estimate_policy_value_momentum_by_snis(
                    trajectory_wise_reward=trajectory_wise_reward[fit_episodes[k]],
                    trajectory_wise_importance_weight=trajectory_wise_importance_weight[
                        fit_episodes[k]
                    ],
                    alpha=alpha_,
                    momentum=2,
                )
                first_order_derivative_ = (
                    -W_1 / (alpha * weight_sum_ * W_0) - np.log(W_0) - delta
                )
                second_order_derivative_ = W_1 ** 2 / (
                    alpha ** 3 * weight_sum_ ** 2 * W_0 ** 2
                ) - W_2 / (alpha ** 3 * weight_sum_ * W_0)

                alpha_prior_ = alpha_
                alpha_ = np.clip(
                    alpha_prior_ - first_order_derivative_ / second_order_derivative_,
                    0,
                    1 / delta,
                )

                if np.abs(alpha_ - alpha_prior_) < epsilon:
                    break

            alpha[k] = alpha_

        return alpha

    def _predict_trajectory_wise_reward_momentum_given_initial_alpha(
        self,
        n_actions: int,
        initial_state: np.ndarray,
        initial_state_action: np.ndarray,
        trajectory_wise_reward: np.ndarray,
        initial_alpha: np.ndarray,
        train_episodes: np.ndarray,
        momentum: int,
    ):
        """Predict trajectory wise reward momentum given alpha (:math:`hat{f}(\\cdot; \\alpha)`).

        Parameters
        -------
        n_actions: int
            Number of the discrete actions.

        initial_state: NDArray, shape (n_episodes, state_dim)
            Initial state observed at each episode.

        initial_state_action: NDArray, shape (n_episodes, )
            Initial action chosen by the behavior policy.

        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        initial_alpha: NDArray, shape (n_folds, )
            Temperature parameter of the exponential function.

        train_episodes: NDArray, shape (n_folds, n_episode // 2)
            Episodes used for training the model.

        momentum: int
            When :math:`n` is given, :math:`W_n` is calculated.
            See Kallus et al. (2022) for the details.

        Return
        -------
        trajectory_wise_reward_prediction: NDArray, shape (n_folds, n_episodes, n_actions)
            Estimated trajectory wise reward (:math:`hat{f}(\\cdot; \\alpha)`).

        """
        n_episodes, state_dim = initial_state.shape

        # prediction set
        X_ = np.zeros(n_episodes * n_actions, state_dim + 1)
        X_[:, -1] = np.tile(np.arange(n_actions), n_episodes)
        for i in range(n_episodes):
            X_[n_actions * i : n_actions * (i + 1), :-1] = np.tile(
                initial_state[i], (n_actions, 1)
            )

        trajectory_wise_reward_prediction = np.zeros(
            (self.n_folds, n_episodes, n_actions)
        )
        for k in range(self.n_folds):
            # train
            X = np.concatenate(
                (
                    initial_state[train_episodes[k]],
                    initial_state_action[train_episodes[k], np.newaxis],
                ),
                axis=1,
            )
            y = trajectory_wise_reward[train_episodes[k]] ** momentum * np.exp(
                -trajectory_wise_reward[train_episodes[k]] / initial_alpha
            )
            self.baseline_estimator.fit(X, y)

            # prediction
            trajectory_wise_reward_prediction[k] = self.baseline_estimator.predict(
                X_
            ).reshape((-1, n_actions))

        return trajectory_wise_reward_prediction

    def _estimate_policy_value_momentum_given_alpha(
        self,
        trajectory_wise_reward: np.ndarray,
        trajectory_wise_importance_weight: np.ndarray,
        trajectory_wise_reward_prediction: np.ndarray,
        initial_state_action: np.ndarray,
        initial_state_action_distribution: np.ndarray,
        alpha: float,
        momentum: int,
    ):
        """Calculate exponential policy value given alpha.

        Parameters
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        trajectory_wise_reward_prediction: NDArray, shape (n_folds, n_episodes, n_actions)
            Estimated trajectory wise reward (:math:`hat{f}(\\cdot; \\alpha)`).

        initial_state_action: NDArray, shape (n_episodes, )
            Initial action chosen by the behavior policy.

        initial_state_action_distribution: NDArray, shape (n_episodes, n_actions)
            Evaluation policy pscore at the initial state of each episode.

        alpha: float
            Temperature parameter of the exponential function.

        momentum: int
            If 0 is given, return :math:`W_0`. If 1 is given, return :math:`W_1`.
            See Kallus et al. (2022) for the details.

        Return
        -------
        estimated_policy_value_momentum: float
            Estimated exponential policy value (:math:`W_0` or :math:`W_1`).

        """
        n_episodes = trajectory_wise_reward.shape[0]

        estimated_policy_value_momentum = np.zeros(self.n_folds)
        for k in self.n_folds:
            predicted_reward_for_the_taken_action_ = trajectory_wise_reward_prediction[
                k, np.arange(n_episodes), initial_state_action
            ]

            baseline_ = (
                initial_state_action_distribution[k]
                * trajectory_wise_reward_prediction[k]
            ).mean(axis=1)
            residual_ = (
                trajectory_wise_reward ** momentum
                * np.exp(-trajectory_wise_reward / alpha)
                - predicted_reward_for_the_taken_action_
            )

            estimated_policy_value_momentum[k] = (
                trajectory_wise_importance_weight * residual_ + baseline_
            ).mean()

        return estimated_policy_value_momentum.mean()

    def _estimate_policy_value_momentum_derivative_given_alpha(
        self,
        trajectory_wise_reward: np.ndarray,
        trajectory_wise_importance_weight: np.ndarray,
        alpha: float,
        momentum: int,
    ):
        """Calculate exponential policy value given alpha.

        Parameters
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        alpha: float
            Temperature parameter of the exponential function.

        momentum: int
            If 0 is given, return :math:`W_0`. If 1 is given, return :math:`W_1`.
            See Kallus et al. (2022) for the details.

        Return
        -------
        estimated_policy_value_momentum_derivative: float
            Estimated exponential policy value (:math:`W_0` or :math:`W_1`).

        """
        trajectory_wise_reward_momentum = trajectory_wise_reward ** (
            momentum + 1
        ) * np.exp(-trajectory_wise_reward / alpha)
        estimated_policy_value_momentum_derivative = (
            trajectory_wise_importance_weight * trajectory_wise_reward_momentum
        ).mean() / alpha ** 2
        return estimated_policy_value_momentum_derivative

    def estimate_worst_case_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        initial_state: np.ndarray,
        initial_state_action: np.ndarray,
        initial_state_action_distribution: np.ndarray,
        gamma: float = 1.0,
        delta: float = 0.05,
        alpha_prior: float = 1.0,
        max_steps: int = 100,
        epsilon: float = 0.01,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> float:
        """Estimate the worst case policy value in a distributionally robust manner.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state: NDArray, shape (n_episodes, state_dim)
            Initial state observed at each episode.

        initial_state_action: NDArray, shape (n_episodes, )
            Initial action chosen by the behavior policy.

        initial_state_action_distribution: NDArray, shape (n_episodes, n_actions)
            Evaluation policy pscore at the initial state of each episode.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        delta: float, default=0.05 (> 0)
            Allowance of the distributional shift.

        alpha_prior: float, default=1.0 (> 0)
            Initial temperature parameter of the exponential function.

        max_steps: int, default=100 (> 0)
            Maximum steps in turning alpha.

        epsilon: float, default=0.01
            Convergence criterion of alpha.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(delta, name="delta", target_type=float, min_val=0.0)
        check_scalar(alpha_prior, name="alpha_prior", target_type=float, min_val=0.0)
        check_scalar(max_steps, name="max_steps", target_type=int, min_val=1)
        check_scalar(epsilon, name="epsilon", target_type=float, min_val=0.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            evaluation_policy_trajectory_wise_pscore,
            name="evaluation_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
            max_val=1.0,
        )
        check_array(
            initial_state,
            name="initial_state",
            expected_dim=2,
        )
        check_array(
            initial_state_action,
            name="initial_state_action",
            expected_dim=1,
        )
        check_array(
            initial_state_action_distribution,
            name="initial_state_action_distribution",
            expected_dim=2,
            min_val=0.0,
            max_val=1.0,
        )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_trajectory_wise_pscore.shape[0]
            == initial_state.shape[0]
            == initial_state_action.shape[0]
            == initial_state_action_distribution[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]"
                "== initial_state.shape[0] == initial_state_action.shape[0] == initial_state_action_distribution.shape[0]`, but found False"
            )
        if random_state is None:
            raise ValueError("random_state must be given")
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )
        n_episodes = trajectory_wise_importance_weight.shape[0]

        fit_episodes = np.zeros((self.n_fold, n_episodes // 2))
        train_episodes = np.zeros((self.n_fold, n_episodes // 2))
        for k in range(self.n_folds):
            fit_episodes[k], train_episodes[k] = train_test_split(
                np.arange(n_episodes),
                test_size=n_episodes // 2,
                train_size=n_episodes // 2,
                random_state=random_state + k,
            )

        initial_alpha = self._initialize_alpha(
            trajectory_wise_reward=trajectory_wise_reward,
            trajectory_wise_importance_weight=trajectory_wise_importance_weight,
            fit_episodes=fit_episodes,
            delta=delta,
            alpha_prior=alpha_prior,
            max_steps=max_steps,
            epsilon=epsilon,
        )
        f_0 = self._predict_trajectory_wise_reward_momentum_given_initial_alpha(
            n_actions=initial_state_action_distribution.shape[1],
            initial_state=initial_state,
            initial_state_action=initial_state_action,
            trajectory_wise_reward=trajectory_wise_reward,
            initial_alpha=initial_alpha,
            train_episodes=train_episodes,
            momentum=0,
        )
        f_1 = self._predict_trajectory_wise_reward_momentum_given_initial_alpha(
            n_actions=initial_state_action_distribution.shape[1],
            initial_state=initial_state,
            initial_state_action=initial_state_action,
            trajectory_wise_reward=trajectory_wise_reward,
            initial_alpha=initial_alpha,
            train_episodes=train_episodes,
            momentum=0,
        )

        alpha = initial_alpha.mean()
        for _ in range(max_steps):
            W_0 = self._estimate_policy_value_momentum_given_alpha(
                trajectory_wise_reward=trajectory_wise_reward,
                trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                trajectory_wise_reward_prediction=f_0,
                initial_state_action=initial_state_action,
                initial_state_action_distribution=initial_state_action_distribution,
                alpha=alpha,
                momentum=0,
            )
            W_1 = self._estimate_policy_value_momentum_given_alpha(
                trajectory_wise_reward=trajectory_wise_reward,
                trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                trajectory_wise_reward_prediction=f_1,
                initial_state_action=initial_state_action,
                initial_state_action_distribution=initial_state_action_distribution,
                alpha=alpha,
                momentum=1,
            )
            W_0_derivative = (
                self._estimate_policy_value_momentum_derivative_given_alpha(
                    trajectory_wise_reward=trajectory_wise_reward,
                    trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                    alpha=alpha,
                    momentum=0,
                )
            )
            W_1_derivative = (
                self._estimate_policy_value_momentum_derivative_given_alpha(
                    trajectory_wise_reward=trajectory_wise_reward,
                    trajectory_wise_importance_weight=trajectory_wise_importance_weight,
                    alpha=alpha,
                    momentum=1,
                )
            )
            objective = -alpha * (np.log(W_0) + delta)
            momentum = -W_1 / (alpha * W_0) - np.log(W_0) - delta
            momentum_derivative = (
                -W_0_derivative / W_0
                - (alpha * W_1_derivative * W_0 - W_1 * (W_0 + alpha * W_0_derivative))
                / (alpha * W_0) ** 2
            )

            alpha_prior = alpha
            alpha = np.clip(alpha_prior - momentum / momentum_derivative, 0, 1 / delta)

            if np.abs(alpha - alpha_prior) < epsilon:
                break

        return objective
