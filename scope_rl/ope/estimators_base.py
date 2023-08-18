# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Abstract base class for Off-Policy Estimator."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, Dict, List

import numpy as np
from d3rlpy.preprocessing import ActionScaler

from ..utils import (
    gaussian_kernel,
    epanechnikov_kernel,
    triangular_kernel,
    cosine_kernel,
    uniform_kernel,
    estimate_confidence_interval_by_bootstrap,
    estimate_confidence_interval_by_hoeffding,
    estimate_confidence_interval_by_empirical_bernstein,
    estimate_confidence_interval_by_t_test,
)


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for (basic) OPE estimators.

    Imported as: :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Note
    -------
    This abstract base class also implements the following private methods.

    *abstract* _estimate_trajectory_value:
        Estimate the trajectory-wise expected reward.

    _calc_behavior_policy_pscore_discrete:
        Calculate the behavior policy pscore (action choice probability) in the case of discrete action spaces.

    _calc_behavior_policy_pscore_continuous:
        Calculate the behavior policy pscore (action choice probability) in the case of continuous action spaces.

    _calc_evaluation_policy_pscore_discrete:
        Calculate the evaluation policy pscore (action choice probability) in the case of discrete action spaces.

    _calc_similarity_weight:
        Calculate the similarity weight (for continuous action case) in the case of continuous action spaces.

    *property* _estimate_confidence_interval:
        Dictionary containing names and functions of ci methods.

        .. code-block:: python

            key: [
                bootstrap,
                hoeffding,
                bernstein,
                ttest,
            ]

    *property* _kernel_function:
        Dictionary containing names and functions of kernels.

        .. code-block:: python

            key: [
                gaussian,
                epanechnikov,
                triangular,
                cosine,
                uniform,
            ]

    """

    @abstractmethod
    def _estimate_trajectory_value(self) -> np.ndarray:
        """Estimate the trajectory-wise expected reward."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of the evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value."""
        raise NotImplementedError

    @property
    def _estimate_confidence_interval(self) -> Dict[str, Callable]:
        """Dictionary containing names and functions of ci methods."""
        return {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    @property
    def _kernel_function(self) -> Dict[str, Callable]:
        """Dictionary containing names and functions of kernels."""
        return {
            "gaussian": gaussian_kernel,
            "epanechnikov": epanechnikov_kernel,
            "triangular": triangular_kernel,
            "cosine": cosine_kernel,
            "uniform": uniform_kernel,
        }

    def _calc_behavior_policy_pscore_discrete(
        self,
        step_per_trajectory: int,
        pscore: np.ndarray,
        pscore_type: str,
    ):
        """Calculate the behavior policy pscore (action choice probability).

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        pscore_type: {"trajectory-wise", "step-wise"}
            Indicates whether to return trajectory-wise pscore or step-wise pscore.

        Return
        -------
        behavior_policy_trajectory_wise_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        behavior_policy_step_wise_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        """
        pscore = pscore.reshape((-1, step_per_trajectory))

        # step-wise pscore
        behavior_policy_pscore = np.cumprod(pscore, axis=1)

        if pscore_type == "trajectory_wise":
            behavior_policy_pscore = np.tile(
                behavior_policy_pscore[:, -1], (step_per_trajectory, 1)
            ).T

        return behavior_policy_pscore

    def _calc_behavior_policy_pscore_continuous(
        self,
        step_per_trajectory: int,
        pscore: np.ndarray,
        pscore_type: str,
    ):
        """Calculate the behavior policy pscore (action choice probability).

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        pscore_type: {"trajectory-wise", "step-wise"}
            Indicates whether to return trajectory-wise pscore or step-wise pscore.

        Return
        -------
        behavior_policy_trajectory_wise_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        behavior_policy_step_wise_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        """
        action_dim = pscore.shape[1]
        pscore = pscore.reshape((-1, step_per_trajectory, action_dim))

        # joint probability
        pscore = np.prod(pscore, axis=2)

        # step-wise pscore
        behavior_policy_pscore = np.cumprod(pscore, axis=1)

        if pscore_type == "trajectory_wise":
            behavior_policy_pscore = np.tile(
                behavior_policy_pscore[:, -1], (step_per_trajectory, 1)
            ).T

        return behavior_policy_pscore

    def _calc_evaluation_policy_pscore_discrete(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        pscore_type: str,
    ):
        """Calculate the evaluation policy pscore (action choice probability).

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        pscore_type: {"trajectory-wise", "step-wise"}
            Indicates whether to return trajectory-wise pscore or step-wise pscore.

        Return
        -------
        evaluation_policy_trajectory_wise_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Trajectory-wise action choice probability of the evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi(a_t \\mid s_t)`

        evaluation_policy_step_wise_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Step-wise action choice probability of the evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi(a_{t'} \\mid s_{t'})`

        """
        evaluation_policy_pscore = evaluation_policy_action_dist[
            np.arange(len(action)), action
        ].reshape((-1, step_per_trajectory))

        # step-wise pscore
        evaluation_policy_pscore = np.cumprod(evaluation_policy_pscore, axis=1)

        if pscore_type == "trajectory_wise":
            evaluation_policy_pscore = np.tile(
                evaluation_policy_pscore[:, -1], (step_per_trajectory, 1)
            ).T
        return evaluation_policy_pscore

    def _calc_similarity_weight(
        self,
        step_per_trajectory: int,
        action: np.ndarray,
        evaluation_policy_action: np.ndarray,
        pscore_type: str,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
    ):
        """Calculate the similarity weight.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        pscore_type: {"trajectory-wise", "step-wise"}
            Indicates whether to return trajectory-wise pscore or step-wise pscore.

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel function.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        trajectory_wise_similarity_weight: ndarray of shape (n_trajectories, step_per_trajectory)
            Trajectory-wise similarity weight between the action chosen by the behavior policy and that chosen by the evaluation policy,
            i.e., :math:`\\prod_{t'=0}^{T-1} K(\\pi(s_t), a_t)` where :math:`K(\\cdot, \\cdot)` is a kernel function.

        step_wise_similarity_weight: ndarray of shape (n_trajectories, step_per_trajectory)
            Step-wise similarity weight between the action chosen by the behavior policy and that chosen by the evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t K(\\pi(s_t), a_t)` where :math:`K(\\cdot, \\cdot)` is a kernel function.

        """
        if action_scaler is not None:
            evaluation_policy_action = action_scaler.transform_numpy(
                evaluation_policy_action
            )
            action = action_scaler.transform_numpy(action)

        similarity_weight = self._kernel_function[kernel](
            evaluation_policy_action,
            action,
            bandwidth=bandwidth,
        ).reshape((-1, step_per_trajectory))

        similarity_weight = np.cumprod(similarity_weight, axis=1)

        if pscore_type == "trajectory_wise":
            similarity_weight = np.tile(
                similarity_weight[:, -1], (step_per_trajectory, 1)
            ).T

        return similarity_weight


@dataclass
class BaseMarginalOPEEstimator(BaseOffPolicyEstimator):
    """Base class for OPE estimators with marginal importance sampling.

    Bases: :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.estimators_base.BaseMarginalOPEEstimator`

    Note
    -------
    This abstract base class also implements the following private methods.

    *abstract* _estimate_trajectory_value:
        Estimate the trajectory-wise expected reward.

    _calc_behavior_policy_pscore_discrete:
        Calculate the behavior policy pscore (action choice probability) in the case of discrete action spaces.

    _calc_behavior_policy_pscore_continuous:
        Calculate the behavior policy pscore (action choice probability) in the case of continuous action spaces.

    _calc_evaluation_policy_pscore_discrete:
        Calculate the evaluation policy pscore (action choice probability) in the case of discrete action spaces.

    _calc_similarity_weight:
        Calculate the similarity weight (for continuous action case) in the case of continuous action spaces.

    _calc_marginal_importance_weight(self):
        Calculate the marginal importance weight.
        (Specified either in :class:`BaseStateMarginalOffPolicyEstimator` or :class:`BaseStateActionMarginalOffPolicyEstimator`)

    *property* _estimate_confidence_interval:
        Dictionary containing names and functions of ci methods.

        .. code-block:: python

            key: [
                bootstrap,
                hoeffding,
                bernstein,
                ttest,
            ]

    """

    def _calc_behavior_policy_pscore_discrete(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        pscore: np.ndarray,
    ):
        """Calculate the behavior policy pscore (action choice probability).

        Parameters
        -------
        n_step_pdis: int (> 0)
            Number of previous steps considered when defining the step-wise importance weight.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        Return
        -------
        behavior_policy_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`
            (adjusted by n_step_pdis)

        """
        pscore = pscore.reshape((-1, step_per_trajectory))

        n_step_pscore = np.zeros_like(pscore)
        for t in range(step_per_trajectory):
            start_id = max(t - n_step_pdis + 1, 0)
            n_step_pscore[:, t] = pscore[:, start_id : t + 1].prod(axis=1)

        return n_step_pscore

    def _calc_behavior_policy_pscore_continuous(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        pscore: np.ndarray,
    ):
        """Calculate the behavior policy pscore (action choice probability).

        Parameters
        -------
        n_step_pdis: int (> 0)
            Number of previous steps considered when defining the step-wise importance weight.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        Return
        -------
        behavior_policy_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`
            (adjusted by n_step_pdis)

        """
        pscore = pscore.prod(axis=1)
        pscore = pscore.reshape((-1, step_per_trajectory))

        n_step_pscore = np.zeros_like(pscore)
        for t in range(step_per_trajectory):
            start_id = max(t - n_step_pdis + 1, 0)
            n_step_pscore[:, t] = pscore[:, start_id : t + 1].prod(axis=1)

        return n_step_pscore

    def _calc_evaluation_policy_pscore_discrete(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        """Calculate the evaluation policy pscore (action choice probability).

        Parameters
        -------
        n_step_pdis: int (> 0)
            Number of previous steps considered when defining the step-wise importance weight.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        Return
        -------
        evaluation_policy_pscore: array-like of shape (n_trajectories, step_per_trajectory)
            Step-wise action choice probability of the evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi(a_{t'} \\mid s_{t'})`
            (adjusted by n_step_pdis)

        """
        evaluation_policy_pscore = evaluation_policy_action_dist[
            np.arange(len(action)), action
        ].reshape((-1, step_per_trajectory))

        n_step_pscore = np.zeros_like(evaluation_policy_pscore)
        for t in range(step_per_trajectory):
            start_id = max(t - n_step_pdis + 1, 0)
            n_step_pscore[:, t] = evaluation_policy_pscore[:, start_id : t + 1].prod(
                axis=1
            )

        return n_step_pscore

    def _calc_similarity_weight(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        action: np.ndarray,
        evaluation_policy_action: np.ndarray,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
    ):
        """Calculate the similarity weight.

        Parameters
        -------
        n_step_pdis: int (> 0)
            Number of previous steps considered when defining the step-wise importance weight.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        similarity_weight: ndarray of shape (n_trajectories, step_per_trajectory)
            Similarity weight between the action chosen by the behavior policy and that chosen by the evaluation policy,
            i.e., :math:`\\prod_{t'=0}^t K(\\pi(s_t), a_t)` where :math:`K(\\cdot, \\cdot)` is a kernel function.
            (adjusted by n_step_pdis)

        """
        if action_scaler is not None:
            evaluation_policy_action = action_scaler.transform_numpy(
                evaluation_policy_action
            )
            action = action_scaler.transform_numpy(action)

        similarity_weight = self._kernel_function[kernel](
            evaluation_policy_action,
            action,
            bandwidth=bandwidth,
        ).reshape((-1, step_per_trajectory))

        n_step_similarity_weight = np.zeros_like(similarity_weight)
        for t in range(step_per_trajectory):
            start_id = max(t - n_step_pdis + 1, 0)
            n_step_similarity_weight[:, t] = similarity_weight[
                :, start_id : t + 1
            ].prod(axis=1)

        return n_step_similarity_weight

    def _calc_marginal_importance_weight(self):
        """Calculate the marginal importance weight."""
        raise NotImplementedError


@dataclass
class BaseStateMarginalOPEEstimator(BaseMarginalOPEEstimator):
    """Base class for State Marginal OPE estimators.

    Bases: :class:`scope_rl.ope.BaseMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.BaseStateMarginalOPEEstimator`

    Note
    -------
    This abstract base class also implements the following private methods.

    *abstract* _estimate_trajectory_value:
        Estimate the trajectory-wise expected reward.

    _calc_behavior_policy_pscore_discrete:
        Calculate the behavior policy pscore (action choice probability) in the case of discrete action spaces.

    _calc_behavior_policy_pscore_continuous:
        Calculate the behavior policy pscore (action choice probability) in the case of continuous action spaces.

    _calc_evaluation_policy_pscore_discrete:
        Calculate the evaluation policy pscore (action choice probability) in the case of discrete action spaces.

    _calc_similarity_weight:
        Calculate the similarity weight (for continuous action case) in the case of continuous action spaces.

    _calc_marginal_importance_weight(self):
        Calculate the marginal importance weight.

    *property* _estimate_confidence_interval:
        Dictionary containing names and functions of ci methods.

        .. code-block:: python

            key: [
                bootstrap,
                hoeffding,
                bernstein,
                ttest,
            ]

    """

    def _calc_marginal_importance_weight(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        state_marginal_importance_weight: np.ndarray,
    ):
        """Calculate the marginal importance weight.

        Parameters
        -------
        n_step_pdis: int (> 0)
            Number of previous steps considered when defining the step-wise importance weight.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state marginal distribution, i.e., :math:`d_{\\pi}(s) / d_{\\pi_b}(s)`

        Return
        -------
        state_marginal_importance_weight: ndarray of shape (n_trajectories, step_per_trajectory)
            Marginal importance weight adjusted by n_step_pdis.

        """
        state_marginal_importance_weight = state_marginal_importance_weight.reshape(
            (-1, step_per_trajectory)
        )
        state_marginal_importance_weight = np.roll(
            state_marginal_importance_weight, n_step_pdis, axis=1
        )
        state_marginal_importance_weight[:, :n_step_pdis] = 1

        return state_marginal_importance_weight


@dataclass
class BaseStateActionMarginalOPEEstimator(BaseMarginalOPEEstimator):
    """Base class for State-Action Marginal OPE estimators.

    Bases: :class:`scope_rl.ope.BaseMarginalOPEEstimator` -> :class:`scope_rl.ope.BaseOffPolicyEstimator`

    Imported as: :class:`scope_rl.ope.BaseStateActionMarginalOPEEstimator`

    Note
    -------
    This abstract base class also implements the following private methods.

    *abstract* _estimate_trajectory_value:
        Estimate the trajectory-wise expected reward.

    _calc_behavior_policy_pscore_discrete:
        Calculate the behavior policy pscore (action choice probability) in the case of discrete action spaces.

    _calc_behavior_policy_pscore_continuous:
        Calculate the behavior policy pscore (action choice probability) in the case of continuous action spaces.

    _calc_evaluation_policy_pscore_discrete:
        Calculate the evaluation policy pscore (action choice probability) in the case of discrete action spaces.

    _calc_similarity_weight:
        Calculate the similarity weight (for continuous action case) in the case of continuous action spaces.

    _calc_marginal_importance_weight(self):
        Calculate the marginal importance weight.

    *property* _estimate_confidence_interval:
        Dictionary containing names and functions of ci methods.

        .. code-block:: python

            key: [
                bootstrap,
                hoeffding,
                bernstein,
                ttest,
            ]

    """

    def _calc_marginal_importance_weight(
        self,
        n_step_pdis: int,
        step_per_trajectory: int,
        state_action_marginal_importance_weight: np.ndarray,
    ):
        """Calculate the marginal importance weight.

        Parameters
        -------
        n_step_pdis: int (> 0)
            Number of previous steps considered when defining the step-wise importance weight.

        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        state_action_marginal_importance_weight: array-like of shape (n_trajectories * step_per_trajectory, )
            Importance weight wrt the state-action marginal distribution, i.e., :math:`d_{\\pi}(s, a) / d_{\\pi_b}(s, a)`

        Return
        -------
        state_action_marginal_importance_weight: ndarray of shape (n_trajectories, step_per_trajectory)
            Marginal importance weight adjusted by n_step_pdis.

        """
        state_action_marginal_importance_weight = (
            state_action_marginal_importance_weight.reshape((-1, step_per_trajectory))
        )
        state_action_marginal_importance_weight = np.roll(
            state_action_marginal_importance_weight, n_step_pdis, axis=1
        )
        state_action_marginal_importance_weight[:, :n_step_pdis] = 1

        return state_action_marginal_importance_weight


@dataclass
class BaseCumulativeDistributionOPEEstimator(metaclass=ABCMeta):
    """Base class for Cumulative Distribution OPE estimators.

    Imported as: :class:`scope_rl.ope.BaseCumulativeDistributionOPEEstimator`

    Note
    -------
    This abstract base class also implements the following private methods.

    _aggregate_trajectory_wise_statistics_discrete:
        Calculate trajectory-wise summary statistics based on step-wise observations in the case of discrete action spaces.

    _aggregate_trajectory_wise_statistics_continuous:
        Calculate trajectory-wise summary statistics based on step-wise observations in the case of continuous action spaces.

    _target_value_given_idx:
        Obtain the reward value corresponding to the given idx when estimating the CDF.

    *property* _kernel_function:
        Dictionary containing names and functions of kernels.

        .. code-block:: python

            key: [
                gaussian,
                epanechnikov,
                triangular,
                cosine,
                uniform,
            ]

    """

    @abstractmethod
    def estimate_cumulative_distribution_function(self) -> Tuple[np.ndarray]:
        """Estimate the cumulative distribution function (CDF) of the policy value."""
        raise NotImplementedError

    @abstractmethod
    def estimate_mean(self) -> float:
        """Estimate the mean of the reward under the evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_variance(self) -> float:
        """Estimate the variance of the reward under the evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_conditional_value_at_risk(self) -> float:
        """Estimate the conditional value at risk (CVaR) of the reward under the evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interquartile_range(self) -> Dict[str, float]:
        """Estimate the interquartile range of the reward under the evaluation policy."""
        raise NotImplementedError

    @property
    def _kernel_function(self) -> Dict[str, Callable]:
        """Dictionary containing names and functions of kernels."""
        return {
            "gaussian": gaussian_kernel,
            "epanechnikov": epanechnikov_kernel,
            "triangular": triangular_kernel,
            "cosine": cosine_kernel,
            "uniform": uniform_kernel,
        }

    def _target_value_given_idx(
        self, idx_: Union[List[int], int], reward_scale: np.ndarray
    ):
        """Obtain the reward value corresponding to the given idx when estimating the CDF.

        Parameters
        -------
        idx_: list of int or int
            Indicating index. When list is given, the average of the two will be returned.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory-wise reward used for x-axis of the CDF plot.

        Return
        -------
        target_value: float
            Value of the given index.

        """
        if len(idx_) == 0 or idx_[0] == len(reward_scale) - 1:
            target_value = reward_scale[-1]
        else:
            target_idx = idx_[0]
            target_value = (reward_scale[target_idx] + reward_scale[target_idx + 1]) / 2
        return target_value

    def _aggregate_trajectory_wise_statistics_discrete(
        self,
        step_per_trajectory: int,
        action: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None,
        pscore: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        state_action_value_prediction: Optional[np.ndarray] = None,
        gamma: float = 1.0,
    ):
        """Calculate trajectory-wise summary statistics based on step-wise observations in the case of discrete action spaces.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_trajectories * step_per_trajectory, )
            Action chosen by the behavior policy.

        reward: ndarray of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, )
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action_dist: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            Conditional action distribution induced by the evaluation policy,
            i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, n_action)
            :math:`\\hat{Q}` for all actions, i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        Return
        -------
        trajectory_wise_reward: ndarray of shape (n_trajectories, )
            Trajectory-wise reward observed under the behavior policy.

        trajectory_wise_importance_weight: ndarray of shape (n_trajectories, )
            Trajectory-wise importance weight.

        initial_state_value_prediction: ndarray of shape (n_trajectories, )
            Estimated initial state value.

        """
        trajectory_wise_importance_weight = None
        trajectory_wise_reward = None
        initial_state_value_prediction = None

        if reward is not None:
            reward = reward.reshape((-1, step_per_trajectory))
            discount = np.full(reward.shape[1], gamma).cumprod()
            trajectory_wise_reward = (reward * discount).sum(axis=1)

        if (
            action is not None
            and pscore is not None
            and evaluation_policy_action_dist is not None
        ):
            pscore = pscore.reshape((-1, step_per_trajectory))
            behavior_policy_pscore = np.cumprod(pscore, axis=1)[:, -1]

            evaluation_policy_pscore = evaluation_policy_action_dist[
                np.arange(len(action)), action
            ].reshape((-1, step_per_trajectory))

            evaluation_policy_pscore = np.cumprod(evaluation_policy_pscore, axis=1)[
                :, -1
            ]

            trajectory_wise_importance_weight = (
                evaluation_policy_pscore / behavior_policy_pscore
            )

        if (
            evaluation_policy_action_dist is not None
            and state_action_value_prediction is not None
        ):
            initial_state_value_prediction = (
                (state_action_value_prediction * evaluation_policy_action_dist)
                .sum(axis=1)
                .reshape((-1, step_per_trajectory))[:, 0]
            )

        return (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        )

    def _aggregate_trajectory_wise_statistics_continuous(
        self,
        step_per_trajectory: int,
        action: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None,
        pscore: Optional[np.ndarray] = None,
        evaluation_policy_action: Optional[np.ndarray] = None,
        state_action_value_prediction: Optional[np.ndarray] = None,
        gamma: float = 1.0,
        kernel: str = "gaussian",
        bandwidth: float = 1.0,
        action_scaler: Optional[ActionScaler] = None,
    ):
        """Calculate trajectory-wise summary statistics based on step-wise observations in the case of continuous action spaces.

        Parameters
        -------
        step_per_trajectory: int (> 0)
            Number of timesteps in an episode.

        action: ndarray of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the behavior policy.

        reward: ndarray of shape (n_trajectories * step_per_trajectory, )
            Observed immediate rewards.

        pscore: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Conditional action choice probability of the behavior policy,
            i.e., :math:`\\pi_0(a \\mid s)`

        evaluation_policy_action: array-like of shape (n_trajectories * step_per_trajectory, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_trajectories * step_per_trajectory, 2)
            :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within (0, 1].

        kernel: {"gaussian", "epanechnikov", "triangular", "cosine", "uniform"}
            Name of the kernel function to smooth importance weights.

        bandwidth: float, default=1.0 (> 0)
            Bandwidth hyperparameter of the kernel.

        action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
            Scaling factor of action.

        Return
        -------
        trajectory_wise_reward: ndarray of shape (n_trajectories, )
            Trajectory-wise reward observed under the behavior policy.

        trajectory_wise_importance_weight: ndarray of shape (n_trajectories, )
            Trajectory-wise importance weight.

        initial_state_value_prediction: ndarray of shape (n_trajectories, )
            Estimated initial state value.

        """
        trajectory_wise_importance_weight = None
        trajectory_wise_reward = None
        initial_state_value_prediction = None

        if reward is not None:
            reward = reward.reshape((-1, step_per_trajectory))
            discount = np.full(reward.shape[1], gamma).cumprod()
            trajectory_wise_reward = (reward * discount).sum(axis=1)

        if (
            action is not None
            and pscore is not None
            and evaluation_policy_action is not None
        ):
            action_dim = action.shape[1]
            pscore = pscore.reshape((-1, step_per_trajectory, action_dim))
            behavior_policy_pscore = pscore.prod(axis=1).prod(axis=1)

            if action_scaler is not None:
                evaluation_policy_action = action_scaler.transform_numpy(
                    evaluation_policy_action
                )
                action = action_scaler.transform_numpy(action)

            similarity_weight = (
                self._kernel_function[kernel](
                    evaluation_policy_action,
                    action,
                    bandwidth=bandwidth,
                )
                .reshape((-1, step_per_trajectory))
                .prod(axis=1)
            )

            trajectory_wise_importance_weight = (
                similarity_weight / behavior_policy_pscore
            )

        if state_action_value_prediction is not None:
            initial_state_value_prediction = state_action_value_prediction.reshape(
                (-1, step_per_trajectory, 2)
            )[:, 0, 1]

        return (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        )
