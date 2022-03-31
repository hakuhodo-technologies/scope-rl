"""Distributional Off-Policy Estimators for Continuous action."""
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import check_scalar

from offlinegym.ope.estimators_base import (
    BaseCumulativeDistributionalOffPolicyEstimator,
    BaseDistributionallyRobustOffPolicyEstimator,
)
from offlinegym.utils import check_array


@dataclass
class ContinuousCumulativeDistributionalDirectMethod(
    BaseCumulativeDistributionalOffPolicyEstimator,
):
    """Direct Method (DM) for estimating cumulative distribution function (CDF) in continuous OPE.

    Note
    -------
    DM estimates CDF using initial state value given by Fitted Q Evaluation (FQE) as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    estimator_name: str, default="cdf_dm"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Alina Beygelzimer and John Langford.
    "The Offset Tree for Learning with Partial Labels.", 2009.

    """

    estimator_name: str = "cdf_dm"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_cumulative_distribution_function: NDArray, shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] // step_per_episode != initial_state_value_prediction:
            raise ValueError(
                "Expected `reward.shape[0] // step_per_episode == initial_state_value_prediction`, but found False"
            )

        density = np.histgram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0]

        return np.insert(density, 0, 0).cumsum()

    def estimate_mean(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )

        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()

        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        estimated_interquartile_range: float
            Estimated conditional value at risk (cVaR) of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        lower_idx = np.argmin(cumulative_density > alpha)

        return (np.diff(cumulative_density) * reward_scale[1:])[: lower_idx + 1].sum()

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        estimated_interquartile_range: Dict[str, float]
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        lower_idx = np.argmin(cumulative_density > alpha)
        upper_idx = np.argmin(cumulative_density > 1 - alpha)

        estimated_interquartile_range = {
            "mean": mean,
            f"{100 * (1. - alpha)}% quartile (lower)": (
                reward_scale[lower_idx] + reward_scale[lower_idx + 1]
            )
            / 2,
            f"{100 * (1. - alpha)}% quartile (upper)": (
                reward_scale[upper_idx] + reward_scale[upper_idx + 1]
            )
            / 2,
        }

        return estimated_interquartile_range


@dataclass
class ContinuousCumulativeDistributionalImportanceSampling(
    BaseCumulativeDistributionalOffPolicyEstimator,
):
    """Importance Sampling (IS) for estimating cumulative distribution function (CDF) in continuous OPE.

    Note
    -------
    IS estimates CDF using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    estimator_name: str, default="cdf_is"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "cdf_is"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_cumulative_distribution_function: NDArray, shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )
        if sigma is not None:
            check_array(sigma, name="sigma", expected_dim=1, min_val=0.0)

            if not action.shape[1] == sigma.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == sigma.shape[0]`, but found False"
                )

        if use_truncated_kernel:
            check_array(action_min, name="action_min", expected_dim=1)
            check_array(action_max, name="action_max", expected_dim=1)

            if not action.shape[1] == action_min.shape[0] == action_max.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == action_min.shape[0] == action_max.shape[0]`, but found False"
                )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )

        n = len(trajectory_wise_reward)

        sort_idxes = trajectory_wise_reward.argsort()
        sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
        cumulative_density = np.clip(sorted_importance_weight.cumsum() / n, 0, 1)

        histogram = np.histgram(
            trajectory_wise_reward, bins=reward_scale, density=True
        )[0]
        cumulative_density = cumulative_density[histogram.cumsum()]

        return np.insert(cumulative_density, 0, 0)

    def estimate_mean(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            reward_scale=reward_scale,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )

        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            reward_scale=reward_scale,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()

        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05
            Proportion of the sided region.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_interquartile_range: float
            Estimated conditional value at risk (cVaR) of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            reward_scale=reward_scale,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )
        lower_idx = np.argmin(cumulative_density > alpha)

        return (np.diff(cumulative_density) * reward_scale[1:])[: lower_idx + 1].sum()

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05
            Proportion of the sided region.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_interquartile_range: Dict[str, float]
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            reward_scale=reward_scale,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        lower_idx = np.argmin(cumulative_density > alpha)
        upper_idx = np.argmin(cumulative_density > 1 - alpha)

        estimated_interquartile_range = {
            "mean": mean,
            f"{100 * (1. - alpha)}% quartile (lower)": (
                reward_scale[lower_idx] + reward_scale[lower_idx + 1]
            )
            / 2,
            f"{100 * (1. - alpha)}% quartile (upper)": (
                reward_scale[upper_idx] + reward_scale[upper_idx + 1]
            )
            / 2,
        }

        return estimated_interquartile_range


@dataclass
class ContinuousCumulativeDistributionalDoublyRobust(
    BaseCumulativeDistributionalOffPolicyEstimator,
):
    """Doubly Robust (DR) for estimating cumulative distribution function (CDF) in continuous OPE.

    Note
    -------
    DR estimates CDF using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    estimator_name: str, default="cdf_dr"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    use_observations_as_reward_scale: bool = False
    estimator_name: str = "cdf_dr"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_cumulative_distribution_function: NDArray, shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )
        if sigma is not None:
            check_array(sigma, name="sigma", expected_dim=1, min_val=0.0)

            if not action.shape[1] == sigma.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == sigma.shape[0]`, but found False"
                )

        if use_truncated_kernel:
            check_array(action_min, name="action_min", expected_dim=1)
            check_array(action_max, name="action_max", expected_dim=1)

            if not action.shape[1] == action_min.shape[0] == action_max.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == action_min.shape[0] == action_max.shape[0]`, but found False"
                )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )

        weighted_residual = np.zeros_like(reward_scale)
        for threshold in reward_scale:
            observation = trajectory_wise_reward <= threshold
            prediction = initial_state_value_prediction <= threshold
            weighted_residual[threshold] = (
                trajectory_wise_importance_weight * (observation - prediction)
            ).mean()

        histogram_baseline = np.histgram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0].cumsum()
        histogram_baseline = np.insert(histogram_baseline, 0, 0)

        cumulative_density = weighted_residual + histogram_baseline

        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1)

    def estimate_mean(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate mean.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )

        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate variance.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()

        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Estimate conditional value at risk.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05
            Proportion of the sided region.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_interquartile_range: float
            Estimated conditional value at risk (cVaR) of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )
        lower_idx = np.argmin(cumulative_density > alpha)

        return (np.diff(cumulative_density) * reward_scale[1:])[: lower_idx + 1].sum()

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        alpha: float = 0.05,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate interquartile range.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05
            Proportion of the sided region.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_interquartile_range: Dict[str, float]
            Estimated interquartile range of the policy value.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)

        cumulative_density = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            initial_state_value_prediction=initial_state_value_prediction,
            reward_scale=reward_scale,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )
        mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
        lower_idx = np.argmin(cumulative_density > alpha)
        upper_idx = np.argmin(cumulative_density > 1 - alpha)

        estimated_interquartile_range = {
            "mean": mean,
            f"{100 * (1. - alpha)}% quartile (lower)": (
                reward_scale[lower_idx] + reward_scale[lower_idx + 1]
            )
            / 2,
            f"{100 * (1. - alpha)}% quartile (upper)": (
                reward_scale[upper_idx] + reward_scale[upper_idx + 1]
            )
            / 2,
        }

        return estimated_interquartile_range


@dataclass
class ContinuousCumulativeDistributionalSelfNormalizedImportanceSampling(
    ContinuousCumulativeDistributionalImportanceSampling,
):
    """Self Normalized Importance Sampling (SNIS) for estimating cumulative distribution function (CDF) in continuous OPE.

    Note
    -------
    SNIS estimates CDF using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    estimator_name: str, default="cdf_is"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "cdf_is"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_cumulative_distribution_function: NDArray, shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )
        if sigma is not None:
            check_array(sigma, name="sigma", expected_dim=1, min_val=0.0)

            if not action.shape[1] == sigma.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == sigma.shape[0]`, but found False"
                )

        if use_truncated_kernel:
            check_array(action_min, name="action_min", expected_dim=1)
            check_array(action_max, name="action_max", expected_dim=1)

            if not action.shape[1] == action_min.shape[0] == action_max.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == action_min.shape[0] == action_max.shape[0]`, but found False"
                )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )

        weight_sum = trajectory_wise_importance_weight.sum()

        sort_idxes = trajectory_wise_reward.argsort()
        sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
        cumulative_density = np.clip(
            sorted_importance_weight.cumsum() / weight_sum, 0, 1
        )

        histogram = np.histgram(
            trajectory_wise_reward, bins=reward_scale, density=True
        )[0]
        cumulative_density = cumulative_density[histogram.cumsum()]

        return np.insert(cumulative_density, 0, 0)


@dataclass
class ContinuousCumulativeDistributionalSelfNormalizedDoublyRobust(
    ContinuousCumulativeDistributionalDoublyRobust,
):
    """Self Normalized Doubly Robust (SNDR) for estimating cumulative distribution function (CDF) in continuous OPE.

    Note
    -------
    SNDR estimates CDF using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    estimator_name: str, default="cdf_dr"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "cdf_dr"

    def __post_init__(self):
        self.action_type = "continuous"

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        initial_state_value_prediction: np.ndarray,
        reward_scale: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Estimate cumulative distribution function.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        reward_scale: NDArray, shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_cumulative_distribution_function: NDArray, shape (n_partition, ) or (n_episode, )
            Estimated cumulative distribution function for the pre-defined reward scale.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        check_array(
            reward_scale,
            name="reward_scale",
            expected_dim=1,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )
        if sigma is not None:
            check_array(sigma, name="sigma", expected_dim=1, min_val=0.0)

            if not action.shape[1] == sigma.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == sigma.shape[0]`, but found False"
                )

        if use_truncated_kernel:
            check_array(action_min, name="action_min", expected_dim=1)
            check_array(action_max, name="action_max", expected_dim=1)

            if not action.shape[1] == action_min.shape[0] == action_max.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == action_min.shape[0] == action_max.shape[0]`, but found False"
                )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )

        weighted_residual = np.zeros_like(reward_scale)
        for threshold in reward_scale:
            observation = trajectory_wise_reward <= threshold
            prediction = initial_state_value_prediction <= threshold
            weighted_residual[threshold] = (
                trajectory_wise_importance_weight * (observation - prediction)
            ).sum() / trajectory_wise_importance_weight.sum()

        histogram_baseline = np.histgram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0].cumsum()
        histogram_baseline = np.insert(histogram_baseline, 0, 0)

        cumulative_density = weighted_residual + histogram_baseline

        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1)


@dataclass
class ContinuousDistributionallyRobustImportanceSampling(
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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "dr_is"

    def __post_init__(self):
        self.action_type = "continuous"

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
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        delta: float = 0.05,
        alpha_prior: float = 1.0,
        max_steps: int = 100,
        epsilon: float = 0.01,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the worst case policy value in a distributionally robust manner.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_scalar(delta, name="delta", type=float, min_val=0.0)
        check_scalar(alpha_prior, name="alpha_prior", type=float, min_val=0.0)
        check_scalar(max_steps, name="max_steps", type=int, min_val=1)
        check_scalar(epsilon, name="epsilon", type=float, min_val=0.0)
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )
        if sigma is not None:
            check_array(sigma, name="sigma", expected_dim=1, min_val=0.0)

            if not action.shape[1] == sigma.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == sigma.shape[0]`, but found False"
                )

        if use_truncated_kernel:
            check_array(action_min, name="action_min", expected_dim=1)
            check_array(action_max, name="action_max", expected_dim=1)

            if not action.shape[1] == action_min.shape[0] == action_max.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == action_min.shape[0] == action_max.shape[0]`, but found False"
                )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
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
class ContinuousDistributionallyRobustSelfNormalizedImportanceSampling(
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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        self.action_type = "continuous"

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
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        delta: float = 0.05,
        alpha_prior: float = 1.0,
        max_steps: int = 100,
        epsilon: float = 0.01,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the worst case policy value in a distributionally robust manner.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_scalar(delta, name="delta", type=float, min_val=0.0)
        check_scalar(alpha_prior, name="alpha_prior", type=float, min_val=0.0)
        check_scalar(max_steps, name="max_steps", type=int, min_val=1)
        check_scalar(epsilon, name="epsilon", type=float, min_val=0.0)
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_trajectory_wise_pscore.shape[0]`, but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )
        if sigma is not None:
            check_array(sigma, name="sigma", expected_dim=1, min_val=0.0)

            if not action.shape[1] == sigma.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == sigma.shape[0]`, but found False"
                )

        if use_truncated_kernel:
            check_array(action_min, name="action_min", expected_dim=1)
            check_array(action_max, name="action_max", expected_dim=1)

            if not action.shape[1] == action_min.shape[0] == action_max.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == action_min.shape[0] == action_max.shape[0]`, but found False"
                )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
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
class ContinuousDistributionallyRobustDoublyRobust(
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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

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
        self.action_type = "continuous"
        check_scalar(self.n_folds, name="n_folds", type=int, min_val=1)

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
            Number of the continuous actions.

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
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        initial_state: np.ndarray,
        initial_state_action: np.ndarray,
        initial_state_action_distribution: np.ndarray,
        gamma: float = 1.0,
        delta: float = 0.05,
        alpha_prior: float = 1.0,
        max_steps: int = 100,
        epsilon: float = 0.01,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> float:
        """Estimate the worst case policy value in a distributionally robust manner.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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

        sigma: Optional[NDArray], shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., band_width hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: Optional[NDArray], shape (action_dim, ), default=None
            Minimum value of action vector.
            When use_truncated_kernel == True, action_min must be given.

        action_max: Optional[NDArray], shape (action_dim, ), default=None
            Maximum value of action vector.
            When use_truncated_kernel == True, action_max must be given.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_scalar(delta, name="delta", type=float, min_val=0.0)
        check_scalar(alpha_prior, name="alpha_prior", type=float, min_val=0.0)
        check_scalar(max_steps, name="max_steps", type=int, min_val=1)
        check_scalar(epsilon, name="epsilon", type=float, min_val=0.0)
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(reward, name="reward", expected_dim=1)
        check_array(
            behavior_policy_trajectory_wise_pscore,
            name="behavior_policy_trajectory_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
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
        if action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if behavior_policy_trajectory_wise_pscore.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `behavior_policy_trajectory_wise_pscore.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if evaluation_policy_action.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `evaluation_policy_action.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if not (
            action.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
            == initial_state.shape[0]
            == initial_state_action.shape[0]
            == initial_state_action_distribution[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_action.shape[0] =="
                " initial_state.shape[0] == initial_state_action.shape[0] == initial_state_action_distribution.shape[0]`, but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )
        if sigma is not None:
            check_array(sigma, name="sigma", expected_dim=1, min_val=0.0)

            if not action.shape[1] == sigma.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == sigma.shape[0]`, but found False"
                )

        if use_truncated_kernel:
            check_array(action_min, name="action_min", expected_dim=1)
            check_array(action_max, name="action_max", expected_dim=1)

            if not action.shape[1] == action_min.shape[0] == action_max.shape[0]:
                raise ValueError(
                    "Expected `action.shape[1] == action_min.shape[0] == action_max.shape[0]`, but found False"
                )

        if random_state is None:
            raise ValueError("random_state must be given")
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_continuous(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
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
