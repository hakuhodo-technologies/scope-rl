"""Distributional Off-Policy Estimators for Continuous action."""
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.utils import check_scalar

from .estimators_base import (
    BaseCumulativeDistributionalOffPolicyEstimator,
)
from ..utils import check_array


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
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
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
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
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
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
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
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
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
        check_scalar(
            step_per_episode, name="step_per_episode", target_type=int, min_val=1
        )
        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
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
