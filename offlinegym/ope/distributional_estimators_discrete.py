"""Distributional Off-Policy Estimators for Discrete action."""
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.utils import check_scalar
import torch
from torch import optim

from offlinegym.ope.estimators_base import (
    BaseCumulativeDistributionalOffPolicyEstimator,
    BaseDistributionallyRobustOffPolicyEstimator,
)
from offlinegym.utils import Optimizer, check_array


@dataclass
class DiscreteCumulativeDistributionalDirectMethod(
    BaseCumulativeDistributionalOffPolicyEstimator,
):
    """Direct Method (DM) for estimating cumulative distribution function (CDF) in discrete OPE.

    Note
    -------
    DM estimates CDF using initial state value given by Fitted Q Evaluation (FQE) as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    r_min: float, default=None
        Minimum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    r_max: float, default=None
        Maximum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    n_partitiion: int, default=None
        Number of partition in reward scale (x-axis of CDF).
        When `use_observations_as_reward_scale == False`, a value must be given.

    use_observations_as_reward_scale: bool, default=False
        Whether to use the reward observed by the behavior policy as the reward scale.
        If True, the reward scale follows the one defined in Chundak et al. (2021).
        If False, the reward scale is uniform, following Huang et al. (2021).

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

    r_min: Optional[float] = None
    r_max: Optional[float] = None
    n_partition: Optional[int] = None
    use_observations_as_reward_scale: bool = False
    estimator_name: str = "cdf_dm"

    def __post_init__(self):
        self.action_type = "discrete"

        if not self.use_observations_as_reward_scale:
            if self.r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`"
                )
            if self.r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`"
                )
            if self.n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`"
                )
            check_scalar(
                self.r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                self.r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                self.n_partition,
                name="n_partition",
                target_type=int,
                min_val=1,
            )

    @property
    def uniform_reward_scale(self) -> np.ndarray:
        return np.linspace(self.r_min, self.r_max, num=self.n_partition)

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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
        if reward.shape[0] % step_per_episode:
            raise ValueError(
                "Expected `reward.shape[0] \\% step_per_episode == 0`, but found False"
            )
        if reward.shape[0] // step_per_episode != initial_state_value_prediction:
            raise ValueError(
                "Expected `reward.shape[0] // step_per_episode == initial_state_value_prediction`, but found False"
            )

        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        if self.use_observations_as_reward_scale:
            reward_scale = np.sort(np.unique(trajectory_wise_reward))
        else:
            reward_scale = self.uniform_reward_scale()

        density = np.histgram(
            initial_state_value_prediction, bins=reward_scale, density=True
        )[0]
        return np.insert(density, 0, 0).cumsum(), reward_scale

    def estimate_mean(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        mean = self.estimate_mean(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        lower_idx = np.argmin(cumulative_density > alpha)
        return (np.diff(cumulative_density) * reward_scale[1:])[: lower_idx + 1].sum()

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        mean = self.estimate_mean(
            step_per_episode=step_per_episode,
            reward=reward,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        lower_idx = np.argmin(cumulative_density > alpha)
        upper_idx = np.argmin(cumulative_density > 1 - alpha)
        return {
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


@dataclass
class DiscreteCumulativeDistributionalImportanceSampling(
    BaseCumulativeDistributionalOffPolicyEstimator,
):
    """Importance Sampling (IS) for estimating cumulative distribution function (CDF) in discrete OPE.

    Note
    -------
    IS estimates CDF using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    r_min: float, default=None
        Minimum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    r_max: float, default=None
        Maximum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    n_partitiion: int, default=None
        Number of partition in reward scale (x-axis of CDF).
        When `use_observations_as_reward_scale == False`, a value must be given.

    use_observations_as_reward_scale: bool, default=False
        Whether to use the reward observed by the behavior policy as the reward scale.
        If True, the reward scale follows the one defined in Chundak et al. (2021).
        If False, the reward scale is uniform, following Huang et al. (2021).

    estimator_name: str, default="cdf_is"
        Name of the estimator.

    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    r_min: float = None
    r_max: float = None
    n_partition: int = 100
    use_observations_as_reward_scale: bool = False
    estimator_name: str = "cdf_is"

    def __post_init__(self):
        self.action_type = "discrete"

        if not self.use_observations_as_reward_scale:
            if self.r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            if self.r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`."
                )
            if self.n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            check_scalar(
                self.r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                self.r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                self.n_partition,
                name="n_partition",
                target_type=int,
                min_val=1,
            )

    @property
    def uniform_reward_scale(self) -> np.ndarray:
        return np.linspace(self.r_min, self.r_max, num=self.n_partition)

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

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
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )

        n = len(trajectory_wise_reward)

        if self.use_observations_as_reward_scale:
            reward_scale = np.sort(np.unique(trajectory_wise_reward))
        else:
            reward_scale = self.uniform_reward_scale()

        sort_idxes = trajectory_wise_reward.argsort()
        sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
        cumulative_density = np.clip(sorted_importance_weight.cumsum() / n, 0, 1)

        histogram = np.histgram(
            trajectory_wise_reward, bins=reward_scale, density=True
        )[0]
        cumulative_density = cumulative_density[histogram.cumsum()]
        return np.insert(cumulative_density, 0, 0), reward_scale

    def estimate_mean(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )
        mean = self.estimate_mean(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

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
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )
        lower_idx = np.argmin(cumulative_density > alpha)
        return (np.diff(cumulative_density) * reward_scale[1:])[: lower_idx + 1].sum()

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

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
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )
        mean = self.estimate_mean(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )
        lower_idx = np.argmin(cumulative_density > alpha)
        upper_idx = np.argmin(cumulative_density > 1 - alpha)
        return {
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


@dataclass
class DiscreteCumulativeDistributionalDoublyRobust(
    BaseCumulativeDistributionalOffPolicyEstimator,
):
    """Doubly Robust (DR) for estimating cumulative distribution function (CDF) in discrete OPE.

    Note
    -------
    DR estimates CDF using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    r_min: float, default=None
        Minimum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    r_max: float, default=None
        Maximum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    n_partitiion: int, default=None
        Number of partition in reward scale (x-axis of CDF).
        When `use_observations_as_reward_scale == False`, a value must be given.

    use_observations_as_reward_scale: bool, default=False
        Whether to use the reward observed by the behavior policy as the reward scale.
        If True, the reward scale follows the one defined in Chundak et al. (2021).
        If False, the reward scale is uniform, following Huang et al. (2021).

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

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    r_min: float = None
    r_max: float = None
    n_partition: int = 100
    use_observations_as_reward_scale: bool = False
    estimator_name: str = "cdf_dr"

    def __post_init__(self):
        self.action_type = "discrete"

        if not self.use_observations_as_reward_scale:
            if self.r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            if self.r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`."
                )
            if self.n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            check_scalar(
                self.r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                self.r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                self.n_partition,
                name="n_partition",
                target_type=int,
                min_val=1,
            )

    @property
    def uniform_reward_scale(self) -> np.ndarray:
        return np.linspace(self.r_min, self.r_max, num=self.n_partition)

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

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
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
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
            reward.shape[0] // step_per_episode
            == behavior_policy_trajectory_wise_pscore.shape[0] // step_per_episode
            == evaluation_policy_trajectory_wise_pscore.shape[0] // step_per_episode
            == initial_state_value_prediction
        ):
            raise ValueError(
                "Expected `reward.shape[0] // step_per_episode == behavior_policy_trajectory_wise_pscore.shape[0] // step_per_episode "
                "== evaluation_policy_trajectory_wise_pscore.shape[0] // step_per_episode == initial_state_value_prediction`, but found False"
            )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )

        n = len(trajectory_wise_reward)

        if self.use_observations_as_reward_scale:
            reward_scale = np.sort(np.unique(trajectory_wise_reward))
        else:
            reward_scale = self.uniform_reward_scale()

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
        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1), reward_scale

    def estimate_mean(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_mean: float
            Estimated mean of the policy value.

        """
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * reward_scale[1:]).sum()

    def estimate_variance(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        estimated_variance: float
            Estimated variance of the policy value.

        """
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        mean = self.estimate_mean(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

    def estimate_conditional_value_at_risk(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

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
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        lower_idx = np.argmin(cumulative_density > alpha)
        return (np.diff(cumulative_density) * reward_scale[1:])[: lower_idx + 1].sum()

    def estimate_interquartile_range(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

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
        (
            cumulative_density,
            reward_scale,
        ) = self.estimate_cumulative_distribution_function(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        mean = self.estimate_mean(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )
        lower_idx = np.argmin(cumulative_density > alpha)
        upper_idx = np.argmin(cumulative_density > 1 - alpha)
        return {
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


@dataclass
class DiscreteCumulativeDistributionalSelfNormalizedImportanceSampling(
    DiscreteCumulativeDistributionalImportanceSampling,
):
    """Self Normalized Importance Sampling (SNIS) for estimating cumulative distribution function (CDF) in discrete OPE.

    Note
    -------
    SNIS estimates CDF using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    r_min: float, default=None
        Minimum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    r_max: float, default=None
        Maximum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    n_partitiion: int, default=None
        Number of partition in reward scale (x-axis of CDF).
        When `use_observations_as_reward_scale == False`, a value must be given.

    use_observations_as_reward_scale: bool, default=False
        Whether to use the reward observed by the behavior policy as the reward scale.
        If True, the reward scale follows the one defined in Chundak et al. (2021).
        If False, the reward scale is uniform, following Huang et al. (2021).

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

    Adith Swaminathan and Thorsten Joachims.
    "The Self-Normalized Estimator for Counterfactual Learning.", 2015.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    r_min: float = None
    r_max: float = None
    n_partition: int = 100
    use_observations_as_reward_scale: bool = False
    estimator_name: str = "cdf_is"

    def __post_init__(self):
        self.action_type = "discrete"

        if not self.use_observations_as_reward_scale:
            if self.r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            if self.r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`."
                )
            if self.n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            check_scalar(
                self.r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                self.r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                self.n_partition,
                name="n_partition",
                target_type=int,
                min_val=1,
            )

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

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
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            gamma=gamma,
        )

        weight_sum = trajectory_wise_importance_weight.sum()

        if self.use_observations_as_reward_scale:
            reward_scale = np.sort(np.unique(trajectory_wise_reward))
        else:
            reward_scale = self.uniform_reward_scale()

        sort_idxes = trajectory_wise_reward.argsort()
        sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
        cumulative_density = np.clip(
            sorted_importance_weight.cumsum() / weight_sum, 0, 1
        )

        histogram = np.histgram(
            trajectory_wise_reward, bins=reward_scale, density=True
        )[0]
        cumulative_density = cumulative_density[histogram.cumsum()]
        return np.insert(cumulative_density, 0, 0), reward_scale


@dataclass
class DiscreteCumulativeDistributionalSelfNormalizedDoublyRobust(
    DiscreteCumulativeDistributionalDoublyRobust,
):
    """Self Normalized Doubly Robust (SNDR) for estimating cumulative distribution function (CDF) in discrete OPE.

    Note
    -------
    SNDR estimates CDF using importance sampling techniques as follows.

    .. math::

        aaa

    where

    Parameters
    -------
    r_min: float, default=None
        Minimum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    r_max: float, default=None
        Maximum reward in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    n_partitiion: int, default=None
        Number of partition in reward scale (x-axis of CDF).
        When `use_observations_as_reward_scale == False`, a value must be given.

    use_observations_as_reward_scale: bool, default=False
        Whether to use the reward observed by the behavior policy as the reward scale.
        If True, the reward scale follows the one defined in Chundak et al. (2021).
        If False, the reward scale is uniform, following Huang et al. (2021).

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

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    r_min: float = None
    r_max: float = None
    n_partition: int = 100
    use_observations_as_reward_scale: bool = False
    estimator_name: str = "cdf_dr"

    def __post_init__(self):
        self.action_type = "discrete"

        if not self.use_observations_as_reward_scale:
            if self.r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            if self.r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`."
                )
            if self.n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            check_scalar(
                self.r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                self.r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                self.n_partition,
                name="n_partition",
                target_type=int,
                min_val=1,
            )

    def estimate_cumulative_distribution_function(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        initial_state_value_prediction: np.ndarray,
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

        behavior_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_b(a_t \\mid s_t)`

        evaluation_policy_trajectory_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of evaluation policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`

        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

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
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
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
            reward.shape[0] // step_per_episode
            == behavior_policy_trajectory_wise_pscore.shape[0] // step_per_episode
            == evaluation_policy_trajectory_wise_pscore.shape[0] // step_per_episode
            == initial_state_value_prediction
        ):
            raise ValueError(
                "Expected `reward.shape[0] // step_per_episode == behavior_policy_trajectory_wise_pscore.shape[0] // step_per_episode "
                "== evaluation_policy_trajectory_wise_pscore.shape[0] // step_per_episode == initial_state_value_prediction`, but found False"
            )
        (
            trajectory_wise_reward,
            trajectory_wise_importance_weight,
            initial_state_value_prediction,
        ) = self._aggregate_trajectory_wise_statistics_discrete(
            step_per_episode=step_per_episode,
            reward=reward,
            behavior_policy_trajectory_wise_pscore=behavior_policy_trajectory_wise_pscore,
            evaluation_policy_trajectory_wise_pscore=evaluation_policy_trajectory_wise_pscore,
            initial_state_value_prediction=initial_state_value_prediction,
            gamma=gamma,
        )

        n = len(trajectory_wise_reward)

        if self.use_observations_as_reward_scale:
            reward_scale = np.sort(np.unique(trajectory_wise_reward))
        else:
            reward_scale = self.uniform_reward_scale()

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
        return np.clip(np.maximum.accumulate(cumulative_density), 0, 1), reward_scale


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
    optimizer: Optimizer, default=Optimizer(SGD, config={"lr": 0.001})
        Optimizer for tuning alpha.

    initial_alpha: float, default=1.0 (> 0)
        Initial temperature parameter of the exponential function.

    max_gradient_steps: int, default=100 (> 0)
        Maximum number of gradient steps in turning alpha.

    delta: float, default=0.1 (> 0)
        Allowance of the distributional shift.

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

    optimizer: Optimizer = Optimizer(optim.SGD, config={"lr": 0.001})
    initial_alpha: float = 1.0
    max_gradient_steps: int = 100
    delta: float = 0.05
    estimator_name: str = "dr_is"

    def __post_init__(self):
        self.action_type = "discrete"

        check_scalar(self.initial_alpha, name="initial_alpha", type=float, min_val=0.0)
        check_scalar(
            self.max_gradient_steps, name="max_gradient_steps", type=int, min_val=1
        )
        check_scalar(self.delta, name="delta", type=float, min_val=0.0)

    def estimate_worst_case_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
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

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
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

        trajectory_wise_reward = torch.tensor(
            trajectory_wise_reward, dtype=torch.float, require_grad=False
        )
        trajectory_wise_importance_weight = torch.tensor(
            trajectory_wise_reward, dtype=torch.float, require_grad=False
        )

        alpha_prior = self.initial_alpha
        alpha = torch.tensor([self.initial_alpha], dtype=torch.float, require_grad=True)
        optimizer = self.optimizer.instantiate([alpha])

        for _ in range(self.max_gradient_steps):
            estimated_exponential_policy_value = (
                trajectory_wise_importance_weight
                * torch.exp(-trajectory_wise_reward / alpha)
            ).mean()
            objective = -alpha * (
                torch.log(estimated_exponential_policy_value) + self.delta
            )
            loss = -objective

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if np.abs(alpha.item() - alpha_prior) < 0.01 * alpha_prior:
                break
            else:
                alpha_prior = alpha.item()

        return objective.item()


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
    optimizer: Optimizer, default=Optimizer(SGD, config={"lr": 0.001})
        Optimizer for tuning alpha.

    initial_alpha: float, default=1.0 (> 0)
        Initial temperature parameter of the exponential function.

    max_gradient_steps: int, default=100 (> 0)
        Maximum number of gradient steps in turning alpha.

    delta: float, default=0.1 (> 0)
        Allowance of the distributional shift.

    estimator_name: str, default="dr_snis"
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

    optimizer: Optimizer = Optimizer(optim.SGD, config={"lr": 0.001})
    initial_alpha: float = 1.0
    max_gradient_steps: int = 100
    delta: float = 0.05
    estimator_name: str = "dr_snis"

    def __post_init__(self):
        self.action_type = "discrete"

        check_scalar(self.initial_alpha, name="initial_alpha", type=float, min_val=0.0)
        check_scalar(
            self.max_gradient_steps, name="max_gradient_steps", type=int, min_val=1
        )
        check_scalar(self.delta, name="delta", type=float, min_val=0.0)

    def estimate_worst_case_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
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

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
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

        weight_sum = trajectory_wise_reward.sum()
        trajectory_wise_reward = torch.tensor(
            trajectory_wise_reward, dtype=torch.float, require_grad=False
        )
        trajectory_wise_importance_weight = torch.tensor(
            trajectory_wise_reward, dtype=torch.float, require_grad=False
        )

        alpha_prior = self.initial_alpha
        alpha = torch.tensor([self.initial_alpha], dtype=torch.float, require_grad=True)
        optimizer = self.optimizer.instantiate([alpha])

        for _ in range(self.max_gradient_steps):
            estimated_exponential_policy_value = (
                trajectory_wise_importance_weight
                * torch.exp(-trajectory_wise_reward / alpha)
            ).sum() / weight_sum
            objective = -alpha * (
                torch.log(estimated_exponential_policy_value) + self.delta
            )
            loss = -objective

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if np.abs(alpha.item() - alpha_prior) < 0.01 * alpha_prior:
                break
            else:
                alpha_prior = alpha.item()

        return objective.item()


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
    optimizer: Optimizer, default=Optimizer(SGD, config={"lr": 0.001})
        Optimizer for tuning alpha.

    initial_alpha: float, default=1.0 (> 0)
        Initial temperature parameter of the exponential function.

    max_gradient_steps: int, default=100 (> 0)
        Maximum number of gradient steps in turning alpha.

    delta: float, default=0.1 (> 0)
        Allowance of the distributional shift.

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

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    optimizer: Optimizer = Optimizer(optim.SGD, config={"lr": 0.001})
    initial_alpha: float = 1.0
    max_gradient_steps: int = 100
    delta: float = 0.05
    n_folds: int = 3
    estimator_name: str = "dr_is"

    def __post_init__(self):
        self.action_type = "discrete"

        check_scalar(self.initial_alpha, name="initial_alpha", type=float, min_val=0.0)
        check_scalar(
            self.max_gradient_steps, name="max_gradient_steps", type=int, min_val=1
        )
        check_scalar(self.delta, name="delta", type=float, min_val=0.0)

    def _calculate_exponential_policy_value_given_alpha(
        self,
        trajectory_wise_reward: np.ndarray,
        trajectory_wise_importance_weight: np.ndarray,
        trajectory_wise_reward_prediction: np.ndarray,
        alpha: float,
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

        moment: int
            If 0 is given, return :math:`W_0`. If 1 is given, return :math:`W_1`.
            See Kallus et al. (2022) for the details.

        Return
        -------
        estimated_worst_case_policy_value: float
            Estimated worst case policy value.

        """

        return -alpha * (np.log(estimated_exponential_policy_value) + self.delta)

    def _calculate_alpha_prior(
        self,
        trajectory_wise_reward: np.ndarray,
        trajectory_wise_importance_weight: np.ndarray,
    ):
        """Use SNIS for finding prior value of alpha for each fold.

        Parameters
        -------
        trajectory_wise_reward: NDArray, shape (n_episodes, )
            Trajectory wise reward observed by the behavior policy.

        trajectory_wise_importance_weight: NDArray, shape (n_episodes, )
            Trajectory wise importance weight.

        Return
        -------
        alpha_prior: float
            Prior value of the temperature parameter of the exponential function..

        """
        weight_sum = trajectory_wise_reward.sum()
        trajectory_wise_reward = torch.tensor(
            trajectory_wise_reward, dtype=torch.float, require_grad=False
        )
        trajectory_wise_importance_weight = torch.tensor(
            trajectory_wise_reward, dtype=torch.float, require_grad=False
        )

        alpha_prior = self.initial_alpha
        alpha = torch.tensor([self.initial_alpha], dtype=torch.float, require_grad=True)
        optimizer = self.optimizer.instantiate([alpha])

        for _ in range(self.max_gradient_steps):
            estimated_exponential_policy_value = (
                trajectory_wise_importance_weight
                * torch.exp(-trajectory_wise_reward / alpha)
            ).sum() / weight_sum
            objective = -alpha * (
                torch.log(estimated_exponential_policy_value) + self.delta
            )
            loss = -objective

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if np.abs(alpha.item() - alpha_prior) < 0.01 * alpha_prior:
                break
            else:
                alpha_prior = alpha.item()

        return alpha.item()

    def estimate_worst_case_policy_value(
        self,
        step_per_episode: int,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_trajectory_wise_pscore: np.ndarray,
        gamma: float = 1.0,
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

        Return
        -------
        estimated_distributionally_robust_worst_case_policy_value: float
            Estimated worst case objective.

        """
        check_scalar(step_per_episode, name="step_per_episode", type=int, min_val=1)
        check_scalar(gamma, name="gamma", type=float, min_val=0.0, max_val=1.0)
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

        weight_sum = trajectory_wise_reward.sum()
        trajectory_wise_reward = torch.tensor(
            trajectory_wise_reward, dtype=torch.float, require_grad=False
        )
        trajectory_wise_importance_weight = torch.tensor(
            trajectory_wise_reward, dtype=torch.float, require_grad=False
        )

        alpha_prior = self.initial_alpha
        alpha = torch.tensor([self.initial_alpha], dtype=torch.float, require_grad=True)
        optimizer = self.optimizer.instantiate([alpha])

        for _ in range(self.max_gradient_steps):
            estimated_exponential_policy_value = (
                trajectory_wise_importance_weight
                * torch.exp(-trajectory_wise_reward / alpha)
            ).sum() / weight_sum
            objective = -alpha * (
                torch.log(estimated_exponential_policy_value) + self.delta
            )
            loss = -objective

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if np.abs(alpha.item() - alpha_prior) < 0.01 * alpha_prior:
                break
            else:
                alpha_prior = alpha.item()

        return objective.item()
