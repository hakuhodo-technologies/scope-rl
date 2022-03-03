"""Distributional Off-Policy Estimators for Discrete action."""
from dataclasses import dataclass
from multiprocessing.sharedctypes import Value
from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.utils import check_scalar

from offlinegym.ope.estimators_base import (
    BaseCumulativeDistributionalOffPolicyEstimator,
    BaseDistributionallyRobustOffPolicyEstimator,
)
from offlinegym.utils import check_array


@dataclass
def DiscreteCumulativeDistributionalDirectMethod(
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

        if not use_observations_as_reward_scale:
            if r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`"
                )
            if r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`"
                )
            if n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`"
                )
            check_scalar(
                r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                n_partition,
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
def DiscreteCumulativeDistributionalImportanceSampling(
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

        if not use_observations_as_reward_scale:
            if r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            if r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`."
                )
            if n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            check_scalar(
                r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                n_partition,
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
def DiscreteCumulativeDistributionalDoublyRobust(
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

        if not use_observations_as_reward_scale:
            if r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            if r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`."
                )
            if n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            check_scalar(
                r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                n_partition,
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
def DiscreteCumulativeDistributionalSelfNormalizedImportanceSampling(
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

        if not use_observations_as_reward_scale:
            if r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            if r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`."
                )
            if n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            check_scalar(
                r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                n_partition,
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
            sorted_importance_weight.cumsum() / trajectory_wise_importance_weight, 0, 1
        )

        histogram = np.histgram(
            trajectory_wise_reward, bins=reward_scale, density=True
        )[0]
        cumulative_density = cumulative_density[histogram.cumsum()]
        return np.insert(cumulative_density, 0, 0), reward_scale


@dataclass
def DiscreteCumulativeDistributionalSelfNormalizedDoublyRobust(
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

        if not use_observations_as_reward_scale:
            if r_min is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            if r_max is None:
                raise ValueError(
                    "r_max must be given when `use_observations_as_reward_scale == False`."
                )
            if n_partition is None:
                raise ValueError(
                    "r_min must be given when `use_observations_as_reward_scale == False`."
                )
            check_scalar(
                r_min,
                name="r_min",
                target_type=float,
            )
            check_scalar(
                r_max,
                name="r_max",
                target_type=float,
            )
            check_scalar(
                n_partition,
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
