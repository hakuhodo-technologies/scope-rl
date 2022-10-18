"""Off-Policy Estimators for Continuous Action (designed for deterministic policies)."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm, truncnorm
from sklearn.utils import check_scalar

from .estimators_base import BaseOffPolicyEstimator
from ..utils import check_array


@dataclass
class ContinuousDirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) for continuous-action OPE (designed for deterministic policies).

    Note
    -------
    DM estimates the policy value using the initial state value given by Fitted Q Evaluation (FQE) as follows.

    .. math::

        \\hat{J}_{\\mathrm{DM}} (\\pi; \\mathcal{D}) := \\mathbb{E}_n [\\mathbb{E}_{a_0 \\sim \\pi(a_0 \\mid s_0)} [\\hat{Q}(s_0, a_0)] ],

    .. math::

        \\hat{J}_{\\mathrm{DM}} (\\pi; \\mathcal{D}) := \\mathbb{E}_n [\\hat{V}(s_0)],

    where :math:`\\mathcal{D}=\\{\\{(s_t, a_t, r_t)\\}_{t=0}^{T-1}\\}_{i=1}^n` is the logged dataset with :math:`n` trajectories of data.
    :math:`T` indicates step per episode. :math:`\\hat{Q}(s_t, a_t)` is the estimated Q value given a state-action pair.
    \\hat{V}(s_t) is the estimated value function given a state.

    There are several ways to estimate :math:`\\hat{Q}(s, a)` such as Fitted Q Evaluation (FQE) (Le et al., 2019) and
    Minimax Q-Function Learning (MQL) (Uehara et al., 2020). :math:`\\hat{V}(s)` is estimated in a similar manner using
    Minimax Value Learning (MVL) (Uehara et al., 2020).

    We use the implementation of FQE provided by d3rlpy (Seno et al., 2021).
    The implementations of Minimax Learning is available in `ofrl/ope/minimax_estimators_continuous.py`.

    Parameters
    -------
    estimator_name: str, default="dm"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Takuma Seno and Michita Imai.
    "d3rlpy: An Offline Deep Reinforcement Library.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Alina Beygelzimer and John Langford.
    "The Offset Tree for Learning with Partial Labels.", 2009.

    """

    estimator_name = "dm"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        initial_state_value_prediction: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        estimated_trajectory_wise_policy_value: array-like of shape (n_episodes, )
            Policy value estimated for each trajectory.
            (Equivalent to initial_state_value_prediction.)

        """
        return initial_state_value_prediction

    def estimate_policy_value(
        self,
        initial_state_value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        estimated_policy_value = self._estimate_trajectory_value(
            initial_state_value_prediction,
        ).mean()
        return estimated_policy_value

    def estimate_interval(
        self,
        initial_state_value_prediction: np.ndarray,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        initial_state_value_prediction: array-like of shape (n_episodes, )
            Estimated initial state value.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: dict
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            initial_state_value_prediction,
            name="initial_state_value_prediction",
            expected_dim=1,
        )
        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            initial_state_value_prediction,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousTrajectoryWiseImportanceSampling(BaseOffPolicyEstimator):
    """Trajectory-wise Importance Sampling (TIS) for continuous-action OPE (designed for deterministic policies).

    Note
    -------
    TIS estimates the policy value via trajectory-wise importance weight as follows.

    .. math::

        \\hat{J}_{\\mathrm{TIS}} (\\pi; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t w_{0:T-1} \\delta(\\pi, a_{0:T-1}) r_t],

    where :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} \\frac{1}{\\pi_0(a_t \\mid s_t)}` is the inverse propensity weight
    and :math:`\\delta(\\pi, a_{0:T-1}) = \\prod_{t=0}^{T-1} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.
    Note that :math:`K(\\cdot)` is a kernel function.

    Parameters
    -------
    estimator_name: str, default="tis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "tis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )
        reward = reward.reshape((-1, step_per_episode))
        behavior_policy_trajectory_wise_pscore = (
            behavior_policy_trajectory_wise_pscore.reshape((-1, step_per_episode))
        )
        discount = np.full(reward.shape[1], gamma).cumprod()

        if sigma is None:
            sigma = np.ones(action.shape[1])

        if use_truncated_kernel:
            similarity_weight = truncnorm.pdf(
                evaluation_policy_action,
                a=(action_min - action) / sigma,
                b=(action_max - action) / sigma,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, -1, 0]
        else:
            similarity_weight = norm.pdf(
                evaluation_policy_action,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, -1, 0]

        similarity_weight = np.tile(
            similarity_weight.reshape((-1, 1)), step_per_episode
        )

        estimated_trajectory_value = (
            discount
            * reward
            * similarity_weight
            / behavior_policy_trajectory_wise_pscore
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

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

        return self._estimate_trajectory_value(
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
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: dict
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
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
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_trajectory_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_trajectory_wise_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

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

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
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
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousPerDecisionImportanceSampling(BaseOffPolicyEstimator):
    """Per-Decision Importance Sampling (PDIS) for continuous-action OPE (designed for deterministic policies).

    Note
    -------
    PDIS estimates the policy value via step-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{PDIS}} (\\pi; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t w_{0:t} \\delta(\\pi, a_{0:t}) r_t],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{1}{\\pi_0(a_{t'} \\mid s_{t'})}`
    and :math:`\\delta(\\pi, a_{0:t}) = \\prod_{t'=0}^t K(\\pi(s_{t'}), a_{t'})` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.
    Note that :math:`K(\\cdot)` is a kernel function.

    Parameters
    -------
    estimator_name: str, default="pdis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Alex Strehl, John Langford, Sham Kakade, and Lihong Li.
    "Learning from Logged Implicit Exploration Data.", 2010.

    Doina Precup, Richard S. Sutton, and Satinder P. Singh.
    "Eligibility Traces for Off-Policy Policy Evaluation.", 2000.

    """

    estimator_name: str = "pdis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )
        reward = reward.reshape((-1, step_per_episode))
        behavior_policy_step_wise_pscore = behavior_policy_step_wise_pscore.reshape(
            (-1, step_per_episode)
        )
        discount = np.full(reward.shape[1], gamma).cumprod()

        if sigma is None:
            sigma = np.ones(action.shape[1])

        if use_truncated_kernel:
            similarity_weight = truncnorm.pdf(
                evaluation_policy_action,
                a=(action_min - action) / sigma,
                b=(action_max - action) / sigma,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, :, 0]
        else:
            similarity_weight = norm.pdf(
                evaluation_policy_action,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, :, 0]

        estimated_trajectory_value = (
            discount * reward * similarity_weight / behavior_policy_step_wise_pscore
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
        check_array(
            behavior_policy_step_wise_pscore,
            name="behavior_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_step_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_step_wise_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

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

        return self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_step_wise_pscore=behavior_policy_step_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: dict
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
        check_array(
            behavior_policy_step_wise_pscore,
            name="behavior_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_step_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_step_wise_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

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

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_step_wise_pscore=behavior_policy_step_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousDoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust (DR) for continuous-action OPE (designed for deterministic policies).

    Note
    -------
    DR estimates the policy value via step-wise importance weighting and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{J}_{\\mathrm{DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t (w_{0:t} \\delta(\\pi, a_{0:t}) (r_t - \\hat{Q}(s_t, a_t))
            + w_{0:t-1} \\delta(\\pi, a_{0:t-1}) \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{1}{\\pi_0(a_{t'} \\mid s_{t'})}`
    and :math:`\\delta(\\pi, a_{0:t}) = \\prod_{t'=1}^t K(\\pi(s_{t'}), a_{t'})` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.
    Note that :math:`K(\\cdot)` is a kernel function.

    Parameters
    -------
    estimator_name: str, default="dr"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name = "dr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by the evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )

        reward = reward.reshape((-1, step_per_episode))
        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_episode)
        )

        pscore = behavior_policy_step_wise_pscore.reshape((-1, step_per_episode))
        pscore_prev = np.roll(pscore, 1, axis=1)
        pscore_prev[:, 0] = 1

        discount = np.full(reward.shape[1], gamma).cumprod()

        if sigma is None:
            sigma = np.ones(action.shape[1])

        if use_truncated_kernel:
            similarity_weight = truncnorm.pdf(
                evaluation_policy_action,
                a=(action_min - action) / sigma,
                b=(action_max - action) / sigma,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, :, 0]
        else:
            similarity_weight = norm.pdf(
                evaluation_policy_action,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, :, 0]

        similarity_weight_prev = np.roll(similarity_weight, 1, axis=1)
        similarity_weight_prev[:, 0] = 1

        estimated_trajectory_value = (
            discount
            * (
                (reward - state_action_value_prediction) * similarity_weight / pscore
                + state_action_value_prediction * similarity_weight_prev / pscore_prev
            )
        ).sum(axis=1)

        return estimated_trajectory_value

    def estimate_policy_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of the evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by the evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If `False`, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        V_hat: ndarray of shape (n_episodes, )
            Estimated policy value.

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
        check_array(
            behavior_policy_step_wise_pscore,
            name="behavior_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_step_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_step_wise_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

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

        return self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_step_wise_pscore=behavior_policy_step_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        ).mean()

    def estimate_interval(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 10000,
        random_state: int = 12345,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value by nonparametric bootstrap.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by the evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        ggamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: dict
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_scalar(
            step_per_episode,
            name="step_per_episode",
            target_type=int,
            min_val=1,
        )
        check_array(
            action,
            name="action",
            expected_dim=2,
        )
        check_array(
            reward,
            name="reward",
            expected_dim=1,
        )
        check_array(
            behavior_policy_step_wise_pscore,
            name="behavior_policy_step_wise_pscore",
            expected_dim=1,
            min_val=0.0,
        )
        check_array(
            evaluation_policy_action,
            name="evaluation_policy_action",
            expected_dim=2,
        )
        if not (
            action.shape[0]
            == reward.shape[0]
            == behavior_policy_step_wise_pscore.shape[0]
            == evaluation_policy_action.shape[0]
        ):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == behavior_policy_step_wise_pscore.shape[0] == evaluation_policy_action.shape[0]`"
                ", but found False"
            )
        if not (action.shape[1] == evaluation_policy_action.shape[1]):
            raise ValueError(
                "Expected `action.shape[1] == evaluation_policy_action.shape[1]`"
                ", but found False"
            )

        check_scalar(gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)

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

        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

        estimated_trajectory_value = self._estimate_trajectory_value(
            step_per_episode=step_per_episode,
            action=action,
            reward=reward,
            behavior_policy_step_wise_pscore=behavior_policy_step_wise_pscore,
            evaluation_policy_action=evaluation_policy_action,
            state_action_value_prediction=state_action_value_prediction,
            gamma=gamma,
            sigma=sigma,
            use_truncated_kernel=use_truncated_kernel,
            action_min=action_min,
            action_max=action_max,
        )
        return self._estimate_confidence_interval[ci](
            samples=estimated_trajectory_value,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ContinuousSelfNormalizedTrajectoryWiseImportanceSampling(
    ContinuousTrajectoryWiseImportanceSampling
):
    """Self-Normalized Trajectory-wise Importance Sampling (SNTIS) for continuous-action OPE (designed for deterministic policies).

    Note
    -------
    SNTIS estimates the policy value via self-normalized trajectory-wise importance weight as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNTIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{0:T-1} \\delta(\\pi, a_{0:T-1})}{\\sum_{n} [w_{1:T-1} \\delta(\\pi, a_{0:T-1})]} r_t],

    where :math:`w_{0:T-1} := \\prod_{t=0}^{T-1} \\frac{1}{\\pi_0(a_t \\mid s_t)}`
    and :math:`\\delta(\\pi, a_{0:T}) = \\prod_{t=0}^{T-1} K(\\pi(s_t), a_t)` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.
    Note that :math:`K(\\cdot)` is a kernel function.

    Parameters
    -------
    estimator_name: str, default="sntis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

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

    estimator_name: str = "sntis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_trajectory_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_trajectory_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Trajectory-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t=0}^T \\pi_0(a_t \\mid s_t)`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )
        reward = reward.reshape((-1, step_per_episode))
        importance_weight = 1 / (
            behavior_policy_trajectory_wise_pscore.reshape((-1, step_per_episode))
        )
        discount = np.full(reward.shape[1], gamma).cumprod()

        if sigma is None:
            sigma = np.ones(action.shape[1])

        if use_truncated_kernel:
            similarity_weight = truncnorm.pdf(
                evaluation_policy_action,
                a=(action_min - action) / sigma,
                b=(action_max - action) / sigma,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, -1, 0]
        else:
            similarity_weight = norm.pdf(
                evaluation_policy_action,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, -1, 0]

        similarity_weight = np.tile(
            similarity_weight.reshape((-1, 1)), step_per_episode
        )

        weight = similarity_weight / importance_weight
        weight_mean = weight.mean(axis=0)
        weight_mean = np.tile(weight_mean, len(weight)).reshape((-1, step_per_episode))
        self_normalized_weight = weight / (weight_mean + 1e-10)

        estimated_trajectory_value = (discount * reward * self_normalized_weight).sum(
            axis=1
        )

        return estimated_trajectory_value


@dataclass
class ContinuousSelfNormalizedPerDecisionImportanceSampling(
    ContinuousPerDecisionImportanceSampling
):
    """Self-Normalized Per-Decision Importance Sampling (SNPDIS) for continuous-action OPE (assume deterministic policies).

    Note
    -------
    SNPDIS estimates the policy value via self-normalized step-wise importance weighting as follows.

    .. math::

        \\hat{J}_{\\mathrm{SNPDIS}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t \\frac{w_{1:t} \\delta(\\pi, a_{0:t})}{\\sum_{n} [w_{1:t} \\delta(\\pi, a_{0:t})]} r_t],

    where :math:`w_{0:t} := \\prod_{t'=1}^t \\frac{1}{\\pi_0(a_{t'} \\mid s_{t'})}`
    and :math:`\\delta(\\pi, a_{0:t}) = \\prod_{t'=1}^t K(\\pi(s_{t'}), a_{t'})` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.
    Note that :math:`K(\\cdot)` is a kernel function.

    Parameters
    -------
    estimator_name: str, default="snpdis"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

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

    estimator_name: str = "snpdis"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )
        reward = reward.reshape((-1, step_per_episode))
        importance_weight = 1 / (
            behavior_policy_step_wise_pscore.reshape((-1, step_per_episode))
        )
        discount = np.full(reward.shape[1], gamma).cumprod()

        if sigma is None:
            sigma = np.ones(action.shape[1])

        if use_truncated_kernel:
            similarity_weight = truncnorm.pdf(
                evaluation_policy_action,
                a=(action_min - action) / sigma,
                b=(action_max - action) / sigma,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, :, 0]
        else:
            similarity_weight = norm.pdf(
                evaluation_policy_action,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, :, 0]

        weight = similarity_weight / importance_weight
        weight_mean = weight.mean(axis=0)
        weight_mean = np.tile(weight_mean, len(weight)).reshape((-1, step_per_episode))
        self_normalized_weight = weight / (weight_mean + 1e-10)

        estimated_trajectory_value = (discount * reward * self_normalized_weight).sum(
            axis=1
        )

        return estimated_trajectory_value


@dataclass
class ContinuousSelfNormalizedDoublyRobust(ContinuousDoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) for continuous-action OPE (assume deterministic policies).

    Note
    -------
    SNDR estimates the policy value via self-normalized step-wise importance weighting and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{J}_{\\mathrm{DR}} (\\pi; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^{T-1} \\gamma^t ( \\frac{w_{0:t} \\delta(\\pi, a_{0:t})}{\\sum_{n} [w_{0:t} \\delta(\\pi, a_{0:t})]} (r_t - \\hat{Q}(s_t, a_t))
            + \\frac{w_{0:t-1} \\delta(\\pi, a_{0:t-1})}{\\sum_{n} [w_{0:t-1} \\delta(\\pi, a_{0:t-1})]} \\mathbb{E}_{a \\sim \\pi(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{1}{\\pi_0(a_{t'} \\mid s_{t'})}`
    and :math:`\\delta(\\pi, a_{0:t}) = \\prod_{t'=1}^t K(\\pi(s_{t'}), a_{t'})` quantifies the similarity between the action logged in the dataset and that taken by the evaluation policy.
    Note that :math:`K(\\cdot)` is a kernel function.

    Parameters
    -------
    estimator_name: str, default="sndr"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    Nan Jiang and Lihong Li.
    "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.", 2016.

    Philip S. Thomas and Emma Brunskill.
    "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.", 2016.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.
    """

    estimator_name = "sndr"

    def __post_init__(self):
        self.action_type = "continuous"

    def _estimate_trajectory_value(
        self,
        step_per_episode: int,
        action: np.ndarray,
        reward: np.ndarray,
        behavior_policy_step_wise_pscore: np.ndarray,
        evaluation_policy_action: np.ndarray,
        state_action_value_prediction: np.ndarray,
        gamma: float = 1.0,
        sigma: Optional[np.ndarray] = None,
        use_truncated_kernel: bool = False,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the behavior policy.

        reward: array-like of shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: array-like of shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of the behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_0(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: array-like of shape (n_episodes * step_per_episode, action_dim)
            Action chosen by the evaluation policy.

        state_action_value_prediction: array-like of shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by the evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

        sigma: array-like of shape (action_dim, ), default=None
            Standard deviation of Gaussian distribution (i.e., `band_width` hyperparameter of gaussian kernel).
            If `None`, sigma is set to 1 for all dimensions.

        use_truncated_kernel: bool, default=False
            Whether to use the Truncated Gaussian kernel or not.
            If False, (normal) Gaussian kernel is used.

        action_min: array-like of shape (action_dim, ), default=None
            Minimum value of action vector.
            When `use_truncated_kernel == True`, action_min must be given.

        action_max: array-like of shape (action_dim, ), default=None
            Maximum value of action vector.
            When `use_truncated_kernel == True`, action_max must be given.

        Return
        -------
        estimated_trajectory_wise_policy_value: ndarray of shape (n_episodes, )
            Policy value estimated for each trajectory.

        """
        action_dim = action.shape[1]
        action = action.reshape((-1, step_per_episode, action_dim))
        evaluation_policy_action = evaluation_policy_action.reshape(
            (-1, step_per_episode, action_dim)
        )

        reward = reward.reshape((-1, step_per_episode))
        state_action_value_prediction = state_action_value_prediction.reshape(
            (-1, step_per_episode)
        )

        pscore = behavior_policy_step_wise_pscore.reshape((-1, step_per_episode))
        pscore_prev = np.roll(pscore, 1, axis=1)
        pscore_prev[:, 0] = 1

        importance_weight = 1 / pscore
        importance_weight_prev = 1 / pscore_prev

        discount = np.full(reward.shape[1], gamma).cumprod()

        if sigma is None:
            sigma = np.ones(action.shape[1])

        if use_truncated_kernel:
            similarity_weight = truncnorm.pdf(
                evaluation_policy_action,
                a=(action_min - action) / sigma,
                b=(action_max - action) / sigma,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, :, 0]
        else:
            similarity_weight = norm.pdf(
                evaluation_policy_action,
                loc=action,
                scale=sigma,
            ).cumprod(axis=1)[:, :, 0]

        similarity_weight_prev = np.roll(similarity_weight, 1, axis=1)
        similarity_weight_prev[:, 0] = 1

        weight = similarity_weight / importance_weight
        weight_mean = weight.mean(axis=0)
        weight_mean = np.tile(weight_mean, len(weight)).reshape((-1, step_per_episode))
        self_normalized_weight = weight / (weight_mean + 1e-10)

        # weight_prev = similarity_weight_prev / importance_weight_prev
        weight_prev = importance_weight_prev
        weight_prev_mean = weight_prev.mean(axis=0)
        weight_prev_mean = np.tile(weight_prev_mean, len(weight_prev)).reshape(
            (-1, step_per_episode)
        )
        self_normalized_weight_prev = weight_prev / (weight_prev_mean + 1e-10)

        estimated_trajectory_value = (discount * reward * self_normalized_weight).sum(
            axis=1
        )

        estimated_trajectory_value = (
            discount
            * (
                (reward - state_action_value_prediction) * self_normalized_weight
                + state_action_value_prediction * self_normalized_weight_prev
            )
        ).sum(axis=1)

        return estimated_trajectory_value
