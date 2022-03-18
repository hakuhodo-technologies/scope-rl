"""Off-Policy Estimators for Continuous action (designed for deterministic policies)."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm, truncnorm
from sklearn.utils import check_scalar

from offlinegym.ope.estimators_base import BaseOffPolicyEstimator
from offlinegym.utils import (
    estimate_confidence_interval_by_bootstrap,
    estimate_confidence_interval_by_hoeffding,
    estimate_confidence_interval_by_empirical_bernstein,
    estimate_confidence_interval_by_t_test,
    check_array,
)


@dataclass
class ContinuousDirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) for continuous OPE (assume deterministic policies).

    Note
    -------
    DM estimates policy value using initial state value given by Fitted Q Evaluation (FQE) as follows.

    .. math::

        \\hat{V}_{\\mathrm{DM}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_n [\\mathbb{E}_{a_0 \\sim \\pi_e(a_0 \\mid s_0)} [\\hat{Q}(x_0, a_0)] ],

    where :math:`\\mathcal{D}=\\{\\{(s_t, a_t, r_t)\\}_{t=0}^T\\}_{i=1}^n` is logged dataset with :math:`n` trajectories of data.
    :math:`T` indicates step per episode. :math:`\\hat{Q}(x_t, a_t)` is estimated Q value given state-action pair.

    Parameters
    -------
    estimator_name: str, default="dm"
        Name of the estimator.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    Alina Beygelzimer and John Langford.
    "The Offset Tree for Learning with Partial Labels.", 2009.

    """

    estimator_name = "dm"

    def __post_init__(self):
        self.action_type = "continuous"

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _estimate_trajectory_value(
        self,
        initial_state_value_prediction: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.
            (Equivalent to initial_state_value_prediction.)

        """
        return initial_state_value_prediction

    def estimate_policy_value(
        self,
        initial_state_value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of evaluation policy.

        Parameters
        -------
        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        Return
        -------
        V_hat: NDArray, shape (n_episodes, )
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
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        initial_state_value_prediction: NDArray, shape (n_episodes, )
            Estimated initial state value.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: Dict[str, float]
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
    """Trajectory-wise Importance Sampling (TIS) for continuous OPE (assume deterministic policies).

    Note
    -------
    TIS estimates policy value using trajectory-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{TIS}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t w_{0:T} \\delta(\\pi_e, a_{0:t}) r_t],

    where :math:`w_{0:T} := \\prod_{t=1}^T \\frac{1}{\\pi_b(a_t \\mid s_t)}` is the inverse propensity weight
    and :math:`\\delta(\\pi_e, a_{0:T}) = \\prod_{t=1}^T K(\\pi_e(s_t), a_t)` indicates similarity between action in dataset and that of evaluation policy.
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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

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
        """Estimate trajectory-wise policy value.

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
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

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

        if self.use_truncated_kernel:
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
        """Estimate policy value of evaluation policy.

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
        V_hat: NDArray, shape (n_episodes, )
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
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

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

        alpha: float, default=0.05 (0, 1)
            Significant level.

        ci: str, default="bootstrap"
            Estimation method for confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: Dict[str, float]
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
class ContinuousStepWiseImportanceSampling(BaseOffPolicyEstimator):
    """Step-wise Importance Sampling (SIS) for continuous OPE (assume deterministic policies).

    Note
    -------
    SIS estimates policy value using step-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SIS}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t w_{0:t} \\delta(\\pi_e, a_{0:t}) r_t],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{1}{\\pi_b(a_{t'} \\mid s_{t'})}`
    and :math:`\\delta(\\pi_e, a_{0:t}) = \\prod_{t'=1}^t K(\\pi_e(s_{t'}), a_{t'})` indicates similarity between action in dataset and that of evaluation policy.
    Note that :math:`K(\\cdot)` is a kernel function.

    Parameters
    -------
    estimator_name: str, default="sis"
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

    estimator_name: str = "sis"

    def __post_init__(self):
        self.action_type = "continuous"

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

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
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

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

        if self.use_truncated_kernel:
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
        """Estimate policy value of evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
        V_hat: NDArray, shape (n_episodes, )
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
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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

        alpha: float, default=0.05 (0, 1)
            Significant level.

        ci: str, default="bootstrap"
            Estimation method for confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: Dict[str, float]
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
    """Doubly Robust (DR) for continuous OPE (assume deterministic policies).

    Note
    -------
    DR estimates policy value using step-wise importance weight and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t (w_{0:t} \\delta(\\pi_e, a_{0:t}) (r_t - \\hat{Q}(s_t, a_t))
            + w_{0:t-1} \\delta(\\pi_e, a_{0:t-1}) \\mathbb{E}_{a \\sim \\pi_e(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{1}{\\pi_b(a_{t'} \\mid s_{t'})}`
    and :math:`\\delta(\\pi_e, a_{0:t}) = \\prod_{t'=1}^t K(\\pi_e(s_{t'}), a_{t'})` indicates similarity between action in dataset and that of evaluation policy.
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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

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
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        state_action_value_prediction: NDArray, shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi_e(a \\mid s_t))`.

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
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

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

        if self.use_truncated_kernel:
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
        """Estimate policy value of evaluation policy.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        state_action_value_prediction: NDArray, shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi_e(a \\mid s_t))`.

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
        V_hat: NDArray, shape (n_episodes, )
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
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        state_action_value_prediction: NDArray, shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi_e(a \\mid s_t))`.

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

        alpha: float, default=0.05 (0, 1)
            Significant level.

        ci: str, default="bootstrap"
            Estimation method for confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        estimated_confidence_interval: Dict[str, float]
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
    """Self-Normalized Trajectory-wise Importance Sampling (SNTIS) for continuous OPE (assume deterministic policies).

    Note
    -------
    SNTIS estimates policy value using self-normalized trajectory-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SNTIS}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t \\frac{w_{1:T} \\delta(\\pi_e, a_{0:T})}{\\mathbb{E}_n [w_{1:T} \\delta(\\pi_e, a_{0:T})]} r_t],

    where :math:`w_{0:T} := \\prod_{t=1}^T \\frac{1}{\\pi_b(a_t \\mid s_t)}`
    and :math:`\\delta(\\pi_e, a_{0:T}) = \\prod_{t=1}^T K(\\pi_e(s_t), a_t)` indicates similarity between action in dataset and that of evaluation policy.
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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

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
        """Estimate trajectory-wise policy value.

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
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

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

        if self.use_truncated_kernel:
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
class ContinuousSelfNormalizedStepWiseImportanceSampling(
    ContinuousStepWiseImportanceSampling
):
    """Self-Normalized Step-wise Importance Sampling (SNSIS) for continuous OPE (assume deterministic policies).

    Note
    -------
    SNSIS estimates policy value using self-normalized step-wise importance weight as follows.

    .. math::

        \\hat{V}_{\\mathrm{SNSIS}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t \\frac{w_{1:t} \\delta(\\pi_e, a_{0:t})}{\\mathbb{E}_n [w_{1:t} \\delta(\\pi_e, a_{0:t})]} r_t],

    where :math:`w_{0:t} := \\prod_{t'=1}^t \\frac{1}{\\pi_b(a_{t'} \\mid s_{t'})}`
    and :math:`\\delta(\\pi_e, a_{0:t}) = \\prod_{t'=1}^t K(\\pi_e(s_{t'}), a_{t'})` indicates similarity between action in dataset and that of evaluation policy.
    Note that :math:`K(\\cdot)` is a kernel function.

    Parameters
    -------
    estimator_name: str, default="snsis"
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

    estimator_name: str = "snsis"

    def __post_init__(self):
        self.action_type = "continuous"

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

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
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

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
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

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

        if self.use_truncated_kernel:
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
    """Self-Normalized Doubly Robust (SNDR) for continuous OPE (assume deterministic policies).

    Note
    -------
    SNDR estimates policy value using self-normalized step-wise importance weight and :math:`\\hat{Q}` as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} [\\sum_{t=0}^T \\gamma^t ( \\frac{w_{0:t} \\delta(\\pi_e, a_{0:t})}{\\mathbb{E}_n[w_{0:t} \\delta(\\pi_e, a_{0:t})]} (r_t - \\hat{Q}(s_t, a_t))
            + \\frac{w_{0:t-1} \\delta(\\pi_e, a_{0:t-1})}{\\mathbb{E}_n[w_{0:t-1} \\delta(\\pi_e, a_{0:t-1})]} \\mathbb{E}_{a \\sim \\pi_e(a \\mid s_t)}[\\hat{Q}(s_t, a)])],

    where :math:`w_{0:t} := \\prod_{t'=0}^t \\frac{1}{\\pi_b(a_{t'} \\mid s_{t'})}`
    and :math:`\\delta(\\pi_e, a_{0:t}) = \\prod_{t'=1}^t K(\\pi_e(s_{t'}), a_{t'})` indicates similarity between action in dataset and that of evaluation policy.
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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

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
        """Estimate trajectory-wise policy value.

        Parameters
        -------
        step_per_episode: int (> 0)
            Number of timesteps in an episode.

        action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by behavior policy.

        reward: NDArray, shape (n_episodes * step_per_episode, )
            Reward observation.

        behavior_policy_step_wise_pscore: NDArray, shape (n_episodes * step_per_episode, )
            Step-wise action choice probability of behavior policy,
            i.e., :math:`\\prod_{t'=0}^t \\pi_b(a_{t'} \\mid s_{t'})`

        evaluation_policy_action: NDArray, shape (n_episodes * step_per_episode, action_dim)
            Action chosen by evaluation policy.

        state_action_value_prediction: NDArray, shape (n_episodes * step_per_episode, )
            :math:`\\hat{Q}` for the action chosen by evaluation policy,
            i.e., :math:`\\hat{Q}(s_t, \\pi_e(a \\mid s_t))`.

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
        estimated_trajectory_wise_policy_value: NDArray, shape (n_episodes, )
            Estimated policy value for each trajectory.

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

        if self.use_truncated_kernel:
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
