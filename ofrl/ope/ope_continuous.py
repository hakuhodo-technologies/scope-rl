"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from collections import defaultdict

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from .estimators_base import BaseOffPolicyEstimator
from ..types import LoggedDataset, OPEInputDict
from ..utils import (
    estimate_confidence_interval_by_bootstrap,
    estimate_confidence_interval_by_hoeffding,
    estimate_confidence_interval_by_empirical_bernstein,
    estimate_confidence_interval_by_t_test,
    defaultdict_to_dict,
    check_array,
    check_logged_dataset,
    check_input_dict,
)


@dataclass
class ContinuousOffPolicyEvaluation:
    """Class to perform a continuous-action OPE by multiple estimators simultaneously.

    Note
    -----------
    OPE estimates the expected policy performance called the policy value.

    .. math::

        V(\\pi) := \\mathbb{E} \\left[ \\sum_{t=0}^{T-1} \\gamma^t r_t \\mid \\pi \\right]

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    ope_estimators: list of BaseOffPolicyEstimator
        List of OPE estimators used to evaluate the policy value of the evaluation policies.
        Estimators must follow the interface of `ofrl.ope.BaseOffPolicyEstimator`.

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

    Examples
    ----------
    .. ::code-block:: python

        # import necessary module from OFRL
        >>> from ofrl.dataset import SyntheticDataset
        >>> from ofrl.policy import ContinuousTruncatedGaussianHead, ContinuousEvalHead
        >>> from ofrl.ope import CreateOPEInput
        >>> from ofrl.ope import ContinuousOffPolicyEvaluation as OPE
        >>> from ofrl.ope import ContinuousTrajectoryWiseImportanceSampling as TIS
        >>> from ofrl.ope import ContinuousPerDecisionImportanceSampling as PDIS

        # import necessary module from other libraries
        >>> import gym
        >>> import rtbgym
        >>> import numpy as np
        >>> from d3rlpy.algos import SAC, RandomPolicy
        >>> from d3rlpy.online.buffers import ReplayBuffer
        >>> from d3rlpy.preprocessing import MinMaxActionScaler

        # initialize environment
        >>> env = gym.make("RTBEnv-continuous-v0")

        # define (RL) agent (i.e., policy) and train on the environment
        >>> sac = SAC(
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,
                    maximum=env.action_space.high,
                ),
            )
        >>> buffer = ReplayBuffer(
                maxlen=10000,
                env=env,
            )
        >>> sac.fit_online(
                env=env,
                buffer=buffer,
                n_steps=10000,
                n_steps_per_epoch=1000,
            )

        # convert ddqn policy to stochastic data collection policy
        >>> behavior_policy = ContinuousTruncatedGaussianHead(
                sac,
                minimum=env.action_space.low,
                maximum=env.action_space.high,
                sigma=np.array([1.0]),
                name="sac_sigma_1.0",
                random_state=12345,
            )

        # initialize dataset class
        >>> dataset = SyntheticDataset(
                env=env,
                behavior_policy=behavior_policy,
                is_rtb_env=True,
                random_state=12345,
            )

        # data collection
        >>> logged_dataset = dataset.obtain_trajectories(n_episodes=100, obtain_info=True)

        # evaluation policy
        >>> sac_ = ContinuousEvalHead(
                base_policy=sac,
                name="sac",
            )
        >>> random_ = ContinuousEvalHead(
                base_policy=RandomPolicy(
                    action_scaler=MinMaxActionScaler(
                        minimum=env.action_space.low,
                        maximum=env.action_space.high,
                    )
                ),
                name="random",
            )

        # create input for off-policy evaluation (OPE)
        >>> prep = CreateOPEInput(
                logged_dataset=logged_dataset,
            )
        >>> input_dict = prep.obtain_whole_inputs(
                evaluation_policies=[sac_, random_],
                env=env,
                n_episodes_on_policy_evaluation=100,
                random_state=12345,
            )

        # OPE
        >>> ope = OPE(
                logged_dataset=logged_dataset,
                ope_estimators=[TIS(), PDIS()],
                use_truncated_kernel=True,
                action_min=env.action_space.low,
                action_max=env.action_space.high,
            )
        >>> policy_value_dict = ope.estimate_policy_value(
                input_dict=input_dict,
            )
        >>> policy_value_dict
        {'sac': {'on_policy': 14.49, 'tis': 7.1201786764411965, 'pdis': 7.405100089592586},
        'random': {'on_policy': 14.41, 'tis': 0.032785084929549006, 'pdis': 4.708319248395723}}

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments.", 2019.

    Josiah P. Hanna, Peter Stone, and Scott Niekum.
    "Bootstrapping with Models: Confidence Intervals for Off-Policy Evaluation.", 2017.

    Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh.
    "High Confidence Policy Improvement.", 2015.

    Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh.
    "High Confidence Off-Policy Evaluation.", 2015.

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]
    sigma: Optional[np.ndarray] = None
    use_truncated_kernel: bool = False
    action_min: Optional[np.ndarray] = None
    action_max: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.step_per_episode = self.logged_dataset["step_per_episode"]

        if self.logged_dataset["action_type"] != "continuous":
            raise ValueError("logged_dataset does not `continuous` action_type")

        self.action_dim = self.logged_dataset["action_dim"]

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != "continuous":
                raise RuntimeError(
                    f"One of the ope_estimators, {estimator.estimator_name} does not match `continuous` action_type"
                )

            if not isinstance(estimator, BaseOffPolicyEstimator):
                raise RuntimeError(
                    f"ope_estimators must be child classes of BaseOffPolicyEstimator, but one of them, {estimator.estimator_name} is not"
                )

        if self.sigma is not None:
            check_array(self.sigma, name="sigma", expected_dim=1, min_val=0.0)

            if self.sigma.shape[0] != self.action_dim:
                raise ValueError(
                    "the length of sigma must be the same with logged_dataset['action_dim'], but found False"
                )

        if self.use_truncated_kernel:
            check_array(self.action_min, name="action_min", expected_dim=1)
            check_array(self.action_max, name="action_max", expected_dim=1)

        if not (
            self.action_dim == self.action_min.shape[0] == self.action_max.shape[0]
        ):
            raise ValueError(
                "expected `logged_dataset['action_dim'] == action_min.shape[0] == action_max.shape[0]`, but found False"
            )

        behavior_policy_pscore = self.logged_dataset["pscore"].reshape(
            (-1, self.step_per_episode)
        )
        behavior_policy_step_wise_pscore = np.cumprod(behavior_policy_pscore, axis=1)
        behavior_policy_trajectory_wise_pscore = np.tile(
            behavior_policy_step_wise_pscore[:, -1], (self.step_per_episode, 1)
        ).T

        self.input_dict_ = {
            "step_per_episode": self.step_per_episode,
            "action": self.logged_dataset["action"].astype(int),
            "reward": self.logged_dataset["reward"],
            "behavior_policy_step_wise_pscore": behavior_policy_step_wise_pscore.flatten(),
            "behavior_policy_trajectory_wise_pscore": behavior_policy_trajectory_wise_pscore.flatten(),
        }

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def estimate_policy_value(
        self,
        input_dict: OPEInputDict,
    ) -> Dict[str, float]:
        """Estimate the policy value of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            Please refer to `CreateOPEInput` class for the detail.
            key: [evaluation_policy_name][
                evaluation_policy_step_wise_pscore,
                evaluation_policy_trajectory_wise_pscore,
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                initial_state_action_distribution,
                on_policy_policy_value,
                gamma,
            ]

        Return
        -------
        policy_value_dict: dict
            Dictionary containing the policy value of each evaluation policy estimated by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        check_input_dict(input_dict)
        policy_value_dict = defaultdict(dict)

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                policy_value_dict[eval_policy]["on_policy"] = input_dict[eval_policy][
                    "on_policy_policy_value"
                ].mean()
            else:
                policy_value_dict[eval_policy]["on_policy"] = None

            for estimator_name, estimator in self.ope_estimators_.items():
                policy_value_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_policy_value(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    sigma=self.sigma,
                    use_truncated_kernel=self.use_truncated_kernel,
                    action_min=self.action_min,
                    action_max=self.action_max,
                )
        return defaultdict_to_dict(policy_value_dict)

    def estimate_intervals(
        self,
        input_dict: OPEInputDict,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate the confidence intervals of the policy value by nonparametric bootstrap.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            Please refer to `CreateOPEInput` class for the detail.
            key: [evaluation_policy_name][
                evaluation_policy_step_wise_pscore,
                evaluation_policy_trajectory_wise_pscore,
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                initial_state_action_distribution,
                on_policy_policy_value,
                gamma,
            ]

        alpha: float, default=0.05 (0, 1)
            Significance level.

        ci: str, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        policy_value_interval_dict: dict
            Dictionary containing the confidence intervals estimated by nonparametric bootstrap.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        check_input_dict(input_dict)
        policy_value_interval_dict = defaultdict(dict)

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                policy_value_interval_dict[eval_policy][
                    "on_policy"
                ] = self._estimate_confidence_interval[ci](
                    input_dict[eval_policy]["on_policy_policy_value"],
                    alpha=alpha,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )
            else:
                policy_value_interval_dict[eval_policy]["on_policy"] = None

            for estimator_name, estimator in self.ope_estimators_.items():
                policy_value_interval_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_interval(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    sigma=self.sigma,
                    use_truncated_kernel=self.use_truncated_kernel,
                    action_min=self.action_min,
                    action_max=self.action_max,
                    alpha=alpha,
                    ci=ci,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

        return defaultdict_to_dict(policy_value_interval_dict)

    def summarize_off_policy_estimates(
        self,
        input_dict: OPEInputDict,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]:
        """Summarize the policy value and their confidence intervals estimated by OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            Please refer to `CreateOPEInput` class for the detail.
            key: [evaluation_policy_name][
                evaluation_policy_step_wise_pscore,
                evaluation_policy_trajectory_wise_pscore,
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                initial_state_action_distribution,
                on_policy_policy_value,
                gamma,
            ]

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        policy_value_dict: dict
            Dictionary containing the policy value of each evaluation policy estimated by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        policy_value_interval_dict: dict
            Dictionary containing the confidence intervals estimated by nonparametric bootstrap.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        check_input_dict(input_dict)
        policy_value_dict = self.estimate_policy_value(input_dict)
        policy_value_interval_dict = self.estimate_intervals(
            input_dict,
            alpha=alpha,
            ci=ci,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

        policy_value_df_dict = dict()
        policy_value_interval_df_dict = dict()

        for eval_policy in input_dict.keys():
            policy_value_df_ = DataFrame(
                policy_value_dict[eval_policy],
                index=["policy_value"],
            ).T

            on_policy_policy_value = None
            if policy_value_dict[eval_policy]["on_policy"] is not None:
                on_policy_policy_value = policy_value_dict[eval_policy][
                    "on_policy"
                ].mean()
            if on_policy_policy_value is not None and on_policy_policy_value > 0:
                policy_value_df_["relative_policy_value"] = (
                    policy_value_df_ / on_policy_policy_value
                )
            else:
                policy_value_df_["relative_policy_value"] = np.nan
            policy_value_dict
            policy_value_df_dict[eval_policy] = policy_value_df_

            policy_value_interval_df_dict[eval_policy] = DataFrame(
                policy_value_interval_dict[eval_policy],
            ).T

        return policy_value_df_dict, policy_value_interval_df_dict

    def visualize_off_policy_estimates(
        self,
        input_dict: OPEInputDict,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        is_relative: bool = False,
        hue: str = "estimator",
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize the policy value estimated by OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            Please refer to `CreateOPEInput` class for the detail.
            key: [evaluation_policy_name][
                evaluation_policy_step_wise_pscore,
                evaluation_policy_trajectory_wise_pscore,
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                initial_state_action_distribution,
                on_policy_policy_value,
                gamma,
            ]

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        is_relative: bool, default=False
            If True, the method visualizes the estimated policy value of the evaluation policies
            relative to the ground-truth policy value of the behavior policy.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        sharey: bool, default=False
            If True, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        check_input_dict(input_dict)
        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )
        if hue not in ["estimator", "policy"]:
            raise ValueError(
                f"hue must be either `estimator` or `policy`, but {hue} is given"
            )
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        if hue == "estimator":
            estimated_trajectory_value_df_dict = dict()
            for eval_policy in input_dict.keys():
                estimated_trajectory_value_dict_ = dict()
                for estimator_name, estimator in self.ope_estimators_.items():
                    estimated_trajectory_value_dict_[
                        estimator_name
                    ] = estimator._estimate_trajectory_value(
                        **input_dict[eval_policy],
                        **self.input_dict_,
                        sigma=self.sigma,
                        use_truncated_kernel=self.use_truncated_kernel,
                        action_min=self.action_min,
                        action_max=self.action_max,
                    )
                estimated_trajectory_value_df_ = DataFrame(
                    estimated_trajectory_value_dict_
                )

                on_policy_policy_value = input_dict[eval_policy][
                    "on_policy_policy_value"
                ]
                if is_relative:
                    if (
                        on_policy_policy_value is not None
                        and on_policy_policy_value > 0
                    ):
                        estimated_trajectory_value_df_dict[eval_policy] = (
                            estimated_trajectory_value_df_
                            / on_policy_policy_value.mean()
                        )
                    else:
                        raise ValueError(
                            f"on_policy_policy_value must be a positive value, but {on_policy_policy_value} is given"
                        )

                estimated_trajectory_value_df_dict[
                    eval_policy
                ] = estimated_trajectory_value_df_

        else:
            estimated_trajectory_value_df_dict = dict()
            for estimator_name, estimator in self.ope_estimators_.items():

                estimated_trajectory_value_dict_ = dict()
                for eval_policy in input_dict.keys():
                    estimated_trajectory_value_dict_[
                        eval_policy
                    ] = estimator._estimate_trajectory_value(
                        **input_dict[estimator_name],
                        **self.input_dict_,
                        sigma=self.sigma,
                        use_truncated_kernel=self.use_truncated_kernel,
                        action_min=self.action_min,
                        action_max=self.action_max,
                    )

                    on_policy_policy_value = input_dict[eval_policy][
                        "on_policy_policy_value"
                    ]
                    if is_relative:
                        if (
                            on_policy_policy_value is not None
                            and on_policy_policy_value > 0
                        ):
                            estimated_trajectory_value_dict_[eval_policy] = (
                                estimated_trajectory_value_dict_[eval_policy]
                                / on_policy_policy_value.mean()
                            )
                        else:
                            raise ValueError(
                                f"on_policy_policy_value must be a positive value, but {on_policy_policy_value} is given"
                            )

                estimated_trajectory_value_df_ = DataFrame(
                    estimated_trajectory_value_dict_
                )
                estimated_trajectory_value_df_dict[
                    estimator_name
                ] = estimated_trajectory_value_df_

        plt.style.use("ggplot")

        if hue == "estimator":
            fig = plt.figure(
                figsize=(2 * len(self.ope_estimators_), 12 * len(input_dict))
            )

            for i, eval_policy in enumerate(input_dict.keys()):
                if i == 0:
                    ax = ax0 = fig.add_subplot(len(self.ope_estimators_), 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(
                        len(self.ope_estimators_), 1, i + 1, sharey=ax0
                    )
                else:
                    ax = fig.add_subplot(len(self.ope_estimators_), 1, i + 1)

                sns.barplot(
                    data=estimated_trajectory_value_df_dict[eval_policy],
                    ax=ax,
                    ci=100 * (1 - alpha),
                    n_boot=n_bootstrap_samples,
                    seed=random_state,
                )
                on_policy_policy_value = input_dict[eval_policy][
                    "on_policy_policy_value"
                ]
                if on_policy_policy_value is not None:
                    on_policy_interval = self._estimate_confidence_interval[ci](
                        samples=on_policy_policy_value,
                        alpha=alpha,
                        n_bootstrap_samples=n_bootstrap_samples,
                        random_state=random_state,
                    )
                if is_relative:
                    ax.axhline(1.0)
                    ax.axhspan(
                        ymin=on_policy_interval[f"{100 * (1. - alpha)}% CI (lower)"]
                        / on_policy_interval["mean"],
                        ymax=on_policy_interval[f"{100 * (1. - alpha)}% CI (upper)"]
                        / on_policy_interval["mean"],
                        alpha=0.3,
                    )
                else:
                    ax.axhline(on_policy_interval["mean"])
                    ax.axhspan(
                        ymin=on_policy_interval[f"{100 * (1. - alpha)}% CI (lower)"],
                        ymax=on_policy_interval[f"{100 * (1. - alpha)}% CI (upper)"],
                        alpha=0.3,
                    )
                ax.set_title(eval_policy, fontsize=16)
                ax.set_ylabel(
                    f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)",
                    fontsize=12,
                )
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)

        else:
            fig = plt.figure(
                figsize=(2 * len(input_dict), 12 * len(self.ope_estimators_))
            )

            for i, estimator in enumerate(self.ope_estimators_.keys()):
                if i == 0:
                    ax = ax0 = fig.add_subplot(len(input_dict), 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(len(input_dict), 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(len(input_dict), 1, i + 1)

                sns.barplot(
                    data=estimated_trajectory_value_df_dict[eval_policy],
                    ax=ax,
                    ci=100 * (1 - alpha),
                    n_boot=n_bootstrap_samples,
                    seed=random_state,
                )
                on_policy_policy_value = input_dict[eval_policy][
                    "on_policy_policy_value"
                ]
                if on_policy_policy_value is not None:
                    on_policy_interval = self._estimate_confidence_interval[ci](
                        samples=on_policy_policy_value,
                        alpha=alpha,
                        n_bootstrap_samples=n_bootstrap_samples,
                        random_state=random_state,
                    )
                if is_relative:
                    ax.axhline(1.0)
                    ax.axhspan(
                        ymin=on_policy_interval[f"{100 * (1. - alpha)}% CI (lower)"]
                        / on_policy_interval["mean"],
                        ymax=on_policy_interval[f"{100 * (1. - alpha)}% CI (upper)"]
                        / on_policy_interval["mean"],
                        alpha=0.3,
                    )
                else:
                    ax.axhline(on_policy_interval["mean"])
                    ax.axhspan(
                        ymin=on_policy_interval[f"{100 * (1. - alpha)}% CI (lower)"],
                        ymax=on_policy_interval[f"{100 * (1. - alpha)}% CI (upper)"],
                        alpha=0.3,
                    )
                ax.set_title(eval_policy, fontsize=16)
                ax.set_ylabel(
                    f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)",
                    fontsize=12,
                )
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def evaluate_performance_of_estimators(
        self,
        input_dict: OPEInputDict,
        metric: str = "relative-ee",
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the estimation performance/accuracy of OPE estimators.

        Note
        -------
        Evaluate the estimation performance/accuracy of OPE estimators by relative estimation error (relative-EE) or squared error (SE).

        .. math ::

            \\mathrm{Relative-EE}(\\hat{V}; \\mathcal{D})
            := \\left| \\frac{\\hat{V}(\\pi; \\mathcal{D}) - V_{\\mathrm{on}}(\\pi)}{V_{\\mathrm{on}}(\\pi)} \\right|,

        .. math ::

            \\mathrm{SE}(\\hat{V}; \\mathcal{D}) := \\left( \\hat{V}(\\pi; \\mathcal{D}) - V_{\\mathrm{on} \\right)^2,

        where :math:`V_{\\mathrm{on}}(\\pi)` is the on-policy policy value of the evaluation policy :math:`\\pi`.
        :math:`\\hat{V}(\\pi; \\mathcal{D})` is the policy value estimated by the OPE estimator :math:`\\hat{V}` and logged dataset :math:`\\mathcal{D}`.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            Please refer to `CreateOPEInput` class for the detail.
            key: [evaluation_policy_name][
                evaluation_policy_step_wise_pscore,
                evaluation_policy_trajectory_wise_pscore,
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                initial_state_action_distribution,
                on_policy_policy_value,
                gamma,
            ]

        metric: {"relative-ee", "se"}, default="relative-ee"
            Evaluation metric used to evaluate and compare the estimation performance/accuracy of OPE estimators.

        Return
        -------
        eval_metric_ope_dict: dict
            Dictionary containing evaluation metric for evaluating the estimation performance/accuracy of OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        check_input_dict(input_dict)
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"metric must be either 'relative-ee' or 'se', but {metric} is given"
            )
        eval_metric_ope_dict = defaultdict(dict)
        policy_value_dict = self.estimate_policy_value(input_dict)

        if metric == "relative-ee":
            for eval_policy in input_dict.keys():
                on_policy_policy_value = input_dict[eval_policy][
                    "on_policy_policy_value"
                ]

                for estimator in self.ope_estimators_.keys():
                    relative_ee_ = (
                        policy_value_dict[eval_policy][estimator]
                        - on_policy_policy_value
                    ) / on_policy_policy_value
                    eval_metric_ope_dict[eval_policy][estimator] = np.abs(relative_ee_)

        else:
            for eval_policy in input_dict.keys():
                on_policy_policy_value = input_dict[eval_policy][
                    "on_policy_policy_value"
                ].mean()

                for estimator in self.ope_estimators_.keys():
                    se_ = (
                        policy_value_dict[eval_policy][estimator]
                        - on_policy_policy_value
                    ) ** 2
                    eval_metric_ope_dict[eval_policy][estimator] = se_

        return defaultdict_to_dict(eval_metric_ope_dict)

    def summarize_estimators_comparison(
        self,
        input_dict: OPEInputDict,
        metric: str = "relative-ee",
    ) -> DataFrame:
        """Summarize performance comparison of OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            Please refer to `CreateOPEInput` class for the detail.
            key: [evaluation_policy_name][
                evaluation_policy_step_wise_pscore,
                evaluation_policy_trajectory_wise_pscore,
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                initial_state_action_distribution,
                on_policy_policy_value,
                gamma,
            ]

        metric: {"relative-ee", "se"}, default="relative-ee"
            Evaluation metric used to evaluate and compare the estimation performance/accuracy of OPE estimators.

        Return
        -------
        eval_metric_ope_df: dataframe
            Dictionary containing evaluation metric for evaluating the estimation performance/accuracy of OPE estimators.

        """
        check_input_dict(input_dict)
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"metric must be either 'relative-ee' or 'se', but {metric} is given"
            )
        eval_metric_ope_df = DataFrame()
        eval_metric_ope_dict = self.evaluate_performance_of_estimators(
            input_dict,
            metric=metric,
        )
        for eval_policy in input_dict.keys():
            eval_metric_ope_df[eval_policy] = DataFrame(
                eval_metric_ope_dict[eval_policy], index=[eval_policy]
            ).T
        return eval_metric_ope_df
