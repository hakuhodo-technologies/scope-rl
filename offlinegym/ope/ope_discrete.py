"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from random import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from collections import defaultdict

import numpy as np
from scipy.stats import norm
from sklearn.utils import check_scalar
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from offlinegym.ope.estimators_base import (
    BaseOffPolicyEstimator,
    BaseCumulativeDistributionalOffPolicyEstimator,
    BaseDistributionallyRobustOffPolicyEstimator,
)
from offlinegym.types import LoggedDataset, OPEInputDict
from offlinegym.utils import (
    estimate_confidence_interval_by_bootstrap,
    estimate_confidence_interval_by_hoeffding,
    estimate_confidence_interval_by_empirical_bernstein,
    estimate_confidence_interval_by_t_test,
    defaultdict_to_dict,
    check_array,
    check_logged_dataset,
)


@dataclass
class DiscreteOffPolicyEvaluation:
    """Class to conduct OPE by multiple estimators simultaneously for discrete action space.

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    ope_estimators: List[BaseOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `offlinegym.ope.BaseOffPolicyEstimator`.

    Examples
    ----------
    .. ::code-block:: python

        # import necessary module from offlinegym
        >>> from offlinegym.dataset import SyntheticDataset
        >>> from offlinegym.policy import DiscreteEpsilonGreedyHead
        >>> from offlinegym.ope import CreateOPEInput
        >>> from offlinegym.ope import OffPolicyEvaluation
        >>> from offlinegym.ope import DiscreteTrajectoryWiseImportanceSampling as TIS
        >>> from offlinegym.ope import DiscreteStepWiseImportanceSampling as SIS

        # import necessary module from other libraries
        >>> from rtbgym import RTBEnv, CustomizedRTBEnv
        >>> from sklearn.linear_model import LogisticRegression
        >>> from d3rlpy.algos import DoubleDQN
        >>> from d3rlpy.online.buffers import ReplayBuffer
        >>> from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        >>> env = RTBEnv(random_state=12345)

        # customize environment from the decision makers' perspective
        >>> env = CustomizedRTBEnv(
                original_env=env,
                reward_predictor=LogisticRegression(),
                action_type="discrete",
            )

        # define (RL) agent (i.e., policy) and train on the environment
        >>> ddqn = DoubleDQN()
        >>> buffer = ReplayBuffer(
                maxlen=10000,
                env=env,
            )
        >>> explorer = ConstantEpsilonGreedy(
                epsilon=0.3,
            )
        >>> ddqn.fit_online(
                env=env,
                buffer=buffer,
                explorer=explorer,
            )

        # convert ddqn policy to stochastic data collection policy
        >>> behavior_policy = DiscreteEpsilonGreedyHead(
                ddqn,
                n_actions=env.action_space.n,
                epsilon=0.3,
                name="ddqn_epsilon_0.3",
                random_state=12345,
            )

        # initialize dataset class
        >>> dataset = SyntheticDataset(
                env=env,
                behavior_policy=behavior_policy,
                random_state=12345,
            )

        # data collection
        >>> logged_dataset = dataset.obtain_trajectories(n_episodes=100, obtain_info=True)

        # evaluation policy
        >>> ddqn_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="ddqn",
            epsilon=0.0,
            random_state=12345
        )
        >>> random_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="random",
            epsilon=1.0,
            random_state=12345
        )

        # create input for off-policy evaluation (OPE)
        >>> prep = CreateOPEInput(
            logged_dataset=logged_dataset,
        )
        >>> input_dict = prep.obtain_whole_inputs(
            evaluation_policies=[ddqn_, random_],
            env=env,
            n_episodes_on_policy_evaluation=100,
            random_state=12345,
        )

        # OPE
        >>> ope = OffPolicyEvaluation(
            logged_dataset=logged_dataset,
            ope_estimators=[TIS(), SIS()],
        )
        >>> policy_value_dict = ope.estimate_policy_value(
            input_dict=input_dict,
        )
        >>> policy_value_dict
        {'ddqn': {'on_policy': 15.5, 'tis': 22.901319216705502, 'sis': 17.970922685707617},
        'random': {'on_policy': 15.5, 'tis': 0.555637908601827, 'sis': 6.108053435521632}}


    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Josiah P. Hanna, Peter Stone, and Scott Niekum.
    "Bootstrapping with Models: Confidence Intervals for Off-Policy Evaluation.", 2017.

    Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh.
    "High Confidence Off-Policy Evaluation.", 2015.

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.step_per_episode = self.logged_dataset["step_per_episode"]

        if self.logged_dataset["action_type"] != "discrete":
            raise RuntimeError("logged_dataset does not `discrete` action_type")

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != "discrete":
                raise RuntimeError(
                    f"One of the ope_estimators, {estimator.estimator_name} does not match `discrete` action_type"
                )

            if not isinstance(estimator, BaseOffPolicyEstimator):
                raise RuntimeError(
                    f"ope_estimators must be child classes of BaseOffPolicyEstimator, but one of them, {estimator.estimator_name} is not"
                )

        self.behavior_policy_value = (
            self.logged_dataset["reward"]
            .reshape((-1, self.step_per_episode))
            .sum(axis=1)
            .mean()
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
        gamma: float = 1.0,
    ) -> Dict[str, float]:
        """Estimate the policy value of evaluation policy.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        policy_value_dict: Dict[str, Dict[str, float]]
            Dictionary containing estimated policy value of each evaluation policy by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
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
                    gamma=gamma,
                )
        return defaultdict_to_dict(policy_value_dict)

    def estimate_intervals(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate confidence intervals of policy value using nonparametric bootstrap procedure.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        ci: str, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=100 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        policy_value_interval_dict: Dict[str, Dict[str, Dict[str, float]]]
            Dictionary containing estimated confidence intervals estimated
            using nonparametric bootstrap procedure.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
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
                    gamma=gamma,
                    alpha=alpha,
                    ci=ci,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

        return defaultdict_to_dict(policy_value_interval_dict)

    def summarize_off_policy_estimates(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]:
        """Summarize policy value and their confidence intervals estimated by OPE estimators.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        ci: str, default="bootstrap"
            Estimation method for confidence intervals.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        (policy_value_df_dict, policy_value_interval_df_dict): Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]
            Dictionary containing policy value and their confidence intervals.
            key: [evaluation_policy_name]

        """
        policy_value_dict = self.estimate_policy_value(input_dict)
        policy_value_interval_dict = self.estimate_intervals(
            input_dict,
            gamma=gamma,
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
        gamma: float = 1.0,
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
        """Visualize policy value estimated by OPE estimators.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        ci: str, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        is_relative: bool, default=False
            If `True`, the method visualizes the estimated policy value of evaluation policy
            relative to the on-policy policy value of the behavior policy.

        hue: str, default="estimator"
            Hue of the plot.
            Choose either from "estimator" or "policy".

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
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

        policy_value_interval_dict = self.estimate_intervals(
            input_dict=input_dict,
            gamma=gamma,
            alpha=alpha,
            ci=ci,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        if hue == "estimator":
            fig = plt.figure(
                figsize=(2 * len(self.ope_estimators_), 4 * len(input_dict))
            )

            for i, eval_policy in enumerate(input_dict.keys()):
                if i == 0:
                    ax = ax0 = fig.add_subplot(len(input_dict), 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(len(input_dict), 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(len(input_dict), 1, i + 1)

                mean = np.zeros(len(self.ope_estimators_))
                lower = np.zeros(len(self.ope_estimators_))
                upper = np.zeros(len(self.ope_estimators_))

                for j, estimator in enumerate(self.ope_estimators_):
                    mean[j] = policy_value_interval_dict[eval_policy][estimator]["mean"]
                    lower[j] = policy_value_interval_dict[eval_policy][estimator][
                        f"{100 * (1. - alpha)}% CI (lower)"
                    ]
                    upper[j] = policy_value_interval_dict[eval_policy][estimator][
                        f"{100 * (1. - alpha)}% CI (upper)"
                    ]

                if is_relative:
                    mean = mean / self.behavior_policy_value
                    lower = lower / self.behavior_policy_value
                    upper = upper / self.behavior_policy_value

                ax.bar(
                    np.arange(len(self.ope_estimators_)),
                    mean,
                    yerr=[upper - mean, mean - lower],
                    color=color,
                    tick_label=list(self.ope_estimators_.keys()),
                )

                on_policy_interval = policy_value_interval_dict[eval_policy][
                    "on_policy"
                ]
                if on_policy_interval is not None:
                    if is_relative:
                        ax.axhline(
                            on_policy_interval["mean"] / self.behavior_policy_value
                        )
                        ax.axhspan(
                            ymin=on_policy_interval[f"{100 * (1. - alpha)}% CI (lower)"]
                            / self.behavior_policy_value,
                            ymax=on_policy_interval[f"{100 * (1. - alpha)}% CI (upper)"]
                            / self.behavior_policy_value,
                            alpha=0.3,
                        )

                    else:
                        ax.axhline(on_policy_interval["mean"])
                        ax.axhspan(
                            ymin=on_policy_interval[
                                f"{100 * (1. - alpha)}% CI (lower)"
                            ],
                            ymax=on_policy_interval[
                                f"{100 * (1. - alpha)}% CI (upper)"
                            ],
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
            visualize_on_policy = True
            for eval_policy in input_dict:
                if input_dict[eval_policy]["on_policy_policy_value"] is None:
                    visualize_on_policy = False

            if visualize_on_policy:
                n_estimators = len(self.ope_estimators_) + 1

            fig = plt.figure(figsize=(2 * len(input_dict), 4 * n_estimators))

            for i, estimator in enumerate(self.ope_estimators_):
                if i == 0:
                    ax = ax0 = fig.add_subplot(n_estimators, 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(n_estimators, 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(n_estimators, 1, i + 1)

                mean = np.zeros(len(input_dict))
                lower = np.zeros(len(input_dict))
                upper = np.zeros(len(input_dict))

                for j, eval_policy in enumerate(input_dict):
                    mean[j] = policy_value_interval_dict[eval_policy][estimator]["mean"]
                    lower[j] = policy_value_interval_dict[eval_policy][estimator][
                        f"{100 * (1. - alpha)}% CI (lower)"
                    ]
                    upper[j] = policy_value_interval_dict[eval_policy][estimator][
                        f"{100 * (1. - alpha)}% CI (upper)"
                    ]

                if is_relative:
                    mean = mean / self.behavior_policy_value
                    lower = lower / self.behavior_policy_value
                    upper = upper / self.behavior_policy_value

                ax.bar(
                    np.arange(len(input_dict)),
                    mean,
                    yerr=[upper - mean, mean - lower],
                    color=color,
                    tick_label=list(input_dict.keys()),
                )

                ax.set_title(estimator, fontsize=16)
                ax.set_ylabel(
                    f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)",
                    fontsize=12,
                )
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)

            if visualize_on_policy:
                if sharey:
                    ax = fig.add_subplot(n_estimators, 1, i + 2, sharey=ax0)
                else:
                    ax = fig.add_subplot(n_estimators, 1, i + 2)

                mean = np.zeros(len(input_dict))
                lower = np.zeros(len(input_dict))
                upper = np.zeros(len(input_dict))

                for j, eval_policy in enumerate(input_dict):
                    mean[j] = policy_value_interval_dict[eval_policy]["on_policy"][
                        "mean"
                    ]
                    lower[j] = policy_value_interval_dict[eval_policy]["on_policy"][
                        f"{100 * (1. - alpha)}% CI (lower)"
                    ]
                    upper[j] = policy_value_interval_dict[eval_policy]["on_policy"][
                        f"{100 * (1. - alpha)}% CI (upper)"
                    ]

                if is_relative:
                    mean = mean / self.behavior_policy_value
                    lower = lower / self.behavior_policy_value
                    upper = upper / self.behavior_policy_value

                ax.bar(
                    np.arange(len(input_dict)),
                    mean,
                    yerr=[upper - mean, mean - lower],
                    color=color,
                    tick_label=list(input_dict.keys()),
                )

                ax.set_title("on_policy", fontsize=16)
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
        gamma: float = 1.0,
        metric: str = "relative-ee",
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate estimation performance of OPE estimators.

        Note
        -------
        Evaluate the estimation performance of OPE estimators by relative estimation error (relative-EE) or squared error (SE).

        .. math ::

            \\mathrm{Relative-EE}(\\hat{V}; \\mathcal{D})
            := \\left| \\frac{\\hat{V}(\\pi_e; \\mathcal{D}) - V_{\\mathrm{on}}(\\pi_e)}{V_{\\mathrm{on}}(\\pi_e)} \\right|,

        .. math ::

            \\mathrm{SE}(\\hat{V}; \\mathcal{D}) := \\left( \\hat{V}(\\pi_e; \\mathcal{D}) - V_{\\mathrm{on} \\right)^2,

        where :math:`V_{\\mathrm{on}}(\\pi_e)` is the on-policy policy value of the evaluation policy :math:`\\pi_e`.
        :math:`\\hat{V}(\\pi_e; \\mathcal{D})` is the estimated policy value by an OPE estimator :math:`\\hat{V}` and logged dataset :math:`\\mathcal{D}`.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        metric: str, default="relative-ee"
            Evaluation metric used to evaluate and compare the estimation performance of OPE estimators.
            Either "relative-ee" or "se".

        Return
        -------
        eval_metric_ope_dict: Dict[str, DIct[str, float]]
            Dictionary containing evaluation metric for evaluating the estimation performance of OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"metric must be either 'relative-ee' or 'se', but {metric} is given"
            )
        eval_metric_ope_dict = defaultdict(dict)
        policy_value_dict = self.estimate_policy_value(input_dict, gamma=gamma)

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
        gamma: float = 1.0,
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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        metric: str, default="relative-ee"
            Evaluation metric used to evaluate and compare the estimation performance of OPE estimators.
            Either "relative-ee" or "se".

        Return
        -------
        eval_metric_ope_df: DataFrame
            Dictionary containing evaluation metric for evaluating the estimation performance of OPE estimators.

        """
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"metric must be either 'relative-ee' or 'se', but {metric} is given"
            )
        eval_metric_ope_df = DataFrame()
        eval_metric_ope_dict = self.evaluate_performance_of_estimators(
            input_dict,
            gamma=gamma,
            metric=metric,
        )
        for eval_policy in input_dict.keys():
            eval_metric_ope_df[eval_policy] = DataFrame(
                eval_metric_ope_dict[eval_policy], index=[eval_policy]
            ).T
        return eval_metric_ope_df


@dataclass
class DiscreteCumulativeDistributionalOffPolicyEvaluation:
    """Class to conduct cumulative distributional OPE by multiple estimators simultaneously in discrete action space.

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    ope_estimators: List[BaseOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `offlinegym.ope.BaseCumulativeDistributionalOffPolicyEstimator`.

    scale_min: float, default=None
        Minimum value of the reward scale in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in CDF.
        When `use_observations_as_reward_scale == False`, a value must be given.

    n_partitiion: int, default=None
        Number of partition in reward scale (x-axis of CDF).
        When `use_observations_as_reward_scale == False`, a value must be given.

    use_observations_as_reward_scale: bool, default=False
        Whether to use the reward observed by the behavior policy as the reward scale.
        If True, the reward scale follows the one defined in Chundak et al. (2021).
        If False, the reward scale is uniform, following Huang et al. (2021).

    Examples
    ----------
    .. ::code-block:: python

        # import necessary module from offlinegym
        >>> from offlinegym.dataset import SyntheticDataset
        >>> from offlinegym.policy import DiscreteEpsilonGreedyHead
        >>> from offlinegym.ope import CreateOPEInput
        >>> from offlinegym.ope import CumulativeDistributionalOffPolicyEvaluation
        >>> from offlinegym.ope import DiscreteCumulativeDistributionalImportanceSampling as CDIS
        >>> from offlinegym.ope import DiscreteCumulativeDistributionalSelfNormalizedImportanceSampling as CDSNIS

        # import necessary module from other libraries
        >>> from rtbgym import RTBEnv, CustomizedRTBEnv
        >>> from sklearn.linear_model import LogisticRegression
        >>> from d3rlpy.algos import DoubleDQN
        >>> from d3rlpy.online.buffers import ReplayBuffer
        >>> from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        >>> env = RTBEnv(random_state=12345)

        # customize environment from the decision makers' perspective
        >>> env = CustomizedRTBEnv(
                original_env=env,
                reward_predictor=LogisticRegression(),
                action_type="discrete",
            )

        # define (RL) agent (i.e., policy) and train on the environment
        >>> ddqn = DoubleDQN()
        >>> buffer = ReplayBuffer(
                maxlen=10000,
                env=env,
            )
        >>> explorer = ConstantEpsilonGreedy(
                epsilon=0.3,
            )
        >>> ddqn.fit_online(
                env=env,
                buffer=buffer,
                explorer=explorer,
            )

        # convert ddqn policy to stochastic data collection policy
        >>> behavior_policy = DiscreteEpsilonGreedyHead(
                ddqn,
                n_actions=env.action_space.n,
                epsilon=0.3,
                name="ddqn_epsilon_0.3",
                random_state=12345,
            )

        # initialize dataset class
        >>> dataset = SyntheticDataset(
                env=env,
                behavior_policy=behavior_policy,
                random_state=12345,
            )

        # data collection
        >>> logged_dataset = dataset.obtain_trajectories(n_episodes=100, obtain_info=True)

        # evaluation policy
        >>> ddqn_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="ddqn",
            epsilon=0.0,
            random_state=12345
        )
        >>> random_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="random",
            epsilon=1.0,
            random_state=12345
        )

        # create input for off-policy evaluation (OPE)
        >>> prep = CreateOPEInput(
            logged_dataset=logged_dataset,
        )
        >>> input_dict = prep.obtain_whole_inputs(
            evaluation_policies=[ddqn_, random_],
            env=env,
            n_episodes_on_policy_evaluation=100,
            random_state=12345,
        )

        # OPE
        >>> ope = OffPolicyEvaluation(
            logged_dataset=logged_dataset,
            ope_estimators=[TIS(), SIS()],
        )
        >>> policy_value_dict = ope.estimate_policy_value(
            input_dict=input_dict,
        )
        >>> policy_value_dict
        {'ddqn': {'on_policy': 15.5, 'tis': 22.901319216705502, 'sis': 17.970922685707617},
        'random': {'on_policy': 15.5, 'tis': 0.555637908601827, 'sis': 6.108053435521632}}


    References
    -------
    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation.", 2021.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits.", 2021.

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None
    n_partition: Optional[int] = None
    use_observations_as_reward_scale: bool = False

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.step_per_episode = self.logged_dataset["step_per_episode"]

        if self.logged_dataset["action_type"] != "discrete":
            raise ValueError("logged_dataset does not `discrete` action_type")

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != "discrete":
                raise RuntimeError(
                    f"One of the ope_estimators, {estimator.estimator_name} does not match `discrete` action_type"
                )

            if not isinstance(
                estimator, BaseCumulativeDistributionalOffPolicyEstimator
            ):
                raise RuntimeError(
                    f"ope_estimators must be child classes of BaseCumulativeDistributionalOffPolicyEstimator, but one of them, {estimator.estimator_name} is not"
                )

        if not self.use_observations_as_reward_scale:
            if self.scale_min is None:
                raise ValueError(
                    "scale_min must be given when `use_observations_as_reward_scale == False`"
                )
            if self.scale_max is None:
                raise ValueError(
                    "scale_max must be given when `use_observations_as_reward_scale == False`"
                )
            if self.n_partition is None:
                raise ValueError(
                    "n_partition must be given when `use_observations_as_reward_scale == False`"
                )
            check_scalar(
                self.scale_min,
                name="scale_min",
                target_type=float,
            )
            check_scalar(
                self.scale_max,
                name="scale_max",
                target_type=float,
            )
            check_scalar(
                self.n_partition,
                name="n_partition",
                target_type=int,
                min_val=1,
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

    def obtain_reward_scale(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
    ):
        """Obtain reward scale (x-axis) for the cumulative distribution function.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        reward_scale: np.ndarray
            Reward Scale (x-axis of the cumulative distribution function).

        """
        if self.use_observations_as_reward_scale:
            reward = (
                self.logged_dataset["reward"]
                .reshape((-1, self.step_per_episode))
                .sum(axis=1)
            )
            reward_scale = np.sort(np.unique(reward))
        else:
            reward_scale = np.linspace(
                self.scale_min, self.scale_max, num=self.n_partition
            )
        return reward_scale

    def estimate_cumulative_distribution_function(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
    ):
        """Estimate the cumulative distribution of the trajectory wise reward of evaluation policy.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        cumulative_distribution_dict: Dict[str, Dict[str, np.ndarray]]
            Dictionary containing estimated cumulative distribution of each evaluation policy by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        cumulative_distribution_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale(input_dict=input_dict, gamma=gamma)

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                on_policy_density = np.histogram(
                    input_dict[eval_policy]["on_policy_policy_value"],
                    bins=reward_scale,
                    density=True,
                )[0]
                cumulative_distribution_dict[eval_policy]["on_policy"] = np.insert(
                    on_policy_density, 0, 0
                ).cumsum()
            else:
                cumulative_distribution_dict[eval_policy]["on_policy"] = None

            for estimator_name, estimator in self.ope_estimators_.items():
                cumulative_distribution_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_cumulative_distribution_function(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                    gamma=gamma,
                )

        return defaultdict_to_dict(cumulative_distribution_dict)

    def estimate_mean(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
    ):
        """Estimate the mean of the trajectory wise reward (i.e., policy value) of evaluation policy.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        mean_dict: Dict[str, Dict[str, float]]
            Dictionary containing estimated mean trajectory wise reward of each evaluation policy by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        mean_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale(input_dict=input_dict, gamma=gamma)

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                density = np.histogram(
                    input_dict[eval_policy]["on_policy_policy_value"],
                    bins=reward_scale,
                    density=True,
                )[0]
                mean_dict[eval_policy]["on_policy"] = (density * reward_scale[1:]).sum()
            else:
                mean_dict[eval_policy]["on_policy"] = None

            for estimator_name, estimator in self.ope_estimators_.items():
                mean_dict[eval_policy][estimator_name] = estimator.estimate_mean(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                    gamma=gamma,
                )

        return defaultdict_to_dict(mean_dict)

    def estimate_variance(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
    ):
        """Estimate the variance of the trajectory wise reward of evaluation policy.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        Return
        -------
        variance_dict: Dict[str, Dict[str, float]]
            Dictionary containing estimated variance of trajectory wise reward of each evaluation policy by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        variance_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale(input_dict=input_dict, gamma=gamma)

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                density = np.histogram(
                    input_dict[eval_policy]["on_policy_policy_value"],
                    bins=reward_scale,
                    density=True,
                )[0]
                mean = (density * reward_scale[1:]).sum()
                variance_dict[eval_policy]["on_policy"] = (
                    density * (reward_scale[1:] - mean) ** 2
                ).sum()
            else:
                variance_dict[eval_policy]["on_policy"] = None

            for estimator_name, estimator in self.ope_estimators_.items():
                variance_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_variance(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                    gamma=gamma,
                )

        return defaultdict_to_dict(variance_dict)

    def estimate_conditional_value_at_risk(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
    ):
        """Estimate the conditional value at risk of the trajectory wise reward of evaluation policy.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        conditional_value_at_risk_dict: Dict[str, Dict[str, float]]
            Dictionary containing estimated conditional value at risk of trajectory wise reward of each evaluation policy by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        check_scalar(alpha, "alpha", target_type=float, min_val=0.0, max_val=1.0)
        conditional_value_at_risk_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale(input_dict=input_dict, gamma=gamma)

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                density = np.histogram(
                    input_dict[eval_policy]["on_policy_policy_value"],
                    bins=reward_scale,
                    density=True,
                )[0]

                idx_ = np.nonzero(density.cumsum() > alpha)[0]
                lower_idx = idx_[0] if len(idx_) else -1

                conditional_value_at_risk_dict[eval_policy]["on_policy"] = (
                    density * reward_scale[1:]
                )[:lower_idx].sum()

            else:
                conditional_value_at_risk_dict[eval_policy]["on_policy"] = None

            for estimator_name, estimator in self.ope_estimators_.items():
                conditional_value_at_risk_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_conditional_value_at_risk(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                    gamma=gamma,
                    alpha=alpha,
                )

        return defaultdict_to_dict(conditional_value_at_risk_dict)

    def estimate_interquartile_range(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
    ):
        """Estimate the interquartile range of the trajectory wise reward of evaluation policy.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05
            Proportion of the sided region.

        Return
        -------
        interquartile_range_dict: Dict[str, Dict[str, Dict[str, float]]]
            Dictionary containing estimated interquartile range at risk of trajectory wise reward of each evaluation policy by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name][quartile_name]

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)
        interquartile_range_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale(input_dict=input_dict, gamma=gamma)

        def target_value_given_idx(idx_):
            if len(idx_):
                target_idx = idx_[0]
                target_value = (
                    reward_scale[target_idx] + reward_scale[target_idx + 1]
                ) / 2
            else:
                target_value = reward_scale[-1]
            return target_value

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                density = np.histogram(
                    input_dict[eval_policy]["on_policy_policy_value"],
                    bins=reward_scale,
                    density=True,
                )[0]

                lower_idx_ = np.nonzero(density.cumsum() > alpha)[0]
                median_idx_ = np.nonzero(density.cumsum() > 0.5)[0]
                upper_idx_ = np.nonzero(density.cumsum() > 1 - alpha)[0]

                interquartile_range_dict[eval_policy]["on_policy"] = {
                    "median": target_value_given_idx(median_idx_),
                    f"{100 * (1. - alpha)}% quartile (lower)": target_value_given_idx(
                        lower_idx_
                    ),
                    f"{100 * (1. - alpha)}% quartile (upper)": target_value_given_idx(
                        upper_idx_
                    ),
                }

            else:
                interquartile_range_dict[eval_policy]["on_policy"] = None

            for estimator_name, estimator in self.ope_estimators_.items():
                interquartile_range_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_interquartile_range(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                    gamma=gamma,
                    alpha=alpha,
                )

        return defaultdict_to_dict(interquartile_range_dict)

    def visualize_cumulative_distribution_function(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_cumulative_distribution_function.png",
    ) -> None:
        """Visualize policy value estimated by OPE estimators.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        hue: str, default="estimator"
            Hue of the plot.
            Choose either from "estimator" or "policy".

        legend: bool, default=True,
            Whether to include legend.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_cumulative_distribution_function.png"
            Name of the bar figure.

        """
        if hue not in ["estimator", "policy"]:
            raise ValueError(
                f"hue must be either `estimator` or `policy`, but {hue} is given"
            )
        if n_cols is not None:
            check_scalar(n_cols, name="n_cols", target_type=int, min_val=1)
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        reward_scale = self.obtain_reward_scale(input_dict=input_dict, gamma=gamma)
        cumulative_distribution_function_dict = (
            self.estimate_cumulative_distribution_function(
                input_dict=input_dict,
                gamma=gamma,
            )
        )

        plt.style.use("ggplot")

        if hue == "estimator":
            n_figs = len(input_dict)
            n_cols = min(3, n_figs) if n_cols is None else n_cols
            n_rows = n_figs // n_cols

            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows)
            )

            if n_rows == 1:
                for i, eval_policy in enumerate(input_dict.keys()):
                    for j, ope_estimator in enumerate(self.ope_estimators_):
                        axes[i].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                ope_estimator
                            ],
                            label=ope_estimator,
                        )

                    if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                        axes[i].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                "on_policy"
                            ],
                            label="on_policy",
                        )

                    axes[i].set_title(eval_policy)
                    axes[i].set_xlabel("trajectory wise reward")
                    axes[i].set_ylabel("cumulative probability")
                    if legend:
                        axes[i].legend()

            else:
                for i, eval_policy in enumerate(input_dict.keys()):
                    for j, ope_estimator in enumerate(self.ope_estimators_):
                        axes[i // n_cols, i % n_cols].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                ope_estimator
                            ],
                            label=ope_estimator,
                        )

                    if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                        axes[i // n_cols, i % n_cols].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                "on_policy"
                            ],
                            label="on_policy",
                        )

                    axes[i // n_cols, i % n_cols].set_title(eval_policy)
                    axes[i // n_cols, i % n_cols].set_xlabel("trajectory wise reward")
                    axes[i // n_cols, i % n_cols].set_ylabel("cumulative probability")
                    if legend:
                        axes[i // n_cols, i % n_cols].legend()

        else:
            visualize_on_policy = True
            for eval_policy in input_dict:
                if input_dict[eval_policy]["on_policy_policy_value"] is None:
                    visualize_on_policy = False

            n_figs = (
                len(self.ope_estimators_) + 1
                if visualize_on_policy
                else len(self.ope_estimators_)
            )
            n_cols = min(3, n_figs) if n_cols is None else n_cols
            n_rows = n_figs // n_cols

            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows)
            )

            if n_rows == 1:
                for i, ope_estimator in enumerate(self.ope_estimators_):
                    for j, eval_policy in enumerate(input_dict.keys()):
                        axes[i].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                ope_estimator
                            ],
                            label=eval_policy,
                        )

                    axes[i].set_title(ope_estimator)
                    axes[i].set_xlabel("trajectory wise reward")
                    axes[i].set_ylabel("cumulative probability")
                    if legend:
                        axes[i].legend()

                if visualize_on_policy:
                    for j, eval_policy in enumerate(input_dict.keys()):
                        axes[i + 1].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                "on_policy"
                            ],
                            label=eval_policy,
                        )

                    axes[i + 1].set_title("on_policy")
                    axes[i + 1].set_xlabel("trajectory wise reward")
                    axes[i + 1].set_ylabel("cumulative probability")
                    if legend:
                        axes[i + 1].legend()

            else:
                for i, ope_estimator in enumerate(self.ope_estimators_):
                    for j, eval_policy in enumerate(input_dict.keys()):
                        axes[i // n_cols, i % n_cols].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                ope_estimator
                            ],
                            label=eval_policy,
                        )

                    axes[i // n_cols, i % n_cols].title(ope_estimator)
                    axes[i // n_cols, i % n_cols].xlabel("trajectory wise reward")
                    axes[i // n_cols, i % n_cols].ylabel("cumulative probability")
                    if legend:
                        axes[i // n_cols, i % n_cols].legend()

                if visualize_on_policy:
                    for j, eval_policy in enumerate(input_dict.keys()):
                        axes[(i + 1) // n_cols, (i + 1) % n_cols].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                "on_policy"
                            ],
                            label=eval_policy,
                        )

                    axes[(i + 1) // n_cols, (i + 1) % n_cols].title("on_policy")
                    axes[(i + 1) // n_cols, (i + 1) % n_cols].xlabel(
                        "trajectory wise reward"
                    )
                    axes[(i + 1) // n_cols, (i + 1) % n_cols].ylabel(
                        "cumulative probability"
                    )
                    if legend:
                        axes[(i + 1) // n_cols, (i + 1) % n_cols].legend()

        fig.tight_layout()
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_policy_value(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
        is_relative: bool = False,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy value estimated by OPE estimators.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        is_relative: bool, default=False
            If `True`, the method visualizes the estimated policy value of evaluation policy
            relative to the ground-truth policy value of behavior policy.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        mean_dict = self.estimate_mean(
            input_dict=input_dict,
            gamma=gamma,
        )
        variance_dict = self.estimate_variance(
            input_dict=input_dict,
            gamma=gamma,
        )

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig = plt.figure(figsize=(2 * len(self.ope_estimators_), 4 * len(input_dict)))

        for i, eval_policy in enumerate(input_dict.keys()):
            if i == 0:
                n = len(input_dict[eval_policy]["on_policy_policy_value"])
                ax = ax0 = fig.add_subplot(len(input_dict), 1, i + 1)
            elif sharey:
                ax = fig.add_subplot(len(input_dict), 1, i + 1, sharey=ax0)
            else:
                ax = fig.add_subplot(len(input_dict), 1, i + 1)

            on_policy_mean = mean_dict[eval_policy].pop("on_policy")
            on_policy_variance = variance_dict[eval_policy].pop("on_policy")
            on_policy_upper, on_policy_lower = norm.interval(
                1 - alpha, loc=on_policy_mean, scale=np.sqrt(on_policy_variance)
            )

            mean = np.array(list(mean_dict[eval_policy].values()), dtype=float)
            variance = np.array(list(variance_dict[eval_policy].values()), dtype=float)
            upper, lower = norm.interval(1 - alpha, loc=mean, scale=np.sqrt(variance))

            if is_relative:
                on_policy_mean = on_policy_mean / (on_policy_mean + 1e-10)
                on_policy_upper = on_policy_upper / (on_policy_mean + 1e-10)
                on_policy_lower = on_policy_lower / (on_policy_mean + 1e-10)

                mean = mean / (on_policy_mean + 1e-10)
                upper = upper / (on_policy_mean + 1e-10)
                lower = lower / (on_policy_mean + 1e-10)

            for i in range(len(self.ope_estimators_)):
                ax.errorbar(
                    np.arange(i, i + 1),
                    mean[i],
                    xerr=[0.4],
                    yerr=[
                        np.array([mean[i] - lower[i]]),
                        np.array([upper[i] - mean[i]]),
                    ],
                    color=color[i],
                    elinewidth=5.0,
                    # fmt="o",
                    # markersize=10.0,
                )

            elines = ax.get_children()
            for i in range(len(self.ope_estimators_)):
                elines[2 * i + 1].set_color("black")
                elines[2 * i + 1].set_linewidth(2.0)

            ax.axhline(on_policy_mean)
            ax.axhspan(
                ymin=on_policy_lower,
                ymax=on_policy_upper,
                alpha=0.3,
            )
            ax.set_title(eval_policy, fontsize=16)
            ax.set_xticks(np.arange(len(self.ope_estimators_)))
            ax.set_xticklabels(list(self.ope_estimators_.keys()))
            ax.set_ylabel(
                f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=12
            )
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlim(-0.5, len(self.ope_estimators_) - 0.5)

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_conditional_value_at_risk(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alphas: np.ndarray = np.arange(0, 1, 20),
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy value estimated by OPE estimators.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alphas: np.ndarray, default=np.arange(0, 1, 20) (0, 1)
            Set of significant levels.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        alphas = np.sort(alphas)

        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        cvar_dict = {}
        for alpha in alphas:
            cvar_dict[alpha] = self.estimate_conditional_value_at_risk(
                input_dict=input_dict,
                gamma=gamma,
                alpha=alpha,
            )

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig = plt.figure(figsize=(2 * len(self.ope_estimators_), 4 * len(input_dict)))

        for i, eval_policy in enumerate(input_dict.keys()):
            if i == 0:
                ax = ax0 = fig.add_subplot(len(input_dict), 1, i + 1)
            elif sharey:
                ax = fig.add_subplot(len(input_dict), 1, i + 1, sharey=ax0)
            else:
                ax = fig.add_subplot(len(input_dict), 1, i + 1)

            on_policy_cvar = cvar_dict[eval_policy].pop("on_policy")
            cvar = np.array(list(cvar_dict[eval_policy].values()), dtype=float)

            for i in range(len(self.ope_estimators_)):
                ax.errorbar(
                    np.arange(i, i + 1),
                    cvar[i],
                    xerr=[0.4],
                    color=color[i],
                    elinewidth=5.0,
                    fmt="o",
                    markersize=10.0,
                )

            ax.axhline(on_policy_cvar)
            ax.set_title(eval_policy, fontsize=16)
            ax.set_xticks(np.arange(len(self.ope_estimators_)))
            ax.set_xticklabels(list(self.ope_estimators_.keys()))
            ax.set_ylabel(f"Estimated Conditional Value at Risk", fontsize=12)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlim(-0.5, len(self.ope_estimators_) - 0.5)

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_interquartile_range(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy value estimated by OPE estimators.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        mean_dict = self.estimate_mean(
            input_dict=input_dict,
            gamma=gamma,
        )
        interquartile_dict = self.estimate_interquartile_range(
            input_dict=input_dict,
            gamma=gamma,
            alpha=alpha,
        )

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig = plt.figure(figsize=(2 * len(self.ope_estimators_), 4 * len(input_dict)))

        for i, eval_policy in enumerate(input_dict.keys()):
            if i == 0:
                ax = ax0 = fig.add_subplot(len(input_dict), 1, i + 1)
            elif sharey:
                ax = fig.add_subplot(len(input_dict), 1, i + 1, sharey=ax0)
            else:
                ax = fig.add_subplot(len(input_dict), 1, i + 1)

            mean = np.zeros(len(self.ope_estimators_))
            median = np.zeros(len(self.ope_estimators_))
            upper = np.zeros(len(self.ope_estimators_))
            lower = np.zeros(len(self.ope_estimators_))

            for i, estimator in enumerate(self.ope_estimators_):
                interquartile_dict_ = interquartile_dict[eval_policy][estimator]
                mean[i] = mean_dict[eval_policy][estimator]
                median[i] = interquartile_dict_["median"]
                upper[i] = interquartile_dict_[
                    f"{100 * (1. - alpha)}% quartile (upper)"
                ]
                lower[i] = interquartile_dict_[
                    f"{100 * (1. - alpha)}% quartile (lower)"
                ]

            ax.bar(
                np.arange(len(self.ope_estimators_)),
                upper - lower,
                bottom=lower,
                color=color,
                edgecolor="black",
                linewidth=0.3,
                tick_label=list(self.ope_estimators_.keys()),
                alpha=0.3,
            )

            for i in range(len(self.ope_estimators_)):
                ax.errorbar(
                    np.arange(i, i + 1),
                    median[i],
                    xerr=[0.4],
                    color=color[i],
                    elinewidth=5.0,
                    fmt="o",
                    markersize=0.1,
                )
                ax.errorbar(
                    np.arange(i, i + 1),
                    mean[i],
                    color=color[i],
                    fmt="o",
                    markersize=10.0,
                )

            # ax.set_linewidth(3.0)
            ax.set_title(eval_policy, fontsize=16)
            ax.set_ylabel(
                f"Estimated {np.int(100*(1 - alpha))}% Interquartile Range",
                fontsize=12,
            )
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlim(-0.5, len(self.ope_estimators_) - 0.5)

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))


@dataclass
class DiscreteDistributionallyRobustOffPolicyEvaluation:
    """Class to conduct distributionally robust OPE by multiple estimators simultaneously in discrete action space.

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    ope_estimators: List[BaseOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `offlinegym.ope.BaseDistributionallyRobustOffPolicyEstimator`.

    alpha_prior: float, default=1.0 (> 0)
        Initial temperature parameter of the exponential function.

    max_steps: int, default=100 (> 0)
        Maximum steps in turning alpha.

    epsilon: float, default=0.01
        Convergence criterion of alpha.

    Examples
    ----------
    # TODO
    .. ::code-block:: python

        # import necessary module from offlinegym
        >>> from offlinegym.dataset import SyntheticDataset
        >>> from offlinegym.policy import DiscreteEpsilonGreedyHead
        >>> from offlinegym.ope import CreateOPEInput
        >>> from offlinegym.ope import OffPolicyEvaluation
        >>> from offlinegym.ope import DiscreteTrajectoryWiseImportanceSampling as TIS
        >>> from offlinegym.ope import DiscreteStepWiseImportanceSampling as SIS

        # import necessary module from other libraries
        >>> from rtbgym import RTBEnv, CustomizedRTBEnv
        >>> from sklearn.linear_model import LogisticRegression
        >>> from d3rlpy.algos import DoubleDQN
        >>> from d3rlpy.online.buffers import ReplayBuffer
        >>> from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        >>> env = RTBEnv(random_state=12345)

        # customize environment from the decision makers' perspective
        >>> env = CustomizedRTBEnv(
                original_env=env,
                reward_predictor=LogisticRegression(),
                action_type="discrete",
            )

        # define (RL) agent (i.e., policy) and train on the environment
        >>> ddqn = DoubleDQN()
        >>> buffer = ReplayBuffer(
                maxlen=10000,
                env=env,
            )
        >>> explorer = ConstantEpsilonGreedy(
                epsilon=0.3,
            )
        >>> ddqn.fit_online(
                env=env,
                buffer=buffer,
                explorer=explorer,
            )

        # convert ddqn policy to stochastic data collection policy
        >>> behavior_policy = DiscreteEpsilonGreedyHead(
                ddqn,
                n_actions=env.action_space.n,
                epsilon=0.3,
                name="ddqn_epsilon_0.3",
                random_state=12345,
            )

        # initialize dataset class
        >>> dataset = SyntheticDataset(
                env=env,
                behavior_policy=behavior_policy,
                random_state=12345,
            )

        # data collection
        >>> logged_dataset = dataset.obtain_trajectories(n_episodes=100, obtain_info=True)

        # evaluation policy
        >>> ddqn_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="ddqn",
            epsilon=0.0,
            random_state=12345
        )
        >>> random_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="random",
            epsilon=1.0,
            random_state=12345
        )

        # create input for off-policy evaluation (OPE)
        >>> prep = CreateOPEInput(
            logged_dataset=logged_dataset,
        )
        >>> input_dict = prep.obtain_whole_inputs(
            evaluation_policies=[ddqn_, random_],
            env=env,
            n_episodes_on_policy_evaluation=100,
            random_state=12345,
        )

        # OPE
        >>> ope = OffPolicyEvaluation(
            logged_dataset=logged_dataset,
            ope_estimators=[TIS(), SIS()],
        )
        >>> policy_value_dict = ope.estimate_policy_value(
            input_dict=input_dict,
        )
        >>> policy_value_dict
        {'ddqn': {'on_policy': 15.5, 'tis': 22.901319216705502, 'sis': 17.970922685707617},
        'random': {'on_policy': 15.5, 'tis': 0.555637908601827, 'sis': 6.108053435521632}}


    References
    -------
    Nathan Kallus, Xiaojie Mao, Kaiwen Wang, and Zhengyuan Zhou.
    "Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning.", 2022.

    Nian Si, Fan Zhang, Zhengyuan Zhou, and Jose Blanchet.
    "Distributional Robust Batch Contextual Bandits.", 2020.

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]
    alpha_prior: float = 1.0
    max_steps: int = 100
    epsilon: float = 0.01

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.step_per_episode = self.logged_dataset["step_per_episode"]

        if self.logged_dataset["action_type"] != "discrete":
            raise ValueError("logged_dataset does not `discrete` action_type")

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != "discrete":
                raise RuntimeError(
                    f"One of the ope_estimators, {estimator.estimator_name} does not match `discrete` action_type"
                )

            if not isinstance(estimator, BaseDistributionallyRobustOffPolicyEstimator):
                raise RuntimeError(
                    f"ope_estimators must be child classes of BaseDistributionallyRobustOffPolicyEstimator, but one of them, {estimator.estimator_name} is not"
                )

        check_scalar(
            self.alpha_prior, name="alpha_prior", target_type=float, min_val=0.0
        )
        check_scalar(self.max_steps, name="max_steps", target_type=int, min_val=1)
        check_scalar(self.epsilon, name="epsilon", target_type=float, min_val=0.0)

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
            "initial_state": self.logged_dataset["state"].reshape(
                (-1, self.step_per_episode)
            )[:, 0],
            "initial_state_action": self.logged_dataset["action"]
            .astype(int)
            .reshape((-1, self.step_per_episode))[:, 0],
        }

    def estimate_worst_case_policy_value(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        delta: float = 0.05,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate the worst case policy value of evaluation policy.

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
                on_policy_policy_value,
            ]

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        delta: float, default=0.05 (> 0)
            Allowance of the distributional shift.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        worst_case_policy_value_dict: Dict[str, Dict[str, float]]
            Dictionary containing estimated policy value of each evaluation policy by OPE estimators.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        worst_case_policy_value_dict = defaultdict(dict)

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                worst_case_policy_value_dict[eval_policy][
                    "on_policy"
                ] = self.estimate_worst_case_on_policy_policy_value(
                    on_policy_policy_value=input_dict[eval_policy][
                        "on_policy_policy_value"
                    ],
                    delta=delta,
                )
            else:
                worst_case_policy_value_dict[eval_policy]["on_policy"] = None

            for estimator_name, estimator in self.ope_estimators_.items():
                worst_case_policy_value_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_worst_case_policy_value(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    step_per_episode=self.step_per_episode,
                    gamma=gamma,
                    delta=delta,
                    alpha_prior=self.alpha_prior,
                    max_steps=self.max_steps,
                    epsilon=self.epsilon,
                    random_state=random_state,
                )

        return defaultdict_to_dict(worst_case_policy_value_dict)
