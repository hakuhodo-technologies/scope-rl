"""Meta class to handle standard and cumulative distribution OPE."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from collections import defaultdict

import numpy as np
from scipy.stats import norm
from sklearn.utils import check_scalar

from pandas import DataFrame
import matplotlib.pyplot as plt

from d3rlpy.preprocessing import ActionScaler

from .estimators_base import (
    BaseOffPolicyEstimator,
    BaseCumulativeDistributionOffPolicyEstimator,
)
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
class OffPolicyEvaluation:
    """Class to perform OPE by multiple estimators simultaneously (applicable to both discrete/continuous action cases).

    Imported as: :class:`ofrl.ope.OffPolicyEvaluation`

    Note
    -----------
    OPE estimates the expected policy performance of a given evaluation policy called the policy value.

    .. math::

        V(\\pi) := \\mathbb{E} \\left[ \\sum_{t=0}^{T-1} \\gamma^t r_t \\mid \\pi \\right]

    where :math:`\\pi` is the evaluation policy, :math:`r_t` is the reward observation at each timestep :math:`t`, 
    :math:`T` is the total number of timesteps in an episode, and :math:`\\gamma` is the discount factor.

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    ope_estimators: list of BaseOffPolicyEstimator
        List of OPE estimators used to evaluate the policy value of the evaluation policies.
        Estimators must follow the interface of :class:`ofrl.ope.BaseOffPolicyEstimator`.

    n_step_pdis: int, default=0 (>= 0)
        Number of previous steps to use per-decision importance weight in marginal OPE estimators.
        If zero is given, the estimator corresponds to the pure state(-action) marginal IS.

    sigma: float, default=1.0 (> 0)
        Bandwidth hyperparameter of gaussian kernel for continuous action space.

    action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
        Scaling factor of action.

    Examples
    ----------

    Preparation:

    .. code-block:: python

        # import necessary module from OFRL
        from ofrl.dataset import SyntheticDataset
        from ofrl.policy import DiscreteEpsilonGreedyHead
        from ofrl.ope import CreateOPEInput
        from ofrl.ope import DiscreteOffPolicyEvaluation as OPE
        from ofrl.ope import DiscreteTrajectoryWiseImportanceSampling as TIS
        from ofrl.ope import DiscretePerDecisionImportanceSampling as PDIS

        # import necessary module from other libraries
        import gym
        import rtbgym
        from d3rlpy.algos import DoubleDQN
        from d3rlpy.online.buffers import ReplayBuffer
        from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        env = gym.make("RTBEnv-discrete-v0")

        # define (RL) agent (i.e., policy) and train on the environment
        ddqn = DoubleDQN()
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env,
        )
        explorer = ConstantEpsilonGreedy(
            epsilon=0.3,
        )
        ddqn.fit_online(
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=10000,
            n_steps_per_epoch=1000,
        )

        # convert ddqn policy to stochastic data collection policy
        behavior_policy = DiscreteEpsilonGreedyHead(
            ddqn,
            n_actions=env.action_space.n,
            epsilon=0.3,
            name="ddqn_epsilon_0.3",
            random_state=12345,
        )

        # initialize dataset class
        dataset = SyntheticDataset(
            env=env,
            behavior_policy=behavior_policy,
            random_state=12345,
        )

        # data collection
        logged_dataset = dataset.obtain_episodes(n_trajectories=100)

    Create Input for OPE:

    .. code-block:: python

        # evaluation policy
        ddqn_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="ddqn",
            epsilon=0.0,
            random_state=12345
        )
        random_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="random",
            epsilon=1.0,
            random_state=12345
        )

        # create input for off-policy evaluation (OPE)
        prep = CreateOPEInput(
            logged_dataset=logged_dataset,
        )
        input_dict = prep.obtain_whole_inputs(
            evaluation_policies=[ddqn_, random_],
            env=env,
            n_trajectories_on_policy_evaluation=100,
            random_state=12345,
        )

    **Off-Policy Evaluation**:

    .. code-block:: python

        # OPE
        ope = OPE(
            logged_dataset=logged_dataset,
            ope_estimators=[TIS(), PDIS()],
        )
        policy_value_dict = ope.estimate_policy_value(
            input_dict=input_dict,
        )

    **Output**:

    .. code-block:: python

        >>> policy_value_dict

        {'ddqn': {'on_policy': 15.95, 'tis': 18.103809657474702, 'pdis': 16.95314065192053},
        'random': {'on_policy': 12.69, 'tis': 0.4885685147584351, 'pdis': 6.2752568547701335}}

    .. seealso::

        * :doc:`Quickstart </documentation/quickstart>`
        * :doc:`Related tutorials </documentation/_autogallery/basic_ope/index>`

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]
    n_step_pdis: int = 0
    sigma: float = 1.0
    action_scaler: Optional[ActionScaler] = None

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.step_per_trajectory = self.logged_dataset["step_per_trajectory"]
        self.action_type = self.logged_dataset["action_type"]

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != self.action_type:
                raise RuntimeError(
                    f"One of the ope_estimators, {estimator.estimator_name} does not match the action_type of logged_dataset (`{self.action_type}`)"
                )

            if not isinstance(estimator, BaseOffPolicyEstimator):
                raise RuntimeError(
                    f"ope_estimators must be child classes of BaseOffPolicyEstimator, but one of them, {estimator.estimator_name} is not"
                )

        self.behavior_policy_value = (
            self.logged_dataset["reward"]
            .reshape((-1, self.step_per_trajectory))
            .sum(axis=1)
            .mean()
        ) + 1e-10  # to avoid devision by zero

        if self.action_type == "discrete":
            self.input_dict_ = {
                "step_per_trajectory": self.step_per_trajectory,
                "action": self.logged_dataset["action"].astype(int),
                "reward": self.logged_dataset["reward"],
                "pscore": self.logged_dataset["pscore"],
            }
        else:
            if self.action_scaler is not None and not isinstance(
                self.action_scaler, ActionScaler
            ):
                raise ValueError(
                    "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
                )
            check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)

            self.input_dict_ = {
                "step_per_trajectory": self.step_per_trajectory,
                "action": self.logged_dataset["action"].astype(int),
                "reward": self.logged_dataset["reward"],
                "pscore": self.logged_dataset["pscore"],
                "action_scaler": self.action_scaler,
                "sigma": self.sigma,
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
        compared_estimators: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Estimate the policy value of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        Return
        -------
        policy_value_dict: dict
            Dictionary containing the policy value of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )

        policy_value_dict = defaultdict(dict)
        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                policy_value_dict[eval_policy]["on_policy"] = input_dict[eval_policy][
                    "on_policy_policy_value"
                ].mean()
            else:
                policy_value_dict[eval_policy]["on_policy"] = None

            for estimator_name in compared_estimators:
                estimator = self.ope_estimators_[estimator_name]
                policy_value_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_policy_value(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    n_step_pdis=self.n_step_pdis,
                )
        return defaultdict_to_dict(policy_value_dict)

    def estimate_intervals(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
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

            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=100 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        policy_value_interval_dict: dict
            Dictionary containing the confidence intervals estimated by nonparametric bootstrap.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        References
        -------
        Josiah P. Hanna, Peter Stone, and Scott Niekum.
        "Bootstrapping with Models: Confidence Intervals for Off-Policy Evaluation." 2017.

        Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh.
        "High Confidence Policy Improvement." 2015.

        Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh.
        "High Confidence Off-Policy Evaluation." 2015.

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
        if ci not in self._estimate_confidence_interval.keys():
            raise ValueError(
                f"ci must be one of 'bootstrap', 'hoeffding', 'bernstein', or 'ttest', but {ci} is given"
            )

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

            for estimator_name in compared_estimators:
                estimator = self.ope_estimators_[estimator_name]
                policy_value_interval_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_interval(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    n_step_pdis=self.n_step_pdis,
                    alpha=alpha,
                    ci=ci,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

        return defaultdict_to_dict(policy_value_interval_dict)

    def summarize_off_policy_estimates(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
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
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

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
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        policy_value_interval_dict: dict
            Dictionary containing the confidence intervals estimated by nonparametric bootstrap.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        check_input_dict(input_dict)
        policy_value_dict = self.estimate_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
        )
        policy_value_interval_dict = self.estimate_intervals(
            input_dict,
            compared_estimators=compared_estimators,
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
        compared_estimators: Optional[List[str]] = None,
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
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        alpha: float, default=0.05
            Significance level. The value should be within (0, 1].

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        is_relative: bool, default=False
            If `True`, the method visualizes the estimated policy value of the evaluation policies
            relative to the on-policy policy value of the behavior policy.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
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
            compared_estimators=compared_estimators,
            alpha=alpha,
            ci=ci,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        n_policies = len(input_dict)
        n_estimators = len(compared_estimators)

        if hue == "estimator":
            fig = plt.figure(figsize=(2 * n_estimators, 4 * n_policies))

            for i, eval_policy in enumerate(input_dict.keys()):
                if i == 0:
                    ax = ax0 = fig.add_subplot(n_policies, 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(n_policies, 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(n_policies, 1, i + 1)

                mean = np.zeros(n_estimators)
                lower = np.zeros(n_estimators)
                upper = np.zeros(n_estimators)

                for j, estimator in enumerate(compared_estimators):
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
                    np.arange(n_estimators),
                    mean,
                    yerr=[upper - mean, mean - lower],
                    color=color,
                    tick_label=compared_estimators,
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
            for eval_policy in input_dict.keys():
                if input_dict[eval_policy]["on_policy_policy_value"] is None:
                    visualize_on_policy = False

            n_policies = len(input_dict)
            n_estimators = (
                len(compared_estimators) + 1
                if visualize_on_policy
                else len(compared_estimators)
            )

            fig = plt.figure(figsize=(2 * n_policies, 4 * n_estimators))

            for i, estimator in enumerate(compared_estimators):
                if i == 0:
                    ax = ax0 = fig.add_subplot(n_estimators, 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(n_estimators, 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(n_estimators, 1, i + 1)

                mean = np.zeros(n_policies)
                lower = np.zeros(n_policies)
                upper = np.zeros(n_policies)

                for j, eval_policy in enumerate(input_dict.keys()):
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
                    np.arange(n_policies),
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

                mean = np.zeros(n_policies)
                lower = np.zeros(n_policies)
                upper = np.zeros(n_policies)

                for j, eval_policy in enumerate(input_dict.keys()):
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
                    np.arange(n_policies),
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

        fig.subplots_adjust(top=1.0)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def evaluate_performance_of_ope_estimators(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        metric: str = "relative-ee",
        return_by_dataframe: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the estimation performance/accuracy of OPE estimators.

        Note
        -------
        Evaluate the estimation performance/accuracy of OPE estimators by relative estimation error (relative-EE) or squared error (SE).

        .. math::

            \\mathrm{Relative-EE}(\\hat{V}; \\mathcal{D})
            := \\left| \\frac{\\hat{V}(\\pi; \\mathcal{D}) - V_{\\mathrm{on}}(\\pi)}{V_{\\mathrm{on}}(\\pi)} \\right|,

        .. math::

            \\mathrm{SE}(\\hat{V}; \\mathcal{D}) := \\left( \\hat{V}(\\pi; \\mathcal{D}) - V_{\\mathrm{on}} \\right)^2,

        where :math:`V_{\\mathrm{on}}(\\pi)` is the on-policy policy value of the evaluation policy :math:`\\pi`.
        :math:`\\hat{V}(\\pi; \\mathcal{D})` is the policy value estimated by the OPE estimator :math:`\\hat{V}` and logged dataset :math:`\\mathcal{D}`.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            Please refer to :class:`CreateOPEInput` class for the detail.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        metric: {"relative-ee", "se"}, default="relative-ee"
            Evaluation metric used to evaluate and compare the estimation performance/accuracy of OPE estimators.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        eval_metric_ope_dict/eval_metric_ope_df: dict or dataframe
            Dictionary/dataframe containing evaluation metric for evaluating the estimation performance/accuracy of OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"metric must be either 'relative-ee' or 'se', but {metric} is given"
            )

        eval_metric_ope_dict = defaultdict(dict)
        policy_value_dict = self.estimate_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
        )

        if metric == "relative-ee":
            for eval_policy in input_dict.keys():
                on_policy_policy_value = input_dict[eval_policy][
                    "on_policy_policy_value"
                ]

                for estimator in compared_estimators:
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

                for estimator in compared_estimators:
                    se_ = (
                        policy_value_dict[eval_policy][estimator]
                        - on_policy_policy_value
                    ) ** 2
                    eval_metric_ope_dict[eval_policy][estimator] = se_

        eval_metric_ope_dict = defaultdict_to_dict(eval_metric_ope_dict)

        if return_by_dataframe:
            eval_metric_ope_df = DataFrame()
            for eval_policy in input_dict.keys():
                eval_metric_ope_df[eval_policy] = DataFrame(
                    eval_metric_ope_dict[eval_policy], index=[eval_policy]
                ).T

        return eval_metric_ope_df if return_by_dataframe else eval_metric_ope_dict

    @property
    def estimators_name(self):
        return list(self.ope_estimators_.keys())


@dataclass
class CumulativeDistributionOffPolicyEvaluation:
    """Class to conduct cumulative distribution OPE by multiple estimators simultaneously (applicable to both discrete/continuous action cases).

    Imported as: :class:`ofrl.ope.CumutiveDistributionOffPolicyEvaluation`

    Note
    -----------
    Cumulative distribution OPE first estimates the following cumulative distribution function, and then estimates some statistics.

    .. math::

        F(m, \\pi) := \\mathbb{E} \\left[ \\mathbb{I} \\left \\{ \\sum_{t=0}^{T-1} \\gamma^t r_t \\leq m \\right \\} \\mid \\pi \\right]

    where :math:`\\pi` is the evaluation policy, :math:`r_t` is the reward observation at each timestep :math:`t`, 
    :math:`T` is the total number of timesteps in an episode, and :math:`\\gamma` is the discount factor.

    CDF is itself informative, but also enables us to calculate the following risk functions.

    * Mean: :math:`\\mu(F) := \\int_{G} G \\, \\mathrm{d}F(G)`
    * Variance: :math:`\\sigma^2(F) := \\int_{G} (G - \\mu(F))^2 \\, \\mathrm{d}F(G)`
    * :math:`\\alpha`-quartile: :math:`Q^{\\alpha}(F) := \\min \\{ G \\mid F(G) \\leq \\alpha \\}`
    * Conditional Value at Risk (CVaR): :math:`\\int_{G} G \\, \mathbb{I}\\{ G \\leq Q^{\\alpha}(F) \\} \\, \\mathrm{d}F(G)`

    where we let :math:`G := \\sum_{t=0}^{T-1} \\gamma^t r_t` to represent the random variable of trajectory wise reward
    and :math:`dF(G) := \\mathrm{lim}_{\\Delta \\rightarrow 0} F(G) - F(G- \\Delta)`.

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    ope_estimators: list of BaseOffPolicyEstimator
        List of OPE estimators used to evaluate the policy value of the evaluation policies.
        Estimators must follow the interface of `ofrl.ope.BaseCumulativeDistributionOffPolicyEstimator`.

    use_custom_reward_scale: bool, default=False
        Whether to use the custom reward scale or the reward observed by the behavior policy.

        If `True`, the reward scale is uniform, following Huang et al. (2021).

        If `False`, the reward scale follows the one defined in Chundak et al. (2021).

    scale_min: float, default=None
        Minimum value of the reward scale in CDF.
        If use_custom_reward_scale is `True`, a value must be given.

    scale_max: float, default=None
        Maximum value of the reward scale in CDF.
        If use_custom_reward_scale is `True`, a value must be given.

    n_partition: int, default=None
        Number of partitions in the reward scale (x-axis of CDF).
        If use_custom_reward_scale is `True`, a value must be given.

    sigma: float, default=1.0 (> 0)
        Bandwidth hyperparameter of gaussian kernel for continuous action space.

    action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
        Scaling factor of action.

    Examples
    ----------

    Preparation:

    .. code-block:: python

        # import necessary module from OFRL
        from ofrl.dataset import SyntheticDataset
        from ofrl.policy import DiscreteEpsilonGreedyHead
        from ofrl.ope import CreateOPEInput
        from ofrl.ope import DiscreteCumulativeDistributionOffPolicyEvaluation as CumulativeDistributionOPE
        from ofrl.ope import DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling as CDIS
        from ofrl.ope import DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling as CDSIS

        # import necessary module from other libraries
        import gym
        import rtbgym
        from d3rlpy.algos import DoubleDQN
        from d3rlpy.online.buffers import ReplayBuffer
        from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        env = gym.make("RTBEnv-discrete-v0")

        # define (RL) agent (i.e., policy) and train on the environment
        ddqn = DoubleDQN()
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env,
        )
        explorer = ConstantEpsilonGreedy(
            epsilon=0.3,
        )
        ddqn.fit_online(
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=10000,
            n_steps_per_epoch=1000,
        )

        # convert ddqn policy to stochastic data collection policy
        behavior_policy = DiscreteEpsilonGreedyHead(
            ddqn,
            n_actions=env.action_space.n,
            epsilon=0.3,
            name="ddqn_epsilon_0.3",
            random_state=12345,
        )

        # initialize dataset class
        dataset = SyntheticDataset(
            env=env,
            behavior_policy=behavior_policy,
            random_state=12345,
        )

        # data collection
        logged_dataset = dataset.obtain_trajectories(n_trajectories=100)

    Create Input for OPE:

    .. code-block:: python

        # evaluation policy
        ddqn_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="ddqn",
            epsilon=0.0,
            random_state=12345
        )
        random_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="random",
            epsilon=1.0,
            random_state=12345
        )

        # create input for off-policy evaluation (OPE)
        prep = CreateOPEInput(
            logged_dataset=logged_dataset,
        )
        input_dict = prep.obtain_whole_inputs(
            evaluation_policies=[ddqn_, random_],
            env=env,
            n_trajectories_on_policy_evaluation=100,
            random_state=12345,
        )

    **Cumulative Distribution OPE**:

    .. code-block:: python

        # OPE
        cd_ope = CumulativeDistributionOPE(
            logged_dataset=logged_dataset,
            ope_estimators=[
                CDIS(estimator_name="cdf_is"),
                CDSIS(estimator_name="cdf_sis"),
            ],
        )
        variance_dict = cd_ope.estimate_variance(
            input_dict=input_dict,
        )

    **Output**:

    .. code-block:: python

        >>> variance_dict

        {'ddqn': {'on_policy': 18.6216, 'cdf_is': 19.201934808340265, 'cdf_snis': 25.315555555555555},
        'random': {'on_policy': 21.512806887023064, 'cdf_is': 13.591854902638273, 'cdf_snis': 7.158545530356914}}

    .. seealso::

        * :doc:`Quickstart </documentation/quickstart>`
        * :doc:`Related tutorials </documentation/_autogallery/cumulative_distribution_ope/index>`

    References
    -------
    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment for Markov Decision Processes." 2022.

    Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli.
    "Off-Policy Risk Assessment in Contextual Bandits." 2021.

    Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas.
    "Universal Off-Policy Evaluation." 2021.

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]
    use_custom_reward_scale: bool = False
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None
    n_partition: Optional[int] = None
    sigma: float = 1.0
    action_scaler: Optional[ActionScaler] = None

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.step_per_trajectory = self.logged_dataset["step_per_trajectory"]
        self.action_type = self.logged_dataset["action_type"]

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != self.action_type:
                raise RuntimeError(
                    f"One of the ope_estimators, {estimator.estimator_name} does not match the action_type of logged_dataset (`{self.action_type}`)"
                )

            if not isinstance(estimator, BaseCumulativeDistributionOffPolicyEstimator):
                raise RuntimeError(
                    f"ope_estimators must be child classes of BaseCumulativeDistributionOffPolicyEstimator, but one of them, {estimator.estimator_name} is not"
                )

        if self.use_custom_reward_scale:
            if self.scale_min is None:
                raise ValueError(
                    "scale_min must be given when `use_custom_reward_scale == True`"
                )
            if self.scale_max is None:
                raise ValueError(
                    "scale_max must be given when `use_custom_reward_scale == True`"
                )
            if self.n_partition is None:
                raise ValueError(
                    "n_partition must be given when `use_custom_reward_scale == True`"
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

        self.behavior_policy_value = (
            self.logged_dataset["reward"]
            .reshape((-1, self.step_per_trajectory))
            .sum(axis=1)
            .mean()
        ) + 1e-10  # to avoid devision by zero

        if self.action_type == "discrete":
            self.input_dict_ = {
                "step_per_trajectory": self.step_per_trajectory,
                "action": self.logged_dataset["action"].astype(int),
                "reward": self.logged_dataset["reward"],
                "pscore": self.logged_dataset["pscore"],
            }
        else:
            if self.action_scaler is not None and not isinstance(
                self.action_scaler, ActionScaler
            ):
                raise ValueError(
                    "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
                )
            check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)

            self.input_dict_ = {
                "step_per_trajectory": self.step_per_trajectory,
                "action": self.logged_dataset["action"].astype(int),
                "reward": self.logged_dataset["reward"],
                "pscore": self.logged_dataset["pscore"],
                "action_scaler": self.action_scaler,
                "sigma": self.sigma,
            }

    def _target_value_given_idx(self, idx_: int, reward_scale: np.ndarray):
        """Obtain target value in reward scale for cumulative distribution estimation.

        Parameters
        -------
        idx_: list of int or int
            Indicating index. If a list is given, the average of the two will be returned.

        reward_scale: array-like of shape (n_partition, )
            Scale of the trajectory wise reward used for x-axis of CDF curve.

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

    def obtain_reward_scale(
        self,
    ):
        """Obtain the reward scale (x-axis) for the cumulative distribution function.

        Return
        -------
        reward_scale: ndarray of shape (n_unique_reward, ) or (n_partition, )
            Reward Scale (x-axis of the cumulative distribution function).

        """
        if self.use_custom_reward_scale:
            reward_scale = np.linspace(
                self.scale_min, self.scale_max, num=self.n_partition
            )
        else:
            reward = (
                self.logged_dataset["reward"]
                .reshape((-1, self.step_per_trajectory))
                .sum(axis=1)
            )
            reward_scale = np.sort(np.unique(reward))
        return reward_scale

    def estimate_cumulative_distribution_function(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
    ):
        """Estimate the cumulative distribution of the trajectory wise reward of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        Return
        -------
        cumulative_distribution_dict: dict
            Dictionary containing the cumulative distribution of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )

        cumulative_distribution_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale()

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

            for estimator_name in compared_estimators:
                estimator = self.ope_estimators_[estimator_name]
                cumulative_distribution_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_cumulative_distribution_function(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                )

        return defaultdict_to_dict(cumulative_distribution_dict)

    def estimate_mean(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
    ):
        """Estimate the mean of the trajectory wise reward (i.e., policy value) of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        Return
        -------
        mean_dict: dict
            Dictionary containing the mean trajectory wise reward of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )

        mean_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale()

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

            for estimator_name in compared_estimators:
                estimator = self.ope_estimators_[estimator_name]
                mean_dict[eval_policy][estimator_name] = estimator.estimate_mean(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                )

        return defaultdict_to_dict(mean_dict)

    def estimate_variance(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
    ):
        """Estimate the variance of the trajectory wise reward of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        Return
        -------
        variance_dict: dict
            Dictionary containing the variance of trajectory wise reward of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )

        variance_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale()

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

            for estimator_name in compared_estimators:
                estimator = self.ope_estimators_[estimator_name]
                variance_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_variance(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                )

        return defaultdict_to_dict(variance_dict)

    def estimate_conditional_value_at_risk(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        alphas: Optional[Union[np.ndarray, float]] = None,
    ):
        """Estimate the conditional value at risk of the trajectory wise reward of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        alphas: {float, array-like of shape (n_alpha, )}, default=None
            Set of proportions of the sided region. The value(s) should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        Return
        -------
        conditional_value_at_risk_dict: dict
            Dictionary containing the conditional value at risk of trajectory wise reward of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
        if alphas is None:
            alphas = np.linspace(0, 1, 21)
        if isinstance(alphas, float):
            check_scalar(
                alphas, name="alphas", target_type=float, min_val=0.0, max_val=1.0
            )
            alphas = np.array([alphas], dtype=float)
        elif isinstance(alphas, np.ndarray):
            check_array(alphas, name="alphas", expected_dim=1, min_val=0.0, max_val=1.0)
        else:
            raise ValueError(
                f"alphas must be float or np.ndarray, but {type(alphas)} is given"
            )

        alphas = np.sort(alphas)
        conditional_value_at_risk_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale()

        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is not None:
                density = np.histogram(
                    input_dict[eval_policy]["on_policy_policy_value"],
                    bins=reward_scale,
                    density=True,
                )[0]

                cvar = np.zeros_like(alphas)
                for i, alpha in enumerate(alphas):
                    idx_ = np.nonzero(density.cumsum() > alpha)[0]
                    lower_idx_ = idx_[0] if len(idx_) else -2
                    cvar[i] = (density * reward_scale[1:])[:lower_idx_].sum()

                conditional_value_at_risk_dict[eval_policy]["on_policy"] = cvar

            else:
                conditional_value_at_risk_dict[eval_policy]["on_policy"] = None

            for estimator_name in compared_estimators:
                estimator = self.ope_estimators_[estimator_name]
                conditional_value_at_risk_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_conditional_value_at_risk(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                    alphas=alphas,
                )

        return defaultdict_to_dict(conditional_value_at_risk_dict)

    def estimate_interquartile_range(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        alpha: float = 0.05,
    ):
        """Estimate the interquartile range of the trajectory wise reward of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        alpha: float, default=0.05
            Proportion of the sided region. The value should be within (0, 1].

        Return
        -------
        interquartile_range_dict: dict
            Dictionary containing the interquartile range of trajectory wise reward of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name][quartile_name]`

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)
        interquartile_range_dict = defaultdict(dict)
        reward_scale = self.obtain_reward_scale()

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
                    "median": self._target_value_given_idx(
                        median_idx_, reward_scale=reward_scale
                    ),
                    f"{100 * (1. - alpha)}% quartile (lower)": self._target_value_given_idx(
                        lower_idx_,
                        reward_scale=reward_scale,
                    ),
                    f"{100 * (1. - alpha)}% quartile (upper)": self._target_value_given_idx(
                        upper_idx_,
                        reward_scale=reward_scale,
                    ),
                }

            else:
                interquartile_range_dict[eval_policy]["on_policy"] = None

            for estimator_name in compared_estimators:
                estimator = self.ope_estimators_[estimator_name]
                interquartile_range_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_interquartile_range(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    reward_scale=reward_scale,
                    alpha=alpha,
                )

        return defaultdict_to_dict(interquartile_range_dict)

    def visualize_cumulative_distribution_function(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_cumulative_distribution_function.png",
    ) -> None:
        """Visualize the cumulative distribution function estimated by OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the figure.

        n_cols: int, default=None
            Number of columns in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_cumulative_distribution_function.png"
            Name of the bar figure.

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
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

        reward_scale = self.obtain_reward_scale()
        cumulative_distribution_function_dict = (
            self.estimate_cumulative_distribution_function(
                input_dict,
                compared_estimators=compared_estimators,
            )
        )

        plt.style.use("ggplot")

        if hue == "estimator":
            n_figs = len(input_dict)
            n_cols = min(3, n_figs) if n_cols is None else n_cols
            n_rows = (n_figs - 1) // n_cols + 1

            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows)
            )

            if n_rows == 1:
                for i, eval_policy in enumerate(input_dict.keys()):
                    for j, estimator in enumerate(compared_estimators):
                        axes[i].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                estimator
                            ],
                            label=estimator,
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

                if legend:
                    handles, labels = axes[0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

            else:
                for i, eval_policy in enumerate(input_dict.keys()):
                    for j, estimator in enumerate(compared_estimators):
                        axes[i // n_cols, i % n_cols].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                estimator
                            ],
                            label=estimator,
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

                if legend:
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

        else:
            visualize_on_policy = True
            for eval_policy in input_dict:
                if input_dict[eval_policy]["on_policy_policy_value"] is None:
                    visualize_on_policy = False

            n_figs = (
                len(compared_estimators) + 1
                if visualize_on_policy
                else len(compared_estimators)
            )
            n_cols = min(3, n_figs) if n_cols is None else n_cols
            n_rows = (n_figs - 1) // n_cols + 1

            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows)
            )

            if n_rows == 1:
                for i, estimator in enumerate(compared_estimators):
                    for j, eval_policy in enumerate(input_dict.keys()):
                        axes[i].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                estimator
                            ],
                            label=eval_policy,
                        )

                    axes[i].set_title(estimator)
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

                if legend:
                    handles, labels = axes[0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

            else:
                for i, estimator in enumerate(compared_estimators):
                    for j, eval_policy in enumerate(input_dict.keys()):
                        axes[i // n_cols, i % n_cols].plot(
                            reward_scale,
                            cumulative_distribution_function_dict[eval_policy][
                                estimator
                            ],
                            label=eval_policy,
                        )

                    axes[i // n_cols, i % n_cols].set_title(estimator)
                    axes[i // n_cols, i % n_cols].set_xlabel("trajectory wise reward")
                    axes[i // n_cols, i % n_cols].set_ylabel("cumulative probability")
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

                    axes[(i + 1) // n_cols, (i + 1) % n_cols].set_title("on_policy")
                    axes[(i + 1) // n_cols, (i + 1) % n_cols].set_xlabel(
                        "trajectory wise reward"
                    )
                    axes[(i + 1) // n_cols, (i + 1) % n_cols].set_ylabel(
                        "cumulative probability"
                    )
                    if legend:
                        axes[(i + 1) // n_cols, (i + 1) % n_cols].legend()

                if legend:
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

        fig.subplots_adjust(hspace=0.35, wspace=0.2)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_policy_value(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        alpha: float = 0.05,
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
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        is_relative: bool, default=False
            If `True`, the method visualizes the estimated policy value of the evaluation policies
            relative to the ground-truth policy value of the behavior policy.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
        if hue not in ["estimator", "policy"]:
            raise ValueError(
                f"hue must be either `estimator` or `policy`, but {hue} is given"
            )
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        mean_dict = self.estimate_mean(
            input_dict,
            compared_estimators=compared_estimators,
        )
        variance_dict = self.estimate_variance(
            input_dict,
            compared_estimators=compared_estimators,
        )

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        n_policies = len(input_dict)
        n_estimators = len(compared_estimators)

        if hue == "estimator":
            fig = plt.figure(figsize=(2 * n_estimators, 4 * n_policies))

            for i, eval_policy in enumerate(input_dict.keys()):
                if i == 0:
                    ax = ax0 = fig.add_subplot(len(input_dict), 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(len(input_dict), 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(len(input_dict), 1, i + 1)

                on_policy_mean = mean_dict[eval_policy].pop("on_policy")
                on_policy_variance = variance_dict[eval_policy].pop("on_policy")

                if on_policy_mean is not None:
                    on_policy_upper, on_policy_lower = norm.interval(
                        1 - alpha, loc=on_policy_mean, scale=np.sqrt(on_policy_variance)
                    )

                mean = np.array(list(mean_dict[eval_policy].values()), dtype=float)
                variance = np.array(
                    list(variance_dict[eval_policy].values()), dtype=float
                )
                lower, upper = norm.interval(
                    1 - alpha, loc=mean, scale=np.sqrt(variance)
                )

                if is_relative:
                    if on_policy_mean is not None:
                        on_policy_mean = on_policy_mean / self.behavior_policy_value
                        on_policy_upper = on_policy_upper / self.behavior_policy_value
                        on_policy_lower = on_policy_lower / self.behavior_policy_value

                    mean = mean / self.behavior_policy_value
                    upper = upper / self.behavior_policy_value
                    lower = lower / self.behavior_policy_value

                for j in range(n_estimators):
                    ax.errorbar(
                        np.arange(j, j + 1),
                        mean[j],
                        xerr=[0.4],
                        yerr=[
                            np.array([mean[j] - lower[j]]),
                            np.array([upper[j] - mean[j]]),
                        ],
                        color=color[j % n_colors],
                        elinewidth=5.0,
                    )

                elines = ax.get_children()
                for j in range(n_estimators):
                    elines[2 * j + 1].set_color("black")
                    elines[2 * j + 1].set_linewidth(2.0)

                if on_policy_mean is not None:
                    ax.axhline(on_policy_mean)
                    ax.axhspan(
                        ymin=on_policy_lower,
                        ymax=on_policy_upper,
                        alpha=0.3,
                    )
                ax.set_title(eval_policy, fontsize=16)
                ax.set_xticks(np.arange(n_estimators))
                ax.set_xticklabels(compared_estimators)
                ax.set_ylabel(
                    f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)",
                    fontsize=12,
                )
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)
                plt.xlim(-0.5, n_estimators - 0.5)

        else:
            visualize_on_policy = True
            for eval_policy in input_dict.keys():
                if input_dict[eval_policy]["on_policy_policy_value"] is None:
                    visualize_on_policy = False

            n_policies = len(input_dict)
            n_estimators = (
                len(compared_estimators) + 1
                if visualize_on_policy
                else len(compared_estimators)
            )

            fig = plt.figure(figsize=(2 * n_policies, 4 * n_estimators))

            for i, estimator in enumerate(self.ope_estimators_):
                if i == 0:
                    ax = ax0 = fig.add_subplot(n_estimators, 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(n_estimators, 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(n_estimators, 1, i + 1)

                mean = np.zeros(len(input_dict))
                variance = np.zeros(len(input_dict))
                for j, eval_policy in enumerate(input_dict.keys()):
                    mean[j] = mean_dict[eval_policy][estimator]
                    variance[j] = variance_dict[eval_policy][estimator]

                lower, upper = norm.interval(
                    1 - alpha, loc=mean, scale=np.sqrt(variance)
                )

                if is_relative:
                    mean = mean / self.behavior_policy_value
                    upper = upper / self.behavior_policy_value
                    lower = lower / self.behavior_policy_value

                for j in range(n_policies):
                    ax.errorbar(
                        np.arange(j, j + 1),
                        mean[j],
                        xerr=[0.4],
                        yerr=[
                            np.array([mean[j] - lower[j]]),
                            np.array([upper[j] - mean[j]]),
                        ],
                        color=color[j % n_colors],
                        elinewidth=5.0,
                    )

                elines = ax.get_children()
                for j in range(n_policies):
                    elines[3 * j + 2].set_color("black")
                    elines[3 * j + 2].set_linewidth(2.0)

                ax.set_title(estimator, fontsize=16)
                ax.set_xticks(np.arange(n_policies))
                ax.set_xticklabels(list(input_dict.keys()))
                ax.set_ylabel(
                    f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)",
                    fontsize=12,
                )
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)
                plt.xlim(-0.5, n_policies - 0.5)

            if visualize_on_policy:
                if sharey:
                    ax = fig.add_subplot(n_estimators, 1, i + 2, sharey=ax0)
                else:
                    ax = fig.add_subplot(n_estimators, 1, i + 2)

                on_policy_mean = mean_dict[eval_policy]["on_policy"]
                on_policy_variance = variance_dict[eval_policy]["on_policy"]
                on_policy_lower, on_policy_upper = norm.interval(
                    1 - alpha, loc=on_policy_mean, scale=np.sqrt(on_policy_variance)
                )

                if is_relative:
                    on_policy_mean = on_policy_mean / self.behavior_policy_value
                    on_policy_upper = on_policy_upper / self.behavior_policy_value
                    on_policy_lower = on_policy_lower / self.behavior_policy_value

                for j in range(n_policies):
                    ax.errorbar(
                        np.arange(j, j + 1),
                        mean[j],
                        xerr=[0.4],
                        yerr=[
                            np.array([mean[j] - lower[j]]),
                            np.array([upper[j] - mean[j]]),
                        ],
                        color=color[j % n_colors],
                        elinewidth=5.0,
                    )

                elines = ax.get_children()
                for j in range(n_policies):
                    elines[3 * j + 2].set_color("black")
                    elines[3 * j + 2].set_linewidth(2.0)

                ax.set_title("on_policy", fontsize=16)
                ax.set_xticks(np.arange(n_policies))
                ax.set_xticklabels(list(input_dict.keys()))
                ax.set_ylabel(
                    f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)",
                    fontsize=12,
                )
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)
                plt.xlim(-0.5, len(input_dict) - 0.5)

        fig.subplots_adjust(top=1.0)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_conditional_value_at_risk(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        alphas: Optional[np.ndarray] = None,
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_conditional_value_at_risk.png",
    ) -> None:
        """Visualize the conditional value at risk estimated by OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        alphas: array-like of shape (n_alpha, ), default=None
            Set of proportions of the sided region. The values should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the figure.

        n_cols: int, default=None
            Number of columns in the figure.

        sharey: bool, default=False
            This parameter is for API consistency.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_conditional_value_at_risk.png"
            Name of the bar figure.

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
        if alphas is None:
            alphas = np.linspace(0, 1, 21)
        if hue not in ["estimator", "policy"]:
            raise ValueError(
                f"hue must be either `estimator` or `policy`, but {hue} is given"
            )
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        visualize_on_policy = True
        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is None:
                visualize_on_policy = False

        cvar_dict = self.estimate_conditional_value_at_risk(
            input_dict,
            compared_estimators=compared_estimators,
            alphas=alphas,
        )

        if visualize_on_policy:
            compared_estimators.append("on_policy")

        plt.style.use("ggplot")

        if hue == "estimator":
            n_figs = len(input_dict)
            n_cols = min(3, n_figs) if n_cols is None else n_cols
            n_rows = (n_figs - 1) // n_cols + 1

            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows)
            )

            if n_rows == 1:
                for i, eval_policy in enumerate(input_dict.keys()):
                    for j, estimator in enumerate(compared_estimators):
                        axes[i].plot(
                            alphas,
                            cvar_dict[eval_policy][estimator],
                            label=estimator,
                        )

                    axes[i].set_title(eval_policy)
                    axes[i].set_xlabel("alpha")
                    axes[i].set_ylabel("CVaR")
                    if legend:
                        axes[i].legend()

                if legend:
                    handles, labels = axes[0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

            else:
                for i, eval_policy in enumerate(input_dict.keys()):
                    for j, estimator in enumerate(compared_estimators):
                        axes[i // n_cols, i % n_cols].plot(
                            alphas,
                            cvar_dict[eval_policy][estimator],
                            label=estimator,
                        )

                    axes[i // n_cols, i % n_cols].set_title(eval_policy)
                    axes[i // n_cols, i % n_cols].set_xlabel("alpha")
                    axes[i // n_cols, i % n_cols].set_ylabel("CVaR")
                    if legend:
                        axes[i // n_cols, i % n_cols].legend()

                if legend:
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

        else:
            n_figs = len(compared_estimators)
            n_cols = min(3, n_figs) if n_cols is None else n_cols
            n_rows = (n_figs - 1) // n_cols + 1

            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows)
            )

            if n_rows == 1:
                for i, estimator in enumerate(compared_estimators):
                    for j, eval_policy in enumerate(input_dict.keys()):
                        axes[i].plot(
                            alphas,
                            cvar_dict[eval_policy][estimator],
                            label=eval_policy,
                        )

                    axes[i].set_title(estimator)
                    axes[i].set_xlabel("alpha")
                    axes[i].set_ylabel("CVaR")
                    if legend:
                        axes[i].legend()

                if legend:
                    handles, labels = axes[0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

            else:
                for i, estimator in enumerate(compared_estimators):
                    for j, eval_policy in enumerate(input_dict.keys()):
                        axes[i // n_cols, i % n_cols].plot(
                            alphas,
                            cvar_dict[eval_policy][estimator],
                            label=eval_policy,
                        )

                    axes[i // n_cols, i % n_cols].set_title(estimator)
                    axes[i // n_cols, i % n_cols].set_xlabel("alpha")
                    axes[i // n_cols, i % n_cols].set_ylabel("CVaR")
                    if legend:
                        axes[i // n_cols, i % n_cols].legend()

            if legend:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                # n_cols shows err
                # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

        fig.subplots_adjust(hspace=0.35, wspace=0.2)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_interquartile_range(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        alpha: float = 0.05,
        hue: str = "estimator",
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_interquartile_range.png",
    ) -> None:
        """Visualize the interquartile range estimated by OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            
            .. code-block:: python

                key: [evaluation_policy_name][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                ]

            .. seealso::

                :class:`ope.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_interquartile_range.png"
            Name of the bar figure.

        """
        check_input_dict(input_dict)
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
        if hue not in ["estimator", "policy"]:
            raise ValueError(
                f"hue must be either `estimator` or `policy`, but {hue} is given"
            )
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        mean_dict = self.estimate_mean(
            input_dict,
            compared_estimators=compared_estimators,
        )
        interquartile_dict = self.estimate_interquartile_range(
            input_dict,
            compared_estimators=compared_estimators,
            alpha=alpha,
        )

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        visualize_on_policy = True
        for eval_policy in input_dict.keys():
            if input_dict[eval_policy]["on_policy_policy_value"] is None:
                visualize_on_policy = False

        n_estimators = (
            len(compared_estimators) + 1
            if visualize_on_policy
            else len(compared_estimators)
        )
        if visualize_on_policy:
            compared_estimators.append("on_policy")

        n_policies = len(input_dict)

        if hue == "estimator":
            fig = plt.figure(figsize=(2 * n_estimators, 4 * n_policies))

            for i, eval_policy in enumerate(input_dict.keys()):
                if i == 0:
                    ax = ax0 = fig.add_subplot(n_policies, 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(n_policies, 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(n_policies, 1, i + 1)

                mean = np.zeros(n_estimators)
                median = np.zeros(n_estimators)
                upper = np.zeros(n_estimators)
                lower = np.zeros(n_estimators)

                for j, estimator in enumerate(compared_estimators):
                    interquartile_dict_ = interquartile_dict[eval_policy][estimator]
                    mean[j] = mean_dict[eval_policy][estimator]
                    median[j] = interquartile_dict_["median"]
                    upper[j] = interquartile_dict_[
                        f"{100 * (1. - alpha)}% quartile (upper)"
                    ]
                    lower[j] = interquartile_dict_[
                        f"{100 * (1. - alpha)}% quartile (lower)"
                    ]

                ax.bar(
                    np.arange(n_estimators),
                    upper - lower,
                    bottom=lower,
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    tick_label=compared_estimators,
                    alpha=0.3,
                )

                for j in range(n_estimators):
                    ax.errorbar(
                        np.arange(j, j + 1),
                        median[j],
                        xerr=[0.4],
                        color=color[j % n_colors],
                        elinewidth=5.0,
                        fmt="o",
                        markersize=0.1,
                    )
                    ax.errorbar(
                        np.arange(j, j + 1),
                        mean[j],
                        color=color[j % n_colors],
                        fmt="o",
                        markersize=10.0,
                    )

                ax.set_title(eval_policy, fontsize=16)
                ax.set_ylabel(
                    f"Estimated {np.int(100*(1 - alpha))}% Interquartile Range",
                    fontsize=12,
                )
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)
                plt.xlim(-0.5, n_estimators - 0.5)

        else:
            fig = plt.figure(figsize=(2 * n_policies, 4 * n_estimators))

            for i, estimator in enumerate(compared_estimators):
                if i == 0:
                    ax = ax0 = fig.add_subplot(n_estimators, 1, i + 1)
                elif sharey:
                    ax = fig.add_subplot(n_estimators, 1, i + 1, sharey=ax0)
                else:
                    ax = fig.add_subplot(n_estimators, 1, i + 1)

                mean = np.zeros(n_policies)
                median = np.zeros(n_policies)
                upper = np.zeros(n_policies)
                lower = np.zeros(n_policies)

                for j, eval_policy in enumerate(input_dict.keys()):
                    interquartile_dict_ = interquartile_dict[eval_policy][estimator]
                    mean[j] = mean_dict[eval_policy][estimator]
                    median[j] = interquartile_dict_["median"]
                    upper[j] = interquartile_dict_[
                        f"{100 * (1. - alpha)}% quartile (upper)"
                    ]
                    lower[j] = interquartile_dict_[
                        f"{100 * (1. - alpha)}% quartile (lower)"
                    ]

                ax.bar(
                    np.arange(n_policies),
                    upper - lower,
                    bottom=lower,
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    tick_label=list(input_dict.keys()),
                    alpha=0.3,
                )

                for j in range(n_policies):
                    ax.errorbar(
                        np.arange(j, j + 1),
                        median[j],
                        xerr=[0.4],
                        color=color[j % n_colors],
                        elinewidth=5.0,
                        fmt="o",
                        markersize=0.1,
                    )
                    ax.errorbar(
                        np.arange(j, j + 1),
                        mean[j],
                        color=color[j % n_colors],
                        fmt="o",
                        markersize=10.0,
                    )

                ax.set_title(estimator, fontsize=16)
                ax.set_ylabel(
                    f"Estimated {np.int(100*(1 - alpha))}% Interquartile Range",
                    fontsize=12,
                )
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)
                plt.xlim(-0.5, n_policies - 0.5)

        fig.subplots_adjust(top=1.0)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    @property
    def estimators_name(self):
        return list(self.ope_estimators_.keys())
