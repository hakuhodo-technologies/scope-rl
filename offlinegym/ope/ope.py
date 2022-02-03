"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from collections import defaultdict
from sklearn.utils.validation import check_scalar
from tqdm.autonotebook import tqdm

import torch
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import gym
from gym.spaces import Box, Discrete
from d3rlpy.dataset import MDPDataset
from d3rlpy.ope import DiscreteFQE
from d3rlpy.ope import FQE as ContinuousFQE
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory

from offlinegym.ope.estimators_discrete import BaseOffPolicyEstimator
from offlinegym.ope.online import rollout_policy_online
from offlinegym.policy.head import BaseHead
from offlinegym.types import LoggedDataset, OPEInputDict
from offlinegym.utils import (
    check_logged_dataset,
    estimate_confidence_interval_by_bootstrap,
    defaultdict_to_dict,
)


@dataclass
class OffPolicyEvaluation:
    """Class to conduct OPE by multiple estimators simultaneously.

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

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.action_type = self.logged_dataset["action_type"]
        self.step_per_episode = self.logged_dataset["step_per_episode"]

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != self.action_type:
                raise RuntimeError(
                    f"One of the ope_estimators, {estimator.estimator_name} does not match action_type in logged_dataset. Please use {self.action_type} type instead"
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
            policy_value_dict[eval_policy]["on_policy"] = input_dict[eval_policy][
                "on_policy_policy_value"
            ].mean()
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

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        policy_value_interval_dict: Dict[str, Dict[str, float]]
            Dictionary containing estimated confidence intervals estimated
            using nonparametric bootstrap procedure.
            key: [evaluation_policy_name][OPE_estimator_name]

        """
        policy_value_interval_dict = defaultdict(dict)

        for eval_policy in input_dict.keys():
            for estimator_name, estimator in self.ope_estimators_.items():
                policy_value_interval_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_interval(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    gamma=gamma,
                    alpha=alpha,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )
        return defaultdict_to_dict(policy_value_interval_dict)

    def summarize_off_policy_estimates(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
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

            on_policy_policy_value = policy_value_dict[eval_policy]["on_policy"].mean()
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
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
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

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

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

        estimated_trajectory_value_df_dict = dict()
        for eval_policy in input_dict.keys():
            estimated_trajectory_value_dict_ = dict()
            for estimator_name, estimator in self.ope_estimators_.items():
                estimated_trajectory_value_dict_[
                    estimator_name
                ] = estimator._estimate_trajectory_value(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    gamma=gamma,
                )
            estimated_trajectory_value_df_ = DataFrame(estimated_trajectory_value_dict_)

            on_policy_policy_value = input_dict[eval_policy]["on_policy_policy_value"]
            if is_relative:
                if on_policy_policy_value is not None and on_policy_policy_value > 0:
                    estimated_trajectory_value_df_dict[eval_policy] = (
                        estimated_trajectory_value_df_ / on_policy_policy_value.mean()
                    )
                else:
                    raise ValueError(
                        f"on_policy_policy_value must be a positive value, but {on_policy_policy_value} is given"
                    )

            estimated_trajectory_value_df_dict[
                eval_policy
            ] = estimated_trajectory_value_df_

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(2 * len(self.ope_estimators_), 12 * len(input_dict)))

        for i, eval_policy in enumerate(input_dict.keys()):
            if i == 0:
                ax = ax0 = fig.add_subplot(len(self.ope_estimators_), 1, i + 1)
            elif sharey:
                ax = fig.add_subplot(len(self.ope_estimators_), 1, i + 1, sharey=ax0)
            else:
                ax = fig.add_subplot(len(self.ope_estimators_), 1, i + 1)

            sns.barplot(
                data=estimated_trajectory_value_df_dict[eval_policy],
                ax=ax,
                ci=100 * (1 - alpha),
                n_boot=n_bootstrap_samples,
                seed=random_state,
            )
            on_policy_policy_value = input_dict[eval_policy]["on_policy_policy_value"]
            if on_policy_policy_value is not None:
                on_policy_interval = estimate_confidence_interval_by_bootstrap(
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
                f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=12
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
class CreateOPEInput:
    """Class to prepare OPE inputs.

    Parameters
    -------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    base_model_args: Optional[Dict[str, Any]], default=None
        Arguments of baseline Fitted Q Evaluation (FQE) model.

    use_base_model: bool, default=False
        Whether to use FQE and obtain :math:`\\hat{Q}`.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Takuma Seno and Michita Imai.
    "d3rlpy: An Offline Deep Reinforcement Library.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    """

    logged_dataset: LoggedDataset
    base_model_args: Optional[Dict[str, Any]] = None
    use_base_model: bool = False

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.n_episodes = self.logged_dataset["n_episodes"]
        self.action_type = self.logged_dataset["action_type"]
        self.n_actions = self.logged_dataset["n_actions"]
        self.action_dim = self.logged_dataset["action_dim"]
        self.state_dim = self.logged_dataset["state_dim"]
        self.step_per_episode = self.logged_dataset["step_per_episode"]

        if self.logged_dataset["action_type"] == "discrete":
            self.mdp_dataset = MDPDataset(
                observations=self.logged_dataset["state"],
                actions=self.logged_dataset["action"],
                rewards=self.logged_dataset["reward"],
                terminals=self.logged_dataset["done"],
                episode_terminals=self.logged_dataset["terminal"],
                discrete_action=True,
            )
        else:
            self.mdp_dataset = MDPDataset(
                observations=self.logged_dataset["state"],
                actions=self.logged_dataset["action"],
                rewards=self.logged_dataset["reward"],
                terminals=self.logged_dataset["done"],
                episode_terminals=self.logged_dataset["terminal"],
            )

        if self.use_base_model:
            self.fqe = {}
            if self.base_model_args is None:
                self.base_model_args = {
                    "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
                    "q_func_factory": MeanQFunctionFactory(),
                    "learning_rate": 1e-4,
                    "use_gpu": torch.cuda.is_available(),
                }

    def build_and_fit_FQE(
        self,
        evaluation_policy: BaseHead,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,  # should be more than n_steps_per_epoch
        n_steps_per_epoch: int = 10000,
    ) -> None:
        """Fit Fitted Q Evaluation (FQE).

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        n_epochs: Optional[int], default=None (> 0)
            Number of epochs to fit FQE.

        n_steps: Optional[int], default=None (> 0)
            Total number pf steps to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        if n_epochs is not None:
            check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        if n_steps is not None:
            check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

        if n_epochs is None and n_steps is None:
            n_steps = n_steps_per_epoch

        if evaluation_policy.name in self.fqe:
            pass

        else:
            if self.action_type == "discrete":
                self.fqe[evaluation_policy.name] = DiscreteFQE(
                    algo=evaluation_policy, **self.base_model_args
                )
            else:
                self.fqe[evaluation_policy.name] = ContinuousFQE(
                    algo=evaluation_policy, **self.base_model_args
                )

            self.fqe[evaluation_policy.name].fit(
                self.mdp_dataset.episodes,
                eval_episodes=self.mdp_dataset.episodes,
                n_epochs=n_epochs,
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                scorers={},
            )

    def predict_state_action_value(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Predict state action value for all actions.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        state_action_value_prediction: NDArray, shape (n_samples, n_actions)
            State action value for observed state and all actions,
            i.e., math`\\hat{Q}(s, a) \\forall a \\in \\mathcal{A}`.

        """
        x = self.logged_dataset["state"]
        x_ = []
        for i in range(x.shape[0]):
            x_.append(np.tile(x[i], (self.n_actions, 1)))
        x_ = np.array(x_).reshape((-1, x.shape[1]))
        a_ = np.tile(np.arange(self.n_actions), x.shape[0])
        return self.fqe[evaluation_policy.name].predict_value(
            x_, a_
        )  # (n_samples, n_actions)

    def obtain_evaluation_policy_action(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain evaluation policy action.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_action: NDArray
            Evaluation policy action :math:`a_t \\sim \\pi_e(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        return evaluation_policy.predict(x=self.logged_dataset["state"])

    def obtain_pscore_for_observed_state_action(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain pscore for observed state action pair.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_pscore: NDArray
            Evaluation policy pscore :math:`\\pi_e(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        return evaluation_policy.calc_pscore_given_action(
            x=self.logged_dataset["state"],
            action=self.logged_dataset["action"],
        )

    def obtain_step_wise_pscore(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain step-wise pscore for the observed state action pair.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_step_wise_pscore: NDArray
            Evaluation policy's step-wise pscore :math:`\\prod_{t'=1}^t \\pi_e(a_{t'} \\mid s_{t'})`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        base_pscore = self.obtain_pscore_for_observed_state_action(
            evaluation_policy
        ).reshape((-1, self.step_per_episode))
        return np.cumprod(base_pscore, axis=1).flatten()

    def obtain_trajectory_wise_pscore(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain trajectory-wise pscore for the observed state action pair.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_trajectory_wise_pscore: NDArray
            Evaluation policy's trajectory-wise pscore :math:`\\prod_{t=1}^T \\pi_e(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        base_pscore = self.obtain_step_wise_pscore(evaluation_policy).reshape(
            (-1, self.step_per_episode)
        )[:, -1]
        return np.tile(base_pscore, (self.step_per_episode, 1)).T.flatten()

    def obtain_action_dist_with_state_action_value_prediction_discrete(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain action choice probability of the discrete evaluation policy and its Q hat for the observed state.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_action_dist: NDArray
            Evaluation policy pscore :math:`\\pi_e(a_t \\mid s_t)`.

        state_action_value_prediction: NDArray, shape (n_samples, n_actions)
            State action value for all observed state and possible action.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        action_dist = evaluation_policy.calc_action_choice_probability(
            self.logged_dataset["state"]
        )
        state_action_value_prediction = (
            self.predict_state_action_value(evaluation_policy)
        ).reshape((-1, self.n_actions))
        return action_dist, state_action_value_prediction  # (n_samples, n_actions)

    def obtain_state_action_value_prediction_continuous(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain Q hat for the continuous (deterministic) evaluation policy.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        state_action_value_prediction: NDArray, shape (n_samples, n_actions)
            State action value for the observed state and action chosen by evaluation policy.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        state = self.logged_dataset["state"]
        action = evaluation_policy.predict(state)
        return self.fqe[evaluation_policy.name].predict_value(state, action)

    def obtain_initial_state_value_prediction_discrete(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain initial state value for the discrete evaluation policy.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        initial_state_value_prediction: NDArray, shape (n_samples, n_actions)
            State action value for the observed state and action chosen by evaluation policy.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        (
            state_action_value,
            pscore,
        ) = self.obtain_action_dist_with_state_action_value_prediction_discrete(
            evaluation_policy
        )
        state_action_value = state_action_value.reshape((-1, self.n_actions))
        state_value = np.sum(state_action_value * pscore, axis=1)
        return state_value.reshape((-1, self.step_per_episode))[:, 0]  # (n_samples, )

    def obtain_initial_state_value_prediction_continuous(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain initial state value for the continuous evaluation policy.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        initial_state_value_prediction: NDArray, shape (n_samples, n_actions)
            State action value for the observed state and action chosen by evaluation policy.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        state_value = self.obtain_state_action_value_prediction_continuous(
            evaluation_policy
        )
        return state_value.reshape((-1, self.step_per_episode))[:, 0]

    def obtain_whole_inputs(
        self,
        evaluation_policies: List[BaseHead],
        env: Optional[gym.Env] = None,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,  # should be more than n_steps_per_epoch
        n_steps_per_epoch: Optional[int] = None,
        n_episodes_on_policy_evaluation: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> OPEInputDict:
        """Obtain input as a dictionary.

        Parameters
        -------
        evaluation_policies: List[BaseHead]
            Evaluation policies.

        env: gym.Env
            Reinforcement learning (RL) environment.

        n_epochs: Optional[int], default=None (> 0)
            Number of epochs to fit FQE.

        n_steps: Optional[int], default=None (> 0)
            Total number pf steps to fit FQE.

        n_steps_per_epoch: int, default=None (> 0)
            Number of steps in an epoch.

        n_episodes_on_policy_evaluation: Optional[int], default=None (> 0)
            Number of episodes to perform on-policy evaluation.

        Return
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.
            key: [evaluation_policy_name][
                evaluation_policy_step_wise_pscore,
                evaluation_policy_trajectory_wise_pscore,
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                on_policy_policy_value,
            ]

            evaluation_policy_step_wise_pscore: Optional[NDArray], shape (n_episodes * step_per_episodes, )
                Step-wise action choice probability of evaluation policy,
                i.e., :math:`\\prod_{t'=0}^t \\pi_e(a_{t'} \\mid s_{t'})`
                If action_type == "continuous", `None` is recorded.

            evaluation_policy_trajectory_wise_pscore: Optional[NDArray], shape (n_episodes * step_per_episodes, )
                Trajectory-wise action choice probability of evaluation policy,
                i.e., :math:`\\prod_{t=0}^T \\pi_e(a_t \\mid s_t)`
                If action_type == "continuous", `None` is recorded.

            evaluation_policy_action: Optional[NDArray], shape (n_episodes * step_per_episodes, )
                Action chosen by the deterministic evaluation policy.
                If action_type == "discrete", `None` is recorded.

            evaluation_policy_action_dist: Optional[NDArray], shape (n_episodes * step_per_episode, n_actions)
                Action choice probability of evaluation policy for all actions,
                i.e., :math:`\\pi_e(a \\mid s_t) \\forall a \\in \\mathcal{A}`
                If action_type == "continuous", `None` is recorded.

            state_action_value_prediction: Optional[NDArray]
                If action_type == "discrete", :math:`\\hat{Q}` for all actions,
                i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.
                shape (n_episodes * step_per_episode, n_actions)

                If action_type == "continuous", :math:`\\hat{Q}` for the action chosen by evaluation policy,
                i.e., :math:`\\hat{Q}(s_t, \\pi_e(a \\mid s_t))`.
                shape (n_episodes * step_per_episode, )

                If use_base_model == False, `None` is recorded.

            initial_state_value_prediction: Optional[NDArray], shape (n_episodes, )
                Estimated initial state value.
                If use_base_model == False, `None` is recorded.

            on_policy_policy_value: Optional[NDArray], shape (n_episodes_on_policy_evaluation, )
                On-policy policy value.
                If env is None, `None` is recorded.

        """
        if env is not None:
            if isinstance(env.action_space, Box) and self.action_type == "discrete":
                raise RuntimeError(
                    "Found mismatch in action_type between env and logged_dataset"
                )
            elif (
                isinstance(env.action_space, Discrete)
                and self.action_type == "continuous"
            ):
                raise RuntimeError(
                    "Found mismatch in action_type between env and logged_dataset"
                )

        for eval_policy in evaluation_policies:
            if eval_policy.action_type != self.action_type:
                raise RuntimeError(
                    f"One of the evaluation_policies, {eval_policy.name} does not match action_type in logged_dataset."
                    " Please use {self.action_type} action type instead."
                )

        if n_episodes_on_policy_evaluation is not None:
            check_scalar(
                n_episodes_on_policy_evaluation,
                name="n_episodes_on_policy_evaluation",
                target_type=int,
                min_val=1,
            )

        if self.use_base_model:
            if n_steps_per_epoch is None:
                n_steps_per_epoch = 10000

            for i in tqdm(
                range(len(evaluation_policies)),
                desc="[fit FQE model]",
                total=len(evaluation_policies),
            ):
                self.build_and_fit_FQE(
                    evaluation_policies[i],
                    n_epochs=n_epochs,
                    n_steps=n_steps,
                    n_steps_per_epoch=n_steps_per_epoch,
                )

        input_dict = defaultdict(dict)

        for i in tqdm(
            range(len(evaluation_policies)),
            desc="[collect input data]",
            total=len(evaluation_policies),
        ):
            # input for IPW, DR
            if self.action_type == "discrete":
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_step_wise_pscore"
                ] = self.obtain_step_wise_pscore(evaluation_policies[i])
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_trajectory_wise_pscore"
                ] = self.obtain_trajectory_wise_pscore(evaluation_policies[i])
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action"
                ] = None
            else:
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_step_wise_pscore"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_trajectory_wise_pscore"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action"
                ] = self.obtain_evaluation_policy_action(evaluation_policies[i])

            # input for DM, DR
            if self.action_type == "discrete":
                if self.use_base_model:
                    (
                        action_dist,
                        state_action_value_prediction,
                    ) = self.obtain_action_dist_with_state_action_value_prediction_discrete(
                        evaluation_policies[i]
                    )
                    input_dict[evaluation_policies[i].name][
                        "evaluation_policy_action_dist"
                    ] = action_dist
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = state_action_value_prediction
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = self.obtain_initial_state_value_prediction_discrete(
                        evaluation_policies[i]
                    )
                else:
                    input_dict[evaluation_policies[i].name][
                        "evaluation_policy_action_dist"
                    ] = None
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = None
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = None
            else:
                if self.use_base_model:
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = self.obtain_state_action_value_prediction_continuous(
                        evaluation_policies[i]
                    )
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = self.obtain_initial_state_value_prediction_continuous(
                        evaluation_policies[i]
                    )
                else:
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = None
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = None

            # input for the evaluation of OPE estimators
            if env is not None:
                if n_episodes_on_policy_evaluation is None:
                    n_episodes_on_policy_evaluation = self.n_episodes

                input_dict[evaluation_policies[i].name][
                    "on_policy_policy_value"
                ] = rollout_policy_online(
                    env,
                    evaluation_policies[i],
                    n_episodes=n_episodes_on_policy_evaluation,
                    random_state=random_state,
                )

            else:
                input_dict[evaluation_policies[i].name]["on_policy_policy_value"] = None

        return defaultdict_to_dict(input_dict)