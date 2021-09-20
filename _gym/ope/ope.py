"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from collections import defaultdict
from tqdm import tqdm

import torch
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import gym
from d3rlpy.ope import DiscreteFQE
from d3rlpy.ope import FQE as ContinuousFQE

from _gym.ope import BaseOffPolicyEstimator
from _gym.policy import BaseHead
from _gym.types import LoggedDataset, OPEInputDict
from _gym.utils import (
    convert_logged_dataset_into_MDPDataset,
    check_base_model_args,
    check_if_valid_env_and_logged_dataset,
    check_input_dict,
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
        Estimators must follow the interface of `_gym.ope.BaseOffPolicyEstimator`.

    Examples
    ----------

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]

    def __post__init__(self) -> None:
        "Initialize class."
        self.action_type = self.logged_dataset["action_type"]
        self.step_per_episode = self.logged_dataset["step_per_episode"]

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != self.action_type:
                raise ValueError(
                    f"one of the ope_estimators, {estimator.estimator_name} does not much action_type in logged_dataset. Please use {self.action_type} type instead."
                )

        behavior_policy_pscore = self.logged_dataset["pscore"].reshape(
            (-1, self.step_per_episode)
        )
        behavior_policy_step_wise_pscore = np.cumprod(behavior_policy_pscore, axis=1)
        behavior_policy_trajectory_wise_pscore = np.tile(
            behavior_policy_step_wise_pscore[:, -1], (self.step_per_episode, 1)
        ).T

        self.input_dict_ = {
            "actions": self.logged_dataset["action"],
            "rewards": self.logged_dataset["reward"],
            "behavior_policy_step_wise_pscore": behavior_policy_step_wise_pscore.flatten(),
            "behavior_policy_trajectory_wise_pscore": behavior_policy_trajectory_wise_pscore.flatten(),
        }

    def estimate_policy_values(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
    ) -> Dict[str, float]:
        check_input_dict(input_dict)

        policy_value_dict = defaultdict(dict)

        for eval_policy in input_dict.keys():
            policy_value_dict[eval_policy]["on_policy"] = input_dict[eval_policy][
                "on_policy_policy_value"
            ]
            for estimator_name, estimator in self.ope_estimators_.items():
                policy_value_dict[eval_policy][
                    estimator_name
                ] = estimator.estimate_policy_value(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    gamma=gamma,
                )
        return policy_value_dict

    def estimate_intervals(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate confidence intervals of policy values using nonparametric bootstrap procedure."""
        check_input_dict(input_dict)

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
        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Summarize policy values and their confidence intervals estimated by OPE estimators."""
        policy_value_dict = self.estimate_policy_values(input_dict)
        policy_value_interval_dict = self.estimate_policy_values(
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

            on_policy_policy_value = policy_value_dict[eval_policy]["on_policy"]
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
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy values estimated by OPE estimators."""
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "fig_dir must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "fig_dir must be a string"
        check_input_dict(input_dict)

        estimated_round_rewards_df_dict = dict()
        for eval_policy in input_dict.keys():
            estimated_round_rewards_dict_ = dict()
            for estimator_name, estimator in self.ope_estimators_.items():
                estimated_round_rewards_dict_[
                    estimator_name
                ] = estimator._estimate_round_rewards(
                    **input_dict[eval_policy],
                    **self.input_dict_,
                    gamma=gamma,
                )
            estimated_round_rewards_df_ = DataFrame(estimated_round_rewards_dict_)

            on_policy_policy_value = input_dict[eval_policy]["on_policy_policy_value"]
            if is_relative:
                if on_policy_policy_value is not None and on_policy_policy_value > 0:
                    estimated_round_rewards_df_dict[eval_policy] = (
                        estimated_round_rewards_df_ / on_policy_policy_value
                    )
                else:
                    raise ValueError()

            else:
                estimated_round_rewards_df_dict[
                    eval_policy
                ] = estimated_round_rewards_df_

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(8, 6.2 * len(input_dict)))

        for i, eval_policy in input_dict.keys():
            ax = fig.add_subplot(len(self.ope_estimators_), 1, i + 1)
            sns.barplot(
                data=estimated_round_rewards_df_dict[eval_policy],
                ax=ax,
                ci=100 * (1 - alpha),
                n_boot=n_bootstrap_samples,
                seed=random_state,
            )
            on_policy_policy_value = input_dict[eval_policy]["on_policy_policy_value"]
            if on_policy_policy_value is not None and not is_relative:
                ax.axhline(on_policy_policy_value)
            ax.set_title(eval_policy, fontsize=20)
            ax.set_ylabel(
                f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=20
            )
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=25 - 2 * len(self.ope_estimators_))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def evaluate_performance_of_estimators(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        metric: str = "relative-ee",
    ) -> Dict[str, float]:
        """Evaluate estimation performance of OPE estimators."""
        eval_metric_ope_dict = defaultdict(dict)
        policy_value_dict = self.estimate_policy_values(input_dict, gamma=gamma)

        if metric == "relative-ee":
            for eval_policy in input_dict.keys():
                on_policy_policy_value = input_dict[eval_policy][
                    "on_policy_policy_value"
                ]

                for estimator in self.ope_estimators_.keys():
                    relative_ee_ = (
                        policy_value_dict[eval_policy] - on_policy_policy_value
                    ) / on_policy_policy_value
                    eval_metric_ope_dict[eval_policy][estimator] = np.abs(relative_ee_)

        else:
            for eval_policy in input_dict.keys():
                on_policy_policy_value = input_dict[eval_policy][
                    "on_policy_policy_value"
                ]

                for estimator in self.ope_estimators_.keys():
                    se_ = (policy_value_dict[eval_policy] - on_policy_policy_value) ** 2
                    eval_metric_ope_dict[eval_policy][estimator] = se_

        return eval_metric_ope_dict

    def summarize_estimators_comparison(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        metric: str = "relative-ee",
    ) -> DataFrame:
        eval_metric_ope_df = DataFrame()
        eval_metric_ope_dict = self.evaluate_performance_of_estimators(
            input_dict,
            gamma=gamma,
            metric=metric,
        )
        for eval_policy in input_dict.keys():
            eval_metric_ope_df[eval_policy] = DataFrame(
                eval_metric_ope_dict[eval_policy]
            ).T
        return eval_metric_ope_df


@dataclass
class CreateOPEInput:
    """Class to prepare OPE inputs.

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    base_model_args: Optional[Dict[str, Any]], default = None
        Arguments of baseline Fitted Q Evaluation (FQE) model.

    """

    logged_dataset: LoggedDataset
    base_model_args: Optional[Dict[str, Any]] = None
    use_base_model: bool = False

    def __post_init__(self) -> None:
        "Initialize class."
        self.action_type = self.logged_dataset["action_type"]
        self.n_actions = self.logged_dataset["n_actions"]
        self.step_per_episode = self.logged_dataset["step_per_episode"]
        self.mdp_dataset = convert_logged_dataset_into_MDPDataset(self.logged_dataset)

        if self.use_base_model:
            self.fqe = dict()
            if self.base_model_args is None:
                self.base_model_args = {
                    "n_epochs": 200,
                    "q_func_factory": "qr",
                    "learning_rate": 1e-4,
                    "use_gpu": torch.cuda.is_available(),
                    "encoder_params": {"hidden_units": [20]},
                }
            check_base_model_args(
                dataset=self.mdp_dataset,
                args=self.base_model_args,
                action_type=self.action_type,
            )

    def construct_FQE(
        self,
        evaluation_policy: BaseHead,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,  # should be more than n_steps_per_epoch
        n_steps_per_epoch: int = 10000,
    ) -> None:
        if n_epochs is None and n_steps is None:
            n_steps = n_steps_per_epoch

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

    def obtain_evaluation_policy_action(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        return evaluation_policy.predict(x=self.logged_dataset["state"])

    def obtain_pscore_for_observed_state_action(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        return evaluation_policy.calculate_pscore_given_action(
            x=self.logged_dataset["state"],
            action=self.logged_dataset["action"],
        )

    def obtain_step_wise_pscore(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        base_pscore = self.obtain_pscore_for_observed_state_action(
            evaluation_policy
        ).reshape((-1, self.step_per_episode))
        return np.cumprod(base_pscore, axis=1).flatten()

    def obtain_trajectory_wise_pscore(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        base_pscore = self.obtain_step_wise_pscore(evaluation_policy).reshape(
            (-1, self.step_per_episode)
        )[:, -1]
        return np.tile(base_pscore, (self.step_per_episode, 1)).T.flatten()

    def obtain_state_action_value_with_pscore(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        state_action_value = (
            evaluation_policy.predict_counterfactual_state_action_value(
                self.logged_dataset["state"]
            )
        )
        pscore = evaluation_policy.calculate_action_choice_probability(
            self.logged_dataset["state"]
        )
        return state_action_value, pscore  # (n_samples, n_actions)

    def obtain_initial_state_value(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        state_action_value, pscore = self.obtain_state_action_value_with_pscore(
            evaluation_policy
        )

        if self.action_type == "discrete":
            state_action_value.reshape(self.n_actions)

        state_value = np.sum(state_action_value * pscore, axis=1)
        return state_value.reshape((-1, self.step_per_episode))[:, 0]  # (n_samples, )

    def evaluate_online(
        env: gym.Env,
        evaluation_policy: BaseHead,
        n_episodes: int = 1000,
    ) -> float:
        total_reward = 0.0
        for _ in tqdm(
            np.arange(n_episodes),
            desc="[calc_on_policy_policy_value]",
            total=n_episodes,
        ):
            state = env.reset()
            done = False

            while not done:
                action = evaluation_policy.sample_action_online(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

        return total_reward / n_episodes

    def obtain_whole_inputs(
        self,
        evaluation_policies: List[BaseHead],
        env: Optional[gym.Env] = None,
    ) -> OPEInputDict:

        if env is not None:
            check_if_valid_env_and_logged_dataset(env, self.logged_dataset)

        for eval_policy in evaluation_policies:
            if eval_policy.action_type != self.action_type:
                raise ValueError(
                    f"one of the evaluation_policies, {eval_policy.name} does not much action_type in logged_dataset. Please use {self.action_type} type instead."
                )

        if self.use_base_model:
            for i in tqdm(
                range(len(evaluation_policies)),
                desc="[fit FQE model]",
                total=len(evaluation_policies),
            ):
                self.construct_FQE(evaluation_policies[i])

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
                    "evaluation_policy_actions"
                ] = None
            else:
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_step_wise_pscore"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_trajectory_wise_pscore"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_actions"
                ] = self.obtain_evaluation_policy_action(evaluation_policies[i])

            # input for DM, DR
            if self.use_base_model:
                state_action_value, pscore = self.obtain_state_action_value_with_pscore(
                    evaluation_policies[i]
                )
                input_dict[evaluation_policies[i].name][
                    "counterfactual_state_action_value"
                ] = state_action_value
                input_dict[evaluation_policies[i].name][
                    "counterfactual_pscore"
                ] = pscore
                input_dict[evaluation_policies[i].name][
                    "initial_state_value"
                ] = self.obtain_initial_state_value(evaluation_policies[i])
            else:
                input_dict[evaluation_policies[i].name][
                    "counterfactual_state_action_value"
                ] = None
                input_dict[evaluation_policies[i].name]["counterfactual_pscore"] = None
                input_dict[evaluation_policies[i].name]["initial_state_value"] = None
            # input for the evaluation of OPE estimators
            if env is not None:
                input_dict[evaluation_policies[i].name][
                    "on_policy_policy_value"
                ] = self.evaluate_online(env, evaluation_policies[i])
            else:
                input_dict[evaluation_policies[i].name]["on_policy_policy_value"] = None

        return input_dict
