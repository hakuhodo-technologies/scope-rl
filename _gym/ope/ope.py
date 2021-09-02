"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

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

        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

            if estimator.action_type != self.action_type:
                raise ValueError(
                    f"one of the ope_estimators, {estimator.estimator_name} does not much action_type in logged_dataset. Please use {self.action_type} type instead."
                )

    def estimate_policy_values(
        self,
        input_dict: OPEInputDict,
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
                )

        return policy_value_dict


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

    def __post__init__(self) -> None:
        "Initialize class."
        self.action_type = self.logged_dataset["action_type"]
        self.step_per_episode = self.logged_dataset["step_per_episode"]
        self.mdp_dataset = convert_logged_dataset_into_MDPDataset(self.logged_dataset)

        if self.use_base_model:
            self.fqe = []
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
        base_pscore = self.obtain_pscore_for_observed_state_action(evaluation_policy)
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
        pscore = evaluation_policy.calculate_action_choice_propability(
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
                action = evaluation_policy.sample_action(state)
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
            input_dict[evaluation_policies[i].name][
                "step_wise_pscore"
            ] = self.obtain_step_wise_pscore(evaluation_policies[i])
            input_dict[evaluation_policies[i].name][
                "trajectory_wise_pscore"
            ] = self.obtain_trajectory_wise_pscore(evaluation_policies[i])
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
