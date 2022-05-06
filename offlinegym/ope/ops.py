from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, List
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_scalar
import matplotlib.pyplot as plt

from offlinegym.ope.ope_discrete import (
    DiscreteOffPolicyEvaluation,
    DiscreteCumulativeDistributionalOffPolicyEvaluation,
    DiscreteDistributionallyRobustOffPolicyEvaluation,
)
from offlinegym.ope.ope_continuous import (
    ContinuousOffPolicyEvaluation,
    ContinuousCumulativeDistributionalOffPolicyEvaluation,
    ContinuousDistributionallyRobustOffPolicyEvaluation,
)
from offlinegym.types import OPEInputDict
from offlinegym.utils import check_array, defaultdict_to_dict


@dataclass
class OffPolicySelection:
    """Class to conduct OPS by multiple estimators simultaneously.

    Parameters
    -----------
    ope: Union[DiscreteOffPolicyEvaluation, ContinuousOffPolicyEvaluation], default=None
        Instance of the (standard) OPE class.

    cumulative_distributional_ope: Union[DiscreteCumulativeDistributionalOffPolicyEvaluation, ContinuousCumulativeDistributionalOffPolicyEvaluation], default=None
        Instance of the cumulative distributional OPE class.

    distributionally_robust_ope: Union[DiscreteDistributionallyRobustOffPolicyEvaluation, ContinuousDistributionallyRobustOffPolicyEvaluation], default=None
        Instance of the distributionally robust OPE class.

    Examples
    ----------


    References
    -------
    Shengpu Tang and Jenna Wiens.
    "Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings.", 2021.

    Justin Fu, Mohammad Norouzi, Ofir Nachum, George Tucker, Ziyu Wang, Alexander Novikov, Mengjiao Yang,
    Michael R. Zhang, Yutian Chen, Aviral Kumar, Cosmin Paduraru, Sergey Levine, and Tom Le Paine.
    "Benchmarks for Deep Off-Policy Evaluation.", 2021.

    Tom Le Paine, Cosmin Paduraru, Andrea Michi, Caglar Gulcehre, Konrad Zolna, Alexander Novikov, Ziyu Wang, and Nando de Freitas.
    "Hyperparameter Selection for Offline Reinforcement Learning.", 2020.

    """

    ope: Optional[
        Union[DiscreteOffPolicyEvaluation, ContinuousOffPolicyEvaluation]
    ] = None
    cumulative_distributional_ope: Optional[
        Union[
            DiscreteCumulativeDistributionalOffPolicyEvaluation,
            ContinuousCumulativeDistributionalOffPolicyEvaluation,
        ]
    ] = None
    distributionally_robust_ope: Optional[
        Union[
            DiscreteDistributionallyRobustOffPolicyEvaluation,
            ContinuousDistributionallyRobustOffPolicyEvaluation,
        ]
    ] = None

    def __post_init__(self):
        if (
            self.ope is None
            and self.cumulative_distributional_ope is None
            and self.distributionally_robust_ope is None
        ):
            raise RuntimeError(
                "one of ope, cumulative_distributional_ope, or distributionally_robust_ope must be given"
            )
        # if not isinstance(
        #     self.ope, (DiscreteOffPolicyEvaluation, ContinuousOffPolicyEvaluation)
        # ):
        #     raise RuntimeError(
        #         "ope must be the instance of either DiscreteOffPolicyEvaluation or ContinuousOffPolicyEvaluation"
        #     )
        # if not isinstance(
        #     self.cumulative_distributional_ope,
        #     (
        #         DiscreteCumulativeDistributionalOffPolicyEvaluation,
        #         ContinuousCumulativeDistributionalOffPolicyEvaluation,
        #     ),
        # ):
        #     raise RuntimeError(
        #         "cumulative_distributional_ope must be the instance of either "
        #         "DiscreteCumulativeDistributionalOffPolicyEvaluation or ContinuousCumulativeDistributionalOffPolicyEvaluation"
        #     )
        # if not isinstance(
        #     self.cumulative_distributional_ope,
        #     (
        #         DiscreteDistributionallyRobustOffPolicyEvaluation,
        #         ContinuousDistributionallyRobustOffPolicyEvaluation,
        #     ),
        # ):
        #     raise RuntimeError(
        #         "distributionally_robust_ope must be the instance of either "
        #         "DiscreteDistributionallyRobustOffPolicyEvaluation or ContinuousDistributionallyRobustOffPolicyEvaluation"
        #     )

        step_per_episode = self.ope.logged_dataset["step_per_episode"]
        check_scalar(
            step_per_episode,
            name="ope.logged_dataset['step_per_episode']",
            target_type=int,
            min_val=1,
        )

        behavior_policy_reward = self.ope.logged_dataset["reward"]
        check_array(
            behavior_policy_reward, name="ope.logged_dataset['reward']", expected_dim=1
        )

        behavior_policy_reward = behavior_policy_reward.reshape((-1, step_per_episode))
        self.behavior_policy_value = behavior_policy_reward.sum(axis=1).mean()

    def obtain_oracle_selection_result(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        return_variance: bool = False,
        return_lower_quartile: bool = False,
        return_conditional_value_at_risk: bool = False,
        return_distributionally_robust_worst_case: bool = False,
        return_by_dataframe: bool = False,
        quartile_alpha: float = 0.05,
        cvar_alpha: float = 0.05,
        distributionally_robust_delta: float = 0.05,
    ):
        """Obtain oracle selection result using ground-truth policy value.

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

        return_variance: bool, default=False
            Whether to return variance.

        return_lower_quartile: bool. default=False
            Whether to return lower interquartile.

        return_conditional_value_at_risk: bool, default=False
            Whether to return conditional value at risk.

        return_distributionally_robust_worst_case: bool, default=False
            Whether to return distributionally robust worst case policy value.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        quartile_alpha: float, default=0.05
            Proportion of the sided region of the interquartile range.

        cvar_alpha: float, default=0.05
            Proportion of the sided region of the conditional value at risk.

        distributionally_robust_delta: float, default=0.05
            Allowance of the distributional shift of the distributionally robust worst case policy value.

        Return
        -------
        ground_truth_dict/ground_truth_df: Union[Dict[str, Any], pd.DataFrame]
            Dictionary/dataframe containing the following ground-truth (on-policy) metrics.

            ranking: List[str]
                Name of the candidate policies sorted by the ground-truth policy value.

            policy_value: List[float]
                Ground-truth policy value of the candidate policies (sorted by ranking).

            relative_policy_value: List[float]
                Ground-truth relative policy value of the candidate policies compared to the behavior policy (sorted by ranking).

            variance: Optional[List[float]]
                Ground-truth variance of the trajectory wise reward of the candidate policies (sorted by ranking).

            ranking_by_lower_quartile: Optional[List[str]]
                Name of the candidate policies sorted by the ground-truth lower quartile of the trajectory wise reward.

            lower_quartile: Optional[List[float]]
                Ground-truth lower quartile of the candidate policies (sorted by ranking_by_lower_quartile).

            ranking_by_conditional_value_at_risk: Optional[List[str]]
                Name of the candidate policies sorted by the ground-truth conditional value at risk.

            conditional_value_at_risk: Optional[List[float]]
                Ground-truth conditional value at risk of the candidate policies (sorted by ranking_by_conditional_value_at_risk).

            ranking_by_distributionally_robust_worst_case: Optional[List[str]]
                Name of the candidate policies sorted by the ground-truth distributionally robust worst case policy value.

            distributionally_robust_worst_case: Optional[List[float]]
                Ground-truth distributionally robust worst case policy value of the candidate policies (sorted by ranking_by_distributionally_robust_worst_case).

            parameters: Dict[str, float]
                Dictionary containing quartile_alpha, cvar_alpha, and distributionally_robust_alpha.

        """
        candidate_policy_names = list(input_dict.keys())
        for eval_policy in candidate_policy_names:
            if input_dict[eval_policy]["on_policy_policy_value"] is None:
                raise ValueError(
                    f"one of the candidate policies, {eval_policy}, does not contain on-policy policy value in input_dict"
                )

        n_policies = len(candidate_policy_names)
        n_samples = len(input_dict[eval_policy]["on_policy_policy_value"])

        policy_value = np.zeros(n_policies)
        for i, eval_policy in enumerate(candidate_policy_names):
            policy_value[i] = input_dict[eval_policy]["on_policy_policy_value"].mean()

        ranking_index = np.argsort(policy_value)[::-1]
        ranking = [candidate_policy_names[ranking_index[i]] for i in range(n_policies)]

        policy_value = np.sort(policy_value)[::-1]
        relative_policy_value = policy_value / self.behavior_policy_value

        if return_variance:
            variance = np.zeros(n_policies)
            for i, eval_policy in enumerate(candidate_policy_names):
                variance[i] = input_dict[eval_policy]["on_policy_policy_value"].var(
                    ddof=1
                )
            variance = variance[ranking_index]

        if return_lower_quartile:
            lower_quartile = np.zeros(n_policies)
            for i, eval_policy in enumerate(candidate_policy_names):
                lower_quartile[i] = np.quantile(
                    input_dict[eval_policy]["on_policy_policy_value"], q=quartile_alpha
                )

            quartile_ranking_index = np.argsort(policy_value)[::-1]
            ranking_by_lower_quartile = [
                candidate_policy_names[quartile_ranking_index[i]]
                for i in range(n_policies)
            ]
            lower_quartile = np.sort(lower_quartile)[::-1]

        if return_conditional_value_at_risk:
            cvar = np.zeros(n_policies)
            for i, eval_policy in enumerate(candidate_policy_names):
                cvar[i] = np.sort(input_dict[eval_policy]["on_policy_policy_value"])[
                    : int(n_samples * cvar_alpha)
                ].mean()

            cvar_ranking_index = np.argsort(cvar)[::-1]
            ranking_by_cvar = [
                candidate_policy_names[cvar_ranking_index[i]] for i in range(n_policies)
            ]
            cvar = np.sort(cvar)[::-1]

        if return_distributionally_robust_worst_case:
            if self.distributionally_robust_ope is None:
                raise RuntimeError(
                    "When using distributionally robust methods, initialize class with distributionally_robust_ope attribute"
                )
            worst_case = np.zeros(n_policies)
            for i, eval_policy in enumerate(candidate_policy_names):
                worst_case[
                    i
                ] = self.distributionally_robust_ope.estimate_worst_case_on_policy_policy_value(
                    input_dict[eval_policy]["on_policy_policy_value"]
                )

            worst_case_ranking_index = np.argsort(worst_case)[::-1]
            ranking_by_worst_case = [
                candidate_policy_names[worst_case_ranking_index[i]]
                for i in range(n_policies)
            ]
            worst_case = np.sort(worst_case)[::-1]

        ground_truth_dict = {
            "ranking": ranking,
            "policy_value": policy_value,
            "relative_policy_value": relative_policy_value,
            "variance": variance if return_variance else None,
            "ranking_by_lower_quartile": ranking_by_lower_quartile
            if return_lower_quartile
            else None,
            "lower_quartile": lower_quartile if return_lower_quartile else None,
            "ranking_by_conditional_value_at_risk": ranking_by_cvar
            if return_conditional_value_at_risk
            else None,
            "conditional_value_at_risk": cvar
            if return_conditional_value_at_risk
            else None,
            "ranking_by_distributionally_robust_worst_case": ranking_by_worst_case
            if return_distributionally_robust_worst_case
            else None,
            "distributionally_robust_worst_case": worst_case
            if return_distributionally_robust_worst_case
            else None,
            "parameters": {
                "quartile_alpha": quartile_alpha if return_lower_quartile else None,
                "cvar_alpha": cvar_alpha if return_conditional_value_at_risk else None,
                "distributionally_robust_delta": distributionally_robust_delta
                if return_distributionally_robust_worst_case
                else None,
            },
        }

        if return_by_dataframe:
            ground_truth_df = pd.DataFrame()
            for key in ground_truth_dict.keys():
                if ground_truth_dict[key] is None or key == "parameters":
                    continue

                ground_truth_df[key] = ground_truth_dict[key]

        return ground_truth_df if return_by_dataframe else ground_truth_dict

    def select_by_policy_value(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_criteria: float = 0.0,
    ):
        """Rank candidate policies by the estimated policy value.

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

        return_metrics: bool, default=False
            Whether to return evaluation metrics including:
            mean-squared-error, rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_criteria: float, default=None [0, 1]
            The relative policy value required for being "safe" candidate policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): Union[Dict[str, Dict[str, Any]], Tuple(Dict[str, pd.DataFrame], pd.DataFrame)]
            Dictionary containing the result of OPS conducted by OPE estimators.
            key: [estimator_name][
                estimated_ranking,
                estimated_policy_value,
                estimated_relative_policy_value,
                mean_squared_error,
                rank_correlation,
                regret,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: List[str]
                Name of the candidate policies sorted by the estimated policy value.

            estimated_policy_value: List[float]
                Estimated policy value of the candidate policies (sorted by estimated_ranking).

            estimated_relative_policy_value: List[float]
                Estimated relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).

            mean_squared_error: Optional[float]
                Mean-squared-error of the estimated policy value.

            rank_correlation: Optional[Tuple[float, float]]
                Rank correlation coefficient and its pvalue between the true ranking and the estimated ranking.

            regret: Optional[float]
                Regret@k and k.

            type_i_error_rate: Optional[float]
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.

            type_ii_error_rate: Optional[float]
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.

            safety_threshold: float
                The required policy value for a safe policy.

        """
        if self.ope is None:
            raise RuntimeError(
                "ope is not given. Please initialize the class with ope attribute"
            )

        estimated_policy_value_dict = self.ope.estimate_policy_value(
            input_dict=input_dict,
            gamma=gamma,
        )

        if return_metrics:
            ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
                gamma=gamma,
            )
            true_ranking = ground_truth_policy_value_dict["ranking"]
            true_policy_value = ground_truth_policy_value_dict["policy_value"]

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = {}
        for i, estimator in enumerate(self.ope.ope_estimators_):

            estimated_policy_value_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_policy_value_[j] = estimated_policy_value_dict[eval_policy][
                    estimator
                ]

            estimated_ranking_index_ = np.argsort(estimated_policy_value_)[::-1]
            estimated_ranking = [
                candidate_policy_names[estimated_ranking_index_[i]]
                for i in range(n_policies)
            ]
            estimated_policy_value = np.sort(estimated_policy_value_)[::-1]
            estimated_relative_policy_value = (
                estimated_policy_value / self.behavior_policy_value
            )

            if return_metrics:
                mse = mean_squared_error(true_policy_value, estimated_policy_value_)
                rankcorr = spearmanr(np.arange(n_policies), estimated_ranking_index_)
                regret = (
                    true_policy_value[:top_k_in_eval_metrics].sum()
                    - true_policy_value[estimated_ranking_index_][
                        :top_k_in_eval_metrics
                    ].sum()
                )

                true_safety = (
                    true_policy_value >= safety_criteria * self.behavior_policy_value
                )
                estimated_safety = (
                    estimated_policy_value_
                    >= safety_criteria * self.behavior_policy_value
                )

                type_i_error_rate = (true_safety > estimated_safety).sum() / (
                    true_safety.sum() + 1e-10
                )
                type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                    1 - true_safety.sum() + 1e-10
                )

            ops_dict[estimator] = {
                "estimated_ranking": estimated_ranking,
                "estimated_policy_value": estimated_policy_value,
                "estimated_relative_policy_value": estimated_relative_policy_value,
                "mean_squared_error": mse if return_metrics else None,
                "rank_correlation": rankcorr if return_metrics else None,
                "regret": (regret, top_k_in_eval_metrics) if return_metrics else None,
                "type_i_error_rate": type_i_error_rate if return_metrics else None,
                "type_ii_error_rate": type_ii_error_rate if return_metrics else None,
                "safety_threshold": safety_criteria * self.behavior_policy_value,
            }

        if return_by_dataframe:
            metric_df = pd.DataFrame()
            mse, rankcorr, pvalue, regret, type_i, type_ii, = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            ranking_df_dict = defaultdict(pd.DataFrame)
            for i, estimator in enumerate(self.ope.ope_estimators_):
                ranking_df_ = pd.DataFrame()
                ranking_df_["estimated_ranking"] = ops_dict[estimator][
                    "estimated_ranking"
                ]
                ranking_df_["estimated_policy_value"] = ops_dict[estimator][
                    "estimated_policy_value"
                ]
                ranking_df_["estimated_relative_policy_value"] = ops_dict[estimator][
                    "estimated_relative_policy_value"
                ]
                ranking_df_dict[estimator] = ranking_df_

                mse.append(ops_dict[estimator]["mean_squared_error"])
                rankcorr.append(ops_dict[estimator]["rank_correlation"][0])
                pvalue.append(ops_dict[estimator]["rank_correlation"][1])
                regret.append(ops_dict[estimator]["regret"][0])
                type_i.append(ops_dict[estimator]["type_i_error_rate"])
                type_ii.append(ops_dict[estimator]["type_ii_error_rate"])

            metric_df["estimator"] = self.ope.ope_estimators_
            metric_df["mean_squared_error"] = mse
            metric_df["rank_correlation"] = rankcorr
            metric_df["pvalue"] = pvalue
            metric_df[f"regret@{top_k_in_eval_metrics}"] = regret
            metric_df["type_i_error_rate"] = type_i
            metric_df["type_ii_error_rate"] = type_ii

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

        return (ranking_df_dict, metric_df) if return_by_dataframe else ops_dict

    def select_by_policy_value_via_cumulative_distributional_ope(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_criteria: float = 0.0,
    ):
        """Rank candidate policies by the estimated policy value via cumulative distributional methods.

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

        return_metrics: bool, default=False
            Whether to return evaluation metrics including:
            mean-squared-error, rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_criteria: float, default=None [0, 1]
            The relative policy value required for being "safe" candidate policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.

        Return
        -------
        ops_dict: Dict[str, Dict[str, Any]]
            Dictionary containing the result of OPS conducted by OPE estimators.
            key: [estimator_name][
                estimated_ranking,
                estimated_policy_value,
                estimated_relative_policy_value,
                mean_squared_error,
                rank_correlation,
                regret,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: List[str]
                Name of the candidate policies sorted by the estimated policy value.

            estimated_policy_value: List[float]
                Estimated policy value of the candidate policies (sorted by estimated_ranking).

            estimated_relative_policy_value: List[float]
                Estimated relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).

            mean_squared_error: Optional[float]
                Mean-squared-error of the estimated policy value.

            rank_correlation: Optional[Tuple[float, float]]
                Rank correlation coefficient and its pvalue between the true ranking and the estimated ranking.

            regret: Optional[float]
                Regret@k and k.

            type_i_error_rate: Optional[float]
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.

            type_ii_error_rate: Optional[float]
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.

            safety_threshold: float
                The required policy value for a safe policy.

        """
        if self.cumulative_distributional_ope is None:
            raise RuntimeError(
                "cumulative_distributional_ope is not given. Please initialize the class with cumulative_distributional_ope attribute"
            )

        estimated_policy_value_dict = self.cumulative_distributional_ope.estimate_mean(
            input_dict=input_dict,
            gamma=gamma,
        )

        if return_metrics:
            ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
                gamma=gamma,
            )
            true_ranking = ground_truth_policy_value_dict["ranking"]
            true_policy_value = ground_truth_policy_value_dict["policy_value"]

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = {}
        for i, estimator in enumerate(
            self.cumulative_distributional_ope.ope_estimators_
        ):

            estimated_policy_value_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_policy_value_[j] = estimated_policy_value_dict[eval_policy][
                    estimator
                ]

            estimated_ranking_index_ = np.argsort(estimated_policy_value_)[::-1]
            estimated_ranking = [
                candidate_policy_names[estimated_ranking_index_[i]]
                for i in range(n_policies)
            ]
            estimated_policy_value = np.sort(estimated_policy_value_)[::-1]
            estimated_relative_policy_value = (
                estimated_policy_value / self.behavior_policy_value
            )

            if return_metrics:
                mse = mean_squared_error(
                    true_policy_value, np.nan_to_num(estimated_policy_value_)
                )
                rankcorr = spearmanr(np.arange(n_policies), estimated_ranking_index_)
                regret = (
                    true_policy_value[:top_k_in_eval_metrics].sum()
                    - true_policy_value[estimated_ranking_index_][
                        :top_k_in_eval_metrics
                    ].sum()
                )

                true_safety = (
                    true_policy_value >= safety_criteria * self.behavior_policy_value
                )
                estimated_safety = (
                    estimated_policy_value_
                    >= safety_criteria * self.behavior_policy_value
                )

                type_i_error_rate = (true_safety > estimated_safety).sum() / (
                    true_safety.sum() + 1e-10
                )
                type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                    1 - true_safety.sum() + 1e-10
                )

            ops_dict[estimator] = {
                "estimated_ranking": estimated_ranking,
                "estimated_policy_value": estimated_policy_value,
                "estimated_relative_policy_value": estimated_relative_policy_value,
                "mean_squared_error": mse if return_metrics else None,
                "rank_correlation": rankcorr if return_metrics else None,
                "regret": (regret, top_k_in_eval_metrics) if return_metrics else None,
                "type_i_error_rate": type_i_error_rate if return_metrics else None,
                "type_ii_error_rate": type_ii_error_rate if return_metrics else None,
                "safety_threshold": safety_criteria * self.behavior_policy_value,
            }

        if return_by_dataframe:
            metric_df = pd.DataFrame()
            mse, rankcorr, pvalue, regret, type_i, type_ii, = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            ranking_df_dict = defaultdict(pd.DataFrame)
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                ranking_df_ = pd.DataFrame()
                ranking_df_["estimated_ranking"] = ops_dict[estimator][
                    "estimated_ranking"
                ]
                ranking_df_["estimated_policy_value"] = ops_dict[estimator][
                    "estimated_policy_value"
                ]
                ranking_df_["estimated_relative_policy_value"] = ops_dict[estimator][
                    "estimated_relative_policy_value"
                ]
                ranking_df_dict[estimator] = ranking_df_

                mse.append(ops_dict[estimator]["mean_squared_error"])
                rankcorr.append(ops_dict[estimator]["rank_correlation"][0])
                pvalue.append(ops_dict[estimator]["rank_correlation"][1])
                regret.append(ops_dict[estimator]["regret"][0])
                type_i.append(ops_dict[estimator]["type_i_error_rate"])
                type_ii.append(ops_dict[estimator]["type_ii_error_rate"])

            metric_df["estimator"] = self.cumulative_distributional_ope.ope_estimators_
            metric_df["mean_squared_error"] = mse
            metric_df["rank_correlation"] = rankcorr
            metric_df["pvalue"] = pvalue
            metric_df[f"regret@{top_k_in_eval_metrics}"] = regret
            metric_df["type_i_error_rate"] = type_i
            metric_df["type_ii_error_rate"] = type_ii

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

        return (ranking_df_dict, metric_df) if return_by_dataframe else ops_dict

    def select_by_policy_value_lower_bound(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_criteria: float = 0.0,
        cis: List[str] = ["bootstrap"],
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ):
        """Rank candidate policies by the estimated policy value lower bound.

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

        return_metrics: bool, default=False
            Whether to return evaluation metrics including:
            rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_criteria: float, default=None [0, 1]
            The relative policy value required for being "safe" candidate policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.

        cis: List[str], default=["bootstrap"]
            Estimation methods for confidence intervals.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        n_bootstrap_samples: int, default=100 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        ops_dict: Dict[str, Dict[str, Dict[str, Any]]]
            Dictionary containing the result of OPS conducted by OPE estimators.
            key: [ci][estimator_name][
                estimated_ranking,
                estimated_policy_value_lower_bound,
                estimated_relative_policy_value_lower_bound,
                rank_correlation,
                regret,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: List[str]
                Name of the candidate policies sorted by the estimated policy value lower bound.

            estimated_policy_value_lower_bound: List[float]
                Estimated policy value lower bound of the candidate policies (sorted by estimated_ranking).

            estimated_relative_policy_value_lower_bound: List[float]
                Estimated relative policy value lower bound of the candidate policies compared to the behavior policy (sorted by estimated_ranking).

            rank_correlation: Optional[Tuple[float, float]]
                Rank correlation coefficient and its pvalue between the true ranking and the estimated ranking.

            regret: Optional[Tuple[float, int]]
                Regret@k and k.

            type_i_error_rate: Optional[float]
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.

            type_ii_error_rate: Optional[float]
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.

            safety_threshold: float
                The required policy value for a safe policy.

        """
        if self.ope is None:
            raise RuntimeError(
                "ope is not given. Please initialize the class with ope attribute"
            )

        if return_metrics:
            ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
                gamma=gamma,
            )
            true_ranking = ground_truth_policy_value_dict["ranking"]
            true_policy_value = ground_truth_policy_value_dict["policy_value"]

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = defaultdict(dict)
        for ci in cis:
            estimated_policy_value_interval_dict = self.ope.estimate_intervals(
                input_dict=input_dict,
                gamma=gamma,
                alpha=alpha,
                ci=ci,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

            for i, estimator in enumerate(self.ope.ope_estimators_):

                estimated_policy_value_lower_bound_ = np.zeros(n_policies)
                for j, eval_policy in enumerate(candidate_policy_names):
                    estimated_policy_value_lower_bound_[
                        j
                    ] = estimated_policy_value_interval_dict[eval_policy][estimator][
                        f"{100 * (1. - alpha)}% CI (lower)"
                    ]

                estimated_ranking_index_ = np.argsort(
                    estimated_policy_value_lower_bound_
                )[::-1]
                estimated_ranking = [
                    candidate_policy_names[estimated_ranking_index_[i]]
                    for i in range(n_policies)
                ]
                estimated_policy_value_lower_bound = np.sort(
                    estimated_policy_value_lower_bound_
                )[::-1]
                estimated_relative_policy_value_lower_bound = (
                    estimated_policy_value_lower_bound / self.behavior_policy_value
                )

                if return_metrics:
                    rankcorr = spearmanr(
                        np.arange(n_policies), estimated_ranking_index_
                    )
                    regret = (
                        true_policy_value[:top_k_in_eval_metrics].sum()
                        - true_policy_value[estimated_ranking_index_][
                            :top_k_in_eval_metrics
                        ].sum()
                    )

                    true_safety = (
                        true_policy_value
                        >= safety_criteria * self.behavior_policy_value
                    )
                    estimated_safety = (
                        estimated_policy_value_lower_bound_
                        >= safety_criteria * self.behavior_policy_value
                    )

                    type_i_error_rate = (
                        true_safety > estimated_safety
                    ).sum() / true_safety.sum()
                    type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                        1 - true_safety
                    ).sum()

                ops_dict[ci][estimator] = {
                    "estimated_ranking": estimated_ranking,
                    "estimated_policy_value_lower_bound": estimated_policy_value_lower_bound,
                    "estimated_relative_policy_value_lower_bound": estimated_relative_policy_value_lower_bound,
                    "rank_correlation": rankcorr if return_metrics else None,
                    "regret": (regret, top_k_in_eval_metrics)
                    if return_metrics
                    else None,
                    "type_i_error_rate": type_i_error_rate if return_metrics else None,
                    "type_ii_error_rate": type_ii_error_rate
                    if return_metrics
                    else None,
                    "safety_threshold": safety_criteria * self.behavior_policy_value,
                }

        ops_dict = defaultdict_to_dict(ops_dict)

        if return_by_dataframe:
            metric_df = pd.DataFrame()
            ci_, estimator_, rankcorr, pvalue, regret, type_i, type_ii, = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            ranking_df_dict = defaultdict(lambda: defaultdict(pd.DataFrame))
            for ci in cis:
                for i, estimator in enumerate(self.ope.ope_estimators_):
                    ranking_df_ = pd.DataFrame()
                    ranking_df_["estimated_ranking"] = ops_dict[ci][estimator][
                        "estimated_ranking"
                    ]
                    ranking_df_["estimated_policy_value_lower_bound"] = ops_dict[ci][
                        estimator
                    ]["estimated_policy_value_lower_bound"]
                    ranking_df_[
                        "estimated_relative_policy_value_lower_bound"
                    ] = ops_dict[ci][estimator][
                        "estimated_relative_policy_value_lower_bound"
                    ]
                    ranking_df_dict[ci][estimator] = ranking_df_

                    ci_.append(ci)
                    estimator_.append(estimator)
                    rankcorr.append(ops_dict[ci][estimator]["rank_correlation"][0])
                    pvalue.append(ops_dict[ci][estimator]["rank_correlation"][1])
                    regret.append(ops_dict[ci][estimator]["regret"][0])
                    type_i.append(ops_dict[ci][estimator]["type_i_error_rate"])
                    type_ii.append(ops_dict[ci][estimator]["type_ii_error_rate"])

            metric_df["ci"] = ci_
            metric_df["estimator"] = estimator_
            metric_df["rank_correlation"] = rankcorr
            metric_df["pvalue"] = pvalue
            metric_df[f"regret@{top_k_in_eval_metrics}"] = regret
            metric_df["type_i_error_rate"] = type_i
            metric_df["type_ii_error_rate"] = type_ii

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

        return (ranking_df_dict, metric_df) if return_by_dataframe else ops_dict

    def select_by_lower_quartile(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank candidate policies by the estimated lower quartile of the trajectory wise reward.

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

        return_metrics: bool, default=False
            Whether to return evaluation metrics including:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        safety_threshold: float, default=None (>= 0)
            The lower quartile required for being "safe" candidate policy.

        Return
        -------
        ops_dict: Dict[str, Dict[str, Any]]
            Dictionary containing the result of OPS conducted by OPE estimators.
            key: [estimator_name][
                estimated_ranking,
                estimated_lower_quartile,
                mean_squared_error,
                rank_correlation,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: List[str]
                Name of the candidate policies sorted by the estimated lower quartile of the trajectory wise reward.

            estimated_lower_quartile: List[float]
                Estimated lower quartile of the trajectory wise reward of the candidate policies (sorted by estimated_ranking).

            mean_squared_error: Optional[float]
                Mean-squared-error of the estimated lower quartile of the trajectory wise reward.

            rank_correlation: Optional[Tuple[float, float]]
                Rank correlation coefficient and its pvalue between the true ranking and the estimated ranking.

            type_i_error_rate: Optional[float]
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.

            type_ii_error_rate: Optional[float]
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.

            safety_threshold: float
                The required lower quartile for a safe policy.

        """
        if self.cumulative_distributional_ope is None:
            raise RuntimeError(
                "cumulative_distributional_ope is not given. Please initialize the class with cumulative_distributional_ope attribute"
            )

        estimated_interquartile_range_dict = (
            self.cumulative_distributional_ope.estimate_interquartile_range(
                input_dict=input_dict,
                gamma=gamma,
                alpha=alpha,
            )
        )

        if return_metrics:
            ground_truth_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
                gamma=gamma,
                return_lower_quartile=True,
                quartile_alpha=alpha,
            )
            true_ranking = ground_truth_dict["ranking_by_lower_quartile"]
            true_lower_quartile = ground_truth_dict["lower_quartile"]

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = {}
        for i, estimator in enumerate(
            self.cumulative_distributional_ope.ope_estimators_
        ):

            estimated_lower_quartile_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_lower_quartile_[j] = estimated_interquartile_range_dict[
                    eval_policy
                ][estimator][f"{100 * (1. - alpha)}% quartile (lower)"]

            estimated_ranking_index_ = np.argsort(estimated_lower_quartile_)[::-1]
            estimated_ranking = [
                candidate_policy_names[estimated_ranking_index_[i]]
                for i in range(n_policies)
            ]
            estimated_lower_quartile = np.sort(estimated_lower_quartile_)[::-1]

            if return_metrics:
                mse = mean_squared_error(true_lower_quartile, estimated_lower_quartile_)
                rankcorr = spearmanr(np.arange(n_policies), estimated_ranking_index_)

                true_safety = true_lower_quartile >= safety_threshold
                estimated_safety = estimated_lower_quartile_ >= safety_threshold

                type_i_error_rate = (true_safety > estimated_safety).sum() / (
                    true_safety.sum() + 1e-10
                )
                type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                    1 - true_safety.sum() + 1e-10
                )

            ops_dict[estimator] = {
                "estimated_ranking": estimated_ranking,
                "estimated_lower_quartile": estimated_lower_quartile,
                "mean_squared_error": mse if return_metrics else None,
                "rank_correlation": rankcorr if return_metrics else None,
                "type_i_error_rate": type_i_error_rate if return_metrics else None,
                "type_ii_error_rate": type_ii_error_rate if return_metrics else None,
                "safety_threshold": safety_threshold,
            }

        if return_by_dataframe:
            metric_df = pd.DataFrame()
            mse, rankcorr, pvalue, type_i, type_ii, = (
                [],
                [],
                [],
                [],
                [],
            )

            ranking_df_dict = defaultdict(pd.DataFrame)
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                ranking_df_ = pd.DataFrame()
                ranking_df_["estimated_ranking"] = ops_dict[estimator][
                    "estimated_ranking"
                ]
                ranking_df_["estimated_lower_quartile"] = ops_dict[estimator][
                    "estimated_lower_quartile"
                ]
                ranking_df_dict[estimator] = ranking_df_

                mse.append(ops_dict[estimator]["mean_squared_error"])
                rankcorr.append(ops_dict[estimator]["rank_correlation"][0])
                pvalue.append(ops_dict[estimator]["rank_correlation"][1])
                type_i.append(ops_dict[estimator]["type_i_error_rate"])
                type_ii.append(ops_dict[estimator]["type_ii_error_rate"])

            metric_df["estimator"] = self.cumulative_distributional_ope.ope_estimators_
            metric_df["mean_squared_error"] = mse
            metric_df["rank_correlation"] = rankcorr
            metric_df["pvalue"] = pvalue
            metric_df["type_i_error_rate"] = type_i
            metric_df["type_ii_error_rate"] = type_ii

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

        return (ranking_df_dict, metric_df) if return_by_dataframe else ops_dict

    def select_by_conditional_value_at_risk(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        alpha: float = 0.05,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank candidate policies by the estimated conditional value at risk.

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

        return_metrics: bool, default=False
            Whether to return evaluation metrics including:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        safety_threshold: float, default=None (>= 0)
            The lower quartile required for being "safe" candidate policy.

        Return
        -------
        ops_dict: Dict[str, Dict[str, Any]]
            Dictionary containing the result of OPS conducted by OPE estimators.
            key: [estimator_name][
                estimated_ranking,
                estimated_lower_quartile,
                mean_squared_error,
                rank_correlation,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: List[str]
                Name of the candidate policies sorted by the estimated conditional value at risk.

            estimated_conditional_value_at_risk: List[float]
                Estimated conditional value at risk of the candidate policies (sorted by estimated_ranking).

            mean_squared_error: Optional[float]
                Mean-squared-error of the estimated conditional value at risk.

            rank_correlation: Optional[Tuple[float, float]]
                Rank correlation coefficient and its pvalue between the true ranking and the estimated ranking.

            type_i_error_rate: Optional[float]
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.

            type_ii_error_rate: Optional[float]
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.

            safety_threshold: float
                The required lower quartile for a safe policy.

        """
        if self.cumulative_distributional_ope is None:
            raise RuntimeError(
                "cumulative_distributional_ope is not given. Please initialize the class with cumulative_distributional_ope attribute"
            )

        estimated_cvar_dict = (
            self.cumulative_distributional_ope.estimate_conditional_value_at_risk(
                input_dict=input_dict,
                gamma=gamma,
                alphas=alpha,
            )
        )

        if return_metrics:
            ground_truth_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
                gamma=gamma,
                return_conditional_value_at_risk=True,
                cvar_alpha=alpha,
            )
            true_ranking = ground_truth_dict["ranking_by_conditional_value_at_risk"]
            true_cvar = ground_truth_dict["conditional_value_at_risk"]

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = {}
        for i, estimator in enumerate(
            self.cumulative_distributional_ope.ope_estimators_
        ):

            estimated_cvar_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_cvar_[j] = estimated_cvar_dict[eval_policy][estimator]

            estimated_ranking_index_ = np.argsort(estimated_cvar_)[::-1]
            estimated_ranking = [
                candidate_policy_names[estimated_ranking_index_[i]]
                for i in range(n_policies)
            ]
            estimated_cvar = np.sort(estimated_cvar_)[::-1]

            if return_metrics:
                mse = mean_squared_error(true_cvar, np.nan_to_num(estimated_cvar_))
                rankcorr = spearmanr(np.arange(n_policies), estimated_ranking_index_)

                true_safety = true_cvar >= safety_threshold
                estimated_safety = estimated_cvar_ >= safety_threshold

                type_i_error_rate = (true_safety > estimated_safety).sum() / (
                    true_safety.sum() + 1e-10
                )
                type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                    1 - true_safety.sum() + 1e-10
                )

            ops_dict[estimator] = {
                "estimated_ranking": estimated_ranking,
                "estimated_conditional_value_at_risk": estimated_cvar,
                "mean_squared_error": mse if return_metrics else None,
                "rank_correlation": rankcorr if return_metrics else None,
                "type_i_error_rate": type_i_error_rate if return_metrics else None,
                "type_ii_error_rate": type_ii_error_rate if return_metrics else None,
                "safety_threshold": safety_threshold,
            }

        if return_by_dataframe:
            metric_df = pd.DataFrame()
            mse, rankcorr, pvalue, type_i, type_ii, = (
                [],
                [],
                [],
                [],
                [],
            )

            ranking_df_dict = defaultdict(pd.DataFrame)
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                ranking_df_ = pd.DataFrame()
                ranking_df_["estimated_ranking"] = ops_dict[estimator][
                    "estimated_ranking"
                ]
                ranking_df_["estimated_conditional_value_at_risk"] = ops_dict[
                    estimator
                ]["estimated_conditional_value_at_risk"]
                ranking_df_dict[estimator] = ranking_df_

                mse.append(ops_dict[estimator]["mean_squared_error"])
                rankcorr.append(ops_dict[estimator]["rank_correlation"][0])
                pvalue.append(ops_dict[estimator]["rank_correlation"][1])
                type_i.append(ops_dict[estimator]["type_i_error_rate"])
                type_ii.append(ops_dict[estimator]["type_ii_error_rate"])

            metric_df["estimator"] = self.cumulative_distributional_ope.ope_estimators_
            metric_df["mean_squared_error"] = mse
            metric_df["rank_correlation"] = rankcorr
            metric_df["pvalue"] = pvalue
            metric_df["type_i_error_rate"] = type_i
            metric_df["type_ii_error_rate"] = type_ii

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

        return (ranking_df_dict, metric_df) if return_by_dataframe else ops_dict

    def select_by_distributionally_robust_worst_case_policy_value(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        delta: float = 0.05,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank candidate policies by the estimated distributionally robust worst case policy value.

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

        delta: float, default=0.05
            Allowance of the distributional shift.

        return_metrics: bool, default=False
            Whether to return evaluation metrics including:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        safety_threshold: float, default=None (>= 0)
            The lower quartile required for being "safe" candidate policy.

        Return
        -------
        ops_dict: Dict[str, Dict[str, Any]]
            Dictionary containing the result of OPS conducted by OPE estimators.
            key: [estimator_name][
                estimated_ranking,
                estimated_lower_quartile,
                mean_squared_error,
                rank_correlation,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: List[str]
                Name of the candidate policies sorted by the estimated conditional value at risk.

            estimated_distributionally_robust_worst_case: List[float]
                Estimated distributionally robust worst case policy value of the candidate policies (sorted by estimated_ranking).

            mean_squared_error: Optional[float]
                Mean-squared-error of the estimated distributionally robust worst case policy value.

            rank_correlation: Optional[Tuple[float, float]]
                Rank correlation coefficient and its pvalue between the true ranking and the estimated ranking.

            type_i_error_rate: Optional[float]
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.

            type_ii_error_rate: Optional[float]
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.

            safety_threshold: float
                The required lower quartile for a safe policy.

        """
        if self.distributionally_robust_ope is None:
            raise RuntimeError(
                "distributionally_robust_ope is not given. Please initialize the class with distributionally_robust_ope attribute"
            )

        estimated_worst_case_policy_value_dict = (
            self.distributionally_robust_ope.estimate_worst_case_policy_value(
                input_dict=input_dict,
                gamma=gamma,
                delta=delta,
            )
        )

        if return_metrics:
            ground_truth_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
                gamma=gamma,
                return_distributionally_robust_worst_case=True,
                distributionally_robust_delta=delta,
            )
            true_ranking = ground_truth_dict[
                "ranking_by_distributionally_robust_worst_case"
            ]
            true_worst_case_policy_value = ground_truth_dict[
                "distributionally_robust_worst_case"
            ]

        candidate_policy_names = true_ranking if return_metrics else input_dict.keys()
        n_policies = len(candidate_policy_names)

        ops_dict = {}
        for estimator in enumerate(self.distributionally_robust_ope.ope_estimators_):

            estimated_worst_case_policy_value_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_worst_case_policy_value_[
                    j
                ] = estimated_worst_case_policy_value_dict[eval_policy][estimator]

            estimated_ranking_index_ = np.argsort(estimated_worst_case_policy_value_)[
                ::-1
            ]
            estimated_ranking = [
                candidate_policy_names[estimated_ranking_index_[i]]
                for i in range(n_policies)
            ]
            estimated_worst_case_policy_value = np.sort(
                estimated_worst_case_policy_value_
            )[::-1]

            if return_metrics:
                mse = mean_squared_error(
                    true_worst_case_policy_value, estimated_worst_case_policy_value_
                )
                rankcorr = spearmanr(np.arange(n_policies), estimated_ranking_index_)

                true_safety = true_worst_case_policy_value >= safety_threshold
                estimated_safety = (
                    estimated_worst_case_policy_value_ >= safety_threshold
                )

                type_i_error_rate = (true_safety > estimated_safety).sum() / (
                    true_safety.sum() + 1e-10
                )
                type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                    1 - true_safety.sum() + 1e-10
                )

            ops_dict[estimator] = {
                "estimated_ranking": estimated_ranking,
                "estimated_distributionally_robust_worst_case": estimated_worst_case_policy_value,
                "mean_squared_error": mse if return_metrics else None,
                "rank_correlation": rankcorr if return_metrics else None,
                "type_i_error_rate": type_i_error_rate if return_metrics else None,
                "type_ii_error_rate": type_ii_error_rate if return_metrics else None,
                "safety_threshold": safety_threshold,
            }

    def visualize_policy_value(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_policy_value.png",
    ):
        """Visualize true policy value and its estimate.

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

        n_cols: int, default=None
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="scatter_policy_value.png"
            Name of the bar figure.

        """
        if n_cols is not None:
            check_scalar(n_cols, name="n_cols", target_type=int, min_val=1)
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

        ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
            input_dict=input_dict,
            gamma=gamma,
        )
        true_ranking = ground_truth_policy_value_dict["ranking"]
        true_policy_value = np.array(ground_truth_policy_value_dict["policy_value"])

        estimated_policy_value_dict = self.select_by_policy_value(
            input_dict=input_dict,
            gamma=gamma,
        )

        plt.style.use("ggplot")
        n_figs = len(self.ope.ope_estimators_)
        n_cols = min(5, n_figs) if n_cols is None else n_cols
        n_rows = (n_figs - 1) // n_cols + 1

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=share_axes,
            sharey=share_axes,
        )

        guide_min, guide_max = 1e5, -1e5
        if n_rows == 1:
            for i, estimator in enumerate(self.ope.ope_estimators_):
                estimated_ranking_ = estimated_policy_value_dict[estimator][
                    "estimated_ranking"
                ]
                estimated_policy_value_ = estimated_policy_value_dict[estimator][
                    "estimated_policy_value"
                ]

                estimated_ranking_dict = {
                    estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
                }
                estimated_policy_value = [
                    estimated_policy_value_[estimated_ranking_dict[true_ranking[i]]]
                    for i in range(len(true_ranking))
                ]
                estimated_policy_value = np.array(estimated_policy_value)

                min_val = np.minimum(
                    np.nanmin(true_policy_value), np.nanmin(estimated_policy_value)
                )
                max_val = np.maximum(
                    np.nanmax(true_policy_value), np.nanmax(estimated_policy_value)
                )
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i].scatter(
                    true_policy_value,
                    estimated_policy_value,
                )
                axes[i].set_title(estimator)
                axes[i].set_xlabel("true policy value")
                axes[i].set_ylabel("estimated policy value")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(self.ope.ope_estimators_):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(self.ope.ope_estimators_):
                estimated_ranking_ = estimated_policy_value_dict[estimator][
                    "estimated_ranking"
                ]
                estimated_policy_value_ = estimated_policy_value_dict[estimator][
                    "estimated_policy_value"
                ]

                estimated_ranking_dict = {
                    estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
                }
                estimated_policy_value = [
                    estimated_policy_value_[estimated_ranking_dict[true_ranking[i]]]
                    for i in range(len(true_ranking))
                ]
                estimated_policy_value = np.array(estimated_policy_value)

                min_val = np.minimum(
                    np.nanmin(true_policy_value), np.nanmin(estimated_policy_value)
                )
                max_val = np.maximum(
                    np.nanmax(true_policy_value), np.nanmax(estimated_policy_value)
                )
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i // n_cols, i % n_cols].scatter(
                    true_policy_value,
                    estimated_policy_value,
                )
                axes[i // n_cols, i % n_cols].set_title(estimator)
                axes[i // n_cols, i % n_cols].set_xlabel("true policy value")
                axes[i // n_cols, i % n_cols].set_ylabel("estimated policy value")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(self.ope.ope_estimators_):
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        fig.tight_layout()
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_policy_value_of_cumulative_distributional_OPE(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_policy_value_of_cumulative_distributional_ope.png",
    ):
        """Visualize true policy value and its estimate obtained by cumulative distributional OPE.

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

        n_cols: int, default=None
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="scatter_policy_value_of_cumulative_distributional_ope.png"
            Name of the bar figure.

        """
        ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
            input_dict=input_dict,
            gamma=gamma,
        )
        true_ranking = ground_truth_policy_value_dict["ranking"]
        true_policy_value = np.array(ground_truth_policy_value_dict["policy_value"])

        estimated_policy_value_dict = (
            self.select_by_policy_value_via_cumulative_distributional_ope(
                input_dict=input_dict,
                gamma=gamma,
            )
        )

        plt.style.use("ggplot")
        n_figs = len(self.cumulative_distributional_ope.ope_estimators_)
        n_cols = min(5, n_figs) if n_cols is None else n_cols
        n_rows = (n_figs - 1) // n_cols + 1

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=share_axes,
            sharey=share_axes,
        )

        guide_min, guide_max = 1e5, -1e5
        if n_rows == 1:
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                estimated_ranking_ = estimated_policy_value_dict[estimator][
                    "estimated_ranking"
                ]
                estimated_policy_value_ = estimated_policy_value_dict[estimator][
                    "estimated_policy_value"
                ]

                estimated_ranking_dict = {
                    estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
                }
                estimated_policy_value = [
                    estimated_policy_value_[estimated_ranking_dict[true_ranking[i]]]
                    for i in range(len(true_ranking))
                ]
                estimated_policy_value = np.array(estimated_policy_value)

                min_val = np.minimum(
                    np.nanmin(true_policy_value), np.nanmin(estimated_policy_value)
                )
                max_val = np.maximum(
                    np.nanmax(true_policy_value), np.nanmax(estimated_policy_value)
                )
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i].scatter(
                    true_policy_value,
                    estimated_policy_value,
                )
                axes[i].set_title(estimator)
                axes[i].set_xlabel("true policy value")
                axes[i].set_ylabel("estimated policy value")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(
                    self.cumulative_distributional_ope.ope_estimators_
                ):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                estimated_ranking_ = estimated_policy_value_dict[estimator][
                    "estimated_ranking"
                ]
                estimated_policy_value_ = estimated_policy_value_dict[estimator][
                    "estimated_policy_value"
                ]

                estimated_ranking_dict = {
                    estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
                }
                estimated_policy_value = [
                    estimated_policy_value_[estimated_ranking_dict[true_ranking[i]]]
                    for i in range(len(true_ranking))
                ]
                estimated_policy_value = np.array(estimated_policy_value)

                min_val = np.minimum(
                    np.nanmin(true_policy_value), np.nanmin(estimated_policy_value)
                )
                max_val = np.maximum(
                    np.nanmax(true_policy_value), np.nanmax(estimated_policy_value)
                )
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i // n_cols, i % n_cols].scatter(
                    true_policy_value,
                    estimated_policy_value,
                )
                axes[i // n_cols, i % n_cols].set_title(estimator)
                axes[i // n_cols, i % n_cols].set_xlabel("true policy value")
                axes[i // n_cols, i % n_cols].set_ylabel("estimated policy value")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(self.ope.ope_estimators_):
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        fig.tight_layout()
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_policy_value_lower_bound(
        self,
        input_dict: OPEInputDict,
        cis: List[str] = ["bootstrap"],
        gamma: float = 1.0,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = 12345,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_policy_value_lower_bound.png",
    ):
        """Visualize true policy value and its estimate lower bound.

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

        cis: List[str], default=["bootstrap"]
            Estimation methods for confidence intervals.

        gamma: float, default=1.0 (0, 1]
            Discount factor.

        alpha: float, default=0.05 (0, 1)
            Significant level.

        n_bootstrap_samples: int, default=100 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        n_cols: int, default=None
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="scatter_policy_value_lower_bound.png"
            Name of the bar figure.

        """
        ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
            input_dict=input_dict,
            gamma=gamma,
        )
        true_ranking = ground_truth_policy_value_dict["ranking"]
        true_policy_value = ground_truth_policy_value_dict["policy_value"]

        estimated_policy_value_dict = self.select_by_policy_value_lower_bound(
            input_dict=input_dict,
            cis=cis,
            gamma=gamma,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

        plt.style.use("ggplot")
        n_figs = len(self.ope.ope_estimators_) * len(cis)
        if len(cis) == 1:
            n_cols = min(5, n_figs) if n_cols is None else n_cols
        else:
            n_cols = len(cis)
        n_rows = (n_figs - 1) // n_cols + 1

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=share_axes,
            sharey=share_axes,
        )

        guide_min, guide_max = 1e5, -1e5
        if len(cis) == 1:
            if n_cols == 1:
                for ci in cis:
                    for i, estimator in enumerate(self.ope.ope_estimators_):
                        estimated_ranking_ = estimated_policy_value_dict[ci][estimator][
                            "estimated_ranking"
                        ]
                        estimated_policy_value_lower_bound_ = (
                            estimated_policy_value_dict[ci][estimator][
                                "estimated_policy_value_lower_bound"
                            ]
                        )

                        estimated_ranking_dict = {
                            estimated_ranking_[i]: i
                            for i in range(len(estimated_ranking_))
                        }
                        estimated_policy_value_lower_bound = [
                            estimated_policy_value_lower_bound_[
                                estimated_ranking_dict[true_ranking[i]]
                            ]
                            for i in range(len(true_ranking))
                        ]

                        min_val = np.minimum(
                            np.nanmin(true_policy_value),
                            np.nanmin(estimated_policy_value_lower_bound),
                        )
                        max_val = np.maximum(
                            np.nanmax(true_policy_value),
                            np.nanmax(estimated_policy_value_lower_bound),
                        )
                        guide_min = min_val if guide_min > min_val else guide_min
                        guide_max = max_val if guide_max < max_val else guide_max

                        if not share_axes:
                            guide = np.linspace(guide_min, guide_max)
                            axes[i].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

                        axes[i].scatter(
                            true_policy_value,
                            estimated_policy_value_lower_bound,
                        )
                        axes[i].set_title(f"({ci}, {estimator})")
                        axes[i].set_xlabel("true policy value")
                        axes[i].set_ylabel("estimated policy value lower bound")

                if share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    for i, estimator in enumerate(self.ope.ope_estimators_):
                        axes[i].plot(
                            guide,
                            guide,
                            color="black",
                            linewidth=1.0,
                        )

            else:
                for ci in cis:
                    for i, estimator in enumerate(self.ope.ope_estimators_):
                        estimated_ranking_ = estimated_policy_value_dict[ci][estimator][
                            "estimated_ranking"
                        ]
                        estimated_policy_value_lower_bound_ = (
                            estimated_policy_value_dict[ci][estimator][
                                "estimated_policy_value_lower_bound"
                            ]
                        )

                        estimated_ranking_dict = {
                            estimated_ranking_[i]: i
                            for i in range(len(estimated_ranking_))
                        }
                        estimated_policy_value_lower_bound = [
                            estimated_policy_value_lower_bound_[
                                estimated_ranking_dict[true_ranking[i]]
                            ]
                            for i in range(len(true_ranking))
                        ]

                        min_val = np.minimum(
                            np.nanmin(true_policy_value),
                            np.nanmin(estimated_policy_value_lower_bound),
                        )
                        max_val = np.maximum(
                            np.nanmax(true_policy_value),
                            np.nanmax(estimated_policy_value_lower_bound),
                        )
                        guide_min = min_val if guide_min > min_val else guide_min
                        guide_max = max_val if guide_max < max_val else guide_max

                        if not share_axes:
                            guide = np.linspace(guide_min, guide_max)
                            axes[i // n_cols, i % n_cols].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

                        axes[i // n_cols, i % n_cols].scatter(
                            true_policy_value,
                            estimated_policy_value_lower_bound,
                        )
                        axes[i // n_cols, i % n_cols].set_title(f"({ci}, {estimator})")
                        axes[i // n_cols, i % n_cols].set_xlabel("true policy value")
                        axes[i // n_cols, i % n_cols].set_ylabel(
                            "estimated policy value"
                        )

                if share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    for i, estimator in enumerate(self.ope.ope_estimators_):
                        axes[i // n_cols, i % n_cols].plot(
                            guide,
                            guide,
                            color="black",
                            linewidth=1.0,
                        )

        else:
            if n_cols == 1:
                for j, ci in enumerate(cis):
                    for estimator in enumerate(self.ope.ope_estimators_):
                        estimated_ranking_ = estimated_policy_value_dict[ci][estimator][
                            "estimated_ranking"
                        ]
                        estimated_policy_value_lower_bound_ = (
                            estimated_policy_value_dict[ci][estimator][
                                "estimated_policy_value_lower_bound"
                            ]
                        )

                        estimated_ranking_dict = {
                            estimated_ranking_[i]: i
                            for i in range(len(estimated_ranking_))
                        }
                        estimated_policy_value_lower_bound = [
                            estimated_policy_value_lower_bound_[
                                estimated_ranking_dict[true_ranking[i]]
                            ]
                            for i in range(len(true_ranking))
                        ]

                        min_val = np.minimum(
                            np.nanmin(true_policy_value),
                            np.nanmin(estimated_policy_value_lower_bound),
                        )
                        max_val = np.maximum(
                            np.nanmax(true_policy_value),
                            np.nanmax(estimated_policy_value_lower_bound),
                        )
                        guide_min = min_val if guide_min > min_val else guide_min
                        guide_max = max_val if guide_max < max_val else guide_max

                        if not share_axes:
                            guide = np.linspace(guide_min, guide_max)
                            axes[j].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

                        axes[j].scatter(
                            true_policy_value,
                            estimated_policy_value_lower_bound,
                        )
                        axes[j].set_title(f"({ci}, {estimator})")
                        axes[j].set_xlabel("true policy value")
                        axes[j].set_ylabel("estimated policy value lower bound")

                if share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    for j, ci in enumerate(cis):
                        axes[j].plot(
                            guide,
                            guide,
                            color="black",
                            linewidth=1.0,
                        )

            else:
                for j, ci in enumerate(cis):
                    for i, estimator in enumerate(self.ope.ope_estimators_):
                        estimated_ranking_ = estimated_policy_value_dict[ci][estimator][
                            "estimated_ranking"
                        ]
                        estimated_policy_value_lower_bound_ = (
                            estimated_policy_value_dict[ci][estimator][
                                "estimated_policy_value_lower_bound"
                            ]
                        )

                        estimated_ranking_dict = {
                            estimated_ranking_[i]: i
                            for i in range(len(estimated_ranking_))
                        }
                        estimated_policy_value_lower_bound = [
                            estimated_policy_value_lower_bound_[
                                estimated_ranking_dict[true_ranking[i]]
                            ]
                            for i in range(len(true_ranking))
                        ]

                        min_val = np.minimum(
                            np.nanmin(true_policy_value),
                            np.nanmin(estimated_policy_value_lower_bound),
                        )
                        max_val = np.maximum(
                            np.nanmax(true_policy_value),
                            np.nanmax(estimated_policy_value_lower_bound),
                        )
                        guide_min = min_val if guide_min > min_val else guide_min
                        guide_max = max_val if guide_max < max_val else guide_max

                        if not share_axes:
                            guide = np.linspace(guide_min, guide_max)
                            axes[i, j].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

                        axes[i, j].scatter(
                            true_policy_value,
                            estimated_policy_value_lower_bound,
                        )
                        axes[i, j].set_title(f"({ci}, {estimator})")
                        axes[i, j].set_xlabel("true policy value")
                        axes[i, j].set_ylabel("estimated policy value")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for j, ci in enumerate(cis):
                    for i, estimator in enumerate(self.ope.ope_estimators_):
                        axes[i, j].plot(
                            guide,
                            guide,
                            color="black",
                            linewidth=1.0,
                        )

        fig.tight_layout()
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_variance(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_variance.png",
    ):
        """Visualize true variance and its estimate.

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

        n_cols: int, default=None
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="scatter_variance.png"
            Name of the bar figure.

        """
        ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
            input_dict=input_dict,
            gamma=gamma,
            return_variance=True,
        )
        candidate_policy_names = ground_truth_policy_value_dict["ranking"]
        true_variance = ground_truth_policy_value_dict["variance"]

        estimated_variance_dict = self.cumulative_distributional_ope.estimate_variance(
            input_dict=input_dict,
            gamma=gamma,
        )

        plt.style.use("ggplot")
        n_figs = len(self.cumulative_distributional_ope.ope_estimators_)
        n_cols = min(5, n_figs) if n_cols is None else n_cols
        n_rows = (n_figs - 1) // n_cols + 1

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=share_axes,
            sharey=share_axes,
        )

        guide_min, guide_max = 1e5, -1e5
        if n_rows == 1:
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                estimated_variance = np.zeros(len(candidate_policy_names))
                for j, eval_policy in enumerate(candidate_policy_names):
                    estimated_variance[j] = estimated_variance_dict[eval_policy][
                        estimator
                    ]

                min_val = np.minimum(
                    np.nanmin(true_variance), np.nanmin(estimated_variance)
                )
                max_val = np.maximum(
                    np.nanmax(true_variance), np.nanmax(estimated_variance)
                )
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i].scatter(
                    true_variance,
                    estimated_variance,
                )
                axes[i].set_title(estimator)
                axes[i].set_xlabel("true variance")
                axes[i].set_ylabel("estimated variance")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(
                    self.cumulative_distributional_ope.ope_estimators_
                ):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                estimated_variance = np.zeros(len(candidate_policy_names))
                for j, eval_policy in enumerate(candidate_policy_names):
                    estimated_variance[j] = estimated_variance_dict[eval_policy][
                        estimator
                    ]

                min_val = np.minimum(
                    np.nanmin(true_variance), np.nanmin(estimated_variance)
                )
                max_val = np.maximum(
                    np.nanmax(true_variance), np.nanmax(estimated_variance)
                )
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i // n_cols, i % n_cols].scatter(
                    true_variance,
                    estimated_variance,
                )
                axes[i // n_cols, i % n_cols].title(estimator)
                axes[i // n_cols, i % n_cols].xlabel("true variance")
                axes[i // n_cols, i % n_cols].ylabel("estimated variance")

            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                axes[i // n_cols, i % n_cols].plot(
                    guide,
                    guide,
                    color="black",
                    linewidth=1.0,
                )

        fig.tight_layout()
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_lower_quartile(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_lower_quartile.png",
    ):
        """Visualize true lower quartile and its estimate.

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

        n_cols: int, default=None
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="scatter_lower_quartile.png"
            Name of the bar figure.

        """
        ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
            input_dict=input_dict,
            gamma=gamma,
            return_lower_quartile=True,
        )
        true_ranking = ground_truth_policy_value_dict["ranking_by_lower_quartile"]
        true_lower_quartile = ground_truth_policy_value_dict["lower_quartile"]

        estimated_policy_value_dict = self.select_by_lower_quartile(
            input_dict=input_dict,
            gamma=gamma,
        )

        plt.style.use("ggplot")
        n_figs = len(self.cumulative_distributional_ope.ope_estimators_)
        n_cols = min(5, n_figs) if n_cols is None else n_cols
        n_rows = (n_figs - 1) // n_cols + 1

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=share_axes,
            sharey=share_axes,
        )

        guide_min, guide_max = 1e5, -1e5
        if n_rows == 1:
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                estimated_ranking_ = estimated_policy_value_dict[estimator][
                    "estimated_ranking"
                ]
                estimated_lower_quartile_ = estimated_policy_value_dict[estimator][
                    "estimated_lower_quartile"
                ]

                estimated_ranking_dict = {
                    estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
                }
                estimated_lower_quartile = [
                    estimated_lower_quartile_[estimated_ranking_dict[true_ranking[i]]]
                    for i in range(len(true_ranking))
                ]

                min_val = np.minimum(
                    np.nanmin(true_lower_quartile), np.nanmin(estimated_lower_quartile)
                )
                max_val = np.maximum(
                    np.nanmax(true_lower_quartile), np.nanmax(estimated_lower_quartile)
                )
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i].scatter(
                    true_lower_quartile,
                    estimated_lower_quartile,
                )
                axes[i].set_title(estimator)
                axes[i].set_xlabel("true lower quartile")
                axes[i].set_ylabel("estimated lower quartile")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(
                    self.cumulative_distributional_ope.ope_estimators_
                ):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                estimated_ranking_ = estimated_policy_value_dict[estimator][
                    "estimated_ranking"
                ]
                estimated_lower_quartile_ = estimated_policy_value_dict[estimator][
                    "estimated_lower_quartile"
                ]

                estimated_ranking_dict = {
                    estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
                }
                estimated_lower_quartile = [
                    estimated_lower_quartile_[estimated_ranking_dict[true_ranking[i]]]
                    for i in range(len(true_ranking))
                ]

                min_val = np.minimum(
                    np.nanmin(true_lower_quartile), np.nanmin(estimated_lower_quartile)
                )
                max_val = np.maximum(
                    np.nanmax(true_lower_quartile), np.nanmax(estimated_lower_quartile)
                )
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i // n_cols, i % n_cols].scatter(
                    true_lower_quartile,
                    estimated_lower_quartile,
                )
                axes[i // n_cols, i % n_cols].set_title(estimator)
                axes[i // n_cols, i % n_cols].set_xlabel("true lower quartile")
                axes[i // n_cols, i % n_cols].set_ylabel("estimated lower quartile")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(
                    self.cumulative_distributional_ope.ope_estimators_
                ):
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        fig.tight_layout()
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_conditional_value_at_risk(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_conditional_value_at_risk.png",
    ):
        """Visualize true conditional value at risk and its estimate.

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

        n_cols: int, default=None
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="scatter_conditional_value_at_risk.png"
            Name of the bar figure.

        """
        ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
            input_dict=input_dict,
            gamma=gamma,
            return_conditional_value_at_risk=True,
        )
        true_ranking = ground_truth_policy_value_dict[
            "ranking_by_conditional_value_at_risk"
        ]
        true_cvar = ground_truth_policy_value_dict["conditional_value_at_risk"]

        estimated_policy_value_dict = self.select_by_conditional_value_at_risk(
            input_dict=input_dict,
            gamma=gamma,
        )

        plt.style.use("ggplot")
        n_figs = len(self.cumulative_distributional_ope.ope_estimators_)
        n_cols = min(5, n_figs) if n_cols is None else n_cols
        n_rows = (n_figs - 1) // n_cols + 1

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=share_axes,
            sharey=share_axes,
        )

        guide_min, guide_max = 1e5, -1e5
        if n_rows == 1:
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                estimated_ranking_ = estimated_policy_value_dict[estimator][
                    "estimated_ranking"
                ]
                estimated_cvar_ = estimated_policy_value_dict[estimator][
                    "estimated_conditional_value_at_risk"
                ]

                estimated_ranking_dict = {
                    estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
                }
                estimated_cvar = [
                    estimated_cvar_[estimated_ranking_dict[true_ranking[i]]]
                    for i in range(len(true_ranking))
                ]

                min_val = np.minimum(np.nanmin(true_cvar), np.nanmin(estimated_cvar))
                max_val = np.maximum(np.nanmax(true_cvar), np.nanmax(estimated_cvar))
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i].scatter(
                    true_cvar,
                    estimated_cvar,
                )
                axes[i].set_title(estimator)
                axes[i].set_xlabel("true CVaR")
                axes[i].set_ylabel("estimated CVaR")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(
                    self.cumulative_distributional_ope.ope_estimators_
                ):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(
                self.cumulative_distributional_ope.ope_estimators_
            ):
                estimated_ranking_ = estimated_policy_value_dict[estimator][
                    "estimated_ranking"
                ]
                estimated_cvar_ = estimated_policy_value_dict[estimator][
                    "estimated_conditional_value_at_risk"
                ]

                estimated_ranking_dict = {
                    estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
                }
                estimated_cvar = [
                    estimated_cvar_[estimated_ranking_dict[true_ranking[i]]]
                    for i in range(len(true_ranking))
                ]

                min_val = np.minimum(np.nanmin(true_cvar), np.nanmin(estimated_cvar))
                max_val = np.maximum(np.nanmax(true_cvar), np.nanmax(estimated_cvar))
                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

                if not share_axes:
                    guide = np.linspace(guide_min, guide_max)
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                axes[i // n_cols, i % n_cols].scatter(
                    true_cvar,
                    estimated_cvar,
                )
                axes[i // n_cols, i % n_cols].plot(
                    guide,
                    guide,
                    color="black",
                    linewidth=1.0,
                )
                axes[i // n_cols, i % n_cols].set_title(estimator)
                axes[i // n_cols, i % n_cols].set_xlabel("true CVaR")
                axes[i // n_cols, i % n_cols].set_ylabel("estimated CVaR")

            if share_axes:
                guide = np.linspace(guide_min, guide_max)
                for i, estimator in enumerate(
                    self.cumulative_distributional_ope.ope_estimators_
                ):
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        fig.tight_layout()
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def visualize_distributionally_robust_worst_case_policy_value(
        self,
        input_dict: OPEInputDict,
        gamma: float = 1.0,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_distributionally_robust_worst_case.png",
    ):
        """Visualize true distributionally robust worst case policy value and its estimate.

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

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="scatter_distributionally_robust_worst_case.png"
            Name of the bar figure.

        """
        ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
            input_dict=input_dict,
            gamma=gamma,
        )
        true_ranking = ground_truth_policy_value_dict[
            "ranking_by_distributionally_robust_worst_case"
        ]
        true_worst_case = ground_truth_policy_value_dict[
            "distributionally_robust_worst_case"
        ]

        estimated_policy_value_dict = (
            self.select_by_distributionally_robust_worst_case_policy_value(
                input_dict=input_dict,
                gamma=gamma,
            )
        )

        plt.style.use("ggplot")
        n_estimators = len(self.distributionally_robust_ope.ope_estimators_)
        fig, axes = plt.subplots(nrows=n_estimators // 5, ncols=min(5, n_estimators))

        for i, estimator in enumerate(self.distributionally_robust_ope.ope_estimators_):
            estimated_ranking_ = estimated_policy_value_dict[estimator][
                "estimated_ranking_by_distributionally_robust_worst_case"
            ]
            estimated_worst_case_ = estimated_policy_value_dict[estimator][
                "estimated_distributionally_robust_worst_case"
            ]

            estimated_ranking_dict = {
                estimated_ranking_[i]: i for i in range(len(estimated_ranking_))
            }
            estimated_worst_case = [
                estimated_worst_case_[estimated_ranking_dict[true_ranking[i]]]
                for i in range(len(true_ranking))
            ]

            guide = np.linspace(
                np.minimum(np.nanmin(true_worst_case), np.nanmin(estimated_worst_case)),
                np.maximum(np.nanmax(true_worst_case), np.nanmax(estimated_worst_case)),
            )

            axes[i // 5, i % 5].scatter(
                true_worst_case,
                estimated_worst_case,
            )
            axes[i // 5, i % 5].plot(
                guide,
                guide,
                color="black",
                linewidth=1.0,
            )
            axes[i // 3, i % 3].title(estimator)
            axes[i // 3, i % 3].xlabel(
                "true distributionally robust worst case policy value"
            )
            axes[i // 3, i % 3].ylabel(
                "estimated distributionally robust worst case policy value"
            )

        fig.tight_layout()
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))
