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

from .ope_discrete import (
    DiscreteOffPolicyEvaluation,
    DiscreteCumulativeDistributionOffPolicyEvaluation,
)
from .ope_continuous import ContinuousOffPolicyEvaluation
from ..types import OPEInputDict
from ..utils import check_array, defaultdict_to_dict


@dataclass
class OffPolicySelection:
    """Class to conduct OPS by multiple estimators simultaneously.

    Note
    -----------
    OPS selects the "best" policy among several candidates based on the policy value or other statistics estimates by OPE.

    (Basic) OPE estimates the expected policy performance called the policy value.

    .. math::

        V(\\pi) := \\mathbb{E} \\left[ \\sum_{t=1}^T \\gamma^{t-1} r_t \\mid \\pi \\right]

    CumulativeDistributionOPE first estimates the following cumulative distribution function,
    and then estimates some statistics including variance, conditional value at risk, and interquartile range.

    .. math::

        F(t, \\pi) := \\mathbb{E} \\left[ \\mathbb{I} \\left \\{ \\sum_{t=1}^T \\gamma^{t-1} r_t \\leq t \\right \\} \\mid \\pi \\right]

    Parameters
    -----------
    ope: {DiscreteOffPolicyEvaluation, ContinuousOffPolicyEvaluation}, default=None
        Instance of the (standard) OPE class.

    cumulative_distribution_ope: DiscreteCumulativeDistributionOffPolicyEvaluation, default=None
        Instance of the cumulative distribution OPE class.

    Examples
    ----------
    .. ::code-block:: python

        # import necessary module from OFRL
        >>> from ofrl.dataset import SyntheticDataset
        >>> from ofrl.policy import DiscreteEpsilonGreedyHead
        >>> from ofrl.ope import CreateOPEInput
        >>> from ofrl.ope import OffPolicySelection
        >>> from ofrl.ope import DiscreteOffPolicyEvaluation as OPE
        >>> from ofrl.ope import DiscreteTrajectoryWiseImportanceSampling as TIS
        >>> from ofrl.ope import DiscretePerDecisionImportanceSampling as PDIS
        >>> from ofrl.ope import DiscreteCumulativeDistributionOffPolicyEvaluation as CumulativeDistributionOPE
        >>> from ofrl.ope import DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling as CDIS
        >>> from ofrl.ope import DiscreteCumulativeDistributionTrajectoryWiseSelfNormalizedImportanceSampling as CDSIS

        # import necessary module from other libraries
        >>> import gym
        >>> import rtbgym
        >>> from d3rlpy.algos import DoubleDQN
        >>> from d3rlpy.online.buffers import ReplayBuffer
        >>> from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        >>> env = gym.make("RTBEnv-discrete-v0")

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
                n_steps=10000,
                n_steps_per_epoch=1000,
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
                is_rtb_env=True,
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

        # OPS
        >>> ope = OPE(
                logged_dataset=logged_dataset,
                ope_estimators=[TIS(), PDIS()],
            )
        >>> cd_ope = CumulativeDistributionOPE(
                logged_dataset=logged_dataset,
                ope_estimators=[
                    CDIS(estimator_name="cdf_is"),
                    CDSIS(estimator_name="cdf_sis"),
                ],
            )
        >>> ops = OffPolicySelection(
                ope=ope,
                cumulative_distribution_ope=cd_ope,
            )
        >>> ops_dict = ops.select_by_policy_value(
                input_dict=input_dict,
                return_metrics=True,
            )
        >>> ops_dict
        {'tis': {'estimated_ranking': ['ddqn', 'random'],
                'estimated_policy_value': array([21.3624954,  0.3827044]),
                'estimated_relative_policy_value': array([1.44732354, 0.02592848]),
                'mean_squared_error': 94.79587393975419,
                'rank_correlation': SpearmanrResult(correlation=0.9999999999999999, pvalue=nan),
                'regret': (0.0, 1),
                'type_i_error_rate': 0.0,
                'type_ii_error_rate': 0.0,
                'safety_threshold': 13.284},
        'pdis': {'estimated_ranking': ['ddqn', 'random'],
                'estimated_policy_value': array([18.02806424,  7.13847486]),
                'estimated_relative_policy_value': array([1.22141357, 0.48363651]),
                'mean_squared_error': 19.45349619733373,
                'rank_correlation': SpearmanrResult(correlation=0.9999999999999999, pvalue=nan),
                'regret': (0.0, 1),
                'type_i_error_rate': 0.0,
                'type_ii_error_rate': 0.0,
                'safety_threshold': 13.284}}


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
    cumulative_distribution_ope: Optional[
        DiscreteCumulativeDistributionOffPolicyEvaluation
    ] = None

    def __post_init__(self):
        if self.ope is None and self.cumulative_distribution_ope is None:
            raise RuntimeError(
                "one of `ope` or `cumulative_distribution_ope` must be given"
            )

        if self.ope is not None and not isinstance(
            self.ope, (DiscreteOffPolicyEvaluation, ContinuousOffPolicyEvaluation)
        ):
            raise RuntimeError(
                "ope must be the instance of either DiscreteOffPolicyEvaluation or ContinuousOffPolicyEvaluation"
            )
        if self.cumulative_distribution_ope is not None and not isinstance(
            self.cumulative_distribution_ope,
            DiscreteCumulativeDistributionOffPolicyEvaluation,
        ):
            raise RuntimeError(
                "cumulative_distribution_ope must be the instance of DiscreteCumulativeDistributionOffPolicyEvaluation"
            )

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
        return_variance: bool = False,
        return_lower_quartile: bool = False,
        return_conditional_value_at_risk: bool = False,
        return_by_dataframe: bool = False,
        quartile_alpha: float = 0.05,
        cvar_alpha: float = 0.05,
    ):
        """Obtain the oracle selection result using the ground-truth policy value.

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

        return_variance: bool, default=False
            Whether to return the variance or not.

        return_lower_quartile: bool. default=False
            Whether to return the lower interquartile or not.

        return_conditional_value_at_risk: bool, default=False
            Whether to return the conditional value at risk or not.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        quartile_alpha: float, default=0.05
            Proportion of the sided region of the interquartile range.

        cvar_alpha: float, default=0.05
            Proportion of the sided region of the conditional value at risk.

        Return
        -------
        ground_truth_dict/ground_truth_df: dict or dataframe
            Dictionary/dataframe containing the following ground-truth (on-policy) metrics.

            ranking: list of str
                Name of the candidate policies sorted by the ground-truth policy value.

            policy_value: list of float
                Ground-truth policy value of the candidate policies (sorted by ranking).

            relative_policy_value: list of float
                Ground-truth relative policy value of the candidate policies compared to the behavior policy (sorted by ranking).

            variance: list of float
                Ground-truth variance of the trajectory wise reward of the candidate policies (sorted by ranking).
                If `return_variance == False`, `None` is recorded.

            ranking_by_lower_quartile: list of str
                Name of the candidate policies sorted by the ground-truth lower quartile of the trajectory wise reward.
                If `return_lower_quartile == False`, `None` is recorded.

            lower_quartile: list of float
                Ground-truth lower quartile of the candidate policies (sorted by ranking_by_lower_quartile).
                If `return_lower_quartile == False`, `None` is recorded.

            ranking_by_conditional_value_at_risk: list of str
                Name of the candidate policies sorted by the ground-truth conditional value at risk.
                If `return_conditional_value_at_risk == False`, `None` is recorded.

            conditional_value_at_risk: list of float
                Ground-truth conditional value at risk of the candidate policies (sorted by ranking_by_conditional_value_at_risk).
                If `return_conditional_value_at_risk == False`, `None` is recorded.

            parameters: dict
                Dictionary containing quartile_alpha, and cvar_alpha.

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
            "parameters": {
                "quartile_alpha": quartile_alpha if return_lower_quartile else None,
                "cvar_alpha": cvar_alpha if return_conditional_value_at_risk else None,
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
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_criteria: float = 0.0,
    ):
        """Rank the candidate policies by their estimated policy value.

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

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics:
            mean-squared-error, rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_criteria: float, default=0.0 (>= 0)
            The relative policy value required to be a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.
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

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value.
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            estimated_policy_value: list of float
                Estimated policy value of the candidate policies (sorted by estimated_ranking).
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            estimated_relative_policy_value: list of float
                Estimated relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            mean_squared_error: float
                Mean-squared-error of the estimated policy value.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            regret: tuple of float and int
                Regret@k and k.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            safety_threshold: float
                The relative policy value required to be a safe policy.

        """
        if self.ope is None:
            raise RuntimeError(
                "ope is not given. Please initialize the class with ope attribute"
            )

        estimated_policy_value_dict = self.ope.estimate_policy_value(
            input_dict=input_dict,
        )

        if return_metrics:
            ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
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

                if true_safety.sum() > 0:
                    type_i_error_rate = (
                        true_safety > estimated_safety
                    ).sum() / true_safety.sum()
                else:
                    type_i_error_rate = 0.0

                if (1 - true_safety).sum() > 0:
                    type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                        1 - true_safety
                    ).sum()
                else:
                    type_ii_error_rate = 0.0

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

    def select_by_policy_value_via_cumulative_distribution_ope(
        self,
        input_dict: OPEInputDict,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_criteria: float = 0.0,
    ):
        """Rank the candidate policies by their estimated policy value via cumulative distribution methods.

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

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics:
            mean-squared-error, rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_criteria: float, default=0.0 (>= 0)
            The relative policy value required to be a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.
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

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value.
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            estimated_policy_value: list of float
                Estimated policy value of the candidate policies (sorted by estimated_ranking).
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            estimated_relative_policy_value: list of float
                Estimated relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            mean_squared_error: float
                Mean-squared-error of the estimated policy value.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            regret: tuple of float and int
                Regret@k and k.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            safety_threshold: float
                The relative policy value required to be a safe policy.

        """
        if self.cumulative_distribution_ope is None:
            raise RuntimeError(
                "cumulative_distribution_ope is not given. Please initialize the class with cumulative_distribution_ope attribute"
            )

        estimated_policy_value_dict = self.cumulative_distribution_ope.estimate_mean(
            input_dict=input_dict,
        )

        if return_metrics:
            ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
            )
            true_ranking = ground_truth_policy_value_dict["ranking"]
            true_policy_value = ground_truth_policy_value_dict["policy_value"]

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = {}
        for i, estimator in enumerate(self.cumulative_distribution_ope.ope_estimators_):

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

                if true_safety.sum() > 0:
                    type_i_error_rate = (
                        true_safety > estimated_safety
                    ).sum() / true_safety.sum()
                else:
                    type_i_error_rate = 0.0

                if (1 - true_safety).sum() > 0:
                    type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                        1 - true_safety
                    ).sum()
                else:
                    type_ii_error_rate = 0.0

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
                self.cumulative_distribution_ope.ope_estimators_
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

            metric_df["estimator"] = self.cumulative_distribution_ope.ope_estimators_
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
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_criteria: float = 0.0,
        cis: List[str] = ["bootstrap"],
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ):
        """Rank the candidate policies by their estimated policy value lower bound.

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

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics:
            rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_criteria: float, default=0.0 (>= 0)
            The relative policy value required to be a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.

        cis: list of {"bootstrap", "hoeffding", "bernstein", "ttest"}, default=["bootstrap"]
            Estimation methods for confidence intervals.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        n_bootstrap_samples: int, default=100 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.
            key: [ci][estimator_name][
                estimated_ranking,
                estimated_policy_value_lower_bound,
                estimated_relative_policy_value_lower_bound,
                mean_squared_error,
                rank_correlation,
                regret,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value lower bound.
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            estimated_policy_value_lower_bound: list of float
                Estimated policy value lower bound of the candidate policies (sorted by estimated_ranking).
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            estimated_relative_policy_value_lower_bound: list of float
                Estimated relative policy value lower bound of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            mean_squared_error: None
                This is for API consistency.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            regret: tuple of float and int
                Regret@k and k.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            safety_threshold: float
                The relative policy value required to be a safe policy.

        """
        if self.ope is None:
            raise RuntimeError(
                "ope is not given. Please initialize the class with ope attribute"
            )

        if return_metrics:
            ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
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

                    if true_safety.sum() > 0:
                        type_i_error_rate = (
                            true_safety > estimated_safety
                        ).sum() / true_safety.sum()
                    else:
                        type_i_error_rate = 0.0

                    if (1 - true_safety).sum() > 0:
                        type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                            1 - true_safety
                        ).sum()
                    else:
                        type_ii_error_rate = 0.0

                ops_dict[ci][estimator] = {
                    "estimated_ranking": estimated_ranking,
                    "estimated_policy_value_lower_bound": estimated_policy_value_lower_bound,
                    "estimated_relative_policy_value_lower_bound": estimated_relative_policy_value_lower_bound,
                    "rank_correlation": rankcorr if return_metrics else None,
                    "mean_squared_error": None,
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
            metric_df["mean_squared_error"] = np.nan
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
        alpha: float = 0.05,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank the candidate policies by their estimated lower quartile of the trajectory wise reward.

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

        alpha: float, default=0.05
            Proportion of the sided region. The value should be within `[0, 0.5]`.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        safety_threshold: float, default=0.0 (>= 0)
            The lower quartile required to be a safe policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.
            key: [estimator_name][
                estimated_ranking,
                estimated_lower_quartile,
                mean_squared_error,
                rank_correlation,
                regret,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated lower quartile of the trajectory wise reward.
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            estimated_lower_quartile: list of float
                Estimated lower quartile of the trajectory wise reward of the candidate policies (sorted by estimated_ranking).
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            mean_squared_error: float
                Mean-squared-error of the estimated lower quartile of the trajectory wise reward.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            regret: None
                This is for API consistency.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            safety_threshold: float
                The lower quartile required to be a safe policy.

        """
        if self.cumulative_distribution_ope is None:
            raise RuntimeError(
                "cumulative_distribution_ope is not given. Please initialize the class with cumulative_distribution_ope attribute"
            )

        estimated_interquartile_range_dict = (
            self.cumulative_distribution_ope.estimate_interquartile_range(
                input_dict=input_dict,
                alpha=alpha,
            )
        )

        if return_metrics:
            ground_truth_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
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
        for i, estimator in enumerate(self.cumulative_distribution_ope.ope_estimators_):

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

                if true_safety.sum() > 0:
                    type_i_error_rate = (
                        true_safety > estimated_safety
                    ).sum() / true_safety.sum()
                else:
                    type_i_error_rate = 0.0

                if (1 - true_safety).sum() > 0:
                    type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                        1 - true_safety
                    ).sum()
                else:
                    type_ii_error_rate = 0.0

            ops_dict[estimator] = {
                "estimated_ranking": estimated_ranking,
                "estimated_lower_quartile": estimated_lower_quartile,
                "mean_squared_error": mse if return_metrics else None,
                "rank_correlation": rankcorr if return_metrics else None,
                "regret": None,
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
                self.cumulative_distribution_ope.ope_estimators_
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

            metric_df["estimator"] = self.cumulative_distribution_ope.ope_estimators_
            metric_df["mean_squared_error"] = mse
            metric_df["rank_correlation"] = rankcorr
            metric_df["pvalue"] = pvalue
            metric_df["regret"] = np.nan
            metric_df["type_i_error_rate"] = type_i
            metric_df["type_ii_error_rate"] = type_ii

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

        return (ranking_df_dict, metric_df) if return_by_dataframe else ops_dict

    def select_by_conditional_value_at_risk(
        self,
        input_dict: OPEInputDict,
        alpha: float = 0.05,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank the candidate policies by their estimated conditional value at risk.

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
            Proportion of the sided region. The value should be within `[0, 1]`.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        safety_threshold: float, default=0.0 (>= 0)
            The conditional value at risk required to be a safe policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.
            key: [estimator_name][
                estimated_ranking,
                estimated_lower_quartile,
                mean_squared_error,
                rank_correlation,
                regret,
                type_i_error_rate,
                type_ii_error_rate,
            ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated conditional value at risk.
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            estimated_conditional_value_at_risk: list of float
                Estimated conditional value at risk of the candidate policies (sorted by estimated_ranking).
                Recorded in `ranking_df_dict` when `return_by_dataframe == True`.

            mean_squared_error: float
                Mean-squared-error of the estimated conditional value at risk.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            rank_correlation: tuple or float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            regret: None
                This is for API consistency.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                If `return_metric == False`, `None` is recorded.
                Recorded in `metric_df` when `return_by_dataframe == True`.

            safety_threshold: float
                The conditional value at risk required to be a safe policy.

        """
        if self.cumulative_distribution_ope is None:
            raise RuntimeError(
                "cumulative_distribution_ope is not given. Please initialize the class with cumulative_distribution_ope attribute"
            )

        estimated_cvar_dict = (
            self.cumulative_distribution_ope.estimate_conditional_value_at_risk(
                input_dict=input_dict,
                alphas=alpha,
            )
        )

        if return_metrics:
            ground_truth_dict = self.obtain_oracle_selection_result(
                input_dict=input_dict,
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
        for i, estimator in enumerate(self.cumulative_distribution_ope.ope_estimators_):

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

                if true_safety.sum() > 0:
                    type_i_error_rate = (
                        true_safety > estimated_safety
                    ).sum() / true_safety.sum()
                else:
                    type_i_error_rate = 0.0

                if (1 - true_safety).sum() > 0:
                    type_ii_error_rate = (true_safety < estimated_safety).sum() / (
                        1 - true_safety
                    ).sum()
                else:
                    type_ii_error_rate = 0.0

            ops_dict[estimator] = {
                "estimated_ranking": estimated_ranking,
                "estimated_conditional_value_at_risk": estimated_cvar,
                "mean_squared_error": mse if return_metrics else None,
                "rank_correlation": rankcorr if return_metrics else None,
                "regret": None,
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
                self.cumulative_distribution_ope.ope_estimators_
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

            metric_df["estimator"] = self.cumulative_distribution_ope.ope_estimators_
            metric_df["mean_squared_error"] = mse
            metric_df["rank_correlation"] = rankcorr
            metric_df["pvalue"] = pvalue
            metric_df["regret"] = np.nan
            metric_df["type_i_error_rate"] = type_i
            metric_df["type_ii_error_rate"] = type_ii

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

        return (ranking_df_dict, metric_df) if return_by_dataframe else ops_dict

    def visualize_policy_value_for_selection(
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
    ):
        """Visualize the policy value estimated by OPE estimators (box plot).

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
            If True, the method visualizes the estimated policy value of the evaluation policy
            relative to the on-policy policy value of the behavior policy.

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
        return self.ope.visualize_off_policy_estimates(
            input_dict=input_dict,
            alpha=alpha,
            ci=ci,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
            is_relative=is_relative,
            hue=hue,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_cumulative_distribution_function_for_selection(
        self,
        input_dict: OPEInputDict,
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_cumulative_distribution_function.png",
    ) -> None:
        """Visualize the cumulative distribution function (cdf plot).

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
        return (
            self.cumulative_distribution_ope.visualize_cumulative_distribution_function(
                input_dict=input_dict,
                hue=hue,
                legend=legend,
                n_cols=n_cols,
                fig_dir=fig_dir,
                fig_name=fig_name,
            )
        )

    def visualize_policy_value_of_cumulative_distribution_ope_for_selection(
        self,
        input_dict: OPEInputDict,
        alpha: float = 0.05,
        is_relative: bool = False,
        hue: str = "estimator",
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize the policy value estimated by cumulative distribution OPE estimators (box plot).

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
            Significance level. The value should bw within `[0, 1)`.

        is_relative: bool, default=False
            If True, the method visualizes the estimated policy value of the evaluation policy
            relative to the ground-truth policy value of the behavior policy.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        sharey: bool, default=False
            If True, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        return self.cumulative_distribution_ope.visualize_policy_value(
            input_dict=input_dict,
            alpha=alpha,
            is_relative=is_relative,
            hue=hue,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_conditional_value_at_risk_for_selection(
        self,
        input_dict: OPEInputDict,
        alphas: np.ndarray = np.linspace(0, 1, 20),
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize the conditional value at risk estimated by cumulative distribution OPE estimators (cdf plot).

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

        alphas: array-like of shape (n_alpha, ), default=np.linspace(0, 1, 20)
            Set of proportions of the sided region.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the figure.

        n_cols: int, default=None
            Number of columns in the figure.

        sharey: bool, default=False
            If True, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        return self.cumulative_distribution_ope.visualize_conditional_value_at_risk(
            input_dict=input_dict,
            alphas=alphas,
            hue=hue,
            legend=legend,
            n_cols=n_cols,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_interquartile_range_for_selection(
        self,
        input_dict: OPEInputDict,
        alpha: float = 0.05,
        hue: str = "estimator",
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize the interquartile range estimated by cumulative distribution OPE estimators (box plot).

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

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        sharey: bool, default=False
            If True, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        return self.cumulative_distribution_ope.visualize_interquartile_range(
            input_dict=input_dict,
            alpha=alpha,
            hue=hue,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_policy_value_for_validation(
        self,
        input_dict: OPEInputDict,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_policy_value.png",
    ):
        """Visualize the true policy value and its estimate (scatter plot).

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
        )
        true_ranking = ground_truth_policy_value_dict["ranking"]
        true_policy_value = np.array(ground_truth_policy_value_dict["policy_value"])

        estimated_policy_value_dict = self.select_by_policy_value(
            input_dict=input_dict,
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
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_policy_value_of_cumulative_distribution_ope_for_validation(
        self,
        input_dict: OPEInputDict,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_policy_value_of_cumulative_distribution_ope.png",
    ):
        """Visualize the true policy value and its estimate obtained by cumulative distribution OPE (scatter plot).

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

        n_cols: int, default=None
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="scatter_policy_value_of_cumulative_distribution_ope.png"
            Name of the bar figure.

        """
        ground_truth_policy_value_dict = self.obtain_oracle_selection_result(
            input_dict=input_dict,
        )
        true_ranking = ground_truth_policy_value_dict["ranking"]
        true_policy_value = np.array(ground_truth_policy_value_dict["policy_value"])

        estimated_policy_value_dict = (
            self.select_by_policy_value_via_cumulative_distribution_ope(
                input_dict=input_dict,
            )
        )

        plt.style.use("ggplot")
        n_figs = len(self.cumulative_distribution_ope.ope_estimators_)
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
                self.cumulative_distribution_ope.ope_estimators_
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
                    self.cumulative_distribution_ope.ope_estimators_
                ):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(
                self.cumulative_distribution_ope.ope_estimators_
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
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_policy_value_lower_bound_for_validation(
        self,
        input_dict: OPEInputDict,
        cis: List[str] = ["bootstrap"],
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = 12345,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_policy_value_lower_bound.png",
    ):
        """Visualize the true policy value and its estimate lower bound (scatter plot).

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

        cis: list of {"bootstrap", "hoeffding", "bernstein", "ttest"}, default=["bootstrap"]
            Estimation methods for confidence intervals.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

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
        )
        true_ranking = ground_truth_policy_value_dict["ranking"]
        true_policy_value = ground_truth_policy_value_dict["policy_value"]

        estimated_policy_value_dict = self.select_by_policy_value_lower_bound(
            input_dict=input_dict,
            cis=cis,
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
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_variance_for_validation(
        self,
        input_dict: OPEInputDict,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_variance.png",
    ):
        """Visualize the true variance and its estimate (scatter plot).

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
            return_variance=True,
        )
        candidate_policy_names = ground_truth_policy_value_dict["ranking"]
        true_variance = ground_truth_policy_value_dict["variance"]

        estimated_variance_dict = self.cumulative_distribution_ope.estimate_variance(
            input_dict=input_dict,
        )

        plt.style.use("ggplot")
        n_figs = len(self.cumulative_distribution_ope.ope_estimators_)
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
                self.cumulative_distribution_ope.ope_estimators_
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
                    self.cumulative_distribution_ope.ope_estimators_
                ):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(
                self.cumulative_distribution_ope.ope_estimators_
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
                self.cumulative_distribution_ope.ope_estimators_
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
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_lower_quartile_for_validation(
        self,
        input_dict: OPEInputDict,
        alpha: float = 0.05,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_lower_quartile.png",
    ):
        """Visualize the true lower quartile and its estimate (scatter plot).

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
            Proportion of the sided region. The value should be within `[0, 0.5]`.

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
            quartile_alpha=alpha,
            return_lower_quartile=True,
        )
        true_ranking = ground_truth_policy_value_dict["ranking_by_lower_quartile"]
        true_lower_quartile = ground_truth_policy_value_dict["lower_quartile"]

        estimated_policy_value_dict = self.select_by_lower_quartile(
            input_dict=input_dict,
            alpha=alpha,
        )

        plt.style.use("ggplot")
        n_figs = len(self.cumulative_distribution_ope.ope_estimators_)
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
                self.cumulative_distribution_ope.ope_estimators_
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
                    self.cumulative_distribution_ope.ope_estimators_
                ):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(
                self.cumulative_distribution_ope.ope_estimators_
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
                    self.cumulative_distribution_ope.ope_estimators_
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
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_conditional_value_at_risk_for_validation(
        self,
        input_dict: OPEInputDict,
        alpha: float = 0.05,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "scatter_conditional_value_at_risk.png",
    ):
        """Visualize the true conditional value at risk and its estimate (scatter plot).

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
            Proportion of the sided region. The value should be within `[0, 1]`.

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
            cvar_alpha=alpha,
            return_conditional_value_at_risk=True,
        )
        true_ranking = ground_truth_policy_value_dict[
            "ranking_by_conditional_value_at_risk"
        ]
        true_cvar = ground_truth_policy_value_dict["conditional_value_at_risk"]

        estimated_policy_value_dict = self.select_by_conditional_value_at_risk(
            input_dict=input_dict,
            alpha=alpha,
        )

        plt.style.use("ggplot")
        n_figs = len(self.cumulative_distribution_ope.ope_estimators_)
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
                self.cumulative_distribution_ope.ope_estimators_
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
                    self.cumulative_distribution_ope.ope_estimators_
                ):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(
                self.cumulative_distribution_ope.ope_estimators_
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
                    self.cumulative_distribution_ope.ope_estimators_
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
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")
