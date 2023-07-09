# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Meta class to handle Off-Policy Selection (OPS) and evaluation of OPE/OPS."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_scalar
import matplotlib.pyplot as plt

from .ope import (
    OffPolicyEvaluation,
    CumulativeDistributionOPE,
)
from ..utils import (
    MultipleInputDict,
    estimate_confidence_interval_by_bootstrap,
    estimate_confidence_interval_by_hoeffding,
    estimate_confidence_interval_by_empirical_bernstein,
    estimate_confidence_interval_by_t_test,
    defaultdict_to_dict,
)
from ..types import OPEInputDict

markers = ["o", "v", "^", "s", "p", "P", "*", "h", "X", "D", "d"]
dkred = "#A60628"


@dataclass
class OffPolicySelection:
    """Class to conduct OPS and evaluation of OPE/OPS with multiple estimators simultaneously.

    Imported as: :class:`scope_rl.ope.OffPolicySelection`

    Note
    -----------
    **Off-Policy Selection (OPS)**

    OPS selects the "best" policy among several candidates based on the policy value or other statistics estimated by OPE.

    .. math::

        \\hat{\\pi} := {\\arg \\max}_{\\pi \\in \\Pi} \hat{J}(\\pi)

    where :math:`\\Pi` is a set of candidate policies and :math:`\hat{J}(\\cdot)` is some OPE estimates of the policy performance. Below, we describe two types of OPE to estimate such policy performance.

    **Off-Policy Evaluation (OPE)**

    (Basic) OPE estimates the expected policy performance called the policy value.

    .. math::

        V(\\pi) := \\mathbb{E} \\left[ \\sum_{t=1}^T \\gamma^{t-1} r_t \\mid \\pi \\right]

    where :math:`r_t` is the reward observed at each timestep :math:`t`,
    :math:`T` is the total number of timesteps in an episode, and :math:`\\gamma` is the discount factor.

    .. seealso::

        :class:`OffPolicyEvaluation`

    **Cumulative Distribution OPE**

    In contrast, cumulative distribution OPE first estimates the following cumulative distribution function.

    .. math::

        F(t, \\pi) := \\mathbb{E} \\left[ \\mathbb{I} \\left \\{ \\sum_{t=1}^T \\gamma^{t-1} r_t \\leq t \\right \\} \\mid \\pi \\right]

    Then, cumulative distribution OPE also estimates some risk functions including variance, conditional value at risk, and interquartile range based on the CDF estimate.

    .. seealso::

        :class:`CumulativeDistributionOPE`

    Parameters
    -----------
    ope: OffPolicyEvaluation, default=None
        Instance of the (standard) OPE class.

    cumulative_distribution_ope: CumulativeDistributionOPE, default=None
        Instance of the cumulative distribution OPE class.

    Examples
    ----------

    Preparation:

    .. code-block:: python

        # import necessary module from SCOPE-RL
        from scope_rl.dataset import SyntheticDataset
        from scope_rl.policy import EpsilonGreedyHead
        from scope_rl.ope import CreateOPEInput
        from scope_rl.ope import OffPolicySelection
        from scope_rl.ope import OffPolicyEvaluation as OPE
        from scope_rl.ope.discrete import TrajectoryWiseImportanceSampling as TIS
        from scope_rl.ope.discrete import PerDecisionImportanceSampling as PDIS
        from scope_rl.ope import CumulativeDistributionOPE
        from scope_rl.ope.discrete import CumulativeDistributionTIS as CD_IS
        from scope_rl.ope.discrete import CumulativeDistributionSNTIS as CD_SNIS

        # import necessary module from other libraries
        import gym
        import rtbgym
        from d3rlpy.algos import DoubleDQN
        from d3rlpy.online.buffers import ReplayBuffer
        from d3rlpy.online.explorers import ConstantEpsilonGreedy

        # initialize environment
        env = gym.make("RTBEnv-discrete-v0")

        # for api compatibility to d3rlpy
        from scope_rl.utils import OldGymAPIWrapper
        env_ = OldGymAPIWrapper(env)

        # define (RL) agent (i.e., policy) and train on the environment
        ddqn = DoubleDQN()
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env_,
        )
        explorer = ConstantEpsilonGreedy(
            epsilon=0.3,
        )
        ddqn.fit_online(
            env=env_,
            buffer=buffer,
            explorer=explorer,
            n_steps=10000,
            n_steps_per_epoch=1000,
        )

        # convert ddqn policy to stochastic data collection policy
        behavior_policy = EpsilonGreedyHead(
            ddqn,
            n_actions=env.action_space.n,
            epsilon=0.3,
            name="ddqn_epsilon_0.3",
            random_state=12345,
        )

        # initialize dataset class
        dataset = SyntheticDataset(
            env=env,
            max_episode_steps=env.step_per_episode,
        )

        # data collection
        logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policy,
            n_trajectories=100,
            random_state=12345,
        )

    Create Input for OPE:

    .. code-block:: python

        # evaluation policy
        ddqn_ = EpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="ddqn",
            epsilon=0.0,
            random_state=12345
        )
        random_ = EpsilonGreedyHead(
            base_policy=ddqn,
            n_actions=env.action_space.n,
            name="random",
            epsilon=1.0,
            random_state=12345
        )

        # create input for off-policy evaluation (OPE)
        prep = CreateOPEInput(
            env=env,
        )
        input_dict = prep.obtain_whole_inputs(
            logged_dataset=logged_dataset,
            evaluation_policies=[ddqn_, random_],
            n_trajectories_on_policy_evaluation=100,
            random_state=12345,
        )

    **Off-Policy Evaluation and Selection**:

    .. code-block:: python

        # OPS
        ope = OPE(
            logged_dataset=logged_dataset,
            ope_estimators=[TIS(), PDIS()],
        )
        cd_ope = CumulativeDistributionOPE(
            logged_dataset=logged_dataset,
            ope_estimators=[
                CD_IS(estimator_name="cd_is"),
                CD_SNIS(estimator_name="cd_snis"),
            ],
        )
        ops = OffPolicySelection(
            ope=ope,
            cumulative_distribution_ope=cd_ope,
        )
        ops_dict = ops.select_by_policy_value(
            input_dict=input_dict,
            return_metrics=True,
        )

    **Output**:

    .. code-block:: python

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

    .. seealso::

        * :doc:`Quickstart </documentation/quickstart>`
        * :doc:`Related tutorials (OPS) </documentation/examples/ops>` and :doc:`related tutorials (assessments) <documentation/examples/assessments>`

    References
    -------
    Vladislav Kurenkov and Sergey Kolesnikov.
    "Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters." 2022.

    Shengpu Tang and Jenna Wiens.
    "Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings." 2021.

    Justin Fu, Mohammad Norouzi, Ofir Nachum, George Tucker, Ziyu Wang, Alexander Novikov, Mengjiao Yang,
    Michael R. Zhang, Yutian Chen, Aviral Kumar, Cosmin Paduraru, Sergey Levine, and Tom Le Paine.
    "Benchmarks for Deep Off-Policy Evaluation." 2021.

    Tom Le Paine, Cosmin Paduraru, Andrea Michi, Caglar Gulcehre, Konrad Zolna, Alexander Novikov, Ziyu Wang, and Nando de Freitas.
    "Hyperparameter Selection for Offline Reinforcement Learning." 2020.

    """

    ope: Optional[OffPolicyEvaluation] = None
    cumulative_distribution_ope: Optional[CumulativeDistributionOPE] = None

    def __post_init__(self):
        if self.ope is None and self.cumulative_distribution_ope is None:
            raise RuntimeError(
                "one of `ope` or `cumulative_distribution_ope` must be given"
            )

        if self.ope is not None and not isinstance(self.ope, OffPolicyEvaluation):
            raise RuntimeError("ope must be the instance of OffPolicyEvaluation")
        if self.cumulative_distribution_ope is not None and not isinstance(
            self.cumulative_distribution_ope, CumulativeDistributionOPE
        ):
            raise RuntimeError(
                "cumulative_distribution_ope must be the instance of CumulativeDistributionOPE"
            )

        self.step_per_trajectory = self.ope.logged_dataset["step_per_trajectory"]
        check_scalar(
            self.step_per_trajectory,
            name="ope.logged_dataset['step_per_trajectory']",
            target_type=int,
            min_val=1,
        )

        self.behavior_policy_reward = {}
        if self.ope.use_multiple_logged_dataset:
            for (
                behavior_policy
            ) in self.ope.multiple_logged_dataset.behavior_policy_names:
                logged_dataset_ = self.ope.multiple_logged_dataset.get(
                    behavior_policy_name=behavior_policy, dataset_id=0
                )
                self.behavior_policy_reward[behavior_policy] = logged_dataset_[
                    "reward"
                ].reshape((-1, self.step_per_trajectory))

                if self.ope.disable_reward_after_done:
                    done = logged_dataset_["done"].reshape(
                        (-1, self.step_per_trajectory)
                    )
                    self.behavior_policy_reward[
                        behavior_policy
                    ] = self.behavior_policy_reward[behavior_policy] * (
                        1 - done
                    ).cumprod(
                        axis=1
                    )

        else:
            behavior_policy = self.ope.logged_dataset["behavior_policy"]
            self.behavior_policy_reward[behavior_policy] = self.ope.logged_dataset[
                "reward"
            ].reshape((-1, self.step_per_trajectory))

            if self.ope.disable_reward_after_done:
                done = self.ope.logged_dataset["done"].reshape(
                    (-1, self.step_per_trajectory)
                )
                self.behavior_policy_reward[
                    behavior_policy
                ] = self.behavior_policy_reward[behavior_policy] * (1 - done).cumprod(
                    axis=1
                )

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _check_compared_estimators(
        self,
        compared_estimators: Optional[List[str]] = None,
        ope_type: str = "standard_ope",
    ):
        if ope_type == "standard_ope":
            if self.ope is None:
                raise RuntimeError(
                    "ope is not given. Please initialize the class with ope attribute"
                )
        else:
            if self.cumulative_distribution_ope is None:
                raise RuntimeError(
                    "cumulative_distribution_ope is not given. Please initialize the class with cumulative_distribution_ope attribute"
                )

        if compared_estimators is None:
            compared_estimators = self.estimators_name[ope_type]
        elif not set(compared_estimators).issubset(self.estimators_name[ope_type]):
            raise ValueError(
                f"compared_estimators must be a subset of self.estimators_name['{ope_type}'], but found False."
            )
        return compared_estimators

    def _check_basic_visualization_inputs(
        self,
        n_cols: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: Optional[str] = None,
    ):
        if n_cols is not None:
            check_scalar(n_cols, name="n_cols", target_type=int, min_val=1)
        if fig_dir is not None and not isinstance(fig_dir, Path):
            raise ValueError(f"fig_dir must be a Path, but {type(fig_dir)} is given")
        if fig_name is not None and not isinstance(fig_name, str):
            raise ValueError(f"fig_dir must be a string, but {type(fig_dir)} is given")

    def _check_topk_inputs(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        max_topk: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
        gamma: Optional[float] = None,
    ):
        if isinstance(input_dict, MultipleInputDict):
            max_topk_ = 100

            if behavior_policy_name is None:
                if dataset_id is None:
                    for n_eval_policies in input_dict.n_eval_policies.values():
                        max_topk_ = min(max_topk_, n_eval_policies.min())
                else:
                    for n_eval_policies in input_dict.n_eval_policies.values():
                        max_topk_ = min(max_topk_, n_eval_policies[dataset_id])
            else:
                if dataset_id is None:
                    max_topk_ = min(
                        max_topk_,
                        input_dict.n_eval_policies[behavior_policy_name].min(),
                    )
                else:
                    max_topk_ = input_dict.n_eval_policies[behavior_policy_name][
                        dataset_id
                    ]
        else:
            behavior_policy_name = input_dict[list(input_dict.keys())[0]][
                "behavior_policy"
            ]
            max_topk_ = len(input_dict)

        if max_topk is None:
            max_topk = int(max_topk_)
        else:
            check_scalar(max_topk, name="max_topk", target_type=int, min_val=1)
            max_topk = min(max_topk, max_topk_)

        if metrics is not None:
            for metric in metrics:
                if metric not in [
                    "k-th",
                    "best",
                    "worst",
                    "mean",
                    "std",
                    "safety_violation_rate",
                    "sharpe_ratio",
                ]:
                    raise ValueError(
                        f"The elements of metrics must be one of 'k-th', 'best', 'worst', 'mean', 'std', 'safety_violation_rate', or 'sharpe_ratio', but {metric} is given."
                    )

        if safety_threshold is None:
            if relative_safety_criteria is not None:
                check_scalar(
                    relative_safety_criteria,
                    name="relative_safety_criteria",
                    target_type=float,
                    min_val=0.0,
                )

                discount = np.full(self.step_per_trajectory, gamma).cumprod() / gamma

                if behavior_policy_name is not None:
                    behavior_policy_reward = self.behavior_policy_reward[
                        behavior_policy_name
                    ]
                    behavior_policy_value = (
                        discount[np.newaxis, :] * behavior_policy_reward
                    ).sum(
                        axis=1
                    ).mean() + 1e-10  # to avoid zero division

                    safety_threshold = relative_safety_criteria * behavior_policy_value
                    safety_threshold = float(safety_threshold)

                elif len(self.behavior_policy_reward) == 1:
                    behavior_policy_reward = list(self.behavior_policy_reward.values())[
                        0
                    ]
                    behavior_policy_value = (
                        discount[np.newaxis, :] * behavior_policy_reward
                    ).sum(
                        axis=1
                    ).mean() + 1e-10  # to avoid zero division

                    safety_threshold = relative_safety_criteria * behavior_policy_value
                    safety_threshold = float(safety_threshold)

                else:
                    safety_threshold = 0.0

            else:
                safety_threshold = 0.0

        check_scalar(
            safety_threshold,
            name="safety_threshold",
            target_type=float,
        )

        return max_topk, safety_threshold

    def _obtain_true_selection_result(
        self,
        input_dict: OPEInputDict,
        return_variance: bool = False,
        return_lower_quartile: bool = False,
        return_conditional_value_at_risk: bool = False,
        return_by_dataframe: bool = False,
        quartile_alpha: float = 0.05,
        cvar_alpha: float = 0.05,
    ):
        """Obtain the oracle selection result based on the ground-truth policy value.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        return_variance: bool, default=False
            Whether to return the variance or not.

        return_lower_quartile: bool. default=False
            Whether to return the lower interquartile or not.

        return_conditional_value_at_risk: bool, default=False
            Whether to return the conditional value at risk or not.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        quartile_alpha: float, default=0.05
            Proportion of the shaded region of the interquartile range.

        cvar_alpha: float, default=0.05
            Proportion of the shaded region of the conditional value at risk.

        Return
        -------
        ground_truth_dict/ground_truth_df: dict or dataframe
            Dictionary/dataframe containing the following ground-truth (on-policy) metrics.

            .. code-block:: python

                key: [
                    ranking,
                    policy_value,
                    relative_policy_value,
                    variance,
                    ranking_by_lower_quartile,
                    lower_quartile,
                    ranking_by_conditional_value_at_risk,
                    conditional_value_at_risk,
                    parameters,  # only when return_by_dataframe == False
                ]

            ranking: list of str
                Name of the candidate policies sorted by the ground-truth policy value.

            policy_value: list of float
                Ground-truth policy value of the candidate policies (sorted by ranking).

            relative_policy_value: list of float
                Ground-truth relative policy value of the candidate policies compared to the behavior policy (sorted by ranking).

            variance: list of float
                Ground-truth variance of the trajectory-wise reward of the candidate policies (sorted by ranking).
                If return_variance is `False`, `None` is recorded.

            ranking_by_lower_quartile: list of str
                Name of the candidate policies sorted by the ground-truth lower quartile of the trajectory-wise reward.
                If return_lower_quartile is `False`, `None` is recorded.

            lower_quartile: list of float
                Ground-truth lower quartile of the candidate policies (sorted by ranking_by_lower_quartile).
                If return_lower_quartile is `False`, `None` is recorded.

            ranking_by_conditional_value_at_risk: list of str
                Name of the candidate policies sorted by the ground-truth conditional value at risk.
                If return_conditional_value_at_risk is `False`, `None` is recorded.

            conditional_value_at_risk: list of float
                Ground-truth conditional value at risk of the candidate policies (sorted by ranking_by_conditional_value_at_risk).
                If return_conditional_value_at_risk is `False`, `None` is recorded.

            parameters: dict
                Dictionary containing quartile_alpha, and cvar_alpha.
                If return_by_dataframe is `True`, parameters will not be returned.

        """
        candidate_policy_names = list(input_dict.keys())
        for eval_policy in candidate_policy_names:
            if input_dict[eval_policy]["on_policy_policy_value"] is None:
                raise ValueError(
                    f"one of the candidate policies, {eval_policy}, does not contain on-policy policy value in input_dict"
                )
        behavior_policy = input_dict[eval_policy]["behavior_policy"]

        n_policies = len(candidate_policy_names)
        n_samples = len(input_dict[eval_policy]["on_policy_policy_value"])

        policy_value = np.zeros(n_policies)
        for i, eval_policy in enumerate(candidate_policy_names):
            policy_value[i] = input_dict[eval_policy]["on_policy_policy_value"].mean()

        ranking_index = np.argsort(policy_value)[::-1]
        ranking = [candidate_policy_names[ranking_index[i]] for i in range(n_policies)]

        gamma = input_dict[eval_policy]["gamma"]
        discount = np.full(self.step_per_trajectory, gamma).cumprod() / gamma

        behavior_policy_reward = self.behavior_policy_reward[behavior_policy]
        behavior_policy_value = (discount[np.newaxis, :] * behavior_policy_reward).sum(
            axis=1
        ).mean() + 1e-10  # to avoid zero division

        policy_value = np.sort(policy_value)[::-1]
        relative_policy_value = policy_value / behavior_policy_value

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

    def _select_by_policy_value(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
    ):
        """Rank the candidate policies by their estimated policy values.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        return_true_values: bool, default=False
            Whether to return the true policy value and corresponding ranking of the candidate policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            mean-squared-error, rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None (>= 0)
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [estimator_name][
                    estimated_ranking,
                    estimated_policy_value,
                    estimated_relative_policy_value,
                    true_ranking,
                    true_policy_value,
                    true_relative_policy_value,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_policy_value: list of float
                Estimated policy value of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_relative_policy_value: list of float
                Estimated relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_policy_value: list of float
                True policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict when return_by_dataframe is `True`.

            true_relative_policy_value: list of float
                True relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: float
                Mean-squared-error of the estimators calculated across candidate evaluation policies.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            regret: tuple of float and int
                Regret@k and k.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df when return_by_dataframe is `True`.

            safety_threshold: float
                A policy whose policy value is below the given threshold is to be considered unsafe.

        """
        behavior_policy_name = list(input_dict.values())[0]["behavior_policy"]
        dataset_id = list(input_dict.values())[0]["dataset_id"]
        gamma = list(input_dict.values())[0]["gamma"]

        discount = np.full(self.step_per_trajectory, gamma).cumprod() / gamma
        behavior_policy_reward = self.behavior_policy_reward[behavior_policy_name]

        behavior_policy_value = (discount[np.newaxis, :] * behavior_policy_reward).sum(
            axis=1
        ).mean() + 1e-10  # to avoid zero division

        if safety_threshold is None:
            if relative_safety_criteria is None:
                safety_threshold = 0.0
            else:
                safety_threshold = relative_safety_criteria * behavior_policy_value

        estimated_policy_value_dict = self.ope.estimate_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        ground_truth_dict = self.obtain_true_selection_result(input_dict)
        true_ranking = ground_truth_dict["ranking"]
        true_policy_value = ground_truth_dict["policy_value"]

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = {}
        for i, estimator in enumerate(compared_estimators):
            estimated_policy_value_ = np.zeros(n_policies)
            true_policy_value_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_policy_value_[j] = estimated_policy_value_dict[eval_policy][
                    estimator
                ]
                true_policy_value_[j] = true_policy_value[j]

            estimated_ranking_index_ = np.argsort(estimated_policy_value_)[::-1]
            true_ranking_index_ = np.argsort(true_policy_value_)[::-1]

            estimated_ranking = [
                candidate_policy_names[estimated_ranking_index_[i]]
                for i in range(n_policies)
            ]
            estimated_policy_value = np.sort(estimated_policy_value_)[::-1]
            estimated_relative_policy_value = (
                estimated_policy_value / behavior_policy_value
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

                true_safety = true_policy_value >= safety_threshold
                estimated_safety = estimated_policy_value_ >= safety_threshold

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
            }
            if return_true_values:
                ops_dict[estimator]["true_ranking"] = true_ranking_index_[
                    estimated_ranking_index_
                ]
                ops_dict[estimator]["true_policy_value"] = true_policy_value_[
                    estimated_ranking_index_
                ]
                ops_dict[estimator]["true_relative_policy_value"] = (
                    true_policy_value_[estimated_ranking_index_] / behavior_policy_value
                )
            if return_metrics:
                ops_dict[estimator]["mean_squared_error"] = mse
                ops_dict[estimator]["rank_correlation"] = rankcorr
                ops_dict[estimator]["regret"] = (regret, top_k_in_eval_metrics)
                ops_dict[estimator]["type_i_error_rate"] = type_i_error_rate
                ops_dict[estimator]["type_ii_error_rate"] = type_ii_error_rate
                ops_dict[estimator]["safety_threshold"] = safety_threshold

        if return_by_dataframe:
            ranking_df_dict = defaultdict(pd.DataFrame)

            for i, estimator in enumerate(compared_estimators):
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

                if return_true_values:
                    ranking_df_["true_ranking"] = ops_dict[estimator]["true_ranking"]
                    ranking_df_["true_policy_value"] = ops_dict[estimator][
                        "true_policy_value"
                    ]
                    ranking_df_["true_relative_policy_value"] = ops_dict[estimator][
                        "true_relative_policy_value"
                    ]

                ranking_df_dict[estimator] = ranking_df_

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

            if return_metrics:
                (
                    mse,
                    rankcorr,
                    pvalue,
                    regret,
                    type_i,
                    type_ii,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for i, estimator in enumerate(compared_estimators):
                    mse.append(ops_dict[estimator]["mean_squared_error"])
                    rankcorr.append(ops_dict[estimator]["rank_correlation"][0])
                    pvalue.append(ops_dict[estimator]["rank_correlation"][1])
                    regret.append(ops_dict[estimator]["regret"][0])
                    type_i.append(ops_dict[estimator]["type_i_error_rate"])
                    type_ii.append(ops_dict[estimator]["type_ii_error_rate"])

                metric_df = pd.DataFrame()
                metric_df["estimator"] = compared_estimators
                metric_df["mean_squared_error"] = mse
                metric_df["rank_correlation"] = rankcorr
                metric_df["pvalue"] = pvalue
                metric_df[f"regret@{top_k_in_eval_metrics}"] = regret
                metric_df["type_i_error_rate"] = type_i
                metric_df["type_ii_error_rate"] = type_ii

            dfs = (ranking_df_dict, metric_df) if return_metrics else ranking_df_dict

        return dfs if return_by_dataframe else ops_dict

    def _select_by_policy_value_via_cumulative_distribution_ope(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
    ):
        """Rank the candidate policies by their estimated policy value via cumulative distribution OPE methods.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        return_true_values: bool, default=False
            Whether to return the true policy value and corresponding ranking of the candidate policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            mean-squared-error, rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None (>= 0)
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [estimator_name][
                    estimated_ranking,
                    estimated_policy_value,
                    estimated_relative_policy_value,
                    true_ranking,
                    true_policy_value,
                    true_relative_policy_value,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_policy_value: list of float
                Estimated policy value of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_relative_policy_value: list of float
                Estimated relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_policy_value: list of float
                True policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_relative_policy_value: list of float
                True relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: float
                Mean-squared-error of the estimators calculated across candidate evaluation policies.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df when return_by_dataframe is `True`.

            regret: tuple of float and int
                Regret@k and k.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df when return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df when return_by_dataframe is `True`.

            safety_threshold: float
                A policy whose policy value is below the given threshold is to be considered unsafe.

        """
        behavior_policy_name = list(input_dict.values())[0]["behavior_policy"]
        dataset_id = list(input_dict.values())[0]["dataset_id"]
        gamma = list(input_dict.values())[0]["gamma"]

        discount = np.full(self.step_per_trajectory, gamma).cumprod() / gamma
        behavior_policy_reward = self.behavior_policy_reward[behavior_policy_name]

        behavior_policy_value = (discount[np.newaxis, :] * behavior_policy_reward).sum(
            axis=1
        ).mean() + 1e-10  # to avoid zero division

        if safety_threshold is None:
            if relative_safety_criteria is None:
                safety_threshold = 0.0
            else:
                safety_threshold = relative_safety_criteria * behavior_policy_value

        estimated_policy_value_dict = self.cumulative_distribution_ope.estimate_mean(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        ground_truth_dict = self.obtain_true_selection_result(input_dict)
        true_ranking = ground_truth_dict["ranking"]
        true_policy_value = ground_truth_dict["policy_value"]

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = {}
        for i, estimator in enumerate(compared_estimators):
            estimated_policy_value_ = np.zeros(n_policies)
            true_policy_value_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_policy_value_[j] = estimated_policy_value_dict[eval_policy][
                    estimator
                ]
                true_policy_value_[j] = true_policy_value[j]

            estimated_ranking_index_ = np.argsort(estimated_policy_value_)[::-1]
            true_ranking_index_ = np.argsort(true_policy_value_)[::-1]

            estimated_ranking = [
                candidate_policy_names[estimated_ranking_index_[i]]
                for i in range(n_policies)
            ]
            estimated_policy_value = np.sort(estimated_policy_value_)[::-1]
            estimated_relative_policy_value = (
                estimated_policy_value / behavior_policy_value
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

                true_safety = true_policy_value >= safety_threshold
                estimated_safety = estimated_policy_value_ >= safety_threshold

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
            }
            if return_true_values:
                ops_dict[estimator]["true_ranking"] = true_ranking_index_[
                    estimated_ranking_index_
                ]
                ops_dict[estimator]["true_policy_value"] = true_policy_value_[
                    estimated_ranking_index_
                ]
                ops_dict[estimator]["true_relative_policy_value"] = (
                    true_policy_value_[estimated_ranking_index_] / behavior_policy_value
                )
            if return_metrics:
                ops_dict[estimator]["mean_squared_error"] = mse
                ops_dict[estimator]["rank_correlation"] = rankcorr
                ops_dict[estimator]["regret"] = (regret, top_k_in_eval_metrics)
                ops_dict[estimator]["type_i_error_rate"] = type_i_error_rate
                ops_dict[estimator]["type_ii_error_rate"] = type_ii_error_rate
                ops_dict[estimator]["safety_threshold"] = safety_threshold

        if return_by_dataframe:
            ranking_df_dict = defaultdict(pd.DataFrame)

            for i, estimator in enumerate(compared_estimators):
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

                if return_true_values:
                    ranking_df_["true_ranking"] = ops_dict[estimator]["true_ranking"]
                    ranking_df_["true_policy_value"] = ops_dict[estimator][
                        "true_policy_value"
                    ]
                    ranking_df_["true_relative_policy_value"] = ops_dict[estimator][
                        "true_relative_policy_value"
                    ]

                ranking_df_dict[estimator] = ranking_df_

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

            if return_metrics:
                (
                    mse,
                    rankcorr,
                    pvalue,
                    regret,
                    type_i,
                    type_ii,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for i, estimator in enumerate(compared_estimators):
                    mse.append(ops_dict[estimator]["mean_squared_error"])
                    rankcorr.append(ops_dict[estimator]["rank_correlation"][0])
                    pvalue.append(ops_dict[estimator]["rank_correlation"][1])
                    regret.append(ops_dict[estimator]["regret"][0])
                    type_i.append(ops_dict[estimator]["type_i_error_rate"])
                    type_ii.append(ops_dict[estimator]["type_ii_error_rate"])

                metric_df = pd.DataFrame()
                metric_df["estimator"] = compared_estimators
                metric_df["mean_squared_error"] = mse
                metric_df["rank_correlation"] = rankcorr
                metric_df["pvalue"] = pvalue
                metric_df[f"regret@{top_k_in_eval_metrics}"] = regret
                metric_df["type_i_error_rate"] = type_i
                metric_df["type_ii_error_rate"] = type_ii

            dfs = (ranking_df_dict, metric_df) if return_metrics else ranking_df_dict

        return dfs if return_by_dataframe else ops_dict

    def _select_by_policy_value_lower_bound(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
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

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        return_true_values: bool, default=False
            Whether to return the true policy value and corresponding ranking of the candidate policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None (>= 0)
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

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

            .. code-block:: python

                key: [ci][estimator_name][
                    estimated_ranking,
                    estimated_policy_value_lower_bound,
                    estimated_relative_policy_value_lower_bound,
                    true_ranking,
                    true_policy_value,
                    true_relative_policy_value,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value lower bound.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_policy_value_lower_bound: list of float
                Estimated policy value lower bound of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_relative_policy_value_lower_bound: list of float
                Estimated relative policy value lower bound of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_policy_value: list of float
                True policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_relative_policy_value: list of float
                True relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: None
                This is for API consistency.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            regret: tuple of float and int
                Regret@k and k.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            safety_threshold: float
                A policy whose policy value is below the given threshold is to be considered unsafe.

        """
        ground_truth_dict = self.obtain_true_selection_result(input_dict)
        true_ranking = ground_truth_dict["ranking"]
        true_policy_value = ground_truth_dict["policy_value"]

        behavior_policy_name = list(input_dict.values())[0]["behavior_policy"]
        dataset_id = list(input_dict.values())[0]["dataset_id"]
        gamma = list(input_dict.values())[0]["gamma"]

        discount = np.full(self.step_per_trajectory, gamma).cumprod() / gamma
        behavior_policy_reward = self.behavior_policy_reward[behavior_policy_name]

        behavior_policy_value = (discount[np.newaxis, :] * behavior_policy_reward).sum(
            axis=1
        ).mean() + 1e-10  # to avoid zero division

        if safety_threshold is None:
            if relative_safety_criteria is None:
                safety_threshold = 0.0
            else:
                safety_threshold = relative_safety_criteria * behavior_policy_value

        candidate_policy_names = (
            true_ranking if return_metrics else list(input_dict.keys())
        )
        n_policies = len(candidate_policy_names)

        ops_dict = defaultdict(dict)
        for ci in cis:
            estimated_policy_value_interval_dict = self.ope.estimate_intervals(
                input_dict,
                compared_estimators=compared_estimators,
                behavior_policy_name=behavior_policy_name,
                dataset_id=dataset_id,
                alpha=alpha,
                ci=ci,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

            for i, estimator in enumerate(compared_estimators):
                estimated_policy_value_lower_bound_ = np.zeros(n_policies)
                true_policy_value_ = np.zeros(n_policies)
                for j, eval_policy in enumerate(candidate_policy_names):
                    estimated_policy_value_lower_bound_[
                        j
                    ] = estimated_policy_value_interval_dict[eval_policy][estimator][
                        f"{100 * (1. - alpha)}% CI (lower)"
                    ]
                    true_policy_value_[j] = true_policy_value[j]

                estimated_ranking_index_ = np.argsort(
                    estimated_policy_value_lower_bound_
                )[::-1]
                true_ranking_index_ = np.argsort(true_policy_value_)[::-1]

                estimated_ranking = [
                    candidate_policy_names[estimated_ranking_index_[i]]
                    for i in range(n_policies)
                ]
                estimated_policy_value_lower_bound = np.sort(
                    estimated_policy_value_lower_bound_
                )[::-1]
                estimated_relative_policy_value_lower_bound = (
                    estimated_policy_value_lower_bound / behavior_policy_value
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

                    true_safety = true_policy_value >= safety_threshold
                    estimated_safety = (
                        estimated_policy_value_lower_bound_ >= safety_threshold
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
                }
                if return_true_values:
                    ops_dict[ci][estimator]["true_ranking"] = true_ranking_index_[
                        estimated_ranking_index_
                    ]
                    ops_dict[ci][estimator]["true_policy_value"] = true_policy_value_[
                        estimated_ranking_index_
                    ]
                    ops_dict[ci][estimator]["true_relative_policy_value"] = (
                        true_policy_value_[estimated_ranking_index_]
                        / behavior_policy_value
                    )
                if return_metrics:
                    ops_dict[ci][estimator]["mean_squared_error"] = None
                    ops_dict[ci][estimator]["rank_correlation"] = rankcorr
                    ops_dict[ci][estimator]["regret"] = (regret, top_k_in_eval_metrics)
                    ops_dict[ci][estimator]["type_i_error_rate"] = type_i_error_rate
                    ops_dict[ci][estimator]["type_ii_error_rate"] = type_ii_error_rate
                    ops_dict[ci][estimator]["safety_threshold"] = safety_threshold

        ops_dict = defaultdict_to_dict(ops_dict)

        if return_by_dataframe:
            ranking_df_dict = defaultdict(lambda: defaultdict(pd.DataFrame))

            for ci in cis:
                for i, estimator in enumerate(compared_estimators):
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

                    if return_true_values:
                        ranking_df_["true_ranking"] = ops_dict[ci][estimator][
                            "true_ranking"
                        ]
                        ranking_df_["true_policy_value"] = ops_dict[ci][estimator][
                            "true_policy_value"
                        ]
                        ranking_df_["true_relative_policy_value"] = ops_dict[ci][
                            estimator
                        ]["true_relative_policy_value"]

                    ranking_df_dict[ci][estimator] = ranking_df_

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

        if return_metrics:
            (
                ci_,
                estimator_,
                rankcorr,
                pvalue,
                regret,
                type_i,
                type_ii,
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for ci in cis:
                for i, estimator in enumerate(compared_estimators):
                    ci_.append(ci)
                    estimator_.append(estimator)
                    rankcorr.append(ops_dict[ci][estimator]["rank_correlation"][0])
                    pvalue.append(ops_dict[ci][estimator]["rank_correlation"][1])
                    regret.append(ops_dict[ci][estimator]["regret"][0])
                    type_i.append(ops_dict[ci][estimator]["type_i_error_rate"])
                    type_ii.append(ops_dict[ci][estimator]["type_ii_error_rate"])

            metric_df = pd.DataFrame()
            metric_df["ci"] = ci_
            metric_df["estimator"] = estimator_
            metric_df["mean_squared_error"] = np.nan
            metric_df["rank_correlation"] = rankcorr
            metric_df["pvalue"] = pvalue
            metric_df[f"regret@{top_k_in_eval_metrics}"] = regret
            metric_df["type_i_error_rate"] = type_i
            metric_df["type_ii_error_rate"] = type_ii

            dfs = (ranking_df_dict, metric_df) if return_metrics else ranking_df_dict

        return dfs if return_by_dataframe else ops_dict

    def _select_by_lower_quartile(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        alpha: float = 0.05,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank the candidate policies by their estimated lower quartile of the trajectory-wise reward.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 0.5]`.

        return_true_values: bool, default=False
            Whether to return the true lower quartile of the trajectory-wise reward
            and corresponding ranking of the candidate evaluation policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        safety_threshold: float, default=0.0 (>= 0)
            The lower quartile required to be considered a safe policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [estimator_name][
                    estimated_ranking,
                    estimated_lower_quartile,
                    true_ranking,
                    true_lower_quartile,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated lower quartile of the trajectory-wise reward.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_lower_quartile: list of float
                Estimated lower quartile of the trajectory-wise reward of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) lower quartile of the trajectory-wise reward of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_lower_quartile: list of float
                True lower quartile of the trajectory-wise reward of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: float
                Mean-squared-error of the estimated lower quartile of the trajectory-wise reward.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            regret: None
                This is for API consistency.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            safety_threshold: float
                The lower quartile required to be considered a safe policy.

        """
        behavior_policy_name = list(input_dict.values())[0]["behavior_policy"]
        dataset_id = list(input_dict.values())[0]["dataset_id"]

        estimated_interquartile_range_dict = (
            self.cumulative_distribution_ope.estimate_interquartile_range(
                input_dict,
                compared_estimators=compared_estimators,
                behavior_policy_name=behavior_policy_name,
                dataset_id=dataset_id,
                alpha=alpha,
            )
        )

        ground_truth_dict = self.obtain_true_selection_result(
            input_dict,
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
        for i, estimator in enumerate(compared_estimators):
            estimated_lower_quartile_ = np.zeros(n_policies)
            true_lower_quartile_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_lower_quartile_[j] = estimated_interquartile_range_dict[
                    eval_policy
                ][estimator][f"{100 * (1. - alpha)}% quartile (lower)"]
                true_lower_quartile_[j] = true_lower_quartile[j]

            estimated_ranking_index_ = np.argsort(estimated_lower_quartile_)[::-1]
            true_ranking_index_ = np.argsort(true_lower_quartile_)[::-1]

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
            }
            if return_true_values:
                ops_dict[estimator]["true_ranking"] = true_ranking_index_[
                    estimated_ranking_index_
                ]
                ops_dict[estimator]["true_lower_quartile"] = true_lower_quartile_[
                    estimated_ranking_index_
                ]
            if return_metrics:
                ops_dict[estimator]["mean_squared_error"] = mse
                ops_dict[estimator]["rank_correlation"] = rankcorr
                ops_dict[estimator]["regret"] = None
                ops_dict[estimator]["type_i_error_rate"] = type_i_error_rate
                ops_dict[estimator]["type_ii_error_rate"] = type_ii_error_rate
                ops_dict[estimator]["safety_threshold"] = safety_threshold

        if return_by_dataframe:
            ranking_df_dict = defaultdict(pd.DataFrame)

            for i, estimator in enumerate(compared_estimators):
                ranking_df_ = pd.DataFrame()
                ranking_df_["estimated_ranking"] = ops_dict[estimator][
                    "estimated_ranking"
                ]
                ranking_df_["estimated_lower_quartile"] = ops_dict[estimator][
                    "estimated_lower_quartile"
                ]

                if return_true_values:
                    ranking_df_["true_ranking"] = ops_dict[estimator]["true_ranking"]
                    ranking_df_["true_lower_quartile"] = ops_dict[estimator][
                        "true_lower_quartile"
                    ]

                ranking_df_dict[estimator] = ranking_df_

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

            if return_metrics:
                (
                    mse,
                    rankcorr,
                    pvalue,
                    type_i,
                    type_ii,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for i, estimator in enumerate(compared_estimators):
                    mse.append(ops_dict[estimator]["mean_squared_error"])
                    rankcorr.append(ops_dict[estimator]["rank_correlation"][0])
                    pvalue.append(ops_dict[estimator]["rank_correlation"][1])
                    type_i.append(ops_dict[estimator]["type_i_error_rate"])
                    type_ii.append(ops_dict[estimator]["type_ii_error_rate"])

                metric_df = pd.DataFrame()
                metric_df["estimator"] = compared_estimators
                metric_df["mean_squared_error"] = mse
                metric_df["rank_correlation"] = rankcorr
                metric_df["pvalue"] = pvalue
                metric_df["regret"] = np.nan
                metric_df["type_i_error_rate"] = type_i
                metric_df["type_ii_error_rate"] = type_ii

            dfs = (ranking_df_dict, metric_df) if return_metrics else ranking_df_dict

        return dfs if return_by_dataframe else ops_dict

    def _select_by_conditional_value_at_risk(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        alpha: float = 0.05,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank the candidate policies by their estimated conditional value at risk.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 1]`.

        return_true_values: bool, default=False
            Whether to return the true conditional value at risk
            and corresponding ranking of the candidate evaluation policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        safety_threshold: float, default=0.0 (>= 0)
            The conditional value at risk required to be considered a safe policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [estimator_name][
                    estimated_ranking,
                    estimated_conditional_value_at_risk,
                    true_ranking,
                    true_conditional_value_at_risk,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated conditional value at risk.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_conditional_value_at_risk: list of float
                Estimated conditional value at risk of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) conditional value at risk of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_conditional_value_at_risk: list of float
                True conditional value at risk of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: float
                Mean-squared-error of the estimated conditional value at risk.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple or float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            regret: None
                This is for API consistency.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is True`.

            safety_threshold: float
                The conditional value at risk required to be considered a safe policy.

        """
        behavior_policy_name = list(input_dict.values())[0]["behavior_policy"]
        dataset_id = list(input_dict.values())[0]["dataset_id"]

        estimated_cvar_dict = (
            self.cumulative_distribution_ope.estimate_conditional_value_at_risk(
                input_dict,
                compared_estimators=compared_estimators,
                behavior_policy_name=behavior_policy_name,
                dataset_id=dataset_id,
                alphas=alpha,
            )
        )

        ground_truth_dict = self.obtain_true_selection_result(
            input_dict,
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
        for i, estimator in enumerate(compared_estimators):
            estimated_cvar_ = np.zeros(n_policies)
            true_cvar_ = np.zeros(n_policies)
            for j, eval_policy in enumerate(candidate_policy_names):
                estimated_cvar_[j] = estimated_cvar_dict[eval_policy][estimator]
                true_cvar_[j] = true_cvar[j]

            estimated_ranking_index_ = np.argsort(estimated_cvar_)[::-1]
            true_ranking_index_ = np.argsort(true_cvar_)[::-1]

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
            }
            if return_true_values:
                ops_dict[estimator]["true_ranking"] = true_ranking_index_[
                    estimated_ranking_index_
                ]
                ops_dict[estimator]["true_conditional_value_at_risk"] = true_cvar_[
                    estimated_ranking_index_
                ]
            if return_metrics:
                ops_dict[estimator]["mean_squared_error"] = mse
                ops_dict[estimator]["rank_correlation"] = rankcorr
                ops_dict[estimator]["regret"] = None
                ops_dict[estimator]["type_i_error_rate"] = type_i_error_rate
                ops_dict[estimator]["type_ii_error_rate"] = type_ii_error_rate
                ops_dict[estimator]["safety_threshold"] = safety_threshold

        if return_by_dataframe:
            ranking_df_dict = defaultdict(pd.DataFrame)

            for i, estimator in enumerate(compared_estimators):
                ranking_df_ = pd.DataFrame()
                ranking_df_["estimated_ranking"] = ops_dict[estimator][
                    "estimated_ranking"
                ]
                ranking_df_["estimated_conditional_value_at_risk"] = ops_dict[
                    estimator
                ]["estimated_conditional_value_at_risk"]

                if return_true_values:
                    ranking_df_["true_ranking"] = ops_dict[estimator]["true_ranking"]
                    ranking_df_["true_conditional_value_at_risk"] = ops_dict[estimator][
                        "true_conditional_value_at_risk"
                    ]

                ranking_df_dict[estimator] = ranking_df_

            ranking_df_dict = defaultdict_to_dict(ranking_df_dict)

            if return_metrics:
                (
                    mse,
                    rankcorr,
                    pvalue,
                    type_i,
                    type_ii,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for i, estimator in enumerate(compared_estimators):
                    mse.append(ops_dict[estimator]["mean_squared_error"])
                    rankcorr.append(ops_dict[estimator]["rank_correlation"][0])
                    pvalue.append(ops_dict[estimator]["rank_correlation"][1])
                    type_i.append(ops_dict[estimator]["type_i_error_rate"])
                    type_ii.append(ops_dict[estimator]["type_ii_error_rate"])

                metric_df = pd.DataFrame()
                metric_df["estimator"] = compared_estimators
                metric_df["mean_squared_error"] = mse
                metric_df["rank_correlation"] = rankcorr
                metric_df["pvalue"] = pvalue
                metric_df["regret"] = np.nan
                metric_df["type_i_error_rate"] = type_i
                metric_df["type_ii_error_rate"] = type_ii

            dfs = (ranking_df_dict, metric_df) if return_metrics else ranking_df_dict

        return dfs if return_by_dataframe else ops_dict

    def obtain_true_selection_result(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        return_variance: bool = False,
        return_lower_quartile: bool = False,
        return_conditional_value_at_risk: bool = False,
        return_by_dataframe: bool = False,
        quartile_alpha: float = 0.05,
        cvar_alpha: float = 0.05,
    ):
        """Obtain the oracle selection result based on the ground-truth policy value.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        return_variance: bool, default=False
            Whether to return the variance or not.

        return_lower_quartile: bool. default=False
            Whether to return the lower interquartile or not.

        return_conditional_value_at_risk: bool, default=False
            Whether to return the conditional value at risk or not.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        quartile_alpha: float, default=0.05
            Proportion of the shaded region of the interquartile range.

        cvar_alpha: float, default=0.05
            Proportion of the shaded region of the conditional value at risk.

        Return
        -------
        ground_truth_dict/ground_truth_df: dict or dataframe (, list of dict or dataframe)
            Dictionary/dataframe containing the following ground-truth (on-policy) metrics.

            .. code-block:: python

                key: [
                    ranking,
                    policy_value,
                    relative_policy_value,
                    variance,
                    ranking_by_lower_quartile,
                    lower_quartile,
                    ranking_by_conditional_value_at_risk,
                    conditional_value_at_risk,
                    parameters,  # only when return_by_dataframe == False
                ]

            ranking: list of str
                Name of the candidate policies sorted by the ground-truth policy value.

            policy_value: list of float
                Ground-truth policy value of the candidate policies (sorted by ranking).

            relative_policy_value: list of float
                Ground-truth relative policy value of the candidate policies compared to the behavior policy (sorted by ranking).

            variance: list of float
                Ground-truth variance of the trajectory-wise reward of the candidate policies (sorted by ranking).
                If return_variance is `False`, `None` is recorded.

            ranking_by_lower_quartile: list of str
                Name of the candidate policies sorted by the ground-truth lower quartile of the trajectory-wise reward.
                If return_lower_quartile is `False`, `None` is recorded.

            lower_quartile: list of float
                Ground-truth lower quartile of the candidate policies (sorted by ranking_by_lower_quartile).
                If return_lower_quartile is `False`, `None` is recorded.

            ranking_by_conditional_value_at_risk: list of str
                Name of the candidate policies sorted by the ground-truth conditional value at risk.
                If return_conditional_value_at_risk is `False`, `None` is recorded.

            conditional_value_at_risk: list of float
                Ground-truth conditional value at risk of the candidate policies (sorted by ranking_by_conditional_value_at_risk).
                If return_conditional_value_at_risk is `False`, `None` is recorded.

            parameters: dict
                Dictionary containing quartile_alpha, and cvar_alpha.
                If return_by_dataframe is `True`, parameters will not be returned.

        """
        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None and dataset_id is None:
                ground_truth = defaultdict(list)

                for (
                    behavior_policy,
                    n_datasets,
                ) in input_dict.n_datasets.items():
                    for dataset_id_ in range(n_datasets):
                        input_dict_ = input_dict.get(
                            behavior_policy_name=behavior_policy,
                            dataset_id=dataset_id_,
                        )
                        ground_truth_ = self._obtain_true_selection_result(
                            input_dict_,
                            return_variance=return_variance,
                            return_lower_quartile=return_lower_quartile,
                            return_conditional_value_at_risk=return_conditional_value_at_risk,
                            return_by_dataframe=return_by_dataframe,
                            quartile_alpha=quartile_alpha,
                            cvar_alpha=cvar_alpha,
                        )
                        ground_truth[behavior_policy].append(ground_truth_)

                ground_truth = defaultdict_to_dict(ground_truth)

            elif behavior_policy_name is None and dataset_id is not None:
                ground_truth = {}
                for behavior_policy in input_dict.behavior_policy_names:
                    input_dict_ = input_dict.get(
                        behavior_policy_name=behavior_policy, dataset_id=dataset_id
                    )
                    ground_truth_ = self._obtain_true_selection_result(
                        input_dict_,
                        return_variance=return_variance,
                        return_lower_quartile=return_lower_quartile,
                        return_conditional_value_at_risk=return_conditional_value_at_risk,
                        return_by_dataframe=return_by_dataframe,
                        quartile_alpha=quartile_alpha,
                        cvar_alpha=cvar_alpha,
                    )
                    ground_truth[behavior_policy] = ground_truth_

            elif behavior_policy_name is not None and dataset_id is None:
                ground_truth = []
                for dataset_id_ in range(input_dict.n_datasets[behavior_policy_name]):
                    input_dict_ = input_dict.get(
                        behavior_policy_name=behavior_policy_name,
                        dataset_id=dataset_id_,
                    )
                    ground_truth_ = self._obtain_true_selection_result(
                        input_dict_,
                        return_variance=return_variance,
                        return_lower_quartile=return_lower_quartile,
                        return_conditional_value_at_risk=return_conditional_value_at_risk,
                        return_by_dataframe=return_by_dataframe,
                        quartile_alpha=quartile_alpha,
                        cvar_alpha=cvar_alpha,
                    )
                    ground_truth.append(ground_truth_)

            else:
                input_dict_ = input_dict.get(
                    behavior_policy_name=behavior_policy_name, dataset_id=dataset_id
                )
                ground_truth = self._obtain_true_selection_result(
                    input_dict_,
                    return_variance=return_variance,
                    return_lower_quartile=return_lower_quartile,
                    return_conditional_value_at_risk=return_conditional_value_at_risk,
                    return_by_dataframe=return_by_dataframe,
                    quartile_alpha=quartile_alpha,
                    cvar_alpha=cvar_alpha,
                )
        else:
            ground_truth = self._obtain_true_selection_result(
                input_dict,
                return_variance=return_variance,
                return_lower_quartile=return_lower_quartile,
                return_conditional_value_at_risk=return_conditional_value_at_risk,
                return_by_dataframe=return_by_dataframe,
                quartile_alpha=quartile_alpha,
                cvar_alpha=cvar_alpha,
            )

        return ground_truth

    def select_by_policy_value(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
    ):
        """Rank the candidate policies by their estimated policy values.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        return_true_values: bool, default=False
            Whether to return the true policy value and corresponding ranking of the candidate policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            mean-squared-error, rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None (>= 0)
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe (, list of dict or dataframe)
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [estimator_name][
                    estimated_ranking,
                    estimated_policy_value,
                    estimated_relative_policy_value,
                    true_ranking,
                    true_policy_value,
                    true_relative_policy_value,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_policy_value: list of float
                Estimated policy value of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_relative_policy_value: list of float
                Estimated relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_policy_value: list of float
                True policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict when return_by_dataframe is `True`.

            true_relative_policy_value: list of float
                True relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: float
                Mean-squared-error of the estimators calculated across candidate evaluation policies.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            regret: tuple of float and int
                Regret@k and k.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df when return_by_dataframe is `True`.

            safety_threshold: float
                A policy whose policy value is below the given threshold is to be considered unsafe.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )

        if self.ope.use_multiple_logged_dataset:
            if isinstance(input_dict, MultipleInputDict):
                if behavior_policy_name is None and dataset_id is None:
                    if (
                        self.ope.multiple_logged_dataset.n_datasets
                        != input_dict.n_datasets
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies and dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = defaultdict(list)
                        metric_df = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_policy_value(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    top_k_in_eval_metrics=top_k_in_eval_metrics,
                                    safety_threshold=safety_threshold,
                                    relative_safety_criteria=relative_safety_criteria,
                                )
                                ranking_df[behavior_policy].append(ops_result_[0])
                                metric_df[behavior_policy].append(ops_result_[1])

                            ops_result = (
                                defaultdict_to_dict(ranking_df),
                                defaultdict_to_dict(metric_df),
                            )

                    else:
                        ops_result = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_policy_value(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    top_k_in_eval_metrics=top_k_in_eval_metrics,
                                    safety_threshold=safety_threshold,
                                    relative_safety_criteria=relative_safety_criteria,
                                )
                                ops_result[behavior_policy].append(ops_result_)

                        ops_result = defaultdict_to_dict(ops_result)

                elif behavior_policy_name is None and dataset_id is not None:
                    if (
                        self.ope.multiple_logged_dataset.behavior_policy_names
                        != input_dict.behavior_policy_names
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = {}
                        metric_df = {}

                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_policy_value(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                            )
                            ranking_df[behavior_policy] = ops_result_[0]
                            metric_df[behavior_policy] = ops_result_[1]

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = {}
                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_policy_value(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                            )
                            ops_result[behavior_policy] = ops_result_

                elif behavior_policy_name is not None and dataset_id is None:
                    if (
                        self.ope.multiple_logged_dataset.n_datasets[
                            behavior_policy_name
                        ]
                        != input_dict.n_datasets[behavior_policy_name]
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = []
                        metric_df = []

                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy_name,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_policy_value(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                            )
                            ranking_df.append(ops_result_[0])
                            metric_df.append(ops_result_[1])

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = []
                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy_name,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_policy_value(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                            )
                            ops_result.append(ops_result_)

                else:
                    input_dict_ = input_dict.get(
                        behavior_policy_name=behavior_policy_name, dataset_id=dataset_id
                    )
                    ops_result = self._select_by_policy_value(
                        input_dict_,
                        compared_estimators=compared_estimators,
                        return_true_values=return_true_values,
                        return_metrics=return_metrics,
                        return_by_dataframe=return_by_dataframe,
                        top_k_in_eval_metrics=top_k_in_eval_metrics,
                        safety_threshold=safety_threshold,
                        relative_safety_criteria=relative_safety_criteria,
                    )

            else:
                ops_result = self._select_by_policy_value(
                    input_dict,
                    compared_estimators=compared_estimators,
                    return_true_values=return_true_values,
                    return_metrics=return_metrics,
                    return_by_dataframe=return_by_dataframe,
                    top_k_in_eval_metrics=top_k_in_eval_metrics,
                    safety_threshold=safety_threshold,
                    relative_safety_criteria=relative_safety_criteria,
                )

        else:
            if isinstance(input_dict, MultipleInputDict):
                raise ValueError(
                    "when using LoggedDataset, please use InputDict instead of MultipleInputDict"
                )

            ops_result = self._select_by_policy_value(
                input_dict,
                compared_estimators=compared_estimators,
                return_true_values=return_true_values,
                return_metrics=return_metrics,
                return_by_dataframe=return_by_dataframe,
                top_k_in_eval_metrics=top_k_in_eval_metrics,
                safety_threshold=safety_threshold,
                relative_safety_criteria=relative_safety_criteria,
            )

        return ops_result

    def select_by_policy_value_via_cumulative_distribution_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
    ):
        """Rank the candidate policies by their estimated policy value via cumulative distribution OPE methods.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        return_true_values: bool, default=False
            Whether to return the true policy value and corresponding ranking of the candidate policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            mean-squared-error, rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None (>= 0)
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe (, list of dict or dataframe)
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [estimator_name][
                    estimated_ranking,
                    estimated_policy_value,
                    estimated_relative_policy_value,
                    true_ranking,
                    true_policy_value,
                    true_relative_policy_value,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_policy_value: list of float
                Estimated policy value of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_relative_policy_value: list of float
                Estimated relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_policy_value: list of float
                True policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_relative_policy_value: list of float
                True relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: float
                Mean-squared-error of the estimators calculated across candidate evaluation policies.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df when return_by_dataframe is `True`.

            regret: tuple of float and int
                Regret@k and k.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df when return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df when return_by_dataframe is `True`.

            safety_threshold: float
                A policy whose policy value is below the given threshold is to be considered unsafe.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )

        if self.cumulative_distribution_ope.use_multiple_logged_dataset:
            if isinstance(input_dict, MultipleInputDict):
                if behavior_policy_name is None and dataset_id is None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.n_datasets
                        != input_dict.n_datasets
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies and dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = defaultdict(list)
                        metric_df = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_policy_value_via_cumulative_distribution_ope(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    top_k_in_eval_metrics=top_k_in_eval_metrics,
                                    safety_threshold=safety_threshold,
                                    relative_safety_criteria=relative_safety_criteria,
                                )
                                ranking_df[behavior_policy].append(ops_result_[0])
                                metric_df[behavior_policy].append(ops_result_[1])

                            ops_result = (
                                defaultdict_to_dict(ranking_df),
                                defaultdict_to_dict(metric_df),
                            )

                    else:
                        ops_result = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_policy_value_via_cumulative_distribution_ope(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    top_k_in_eval_metrics=top_k_in_eval_metrics,
                                    safety_threshold=safety_threshold,
                                    relative_safety_criteria=relative_safety_criteria,
                                )
                                ops_result[behavior_policy].append(ops_result_)

                        ops_result = defaultdict_to_dict(ops_result)

                elif behavior_policy_name is None and dataset_id is not None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.behavior_policy_names
                        != input_dict.behavior_policy_names
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = {}
                        metric_df = {}

                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_policy_value_via_cumulative_distribution_ope(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                            )
                            ranking_df[behavior_policy] = ops_result_[0]
                            metric_df[behavior_policy] = ops_result_[1]

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = {}
                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_policy_value_via_cumulative_distribution_ope(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                            )
                            ops_result[behavior_policy] = ops_result_

                elif behavior_policy_name is not None and dataset_id is None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.n_datasets[
                            behavior_policy_name
                        ]
                        != input_dict.n_datasets[behavior_policy_name]
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = []
                        metric_df = []

                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy_name,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_policy_value_via_cumulative_distribution_ope(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                            )
                            ranking_df.append(ops_result_[0])
                            metric_df.append(ops_result_[1])

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = []
                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy_name,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_policy_value_via_cumulative_distribution_ope(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                            )
                            ops_result.append(ops_result_)

                else:
                    input_dict_ = input_dict.get(
                        behavior_policy_name=behavior_policy_name, dataset_id=dataset_id
                    )
                    ops_result = (
                        self._select_by_policy_value_via_cumulative_distribution_ope(
                            input_dict_,
                            compared_estimators=compared_estimators,
                            return_true_values=return_true_values,
                            return_metrics=return_metrics,
                            return_by_dataframe=return_by_dataframe,
                            top_k_in_eval_metrics=top_k_in_eval_metrics,
                            safety_threshold=safety_threshold,
                            relative_safety_criteria=relative_safety_criteria,
                        )
                    )

            else:
                ops_result = (
                    self._select_by_policy_value_via_cumulative_distribution_ope(
                        input_dict,
                        compared_estimators=compared_estimators,
                        return_true_values=return_true_values,
                        return_metrics=return_metrics,
                        return_by_dataframe=return_by_dataframe,
                        top_k_in_eval_metrics=top_k_in_eval_metrics,
                        safety_threshold=safety_threshold,
                        relative_safety_criteria=relative_safety_criteria,
                    )
                )

        else:
            if isinstance(input_dict, MultipleInputDict):
                raise ValueError(
                    "when using LoggedDataset, please use InputDict instead of MultipleInputDict"
                )

            ops_result = self._select_by_policy_value_via_cumulative_distribution_ope(
                input_dict,
                compared_estimators=compared_estimators,
                return_true_values=return_true_values,
                return_metrics=return_metrics,
                return_by_dataframe=return_by_dataframe,
                top_k_in_eval_metrics=top_k_in_eval_metrics,
                safety_threshold=safety_threshold,
                relative_safety_criteria=relative_safety_criteria,
            )

        return ops_result

    def select_by_policy_value_lower_bound(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        top_k_in_eval_metrics: int = 1,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
        cis: List[str] = ["bootstrap"],
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ):
        """Rank the candidate policies by their estimated policy value lower bound.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        return_true_values: bool, default=False
            Whether to return the true policy value and corresponding ranking of the candidate policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            rank-correlation, regret@k, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        top_k_in_eval_metrics: int, default=1
            How many candidate policies are included in regret@k.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None (>= 0)
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

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
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe (, list of dict or dataframe)
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [ci][estimator_name][
                    estimated_ranking,
                    estimated_policy_value_lower_bound,
                    estimated_relative_policy_value_lower_bound,
                    true_ranking,
                    true_policy_value,
                    true_relative_policy_value,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated policy value lower bound.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_policy_value_lower_bound: list of float
                Estimated policy value lower bound of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_relative_policy_value_lower_bound: list of float
                Estimated relative policy value lower bound of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_policy_value: list of float
                True policy value of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_relative_policy_value: list of float
                True relative policy value of the candidate policies compared to the behavior policy (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: None
                This is for API consistency.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            regret: tuple of float and int
                Regret@k and k.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            safety_threshold: float
                A policy whose policy value is below the given threshold is to be considered unsafe.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )

        if self.ope.use_multiple_logged_dataset:
            if isinstance(input_dict, MultipleInputDict):
                if behavior_policy_name is None and dataset_id is None:
                    if (
                        self.ope.multiple_logged_dataset.n_datasets
                        != input_dict.n_datasets
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies and dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = defaultdict(list)
                        metric_df = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_policy_value_lower_bound(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    top_k_in_eval_metrics=top_k_in_eval_metrics,
                                    safety_threshold=safety_threshold,
                                    relative_safety_criteria=relative_safety_criteria,
                                    cis=cis,
                                    alpha=alpha,
                                    n_bootstrap_samples=n_bootstrap_samples,
                                    random_state=random_state,
                                )
                                ranking_df[behavior_policy].append(ops_result_[0])
                                metric_df[behavior_policy].append(ops_result_[1])

                            ops_result = (
                                defaultdict_to_dict(ranking_df),
                                defaultdict_to_dict(metric_df),
                            )

                    else:
                        ops_result = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_policy_value_lower_bound(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    top_k_in_eval_metrics=top_k_in_eval_metrics,
                                    safety_threshold=safety_threshold,
                                    relative_safety_criteria=relative_safety_criteria,
                                    cis=cis,
                                    alpha=alpha,
                                    n_bootstrap_samples=n_bootstrap_samples,
                                    random_state=random_state,
                                )
                                ops_result[behavior_policy].append(ops_result_)

                        ops_result = defaultdict_to_dict(ops_result)

                elif behavior_policy_name is None and dataset_id is not None:
                    if (
                        self.ope.multiple_logged_dataset.behavior_policy_names
                        != input_dict.behavior_policy_names
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = {}
                        metric_df = {}

                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_policy_value_lower_bound(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                                cis=cis,
                                alpha=alpha,
                                n_bootstrap_samples=n_bootstrap_samples,
                                random_state=random_state,
                            )
                            ranking_df[behavior_policy] = ops_result_[0]
                            metric_df[behavior_policy] = ops_result_[1]

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = {}
                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_policy_value_lower_bound(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                                cis=cis,
                                alpha=alpha,
                                n_bootstrap_samples=n_bootstrap_samples,
                                random_state=random_state,
                            )
                            ops_result[behavior_policy] = ops_result_

                elif behavior_policy_name is not None and dataset_id is None:
                    if (
                        self.ope.multiple_logged_dataset.n_datasets[
                            behavior_policy_name
                        ]
                        != input_dict.n_datasets[behavior_policy_name]
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = []
                        metric_df = []

                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_policy_value_lower_bound(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                                cis=cis,
                                alpha=alpha,
                                n_bootstrap_samples=n_bootstrap_samples,
                                random_state=random_state,
                            )
                            ranking_df.append(ops_result_[0])
                            metric_df.append(ops_result_[1])

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = []
                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy_name,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_policy_value_lower_bound(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                top_k_in_eval_metrics=top_k_in_eval_metrics,
                                safety_threshold=safety_threshold,
                                relative_safety_criteria=relative_safety_criteria,
                                cis=cis,
                                alpha=alpha,
                                n_bootstrap_samples=n_bootstrap_samples,
                                random_state=random_state,
                            )
                            ops_result.append(ops_result_)

                else:
                    input_dict_ = input_dict.get(
                        behavior_policy_name=behavior_policy_name,
                        dataset_id=dataset_id,
                    )
                    ops_result = self._select_by_policy_value_lower_bound(
                        input_dict_,
                        compared_estimators=compared_estimators,
                        return_true_values=return_true_values,
                        return_metrics=return_metrics,
                        return_by_dataframe=return_by_dataframe,
                        top_k_in_eval_metrics=top_k_in_eval_metrics,
                        safety_threshold=safety_threshold,
                        relative_safety_criteria=relative_safety_criteria,
                        cis=cis,
                        alpha=alpha,
                        n_bootstrap_samples=n_bootstrap_samples,
                        random_state=random_state,
                    )

            else:
                ops_result = self._select_by_policy_value_lower_bound(
                    input_dict,
                    compared_estimators=compared_estimators,
                    return_true_values=return_true_values,
                    return_metrics=return_metrics,
                    return_by_dataframe=return_by_dataframe,
                    top_k_in_eval_metrics=top_k_in_eval_metrics,
                    safety_threshold=safety_threshold,
                    relative_safety_criteria=relative_safety_criteria,
                    cis=cis,
                    alpha=alpha,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

        else:
            if isinstance(input_dict, MultipleInputDict):
                raise ValueError(
                    "when using LoggedDataset, please use InputDict instead of MultipleInputDict"
                )

            ops_result = self._select_by_policy_value_lower_bound(
                input_dict,
                compared_estimators=compared_estimators,
                return_true_values=return_true_values,
                return_metrics=return_metrics,
                return_by_dataframe=return_by_dataframe,
                top_k_in_eval_metrics=top_k_in_eval_metrics,
                safety_threshold=safety_threshold,
                relative_safety_criteria=relative_safety_criteria,
                cis=cis,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return ops_result

    def select_by_lower_quartile(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        alpha: float = 0.05,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank the candidate policies by their estimated lower quartile of the trajectory-wise reward.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 0.5]`.

        return_true_values: bool, default=False
            Whether to return the true lower quartile of the trajectory-wise reward
            and corresponding ranking of the candidate evaluation policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        safety_threshold: float, default=0.0 (>= 0)
            The lower quartile required to be considered a safe policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [estimator_name][
                    estimated_ranking,
                    estimated_lower_quartile,
                    true_ranking,
                    true_lower_quartile,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated lower quartile of the trajectory-wise reward.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_lower_quartile: list of float
                Estimated lower quartile of the trajectory-wise reward of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) lower quartile of the trajectory-wise reward of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_lower_quartile: list of float
                True lower quartile of the trajectory-wise reward of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: float
                Mean-squared-error of the estimated lower quartile of the trajectory-wise reward.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple of float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            regret: None
                This is for API consistency.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            safety_threshold: float
                The lower quartile required to be considered a safe policy.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )

        if self.cumulative_distribution_ope.use_multiple_logged_dataset:
            if isinstance(input_dict, MultipleInputDict):
                if behavior_policy_name is None and dataset_id is None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.n_datasets
                        != input_dict.n_datasets
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies and dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = defaultdict(list)
                        metric_df = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_lower_quartile(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    alpha=alpha,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    safety_threshold=safety_threshold,
                                )
                                ranking_df[behavior_policy].append(ops_result_[0])
                                metric_df[behavior_policy].append(ops_result_[1])

                            ops_result = (
                                defaultdict_to_dict(ranking_df),
                                defaultdict_to_dict(metric_df),
                            )

                    else:
                        ops_result = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = (
                                    ops_result_
                                ) = self._select_by_lower_quartile(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    alpha=alpha,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    safety_threshold=safety_threshold,
                                )
                                ops_result[behavior_policy].append(ops_result_)

                        ops_result = defaultdict_to_dict(ops_result)

                elif behavior_policy_name is None and dataset_id is not None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.behavior_policy_names
                        != input_dict.behavior_policy_names
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = {}
                        metric_df = {}

                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_lower_quartile(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                alpha=alpha,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                safety_threshold=safety_threshold,
                            )
                            ranking_df[behavior_policy] = ops_result_[0]
                            metric_df[behavior_policy] = ops_result_[1]

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = {}
                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_lower_quartile(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                alpha=alpha,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                safety_threshold=safety_threshold,
                            )
                            ops_result[behavior_policy] = ops_result_

                elif behavior_policy_name is not None and dataset_id is None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.n_datasets[
                            behavior_policy_name
                        ]
                        != input_dict.n_datasets[behavior_policy_name]
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = []
                        metric_df = []

                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_lower_quartile(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                alpha=alpha,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                safety_threshold=safety_threshold,
                            )
                            ranking_df.append(ops_result_[0])
                            metric_df.append(ops_result_[1])

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = []
                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy_name,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_lower_quartile(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                alpha=alpha,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                safety_threshold=safety_threshold,
                            )
                            ops_result.append(ops_result_)

                else:
                    input_dict_ = input_dict.get(
                        behavior_policy_name=behavior_policy_name, dataset_id=dataset_id
                    )
                    ops_result = self._select_by_lower_quartile(
                        input_dict_,
                        compared_estimators=compared_estimators,
                        alpha=alpha,
                        return_true_values=return_true_values,
                        return_metrics=return_metrics,
                        return_by_dataframe=return_by_dataframe,
                        safety_threshold=safety_threshold,
                    )

            else:
                ops_result = self._select_by_lower_quartile(
                    input_dict,
                    compared_estimators=compared_estimators,
                    alpha=alpha,
                    return_true_values=return_true_values,
                    return_metrics=return_metrics,
                    return_by_dataframe=return_by_dataframe,
                    safety_threshold=safety_threshold,
                )

        else:
            if isinstance(input_dict, MultipleInputDict):
                raise ValueError(
                    "when using LoggedDataset, please use InputDict instead of MultipleInputDict"
                )

            ops_result = self._select_by_lower_quartile(
                input_dict,
                compared_estimators=compared_estimators,
                alpha=alpha,
                return_true_values=return_true_values,
                return_metrics=return_metrics,
                return_by_dataframe=return_by_dataframe,
                safety_threshold=safety_threshold,
            )

        return ops_result

    def select_by_conditional_value_at_risk(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        alpha: float = 0.05,
        return_true_values: bool = False,
        return_metrics: bool = False,
        return_by_dataframe: bool = False,
        safety_threshold: float = 0.0,
    ):
        """Rank the candidate policies by their estimated conditional value at risk.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleLoggedDataset
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 1]`.

        return_true_values: bool, default=False
            Whether to return the true conditional value at risk
            and corresponding ranking of the candidate evaluation policies.

        return_metrics: bool, default=False
            Whether to return the following evaluation metrics in terms of OPE and OPS:
            mean-squared-error, rank-correlation, and Type I and Type II error rate.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        safety_threshold: float, default=0.0 (>= 0)
            The conditional value at risk required to be considered a safe policy.

        Return
        -------
        ops_dict/(ranking_df_dict, metric_df): dict or dataframe (, list of dict or dataframe)
            Dictionary/dataframe containing the result of OPS conducted by OPE estimators.

            .. code-block:: python

                key: [estimator_name][
                    estimated_ranking,
                    estimated_conditional_value_at_risk,
                    true_ranking,
                    true_conditional_value_at_risk,
                    mean_squared_error,
                    rank_correlation,
                    regret,
                    type_i_error_rate,
                    type_ii_error_rate,
                ]

            estimated_ranking: list of str
                Name of the candidate policies sorted by the estimated conditional value at risk.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            estimated_conditional_value_at_risk: list of float
                Estimated conditional value at risk of the candidate policies (sorted by estimated_ranking).
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_ranking: list of int
                Ranking index of the (true) conditional value at risk of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            true_conditional_value_at_risk: list of float
                True conditional value at risk of the candidate policies (sorted by estimated_ranking).
                Recorded only when return_true_values is `True`.
                Recorded in ranking_df_dict if return_by_dataframe is `True`.

            mean_squared_error: float
                Mean-squared-error of the estimated conditional value at risk.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            rank_correlation: tuple or float
                Rank correlation coefficient between the true ranking and the estimated ranking, and its pvalue.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            regret: None
                This is for API consistency.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_i_error_rate: float
                Type I error rate of the hypothetical test. True Negative when the policy is safe but estimated as unsafe.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is `True`.

            type_ii_error_rate: float
                Type II error rate of the hypothetical test. False Positive when the policy is unsafe but undetected.
                Recorded only when return_metric is `True`.
                Recorded in metric_df if return_by_dataframe is True`.

            safety_threshold: float
                The conditional value at risk required to be considered a safe policy.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )

        if self.cumulative_distribution_ope.use_multiple_logged_dataset:
            if isinstance(input_dict, MultipleInputDict):
                if behavior_policy_name is None and dataset_id is None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.n_datasets
                        != input_dict.n_datasets
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies and dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = defaultdict(list)
                        metric_df = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_conditional_value_at_risk(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    alpha=alpha,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    safety_threshold=safety_threshold,
                                )
                                ranking_df[behavior_policy].append(ops_result_[0])
                                metric_df[behavior_policy].append(ops_result_[1])

                            ops_result = (
                                defaultdict_to_dict(ranking_df),
                                defaultdict_to_dict(metric_df),
                            )

                    else:
                        ops_result = defaultdict(list)

                        for (
                            behavior_policy,
                            n_datasets,
                        ) in input_dict.n_datasets.items():
                            for dataset_id_ in range(n_datasets):
                                input_dict_ = input_dict.get(
                                    behavior_policy_name=behavior_policy,
                                    dataset_id=dataset_id_,
                                )
                                ops_result_ = self._select_by_conditional_value_at_risk(
                                    input_dict_,
                                    compared_estimators=compared_estimators,
                                    alpha=alpha,
                                    return_true_values=return_true_values,
                                    return_metrics=return_metrics,
                                    return_by_dataframe=return_by_dataframe,
                                    safety_threshold=safety_threshold,
                                )
                                ops_result[behavior_policy].append(ops_result_)

                        ops_result = defaultdict_to_dict(ops_result)

                elif behavior_policy_name is None and dataset_id is not None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.behavior_policy_names
                        != input_dict.behavior_policy_names
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same behavior policies, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = {}
                        metric_df = {}

                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_conditional_value_at_risk(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                alpha=alpha,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                safety_threshold=safety_threshold,
                            )
                            ranking_df[behavior_policy] = ops_result_[0]
                            metric_df[behavior_policy] = ops_result_[1]

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = {}
                        for behavior_policy in input_dict.behavior_policy_names:
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy,
                                dataset_id=dataset_id,
                            )
                            ops_result_ = self._select_by_conditional_value_at_risk(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                alpha=alpha,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                safety_threshold=safety_threshold,
                            )
                            ops_result[behavior_policy] = ops_result_

                elif behavior_policy_name is not None and dataset_id is None:
                    if (
                        self.cumulative_distribution_ope.multiple_logged_dataset.n_datasets[
                            behavior_policy_name
                        ]
                        != input_dict.n_datasets[behavior_policy_name]
                    ):
                        raise ValueError(
                            "Expected that logged datasets and input dicts consists of the same dataset ids, but found False."
                        )

                    if return_metrics and return_by_dataframe:
                        ranking_df = []
                        metric_df = []

                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy_name,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_conditional_value_at_risk(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                alpha=alpha,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                safety_threshold=safety_threshold,
                            )
                            ranking_df.append(ops_result_[0])
                            metric_df.append(ops_result_[1])

                        ops_result = (ranking_df, metric_df)

                    else:
                        ops_result = []
                        for dataset_id_ in range(
                            input_dict.n_datasets[behavior_policy_name]
                        ):
                            input_dict_ = input_dict.get(
                                behavior_policy_name=behavior_policy_name,
                                dataset_id=dataset_id_,
                            )
                            ops_result_ = self._select_by_conditional_value_at_risk(
                                input_dict_,
                                compared_estimators=compared_estimators,
                                alpha=alpha,
                                return_true_values=return_true_values,
                                return_metrics=return_metrics,
                                return_by_dataframe=return_by_dataframe,
                                safety_threshold=safety_threshold,
                            )
                            ops_result.append(ops_result_)

                else:
                    input_dict_ = input_dict.get(
                        behavior_policy_name=behavior_policy_name, dataset_id=dataset_id
                    )
                    ops_result = self._select_by_conditional_value_at_risk(
                        input_dict_,
                        compared_estimators=compared_estimators,
                        alpha=alpha,
                        return_true_values=return_true_values,
                        return_metrics=return_metrics,
                        return_by_dataframe=return_by_dataframe,
                        safety_threshold=safety_threshold,
                    )

            else:
                ops_result = self._select_by_conditional_value_at_risk(
                    input_dict,
                    compared_estimators=compared_estimators,
                    alpha=alpha,
                    return_true_values=return_true_values,
                    return_metrics=return_metrics,
                    return_by_dataframe=return_by_dataframe,
                    safety_threshold=safety_threshold,
                )

        else:
            if isinstance(input_dict, MultipleInputDict):
                raise ValueError(
                    "when using LoggedDataset, please use InputDict instead of MultipleInputDict"
                )

            ops_result = self._select_by_conditional_value_at_risk(
                input_dict,
                compared_estimators=compared_estimators,
                alpha=alpha,
                return_true_values=return_true_values,
                return_metrics=return_metrics,
                return_by_dataframe=return_by_dataframe,
                safety_threshold=safety_threshold,
            )

        return ops_result

    def visualize_policy_value_for_selection(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        is_relative: bool = False,
        hue: str = "estimator",
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value_standard_ope.png",
    ):
        """Visualize the policy value estimated by OPE estimators (box plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        is_relative: bool, default=False
            If `True`, the method visualizes the estimated policy value of the evaluation policy
            relative to the on-policy policy value of the behavior policy.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value_standard_ope.png"
            Name of the bar figure.

        """
        self.ope.visualize_off_policy_estimates(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
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
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_cumulative_distribution_function.png",
    ) -> None:
        """Visualize the cumulative distribution function (cdf plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleLoggedDataset
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the figure.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_cumulative_distribution_function.png"
            Name of the bar figure.

        """
        self.cumulative_distribution_ope.visualize_cumulative_distribution_function(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            hue=hue,
            legend=legend,
            n_cols=n_cols,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_policy_value_of_cumulative_distribution_ope_for_selection(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        hue: str = "estimator",
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value_cumulative_distribution_ope.png",
    ) -> None:
        """Visualize the policy value estimated by cumulative distribution OPE estimators (box plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        alpha: float, default=0.05
            Significance level. The value should bw within `[0, 1)`.

        is_relative: bool, default=False
            If `True`, the method visualizes the estimated policy value of the evaluation policy
            relative to the ground-truth policy value of the behavior policy.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value_cumulative_distribution_ope.png"
            Name of the bar figure.

        """
        self.cumulative_distribution_ope.visualize_policy_value(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            alpha=alpha,
            is_relative=is_relative,
            hue=hue,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_conditional_value_at_risk_for_selection(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        alphas: Optional[np.ndarray] = None,
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_conditional_value_at_risk.png",
    ) -> None:
        """Visualize the conditional value at risk estimated by cumulative distribution OPE estimators (cdf plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        alphas: array-like of shape (n_alpha, ), default=None
            Set of proportions of the shaded region. The values should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the figure.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_conditional_value_at_risk.png"
            Name of the bar figure.

        """
        self.cumulative_distribution_ope.visualize_conditional_value_at_risk(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
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
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        alpha: float = 0.05,
        hue: str = "estimator",
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_interquartile_range.png",
    ) -> None:
        """Visualize the interquartile range estimated by cumulative distribution OPE estimators (box plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

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
        self.cumulative_distribution_ope.visualize_interquartile_range(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            alpha=alpha,
            hue=hue,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_policy_value_with_multiple_estimates_standard_ope(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        plot_type: str = "ci",
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value_multiple_standard_ope.png",
    ) -> None:
        """Visualize the policy value estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, we get the empirical average of the estimated values with their estimated confidence intervals.
            If "scatter" is given, we get a scatter plot of estimated values.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value_multiple.png"
            Name of the bar figure.

        """
        self.ope.visualize_policy_value_with_multiple_estimates(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            plot_type=plot_type,
            hue=hue,
            legend=legend,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_cumulative_distribution_function_with_multiple_estimates(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        scale_min: Optional[float] = None,
        scale_max: Optional[float] = None,
        n_partition: Optional[int] = None,
        plot_type: str = "ci_hue",
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value_multiple.png",
    ) -> None:
        """Visualize the policy value estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        This function is not applicable when the data-driven reward scaler is used.
        Please set ``scale_min``, ``scale_max``, and ``n_partition`` to use.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        scale_min: float, default=None
            Minimum value of the reward scale in the CDF.

        scale_max: float, default=None
            Maximum value of the reward scale in the CDF.

        n_partition: int, default=None
            Number of partitions in the reward scale (x-axis of the CDF).

        plot_type: {"ci_hue", "ci_behavior_policy", "enumerate"}, default="ci_hue"
            Type of plot.
            If "ci" is given, the method visualizes the average policy value and its 95% confidence intervals based on the multiple estimate.
            If "enumerate" is given, we get a scatter plot of estimated values.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value_multiple.png"
            Name of the bar figure.

        """
        self.cumulative_distribution_ope.visualize_cumulative_distribution_function_with_multiple_estimates(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            scale_min=scale_min,
            scale_max=scale_max,
            n_partition=n_partition,
            plot_type=plot_type,
            hue=hue,
            legend=legend,
            n_cols=n_cols,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_policy_value_with_multiple_estimates_cumulative_distribution_ope(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        plot_type: str = "ci",
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value_multiple_cumulative_distribution_ope.png",
    ) -> None:
        """Visualize the policy value estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, we get the empirical average of the estimated values with their estimated confidence intervals.
            If "scatter" is given, we get a scatter plot of estimated values.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value_multiple.png"
            Name of the bar figure.

        """
        self.cumulative_distribution_ope.visualize_policy_value_with_multiple_estimates(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            plot_type=plot_type,
            hue=hue,
            legend=legend,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_variance_with_multiple_estimates(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        plot_type: str = "ci",
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_variance_multiple.png",
    ) -> None:
        """Visualize the variance of the trajectory-wise reward under the evaluation policy estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, we get the empirical average of the estimated values with their estimated confidence intervals.
            If "scatter" is given, we get a scatter plot of estimated values.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_variance_multiple.png"
            Name of the bar figure.

        """
        self.cumulative_distribution_ope.visualize_variance_with_multiple_estimates(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            plot_type=plot_type,
            hue=hue,
            legend=legend,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_conditional_value_at_risk_with_multiple_estimates(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        alpha: float = 0.05,
        plot_type: str = "ci",
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_conditional_value_at_risk_multiple.png",
    ) -> None:
        """Visualize the conditional value at risk of the trajectory-wise reward under the evaluation policy estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        alpha: float = 0.05.
            Proportion of the shaded region in CVaR estimate. The value should be within `[0, 1)`.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, we get the empirical average of the estimated values with their estimated confidence intervals.
            If "scatter" is given, we get a scatter plot of estimated values.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_conditional_value_at_risk_multiple.png"
            Name of the bar figure.

        """
        self.cumulative_distribution_ope.visualize_conditional_value_at_risk_with_multiple_estimates(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            alpha=alpha,
            plot_type=plot_type,
            hue=hue,
            legend=legend,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_lower_quartile_with_multiple_estimates(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        alpha: float = 0.05,
        plot_type: str = "ci",
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_conditional_value_at_risk_multiple.png",
    ) -> None:
        """Visualize the lower quartile of the trajectory-wise reward under the evaluation policy estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        alpha: float = 0.05.
            Proportion of the shaded region in CVaR estimate. The value should be within `[0, 1)`.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, we get the empirical average of the estimated values with their estimated confidence intervals.
            If "scatter" is given, we get a scatter plot of estimated values.

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="estimated_conditional_value_at_risk_multiple.png"
            Name of the bar figure.

        """
        self.cumulative_distribution_ope.visualize_lower_quartile_with_multiple_estimates(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            alpha=alpha,
            plot_type=plot_type,
            hue=hue,
            legend=legend,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def _obtain_topk_policy_performance(
        self,
        true_dict: Dict,
        estimation_dict: Dict,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        true_dict_ranking_arg: str,
        true_dict_value_arg: str,
        estimation_dict_ranking_arg: str,
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        max_topk: Optional[int] = None,
        ope_alpha: Optional[float] = None,
        return_safety_violation_rate: bool = False,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        return_by_dataframe: bool = False,
    ):
        """Calculate top-k policy deployment performances.

        Parameters
        -------
        true_dict: dict
            Dictionary containing the true policy performance.

        estimation_dict: dict
            Dictionary containing the estimated policy performance.

        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        true_dict_ranking_arg: str
            Name of the key indicating the ranked list of the candidate policies in true_dict.

        true_dict_value_arg: str
            Name of the key indicating the true policy performance of the candidate policies in true_dict.

        estimation_dict_ranking_arg: str
            Name of the ley indicaing the estimated ranking of the candidate policies in true_dict.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        behavior_policy: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        max_topk: int, default=None
            Maximum number of policies to be deployed.

        ope_alpha: float, default=None
            Significance level. The value should be within `[0, 1)`.

        return_safety_violation_rate: bool, default=False.
            Whether to calculate and return the safety violate.

        safety_threshold: float, default=0.0 (>= 0)
            The conditional value at risk required to be considered a safe policy.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that when returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if return_safety_violation_rate:
            metrics = ["k-th", "best", "worst", "mean", "std", "safety_violation_rate"]
        else:
            metrics = ["k-th", "best", "worst", "mean", "std"]

        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None and dataset_id is None:
                ranking_dict = defaultdict(list)
                for behavior_policy, n_datasets in input_dict.n_datasets.items():
                    for dataset_id_ in range(n_datasets):
                        true_dict[behavior_policy][dataset_id_] = dict(
                            zip(
                                true_dict[behavior_policy][dataset_id_][
                                    true_dict_ranking_arg
                                ],
                                true_dict[behavior_policy][dataset_id_][
                                    true_dict_value_arg
                                ],
                            )
                        )

                        tmp_ranking_dict = dict()
                        for i, estimator in enumerate(compared_estimators):
                            policy_performance = np.zeros(
                                input_dict.n_eval_policies[behavior_policy][dataset_id_]
                            )
                            estimated_ranking = estimation_dict[behavior_policy][
                                dataset_id_
                            ][estimator]["estimated_ranking"]

                            for i, eval_policy in enumerate(estimated_ranking):
                                policy_performance[i] = true_dict[behavior_policy][
                                    dataset_id_
                                ][eval_policy]

                            tmp_ranking_dict[estimator] = policy_performance

                        ranking_dict[behavior_policy].append(tmp_ranking_dict)

            elif behavior_policy_name is None and dataset_id is not None:
                ranking_dict = {}
                for behavior_policy in input_dict.behavior_policy_names:
                    true_dict[behavior_policy] = dict(
                        zip(
                            true_dict[behavior_policy][true_dict_ranking_arg],
                            true_dict[behavior_policy][true_dict_value_arg],
                        )
                    )

                    tmp_ranking_dict = dict()
                    for i, estimator in enumerate(compared_estimators):
                        policy_performance = np.zeros(
                            input_dict.n_eval_policies[behavior_policy][dataset_id]
                        )
                        estimated_ranking = estimation_dict[behavior_policy][estimator][
                            estimation_dict_ranking_arg
                        ]

                        for i, eval_policy in enumerate(estimated_ranking):
                            policy_performance[i] = true_dict[behavior_policy][
                                eval_policy
                            ]

                        tmp_ranking_dict[estimator] = policy_performance

                    ranking_dict[behavior_policy] = tmp_ranking_dict

            elif behavior_policy_name is not None and dataset_id is None:
                ranking_dict = []
                for dataset_id_ in range(input_dict.n_datasets[behavior_policy_name]):
                    true_dict[dataset_id_] = dict(
                        zip(
                            true_dict[dataset_id_][true_dict_ranking_arg],
                            true_dict[dataset_id_][true_dict_value_arg],
                        )
                    )

                    tmp_ranking_dict = dict()
                    for i, estimator in enumerate(compared_estimators):
                        policy_performance = np.zeros(
                            input_dict.n_eval_policies[behavior_policy_name][
                                dataset_id_
                            ]
                        )
                        estimated_ranking = estimation_dict[dataset_id_][estimator][
                            estimation_dict_ranking_arg
                        ]

                        for i, eval_policy in enumerate(estimated_ranking):
                            policy_performance[i] = true_dict[dataset_id_][eval_policy]

                        tmp_ranking_dict[estimator] = policy_performance

                    ranking_dict.append(tmp_ranking_dict)

            else:
                true_dict = dict(
                    zip(
                        true_dict[true_dict_ranking_arg],
                        true_dict[true_dict_value_arg],
                    )
                )

                ranking_dict = dict()
                for i, estimator in enumerate(compared_estimators):
                    policy_performance = np.zeros(
                        input_dict.n_eval_policies[behavior_policy_name][dataset_id]
                    )
                    estimated_ranking = estimation_dict[estimator][
                        estimation_dict_ranking_arg
                    ]

                    for i, eval_policy in enumerate(estimated_ranking):
                        policy_performance[i] = true_dict[eval_policy]

                    ranking_dict[estimator] = policy_performance

        else:
            true_dict = dict(
                zip(
                    true_dict[true_dict_ranking_arg],
                    true_dict[true_dict_value_arg],
                )
            )

            ranking_dict = dict()
            for i, estimator in enumerate(compared_estimators):
                policy_performance = np.zeros((len(input_dict),))
                estimated_ranking = estimation_dict[estimator][
                    estimation_dict_ranking_arg
                ]

                for i, eval_policy in enumerate(estimated_ranking):
                    policy_performance[i] = true_dict[eval_policy]

                ranking_dict[estimator] = policy_performance

        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        discount = np.full(self.step_per_trajectory, gamma).cumprod() / gamma

        behavior_policy_cum_reward = {}
        behavior_policy_value = {}
        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None:
                for behavior_policy in input_dict.behavior_policy_names:
                    behavior_policy_reward = self.behavior_policy_reward[
                        behavior_policy
                    ]
                    behavior_policy_cum_reward[behavior_policy] = (
                        discount[np.newaxis, :] * behavior_policy_reward
                    ).sum(
                        axis=1
                    ) + 1e-10  # to avoid zero division
                    behavior_policy_value[behavior_policy] = (
                        discount[np.newaxis, :] * behavior_policy_reward
                    ).sum(
                        axis=1
                    ).mean() + 1e-10  # to avoid zero division
            else:
                behavior_policy_reward = self.behavior_policy_reward[
                    behavior_policy_name
                ]
                behavior_policy_cum_reward[behavior_policy_name] = (
                    discount[np.newaxis, :] * behavior_policy_reward
                ).sum(
                    axis=1
                ) + 1e-10  # to avoid zero division
                behavior_policy_value[behavior_policy_name] = (
                    discount[np.newaxis, :] * behavior_policy_reward
                ).sum(
                    axis=1
                ).mean() + 1e-10  # to avoid zero division
        else:
            behavior_policy = input_dict[list(input_dict.keys())[0]]["behavior_policy"]
            behavior_policy_reward = self.behavior_policy_reward[behavior_policy]
            behavior_policy_cum_reward[behavior_policy] = (
                discount[np.newaxis, :] * behavior_policy_reward
            ).sum(
                axis=1
            ) + 1e-10  # to avoid zero division
            behavior_policy_value[behavior_policy] = (
                discount[np.newaxis, :] * behavior_policy_reward
            ).sum(
                axis=1
            ).mean() + 1e-10  # to avoid zero division

        metric_dict = defaultdict(dict)
        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None and dataset_id is None:
                n_datasets = input_dict.n_datasets
                total_n_datasets = np.array(list(n_datasets.values())).sum()
                baseline = np.zeros(total_n_datasets)

                for i, estimator in enumerate(compared_estimators):
                    for j, metric in enumerate(metrics):
                        topk_metric = np.zeros((max_topk, total_n_datasets))

                        for topk in range(max_topk):
                            l = 0
                            for behavior_policy in input_dict.behavior_policy_names:
                                for dataset_id_ in range(n_datasets[behavior_policy]):
                                    if i == 0:
                                        if true_dict_value_arg == "policy_value":
                                            baseline[l] = behavior_policy_value[
                                                behavior_policy
                                            ]
                                        elif (
                                            true_dict_value_arg
                                            == "conditional_value_at_risk"
                                        ):
                                            baseline_reward = (
                                                behavior_policy_cum_reward[
                                                    behavior_policy
                                                ]
                                            )
                                            baseline[l] = np.sort(baseline_reward)[
                                                : int(len(baseline_reward) * ope_alpha)
                                            ].mean()
                                        elif true_dict_value_arg == "lower_quartile":
                                            baseline_reward = (
                                                behavior_policy_cum_reward[
                                                    behavior_policy
                                                ]
                                            )
                                            baseline[l] = np.quantile(
                                                baseline_reward,
                                                q=ope_alpha,
                                            )

                                    topk_values = ranking_dict[behavior_policy][
                                        dataset_id_
                                    ][estimator][: topk + 1]

                                    if metric == "k-th":
                                        topk_metric[topk, l] = topk_values[-1]
                                    elif metric == "best":
                                        topk_metric[topk, l] = topk_values.max()
                                    elif metric == "worst":
                                        topk_metric[topk, l] = topk_values.min()
                                    elif metric == "mean":
                                        topk_metric[topk, l] = topk_values.mean()
                                    elif metric == "std":
                                        topk_metric[topk, l] = topk_values.std(ddof=1)
                                    else:
                                        topk_metric[topk, l] = (
                                            topk_values < safety_threshold
                                        ).sum() / (topk + 1)

                                    l += 1

                        metric_dict[estimator][metric] = topk_metric

                    if i == 0:
                        baseline = np.tile(baseline, (max_topk, 1))

                    sharpe_ratio = (
                        np.clip(metric_dict[estimator]["best"] - baseline, 0, None)
                        / metric_dict[estimator]["std"]
                    )

                    if clip_sharpe_ratio:
                        sharpe_ratio[1:] = np.nan_to_num(sharpe_ratio[1:], posinf=1e2)
                        sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                    metric_dict[estimator]["sharpe_ratio"] = sharpe_ratio

            elif behavior_policy_name is None and dataset_id is not None:
                total_n_datasets = len(input_dict.behavior_policy_names)
                baseline = np.zeros(total_n_datasets)

                for i, estimator in enumerate(compared_estimators):
                    for j, metric in enumerate(metrics):
                        topk_metric = np.zeros((max_topk, total_n_datasets))

                        for topk in range(max_topk):
                            for l, behavior_policy in enumerate(
                                input_dict.behavior_policy_names
                            ):
                                if i == 0:
                                    if true_dict_value_arg == "policy_value":
                                        baseline[l] = behavior_policy_value[
                                            behavior_policy
                                        ]
                                    elif (
                                        true_dict_value_arg
                                        == "conditional_value_at_risk"
                                    ):
                                        baseline_reward = behavior_policy_cum_reward[
                                            behavior_policy
                                        ]
                                        baseline[l] = np.sort(baseline_reward)[
                                            : int(len(baseline_reward) * ope_alpha)
                                        ].mean()
                                    elif true_dict_value_arg == "lower_quartile":
                                        baseline_reward = behavior_policy_cum_reward[
                                            behavior_policy
                                        ]
                                        baseline[l] = np.quantile(
                                            baseline_reward,
                                            q=ope_alpha,
                                        )

                                topk_values = ranking_dict[behavior_policy][estimator][
                                    : topk + 1
                                ]

                                if metric == "k-th":
                                    topk_metric[topk, l] = topk_values[-1]
                                elif metric == "best":
                                    topk_metric[topk, l] = topk_values.max()
                                elif metric == "worst":
                                    topk_metric[topk, l] = topk_values.min()
                                elif metric == "mean":
                                    topk_metric[topk, l] = topk_values.mean()
                                elif metric == "std":
                                    topk_metric[topk, l] = topk_values.std(ddof=1)
                                else:
                                    topk_metric[topk, l] = (
                                        topk_values < safety_threshold
                                    ).sum() / (topk + 1)

                        metric_dict[estimator][metric] = topk_metric

                    if i == 0:
                        baseline = np.tile(baseline, (max_topk, 1))

                    sharpe_ratio = (
                        np.clip(metric_dict[estimator]["best"] - baseline, 0, None)
                        / metric_dict[estimator]["std"]
                    )

                    if clip_sharpe_ratio:
                        sharpe_ratio[1:] = np.nan_to_num(sharpe_ratio[1:], posinf=1e2)
                        sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                    metric_dict[estimator]["sharpe_ratio"] = sharpe_ratio

            elif behavior_policy_name is not None and dataset_id is None:
                total_n_datasets = input_dict.n_datasets[behavior_policy_name]
                if true_dict_value_arg == "policy_value":
                    baseline = behavior_policy_value[behavior_policy_name]
                elif true_dict_value_arg == "conditional_value_at_risk":
                    baseline_reward = behavior_policy_cum_reward[behavior_policy_name]
                    baseline = np.sort(baseline_reward)[
                        : int(len(baseline_reward) * ope_alpha)
                    ].mean()
                elif true_dict_value_arg == "lower_quartile":
                    baseline_reward = behavior_policy_cum_reward[behavior_policy_name]
                    baseline = np.quantile(
                        baseline_reward,
                        q=ope_alpha,
                    )

                for i, estimator in enumerate(compared_estimators):
                    for j, metric in enumerate(metrics):
                        topk_metric = np.zeros((max_topk, total_n_datasets))

                        for topk in range(max_topk):
                            for l in range(total_n_datasets):
                                topk_values = ranking_dict[l][estimator][: topk + 1]

                                if metric == "k-th":
                                    topk_metric[topk, l] = topk_values[-1]
                                elif metric == "best":
                                    topk_metric[topk, l] = topk_values.max()
                                elif metric == "worst":
                                    topk_metric[topk, l] = topk_values.min()
                                elif metric == "mean":
                                    topk_metric[topk, l] = topk_values.mean()
                                elif metric == "std":
                                    topk_metric[topk, l] = topk_values.std(ddof=1)
                                else:
                                    topk_metric[topk, l] = (
                                        topk_values < safety_threshold
                                    ).sum() / (topk + 1)

                        metric_dict[estimator][metric] = topk_metric

                    sharpe_ratio = (
                        np.clip(metric_dict[estimator]["best"] - baseline, 0, None)
                        / metric_dict[estimator]["std"]
                    )

                    if clip_sharpe_ratio:
                        sharpe_ratio[1:] = np.nan_to_num(sharpe_ratio[1:], posinf=1e2)
                        sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                    metric_dict[estimator]["sharpe_ratio"] = sharpe_ratio

            else:
                total_n_datasets = 1
                if true_dict_value_arg == "policy_value":
                    baseline = behavior_policy_value[behavior_policy_name]
                elif true_dict_value_arg == "conditional_value_at_risk":
                    baseline_reward = behavior_policy_cum_reward[behavior_policy_name]
                    baseline = np.sort(baseline_reward)[
                        : int(len(baseline_reward) * ope_alpha)
                    ].mean()
                elif true_dict_value_arg == "lower_quartile":
                    baseline_reward = behavior_policy_cum_reward[behavior_policy_name]
                    baseline = np.quantile(
                        baseline_reward,
                        q=ope_alpha,
                    )

                for i, estimator in enumerate(compared_estimators):
                    for j, metric in enumerate(metrics):
                        topk_metric = np.zeros((max_topk, total_n_datasets))

                        for topk in range(max_topk):
                            topk_values = ranking_dict[estimator][: topk + 1]

                            if metric == "k-th":
                                topk_metric[topk, 0] = topk_values[-1]
                            elif metric == "best":
                                topk_metric[topk, 0] = topk_values.max()
                            elif metric == "worst":
                                topk_metric[topk, 0] = topk_values.min()
                            elif metric == "mean":
                                topk_metric[topk, 0] = topk_values.mean()
                            elif metric == "std":
                                topk_metric[topk, 0] = topk_values.std(ddof=1)
                            else:
                                topk_metric[topk, 0] = (
                                    topk_values < safety_threshold
                                ).sum() / (topk + 1)

                        metric_dict[estimator][metric] = topk_metric

                    sharpe_ratio = (
                        np.clip(metric_dict[estimator]["best"] - baseline, 0, None)
                        / metric_dict[estimator]["std"]
                    )

                    if clip_sharpe_ratio:
                        sharpe_ratio[1:] = np.nan_to_num(sharpe_ratio[1:], posinf=1e2)
                        sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                    metric_dict[estimator]["sharpe_ratio"] = sharpe_ratio

        else:
            behavior_policy = input_dict[list(input_dict.keys())[0]]["behavior_policy"]
            if true_dict_value_arg == "policy_value":
                baseline = behavior_policy_value[behavior_policy]
            elif true_dict_value_arg == "conditional_value_at_risk":
                baseline_reward = behavior_policy_cum_reward[behavior_policy]
                baseline = np.sort(baseline_reward)[
                    : int(len(baseline_reward) * ope_alpha)
                ].mean()
            elif true_dict_value_arg == "lower_quartile":
                baseline_reward = behavior_policy_cum_reward[behavior_policy]
                baseline = np.quantile(
                    baseline_reward,
                    q=ope_alpha,
                )

            for i, estimator in enumerate(compared_estimators):
                for j, metric in enumerate(metrics):
                    topk_metric = np.zeros((max_topk, 1))

                    for topk in range(max_topk):
                        topk_values = ranking_dict[estimator][: topk + 1]

                        if metric == "k-th":
                            topk_metric[topk, 0] = topk_values[-1]
                        elif metric == "best":
                            topk_metric[topk, 0] = topk_values.max()
                        elif metric == "worst":
                            topk_metric[topk, 0] = topk_values.min()
                        elif metric == "mean":
                            topk_metric[topk, 0] = topk_values.mean()
                        elif metric == "std":
                            topk_metric[topk, 0] = topk_values.std(ddof=1)
                        else:
                            topk_metric[topk, 0] = (
                                topk_values < safety_threshold
                            ).sum() / (topk + 1)

                    metric_dict[estimator][metric] = topk_metric

                sharpe_ratio = (
                    np.clip(metric_dict[estimator]["best"] - baseline, 0, None)
                    / metric_dict[estimator]["std"]
                )

                if clip_sharpe_ratio:
                    sharpe_ratio[1:] = np.nan_to_num(sharpe_ratio[1:], posinf=1e2)
                    sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                metric_dict[estimator]["sharpe_ratio"] = sharpe_ratio

        metric_dict = defaultdict_to_dict(metric_dict)

        if return_by_dataframe:
            metrics.extend(["sharpe_ratio"])
            metric_df = []

            for i, estimator in enumerate(compared_estimators):
                metric_df_ = pd.DataFrame()
                metric_df_["topk"] = np.arange(max_topk)
                metric_df_["estimator"] = estimator
                metric_df_ = metric_df_[["estimator", "topk"]]

                for metric in metrics:
                    metric_df_[metric] = metric_dict[estimator][metric].mean(axis=1)

                metric_df.append(metric_df_)

            metric = pd.concat(metric_df, axis=0)

        else:
            metric = metric_dict

        return metric

    def obtain_topk_policy_value_selected_by_standard_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        max_topk: Optional[int] = None,
        return_safety_violation_rate: bool = False,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
        return_by_dataframe: bool = False,
    ):
        """Obtain the topk deployment result (policy value) selected by standard OPE.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        return_safety_violation_rate: bool, default=False.
            Whether to calculate and return the safety violate.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that policy performance refers to the (standard) policy value here. When returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            relative_safety_criteria=relative_safety_criteria,
            gamma=gamma,
        )

        true_dict = self.obtain_true_selection_result(
            input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        estimation_dict = self.select_by_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        return self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking",
            true_dict_value_arg="policy_value",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            return_safety_violation_rate=return_safety_violation_rate,
            safety_threshold=safety_threshold,
            return_by_dataframe=return_by_dataframe,
        )

    def obtain_topk_policy_value_selected_by_cumulative_distribution_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        max_topk: Optional[int] = None,
        return_safety_violation_rate: bool = False,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
        return_by_dataframe: bool = False,
    ):
        """Obtain the topk deployment result (policy value) selected by cumulative distribution OPE.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        return_safety_violation_rate: bool, default=False.
            Whether to calculate and return the safety violate.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that policy performance refers to the (standard) policy value here. When returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            relative_safety_criteria=relative_safety_criteria,
            gamma=gamma,
        )

        true_dict = self.obtain_true_selection_result(
            input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        estimation_dict = self.select_by_policy_value_via_cumulative_distribution_ope(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        return self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking",
            true_dict_value_arg="policy_value",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            return_safety_violation_rate=return_safety_violation_rate,
            safety_threshold=safety_threshold,
            return_by_dataframe=return_by_dataframe,
        )

    def obtain_topk_policy_value_selected_by_lower_bound(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        max_topk: Optional[int] = None,
        return_safety_violation_rate: bool = False,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        cis: List[str] = ["bootstrap"],
        ope_alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        return_by_dataframe: bool = False,
    ):
        """Obtain the topk deployment (policy value) result selected by its estimated lower bound.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        return_safety_violation_rate: bool, default=False.
            Whether to calculate and return the safety violate.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        cis: list of {"bootstrap", "hoeffding", "bernstein", "ttest"}, default=["bootstrap"]
            Estimation methods for confidence intervals.

        ope_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        n_bootstrap_samples: int, default=100 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that policy performance refers to the (standard) policy value here. When returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            relative_safety_criteria=relative_safety_criteria,
            gamma=gamma,
        )
        if return_safety_violation_rate:
            metrics = ["k-th", "best", "worst", "mean", "std", "safety_violation_rate"]
        else:
            metrics = ["k-th", "best", "worst", "mean", "std"]

        policy_value_dict = self.select_by_policy_value_lower_bound(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            return_true_values=True,
            cis=cis,
            alpha=ope_alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

        discount = np.full(self.step_per_trajectory, gamma).cumprod() / gamma

        behavior_policy_cum_reward = {}
        behavior_policy_value = {}
        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None:
                for behavior_policy in input_dict.behavior_policy_names:
                    behavior_policy_reward = self.behavior_policy_reward[
                        behavior_policy
                    ]
                    behavior_policy_cum_reward[behavior_policy] = (
                        discount[np.newaxis, :] * behavior_policy_reward
                    ).sum(
                        axis=1
                    ) + 1e-10  # to avoid zero division
                    behavior_policy_value[behavior_policy] = (
                        discount[np.newaxis, :] * behavior_policy_reward
                    ).sum(
                        axis=1
                    ).mean() + 1e-10  # to avoid zero division
            else:
                behavior_policy_reward = self.behavior_policy_reward[
                    behavior_policy_name
                ]
                behavior_policy_cum_reward[behavior_policy_name] = (
                    discount[np.newaxis, :] * behavior_policy_reward
                ).sum(
                    axis=1
                ) + 1e-10  # to avoid zero division
                behavior_policy_value[behavior_policy_name] = (
                    discount[np.newaxis, :] * behavior_policy_reward
                ).sum(
                    axis=1
                ).mean() + 1e-10  # to avoid zero division
        else:
            behavior_policy = input_dict[list(input_dict.keys())[0]]["behavior_policy"]
            behavior_policy_reward = self.behavior_policy_reward[behavior_policy]
            behavior_policy_cum_reward[behavior_policy] = (
                discount[np.newaxis, :] * behavior_policy_reward
            ).sum(
                axis=1
            ) + 1e-10  # to avoid zero division
            behavior_policy_value[behavior_policy] = (
                discount[np.newaxis, :] * behavior_policy_reward
            ).sum(
                axis=1
            ).mean() + 1e-10  # to avoid zero division

        metric_dict = defaultdict(lambda: defaultdict(dict))

        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None and dataset_id is None:
                n_datasets = input_dict.n_datasets
                total_n_datasets = np.array(list(n_datasets.values())).sum()
                baseline = np.zeros(total_n_datasets)

                for ci in cis:
                    for i, estimator in enumerate(compared_estimators):
                        for j, metric in enumerate(metrics):
                            topk_metric = np.zeros((max_topk, total_n_datasets))

                            for topk in range(max_topk):
                                l = 0
                                for behavior_policy in input_dict.behavior_policy_names:
                                    for dataset_id_ in range(
                                        n_datasets[behavior_policy]
                                    ):
                                        if i == 0 and ci == cis[0]:
                                            baseline[l] = behavior_policy_value[
                                                behavior_policy
                                            ]

                                        topk_values = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "true_policy_value"
                                        ][
                                            : topk + 1
                                        ]

                                        if metric == "k-th":
                                            topk_metric[topk, l] = topk_values[-1]
                                        elif metric == "best":
                                            topk_metric[topk, l] = topk_values.max()
                                        elif metric == "worst":
                                            topk_metric[topk, l] = topk_values.min()
                                        elif metric == "mean":
                                            topk_metric[topk, l] = topk_values.mean()
                                        elif metric == "std":
                                            topk_metric[topk, l] = topk_values.std(
                                                ddof=1
                                            )
                                        else:
                                            topk_metric[topk, l] = (
                                                topk_values < safety_threshold
                                            ).sum() / (topk + 1)

                                        l += 1

                            metric_dict[ci][estimator][metric] = topk_metric

                        if i == 0 and ci == cis[0]:
                            baseline = np.tile(baseline, (max_topk, 1))

                        sharpe_ratio = (
                            np.clip(
                                metric_dict[ci][estimator]["best"] - baseline, 0, None
                            )
                            / metric_dict[ci][estimator]["std"]
                        )

                        if clip_sharpe_ratio:
                            sharpe_ratio[1:] = np.nan_to_num(
                                sharpe_ratio[1:], posinf=1e2
                            )
                            sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                        metric_dict[ci][estimator]["sharpe_ratio"] = sharpe_ratio

            elif behavior_policy_name is None and dataset_id is not None:
                total_n_datasets = len(input_dict.behavior_policy_names)
                baseline = np.zeros(total_n_datasets)

                for ci in cis:
                    for i, estimator in enumerate(compared_estimators):
                        for j, metric in enumerate(metrics):
                            topk_metric = np.zeros((max_topk, total_n_datasets))

                            for topk in range(max_topk):
                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    if i == 0 and ci == cis[0]:
                                        baseline[l] = behavior_policy_value[
                                            behavior_policy
                                        ]

                                    topk_values = policy_value_dict[behavior_policy][
                                        ci
                                    ][estimator]["true_policy_value"][: topk + 1]

                                    if metric == "k-th":
                                        topk_metric[topk, l] = topk_values[-1]
                                    elif metric == "best":
                                        topk_metric[topk, l] = topk_values.max()
                                    elif metric == "worst":
                                        topk_metric[topk, l] = topk_values.min()
                                    elif metric == "mean":
                                        topk_metric[topk, l] = topk_values.mean()
                                    elif metric == "std":
                                        topk_metric[topk, l] = topk_values.std(ddof=1)
                                    else:
                                        topk_metric[topk, l] = (
                                            topk_values < safety_threshold
                                        ).sum() / (topk + 1)

                            metric_dict[ci][estimator][metric] = topk_metric

                        if i == 0 and ci == cis[0]:
                            baseline = np.tile(baseline, (max_topk, 1))

                        sharpe_ratio = (
                            np.clip(
                                metric_dict[ci][estimator]["best"] - baseline, 0, None
                            )
                            / metric_dict[ci][estimator]["std"]
                        )

                        if clip_sharpe_ratio:
                            sharpe_ratio[1:] = np.nan_to_num(
                                sharpe_ratio[1:], posinf=1e2
                            )
                            sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                        metric_dict[ci][estimator]["sharpe_ratio"] = sharpe_ratio

            elif behavior_policy_name is not None and dataset_id is None:
                total_n_datasets = input_dict.n_datasets[behavior_policy_name]
                baseline = behavior_policy_value[behavior_policy_name]

                for ci in cis:
                    for i, estimator in enumerate(compared_estimators):
                        for j, metric in enumerate(metrics):
                            topk_metric = np.zeros((max_topk, total_n_datasets))

                            for topk in range(max_topk):
                                for l in range(total_n_datasets):
                                    topk_values = policy_value_dict[l][ci][estimator][
                                        "true_policy_value"
                                    ][: topk + 1]

                                    if metric == "k-th":
                                        topk_metric[topk, l] = topk_values[-1]
                                    elif metric == "best":
                                        topk_metric[topk, l] = topk_values.max()
                                    elif metric == "worst":
                                        topk_metric[topk, l] = topk_values.min()
                                    elif metric == "mean":
                                        topk_metric[topk, l] = topk_values.mean()
                                    elif metric == "std":
                                        topk_metric[topk, l] = topk_values.std(ddof=1)
                                    else:
                                        topk_metric[topk, l] = (
                                            topk_values < safety_threshold
                                        ).sum() / (topk + 1)

                            metric_dict[ci][estimator][metric] = topk_metric

                        sharpe_ratio = (
                            np.clip(
                                metric_dict[ci][estimator]["best"] - baseline, 0, None
                            )
                            / metric_dict[ci][estimator]["std"]
                        )

                        if clip_sharpe_ratio:
                            sharpe_ratio[1:] = np.nan_to_num(
                                sharpe_ratio[1:], posinf=1e2
                            )
                            sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                        metric_dict[ci][estimator]["sharpe_ratio"] = sharpe_ratio

            else:
                total_n_datasets = 1
                baseline = behavior_policy_value[behavior_policy_name]

                for ci in cis:
                    for i, estimator in enumerate(compared_estimators):
                        for j, metric in enumerate(metrics):
                            topk_metric = np.zeros((max_topk, total_n_datasets))

                            for topk in range(max_topk):
                                topk_values = policy_value_dict[ci][estimator][
                                    "true_policy_value"
                                ][: topk + 1]

                                if metric == "k-th":
                                    topk_metric[topk, 0] = topk_values[-1]
                                elif metric == "best":
                                    topk_metric[topk, 0] = topk_values.max()
                                elif metric == "worst":
                                    topk_metric[topk, 0] = topk_values.min()
                                elif metric == "mean":
                                    topk_metric[topk, 0] = topk_values.mean()
                                elif metric == "std":
                                    topk_metric[topk, 0] = topk_values.std(ddof=1)
                                else:
                                    topk_metric[topk, 0] = (
                                        topk_values < safety_threshold
                                    ).sum() / (topk + 1)

                            metric_dict[ci][estimator][metric] = topk_metric

                        sharpe_ratio = (
                            np.clip(
                                metric_dict[ci][estimator]["best"] - baseline, 0, None
                            )
                            / metric_dict[ci][estimator]["std"]
                        )

                        if clip_sharpe_ratio:
                            sharpe_ratio[1:] = np.nan_to_num(
                                sharpe_ratio[1:], posinf=1e2
                            )
                            sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                        metric_dict[ci][estimator]["sharpe_ratio"] = sharpe_ratio

        else:
            behavior_policy = input_dict[list(input_dict.keys())[0]]["behavior_policy"]
            baseline = behavior_policy_value[behavior_policy]

            for ci in cis:
                for i, estimator in enumerate(compared_estimators):
                    for j, metric in enumerate(metrics):
                        topk_metric = np.zeros((max_topk, 1))

                        for topk in range(max_topk):
                            topk_values = policy_value_dict[ci][estimator][
                                "true_policy_value"
                            ][: topk + 1]

                            if metric == "k-th":
                                topk_metric[topk, 0] = topk_values[-1]
                            elif metric == "best":
                                topk_metric[topk, 0] = topk_values.max()
                            elif metric == "worst":
                                topk_metric[topk, 0] = topk_values.min()
                            elif metric == "mean":
                                topk_metric[topk, 0] = topk_values.mean()
                            elif metric == "std":
                                topk_metric[topk, 0] = topk_values.std(ddof=1)
                            else:
                                topk_metric[topk, 0] = (
                                    topk_values < safety_threshold
                                ).sum() / (topk + 1)

                        metric_dict[ci][estimator][metric] = topk_metric

                    sharpe_ratio = (
                        np.clip(metric_dict[ci][estimator]["best"] - baseline, 0, None)
                        / metric_dict[ci][estimator]["std"]
                    )

                    if clip_sharpe_ratio:
                        sharpe_ratio[1:] = np.nan_to_num(sharpe_ratio[1:], posinf=1e2)
                        sharpe_ratio[1:] = np.clip(sharpe_ratio[1:], 0.0, 1e2)

                    metric_dict[ci][estimator]["sharpe_ratio"] = sharpe_ratio

        metric_dict = defaultdict_to_dict(metric_dict)

        if return_by_dataframe:
            metrics.extend(["sharpe_ratio"])
            metric_df = []

            for ci in cis:
                for estimator in compared_estimators:
                    metric_df_ = pd.DataFrame()
                    metric_df_["topk"] = np.arange(max_topk)
                    metric_df_["estimator"] = estimator
                    metric_df_["ci"] = ci
                    metric_df_ = metric_df_[["ci", "estimator", "topk"]]

                    for metric in metrics:
                        metric_df_[metric] = metric_dict[ci][estimator][metric].mean(
                            axis=1
                        )

                    metric_df.append(metric_df_)

            metric = pd.concat(metric_df, axis=0)

        else:
            metric = metric_dict

        return metric

    def obtain_topk_conditional_value_at_risk_selected_by_standard_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        ope_alpha: float = 0.05,
        max_topk: Optional[int] = None,
        return_safety_violation_rate: bool = False,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        return_by_dataframe: bool = False,
    ):
        """Obtain the topk deployment result (CVaR) selected by standard OPE.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        ope_alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 1]`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        return_safety_violation_rate: bool, default=False.
            Whether to calculate and return the safety violate.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that policy performance refers to CVaR here. When returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            gamma=gamma,
        )

        true_dict = self.obtain_true_selection_result(
            input_dict,
            return_conditional_value_at_risk=True,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            cvar_alpha=ope_alpha,
        )
        estimation_dict = self.select_by_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        return self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking_by_conditional_value_at_risk",
            true_dict_value_arg="conditional_value_at_risk",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            ope_alpha=ope_alpha,
            return_safety_violation_rate=return_safety_violation_rate,
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
            return_by_dataframe=return_by_dataframe,
        )

    def obtain_topk_conditional_value_at_risk_selected_by_cumulative_distribution_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        ope_alpha: float = 0.05,
        max_topk: Optional[int] = None,
        return_safety_violation_rate: bool = False,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        return_by_dataframe: bool = False,
    ):
        """Obtain the topk deployment result (CVaR) selected by cumulative distribution OPE.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        ope_alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 1]`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        return_safety_violation_rate: bool, default=False.
            Whether to calculate and return the safety violate.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that policy performance refers to CVaR here. When returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            gamma=gamma,
        )

        true_dict = self.obtain_true_selection_result(
            input_dict,
            return_conditional_value_at_risk=True,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            cvar_alpha=ope_alpha,
        )
        estimation_dict = self.select_by_conditional_value_at_risk(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            alpha=ope_alpha,
        )

        return self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking_by_conditional_value_at_risk",
            true_dict_value_arg="conditional_value_at_risk",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            ope_alpha=ope_alpha,
            return_safety_violation_rate=return_safety_violation_rate,
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
            return_by_dataframe=return_by_dataframe,
        )

    def obtain_topk_lower_quartile_selected_by_standard_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        ope_alpha: float = 0.05,
        max_topk: Optional[int] = None,
        return_safety_violation_rate: bool = False,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        return_by_dataframe: bool = False,
    ):
        """Obtain the topk deployment result (lower quartile) selected by standard OPE.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 0.5]`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        return_safety_violation_rate: bool, default=False.
            Whether to calculate and return the safety violate.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that policy performance refers to the lower quartile here. When returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            gamma=gamma,
        )

        true_dict = self.obtain_true_selection_result(
            input_dict,
            return_lower_quartile=True,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            quartile_alpha=ope_alpha,
        )
        estimation_dict = self.select_by_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        return self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking_by_lower_quartile",
            true_dict_value_arg="lower_quartile",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            ope_alpha=ope_alpha,
            return_safety_violation_rate=return_safety_violation_rate,
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
            return_by_dataframe=return_by_dataframe,
        )

    def obtain_topk_lower_quartile_selected_by_cumulative_distribution_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        ope_alpha: float = 0.05,
        max_topk: Optional[int] = None,
        return_safety_violation_rate: bool = False,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        return_by_dataframe: bool = False,
    ):
        """Obtain the topk deployment result (lower quartile) selected by cumulative distribution OPE.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 0.5]`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        return_safety_violation_rate: bool, default=False.
            Whether to calculate and return the safety violate.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that policy performance refers to the lower quartile here. When returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            gamma=gamma,
        )

        true_dict = self.obtain_true_selection_result(
            input_dict,
            return_lower_quartile=True,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            quartile_alpha=ope_alpha,
        )
        estimation_dict = self.select_by_lower_quartile(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            alpha=ope_alpha,
        )

        return self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking_by_lower_quartile",
            true_dict_value_arg="lower_quartile",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            ope_alpha=ope_alpha,
            return_safety_violation_rate=return_safety_violation_rate,
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
            return_by_dataframe=return_by_dataframe,
        )

    def _obtain_min_max_val_for_topk_visualization(
        self,
        true_dict: Dict,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
    ):
        """Obtain minimum and maximum policy performance for topk visualization.

        Parameters
        -------
        true_dict: dict
            Dictionary containing the true deployment result.

        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        Return
        -------
        topk_metric_dict/topk_metric_df: dict or dataframe
            Dictionary/dataframe containing the following top-k risk return tradeoff metrics.
            Note that when returning dataframe, the average value will be returned.

            .. code-block:: python

                key: [estimator][
                    k-th,
                    best,  # return
                    worst,  # risk
                    mean,   # risk
                    std,    # risk
                    safety_violation_rate,  # risk
                    sharpe_ratio,  # risk-return tradeoff
                ]

            k-th: ndarray of shape (max_topk, total_n_datasets)
                Policy performance of the k-th deployment policy.

            best: ndarray of shape (max_topk, total_n_datasets)
                Best policy performance among the top-k deployment policies.

            worst: ndarray of shape (max_topk, total_n_datasets)
                Wosrt policy performance among the top-k deployment policies.

            mean: ndarray of shape (max_topk, total_n_datasets)
                Mean policy performance of the top-k deployment policies.

            std: ndarray of shape (max_topk, total_n_datasets)
                Standard deviation of the policy performance among the top-k deployment policies.

            safety_violation_rate: ndarray of shape (max_topk, total_n_datasets)
                Safety violation rate regarding the policy performance of the top-k deployment policies.

            sharpe_ratio: ndarray of shape (max_topk, total_n_datasets)
                Risk-return tradeoff metrics defined as follows: :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        """
        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None and dataset_id is None:
                n_datasets = input_dict.n_datasets
                total_n_datasets = np.array(list(n_datasets.values())).sum()
            elif behavior_policy_name is None and dataset_id is not None:
                total_n_datasets = len(input_dict.behavior_policy_names)
            elif behavior_policy_name is not None and dataset_id is None:
                total_n_datasets = input_dict.n_datasets[behavior_policy_name]
            else:
                total_n_datasets = 1

        if isinstance(input_dict, MultipleInputDict):
            min_vals = np.zeros(total_n_datasets)
            max_vals = np.zeros(total_n_datasets)

            if behavior_policy_name is None and dataset_id is None:
                l = 0
                for behavior_policy, n_datasets in input_dict.n_datasets.items():
                    for dataset_id_ in range(n_datasets):
                        min_vals[l] = np.array(
                            list(true_dict[behavior_policy][dataset_id_].values())
                        ).min()
                        max_vals[l] = np.array(
                            list(true_dict[behavior_policy][dataset_id_].values())
                        ).max()
                        l += 1

            elif behavior_policy_name is None and dataset_id is not None:
                for l, behavior_policy in enumerate(input_dict.behavior_policy_names):
                    min_vals[l] = np.array(
                        list(true_dict[behavior_policy].values())
                    ).min()
                    max_vals[l] = np.array(
                        list(true_dict[behavior_policy].values())
                    ).max()

            elif behavior_policy_name is not None and dataset_id is None:
                for l in range(total_n_datasets):
                    min_vals[l] = np.array(list(true_dict[l].values())).min()
                    max_vals[l] = np.array(list(true_dict[l].values())).max()

            else:
                min_vals[0] = np.array(list(true_dict.values())).min()
                max_vals[0] = np.array(list(true_dict.values())).max()

            min_val = min_vals.mean()
            max_val = max_vals.mean()

        else:
            min_val = np.array(list(true_dict.values())).min()
            max_val = np.array(list(true_dict.values())).max()

        return min_val, max_val

    def _visualize_topk_policy_performance(
        self,
        metric_dict: Dict,
        min_val: float,
        max_val: float,
        compared_estimators: Optional[List[str]] = None,
        metrics: List[str] = [
            "k-th",
            "best",
            "worst",
            "mean",
            "std",
            "safety_violation_rate",
            "sharpe_ratio",
        ],
        max_topk: Optional[int] = None,
        safety_threshold: Optional[float] = None,
        visualize_ci: bool = False,
        ci: str = "bootstrap",
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        legend: bool = True,
        ylabel: str = "policy performance",
        ymax_sharpe_ratio: Optional[float] = None,
        fig_dir: Optional[Path] = None,
        fig_name: Optional[str] = None,
    ):
        """Visualize top-k policy deployment performances.

        Parameters
        -------
        metric_dict: dict
            Dictionary containing the top-k risk return tradeoff metrics.

        min_val: float
            Minimum value in the plot.

        max_val: float
            Maximum value in the plot.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        metrics: list of {"k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"}, default=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"]
            Indicate which of the policy performance among {"best", "worst", "mean", "std"}, sharpe ratio, and safety violation rate to report.
            For "k-th", it means that the policy performance of the (estimated) k-th policy will be visualized.

            We define the sharpe ratio for OPE as :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        visualize_ci: bool, default=False
            Whether to visualize ci.

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        legend: bool, default=True
            Whether to include a legend in the figure.

        ylabel: str, default="policy performance"
            Label of the y-axis.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="topk_policy_value_standard_ope.png"
            Name of the bar figure.

        """
        yaxis_min_val = (
            min_val if safety_threshold is None else min(min_val, safety_threshold)
        )
        yaxis_max_val = (
            max_val if safety_threshold is None else max(max_val, safety_threshold)
        )
        margin = (yaxis_max_val - yaxis_min_val) * 0.05

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        n_figs = len(metrics)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_figs,
            figsize=(6 * n_figs, 4),
        )

        if len(metrics) == 1:
            for i, estimator in enumerate(compared_estimators):
                axes.plot(
                    np.arange(1, max_topk + 1),
                    metric_dict[estimator][metric].mean(axis=1),
                    color=color[i % n_colors],
                    marker=markers[i],
                    label=estimator,
                )

                if visualize_ci:
                    lower = np.zeros(max_topk)
                    upper = np.zeros(max_topk)

                    for topk in range(max_topk):
                        ci_ = self._estimate_confidence_interval[ci](
                            metric_dict[estimator][metric][topk],
                            alpha=alpha,
                            n_bootstrap_samples=n_bootstrap_samples,
                            random_state=random_state,
                        )
                        lower[topk] = ci_[f"{100 * (1. - alpha)}% CI (lower)"]
                        upper[topk] = ci_[f"{100 * (1. - alpha)}% CI (upper)"]

                    axes.fill_between(
                        np.arange(1, max_topk + 1),
                        lower,
                        upper,
                        color=color[i % n_colors],
                        alpha=0.3,
                    )

            if metric in ["k-th", "best", "worst", "mean"]:
                if safety_threshold is not None:
                    axes.plot(
                        np.arange(1, max_topk + 1),
                        np.full(max_topk, safety_threshold),
                        color=dkred,
                        label="safety threshold",
                    )
                    axes.plot(
                        np.arange(1, max_topk + 1),
                        np.full(max_topk, max_val),
                        color="black",
                        linewidth=0.5,
                    )
                    axes.plot(
                        np.arange(1, max_topk + 1),
                        np.full(max_topk, min_val),
                        color="black",
                        linewidth=0.5,
                    )

                axes.set_title(f"{metric}")
                axes.set_ylabel(f"{metric} {ylabel}")
                axes.set_ylim(yaxis_min_val - margin, yaxis_max_val + margin)

            elif metric == "std":
                axes.set_title("std")
                axes.set_ylabel("standard deviation")

            elif metric == "sharpe_ratio":
                axes.plot(
                    np.arange(2, max_topk + 1),
                    np.zeros(max_topk - 1),
                    color="black",
                    linewidth=0.5,
                )

                axes.set_title("sharpe ratio")
                axes.set_ylabel("sharpe ratio")
                axes.set_ylim(0.0, ymax_sharpe_ratio)

            else:
                axes.set_title("safety violation")
                axes.set_ylabel("safety violation rate")
                axes.set_ylim(-0.05, 1.05)

            axes.set_xlabel("# of policies deployed")

            if legend:
                axes.legend(loc="upper right")

        else:
            for j, metric in enumerate(metrics):
                for i, estimator in enumerate(compared_estimators):
                    axes[j].plot(
                        np.arange(1, max_topk + 1),
                        metric_dict[estimator][metric].mean(axis=1),
                        color=color[i % n_colors],
                        marker=markers[i],
                        label=estimator,
                    )

                    if visualize_ci:
                        lower = np.zeros(max_topk)
                        upper = np.zeros(max_topk)

                        for topk in range(max_topk):
                            ci_ = self._estimate_confidence_interval[ci](
                                metric_dict[estimator][metric][topk],
                                alpha=alpha,
                                n_bootstrap_samples=n_bootstrap_samples,
                                random_state=random_state,
                            )
                            lower[topk] = ci_[f"{100 * (1. - alpha)}% CI (lower)"]
                            upper[topk] = ci_[f"{100 * (1. - alpha)}% CI (upper)"]

                        axes[j].fill_between(
                            np.arange(1, max_topk + 1),
                            lower,
                            upper,
                            color=color[i % n_colors],
                            alpha=0.3,
                        )

                if metric in ["k-th", "best", "worst", "mean"]:
                    if safety_threshold is not None:
                        axes[j].plot(
                            np.arange(1, max_topk + 1),
                            np.full(max_topk, safety_threshold),
                            color=dkred,
                            label="safety threshold",
                        )
                        axes[j].plot(
                            np.arange(1, max_topk + 1),
                            np.full(max_topk, max_val),
                            color="black",
                            linewidth=0.5,
                        )
                        axes[j].plot(
                            np.arange(1, max_topk + 1),
                            np.full(max_topk, min_val),
                            color="black",
                            linewidth=0.5,
                        )

                    axes[j].set_title(f"{metric}")
                    axes[j].set_ylabel(f"{metric} {ylabel}")
                    axes[j].set_ylim(yaxis_min_val - margin, yaxis_max_val + margin)

                elif metric == "std":
                    axes[j].set_title("std")
                    axes[j].set_ylabel("standard deviation")

                elif metric == "sharpe_ratio":
                    axes[j].plot(
                        np.arange(2, max_topk + 1),
                        np.zeros(max_topk - 1),
                        color="black",
                        linewidth=0.5,
                    )

                    axes[j].set_title("sharpe ratio")
                    axes[j].set_ylabel("sharpe ratio")
                    axes[j].set_ylim(0.0, ymax_sharpe_ratio)

                else:
                    axes[j].set_title("safety violation")
                    axes[j].set_ylabel("safety violation rate")
                    axes[j].set_ylim(-0.05, 1.05)

                axes[j].set_xlabel("# of policies deployed")

                if legend:
                    axes[j].legend(loc="upper right")

            if legend:
                handles, labels = axes[0].get_legend_handles_labels()
                # n_cols shows err
                # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

        fig.subplots_adjust(hspace=0.35, wspace=0.2)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_topk_policy_value_selected_by_standard_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        metrics: List[str] = [
            "k-th",
            "best",
            "worst",
            "mean",
            "std",
            "safety_violation_rate",
            "sharpe_ratio",
        ],
        max_topk: Optional[int] = None,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        ymax_sharpe_ratio: Optional[float] = None,
        visualize_ci: bool = False,
        plot_ci: str = "bootstrap",
        plot_alpha: float = 0.05,
        plot_n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "topk_policy_value_standard_ope.png",
    ):
        """Visualize the topk deployment result (policy value) selected by standard OPE.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        metrics: list of {"k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"}, default=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"]
            Indicate which of the policy performance among {"best", "worst", "mean", "std"}, sharpe ratio, and safety violation rate to report.
            For "k-th", it means that the policy performance of the (estimated) k-th policy will be visualized.

            We define the sharpe ratio for OPE as :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.
            Only applicable when using a single behavior policy.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        visualize_ci: bool, default=False
            Whether to visualize ci. (Only applicable when :class:`MultipleInputDict` is given.)

        plot_ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        plot_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        plot_n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        legend: bool, default=True
            Whether to include a legend in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="topk_policy_value_standard_ope.png"
            Name of the bar figure.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            metrics=metrics,
            safety_threshold=safety_threshold,
            relative_safety_criteria=relative_safety_criteria,
            gamma=gamma,
        )
        self._check_basic_visualization_inputs(fig_dir=fig_dir, fig_name=fig_name)

        true_dict = self.obtain_true_selection_result(
            input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        estimation_dict = self.select_by_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        # note: true_dict is transformed in this function, as it is passed by reference
        metric_dict = self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking",
            true_dict_value_arg="policy_value",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            return_safety_violation_rate=("safety_violation_rate" in metrics),
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
        )
        # in the case with single input_dict, true_dict has not been transformed
        if not isinstance(input_dict, MultipleInputDict) or (
            behavior_policy_name is not None and dataset_id is not None
        ):
            true_dict = dict(
                zip(
                    true_dict["ranking"],
                    true_dict["policy_value"],
                )
            )

        min_val, max_val = self._obtain_min_max_val_for_topk_visualization(
            true_dict=true_dict,
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        self._visualize_topk_policy_performance(
            metric_dict=metric_dict,
            min_val=min_val,
            max_val=max_val,
            compared_estimators=compared_estimators,
            metrics=metrics,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            visualize_ci=visualize_ci,
            ci=plot_ci,
            alpha=plot_alpha,
            n_bootstrap_samples=plot_n_bootstrap_samples,
            random_state=random_state,
            legend=legend,
            ylabel="policy value",
            ymax_sharpe_ratio=ymax_sharpe_ratio,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_topk_policy_value_selected_by_cumulative_distribution_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        metrics: List[str] = [
            "k-th",
            "best",
            "worst",
            "mean",
            "std",
            "safety_violation_rate",
            "sharpe_ratio",
        ],
        max_topk: Optional[int] = None,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        ymax_sharpe_ratio: Optional[float] = None,
        visualize_ci: bool = False,
        plot_ci: str = "bootstrap",
        plot_alpha: float = 0.05,
        plot_n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "topk_policy_value_cumulative_distribution_ope.png",
    ):
        """Visualize the topk deployment result (policy value) selected by cumulative distribution OPE.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        metrics: list of {"k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"}, default=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"]
            Indicate which of the policy performance among {"best", "worst", "mean", "std"}, sharpe ratio, and safety violation rate to report.
            For "k-th", it means that the policy performance of the (estimated) k-th policy will be visualized.

            We define the sharpe ratio for OPE as :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        visualize_ci: bool, default=False
            Whether to visualize ci. (Only applicable when :class:`MultipleInputDict` is given.)

        plot_ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        plot_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        plot_n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        legend: bool, default=True
            Whether to include a legend in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="topk_policy_value_cumulative_distribution_ope.png"
            Name of the bar figure.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            metrics=metrics,
            safety_threshold=safety_threshold,
            relative_safety_criteria=relative_safety_criteria,
            gamma=gamma,
        )
        self._check_basic_visualization_inputs(fig_dir=fig_dir, fig_name=fig_name)

        true_dict = self.obtain_true_selection_result(
            input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        estimation_dict = self.select_by_policy_value_via_cumulative_distribution_ope(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        # note: true_dict is transformed in this function, as it is passed by reference
        metric_dict = self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking",
            true_dict_value_arg="policy_value",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            return_safety_violation_rate=("safety_violation_rate" in metrics),
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
        )
        # in the case with single input_dict, true_dict has not been transformed
        if not isinstance(input_dict, MultipleInputDict) or (
            behavior_policy_name is not None and dataset_id is not None
        ):
            true_dict = dict(
                zip(
                    true_dict["ranking"],
                    true_dict["policy_value"],
                )
            )

        min_val, max_val = self._obtain_min_max_val_for_topk_visualization(
            true_dict=true_dict,
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        self._visualize_topk_policy_performance(
            metric_dict=metric_dict,
            min_val=min_val,
            max_val=max_val,
            compared_estimators=compared_estimators,
            metrics=metrics,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            visualize_ci=visualize_ci,
            ci=plot_ci,
            alpha=plot_alpha,
            n_bootstrap_samples=plot_n_bootstrap_samples,
            random_state=random_state,
            legend=legend,
            ylabel="policy value",
            ymax_sharpe_ratio=ymax_sharpe_ratio,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_topk_policy_value_selected_by_lower_bound(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        metrics: List[str] = [
            "k-th",
            "best",
            "worst",
            "mean",
            "std",
            "safety_violation_rate",
            "sharpe_ratio",
        ],
        max_topk: Optional[int] = None,
        safety_threshold: Optional[float] = None,
        relative_safety_criteria: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        ymax_sharpe_ratio: Optional[float] = None,
        ope_cis: List[str] = ["bootstrap"],
        ope_alpha: float = 0.05,
        ope_n_bootstrap_samples: int = 100,
        visualize_ci: bool = False,
        plot_ci: str = "bootstrap",
        plot_alpha: float = 0.05,
        plot_n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "topk_policy_value_standard_ope_lower_bound.png",
    ):
        """Visualize the topk deployment result (policy value) selected by its estimated lower bound.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        metrics: list of {"k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"}, default=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"]
            Indicate which of the policy performance among {"best", "worst", "mean", "std"}, sharpe ratio, and safety violation rate to report.
            For "k-th", it means that the policy performance of the (estimated) k-th policy will be visualized.

            We define the sharpe ratio for OPE as :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        safety_threshold: float, default=None.
            A policy whose policy value is below the given threshold is to be considered unsafe.

        relative_safety_criteria: float, default=None
            The relative policy value required to be considered a safe policy.
            For example, when 0.9 is given, candidate policy must exceed 90\\% of the behavior policy performance.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        ope_cis: list of {"bootstrap", "hoeffding", "bernstein", "ttest"}, default=["bootstrap"]
            Estimation methods for confidence intervals.

        ope_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        ope_n_bootstrap_samples: int, default=100 (> 0)
            Number of resampling performed in the bootstrap procedure.

        visualize_ci: bool, default=False
            Whether to visualize ci. (Only applicable when :class:`MultipleInputDict` is given.)

        plot_ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        plot_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        plot_n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        legend: bool, default=True
            Whether to include a legend in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="topk_policy_value_standard_ope_lower_bound.png"
            Name of the bar figure.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            metrics=metrics,
            safety_threshold=safety_threshold,
            relative_safety_criteria=relative_safety_criteria,
            gamma=gamma,
        )
        self._check_basic_visualization_inputs(fig_dir=fig_dir, fig_name=fig_name)

        true_dict = self.obtain_true_selection_result(
            input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        metric_dict = self.obtain_topk_policy_value_selected_by_lower_bound(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            return_safety_violation_rate=("safety_violation_rate" in metrics),
            safety_threshold=safety_threshold,
            relative_safety_criteria=relative_safety_criteria,
            clip_sharpe_ratio=clip_sharpe_ratio,
            cis=ope_cis,
            ope_alpha=ope_alpha,
            n_bootstrap_samples=ope_n_bootstrap_samples,
            random_state=random_state,
        )

        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None and dataset_id is None:
                for behavior_policy, n_datasets in input_dict.n_datasets.items():
                    for dataset_id_ in range(n_datasets):
                        true_dict[behavior_policy][dataset_id_] = dict(
                            zip(
                                true_dict[behavior_policy][dataset_id_]["policy_value"],
                                true_dict[behavior_policy][dataset_id_]["policy_value"],
                            )
                        )
            elif behavior_policy_name is None and dataset_id is not None:
                for behavior_policy in input_dict.behavior_policy_names:
                    true_dict[behavior_policy] = dict(
                        zip(
                            true_dict[behavior_policy]["policy_value"],
                            true_dict[behavior_policy]["policy_value"],
                        )
                    )
            elif behavior_policy_name is not None and dataset_id is None:
                for dataset_id_ in range(input_dict.n_datasets[behavior_policy_name]):
                    true_dict[dataset_id_] = dict(
                        zip(
                            true_dict[dataset_id_]["policy_value"],
                            true_dict[dataset_id_]["policy_value"],
                        )
                    )
            else:
                true_dict = dict(
                    zip(
                        true_dict["policy_value"],
                        true_dict["policy_value"],
                    )
                )
        else:
            true_dict = dict(
                zip(
                    true_dict["policy_value"],
                    true_dict["policy_value"],
                )
            )

        min_val, max_val = self._obtain_min_max_val_for_topk_visualization(
            true_dict=true_dict,
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        yaxis_min_val = (
            min_val if safety_threshold is None else min(min_val, safety_threshold)
        )
        yaxis_max_val = (
            max_val if safety_threshold is None else max(max_val, safety_threshold)
        )
        margin = (yaxis_max_val - yaxis_min_val) * 0.05

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        n_rows = len(ope_cis)
        n_cols = len(metrics)

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
        )

        if n_rows == 1:
            ope_ci = ope_cis[0]

            if len(metrics) == 1:
                for i, estimator in enumerate(compared_estimators):
                    axes.plot(
                        np.arange(1, max_topk + 1),
                        metric_dict[ope_ci][estimator][metric].mean(axis=1),
                        color=color[i % n_colors],
                        marker=markers[i],
                        label=estimator,
                    )

                    if visualize_ci:
                        lower = np.zeros(max_topk)
                        upper = np.zeros(max_topk)

                        for topk in range(max_topk):
                            ci = self._estimate_confidence_interval[plot_ci](
                                metric_dict[ope_ci][estimator][metric][topk],
                                alpha=plot_alpha,
                                n_bootstrap_samples=plot_n_bootstrap_samples,
                                random_state=random_state,
                            )
                            lower[topk] = ci[f"{100 * (1. - plot_alpha)}% CI (lower)"]
                            upper[topk] = ci[f"{100 * (1. - plot_alpha)}% CI (upper)"]

                        axes.fill_between(
                            np.arange(1, max_topk + 1),
                            lower,
                            upper,
                            color=color[i % n_colors],
                            alpha=0.3,
                        )

                if metric in ["k-th", "best", "worst", "mean"]:
                    if safety_threshold is not None:
                        axes.plot(
                            np.arange(1, max_topk + 1),
                            np.full(max_topk, safety_threshold),
                            color=dkred,
                            label="safety threshold",
                        )
                        axes.plot(
                            np.arange(1, max_topk + 1),
                            np.full(max_topk, max_val),
                            color="black",
                            linewidth=0.5,
                        )
                        axes.plot(
                            np.arange(1, max_topk + 1),
                            np.full(max_topk, min_val),
                            color="black",
                            linewidth=0.5,
                        )

                    axes.set_title(f"{metric}")
                    axes.set_ylabel(f"{metric} policy value")
                    axes.set_ylim(yaxis_min_val - margin, yaxis_max_val + margin)

                elif metric == "std":
                    axes.set_title("std")
                    axes.set_ylabel("standard deviation")

                elif metric == "sharpe_ratio":
                    axes.plot(
                        np.arange(2, max_topk + 1),
                        np.zeros(max_topk - 1),
                        color="black",
                        linewidth=0.5,
                    )

                    axes.set_title("sharpe ratio")
                    axes.set_ylabel("sharpe ratio")
                    axes.set_ylim(0.0, ymax_sharpe_ratio)

                else:
                    axes.set_title("safety violation")
                    axes.set_ylabel("safety violation rate")
                    axes.set_ylim(-0.05, 1.05)

                axes.set_xlabel("# of policies deployed")

                if legend:
                    axes.legend(loc="upper right")

            if legend:
                handles, labels = axes.get_legend_handles_labels()
                # n_cols shows err
                # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

            else:
                for j, metric in enumerate(metrics):
                    for i, estimator in enumerate(compared_estimators):
                        axes[j].plot(
                            np.arange(1, max_topk + 1),
                            metric_dict[ope_ci][estimator][metric].mean(axis=1),
                            color=color[i % n_colors],
                            marker=markers[i],
                            label=estimator,
                        )

                        if visualize_ci:
                            lower = np.zeros(max_topk)
                            upper = np.zeros(max_topk)

                            for topk in range(max_topk):
                                ci = self._estimate_confidence_interval[plot_ci](
                                    metric_dict[ope_ci][estimator][metric][topk],
                                    alpha=plot_alpha,
                                    n_bootstrap_samples=plot_n_bootstrap_samples,
                                    random_state=random_state,
                                )
                                lower[topk] = ci[
                                    f"{100 * (1. - plot_alpha)}% CI (lower)"
                                ]
                                upper[topk] = ci[
                                    f"{100 * (1. - plot_alpha)}% CI (upper)"
                                ]

                            axes[j].fill_between(
                                np.arange(1, max_topk + 1),
                                lower,
                                upper,
                                color=color[i % n_colors],
                                alpha=0.3,
                            )

                    if metric in ["k-th", "best", "worst", "mean"]:
                        if safety_threshold is not None:
                            axes[j].plot(
                                np.arange(1, max_topk + 1),
                                np.full(max_topk, safety_threshold),
                                color=dkred,
                                label="safety threshold",
                            )
                            axes[j].plot(
                                np.arange(1, max_topk + 1),
                                np.full(max_topk, max_val),
                                color="black",
                                linewidth=0.5,
                            )
                            axes[j].plot(
                                np.arange(1, max_topk + 1),
                                np.full(max_topk, min_val),
                                color="black",
                                linewidth=0.5,
                            )

                        axes[j].set_title(f"{metric}")
                        axes[j].set_ylabel(f"{metric} policy value")
                        axes[j].set_ylim(yaxis_min_val - margin, yaxis_max_val + margin)

                    elif metric == "std":
                        axes[j].set_title("std")
                        axes[j].set_ylabel("standard deviation")

                    elif metric == "sharpe_ratio":
                        axes.plot(
                            np.arange(2, max_topk + 1),
                            np.zeros(max_topk - 1),
                            color="black",
                            linewidth=0.5,
                        )

                        axes[j].set_title("sharpe ratio")
                        axes[j].set_ylabel("sharpe ratio")
                        axes[j].set_ylim(0.0, ymax_sharpe_ratio)

                    else:
                        axes[j].set_title("safety violation")
                        axes[j].set_ylabel("safety violation rate")
                        axes[j].set_ylim(-0.05, 1.05)

                    axes[j].set_xlabel("# of policies deployed")

                    if legend:
                        axes[j].legend(loc="upper right")

                if legend:
                    handles, labels = axes[0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

        else:
            if len(metrics) == 1:
                for l, ope_ci in enumerate(ope_cis):
                    for i, estimator in enumerate(compared_estimators):
                        axes[l].plot(
                            np.arange(1, max_topk + 1),
                            metric_dict[ope_ci][estimator][metric].mean(axis=1),
                            color=color[i % n_colors],
                            marker=markers[i],
                            label=estimator,
                        )

                        if visualize_ci:
                            lower = np.zeros(max_topk)
                            upper = np.zeros(max_topk)

                            for topk in range(max_topk):
                                ci = self._estimate_confidence_interval[plot_ci](
                                    metric_dict[ope_ci][estimator][metric][topk],
                                    alpha=plot_alpha,
                                    n_bootstrap_samples=plot_n_bootstrap_samples,
                                    random_state=random_state,
                                )
                                lower[topk] = ci[
                                    f"{100 * (1. - plot_alpha)}% CI (lower)"
                                ]
                                upper[topk] = ci[
                                    f"{100 * (1. - plot_alpha)}% CI (upper)"
                                ]

                            axes[l].fill_between(
                                np.arange(1, max_topk + 1),
                                lower,
                                upper,
                                color=color[i % n_colors],
                                alpha=0.3,
                            )

                    if metric in ["k-th", "best", "worst", "mean"]:
                        if safety_threshold is not None:
                            axes[l].plot(
                                np.arange(1, max_topk + 1),
                                np.full(max_topk, safety_threshold),
                                color=dkred,
                                label="safety threshold",
                            )
                            axes[l].plot(
                                np.arange(1, max_topk + 1),
                                np.full(max_topk, max_val),
                                color="black",
                                linewidth=0.5,
                            )
                            axes[l].plot(
                                np.arange(1, max_topk + 1),
                                np.full(max_topk, min_val),
                                color="black",
                                linewidth=0.5,
                            )

                        axes[l].set_title(f"{metric}")
                        axes[l].set_ylabel(f"{metric} policy value")
                        axes[l].set_ylim(yaxis_min_val - margin, yaxis_max_val + margin)

                    elif metric == "std":
                        axes[l].set_title("std")
                        axes[l].set_ylabel("standard deviation")

                    elif metric == "sharpe_ratio":
                        axes[l].plot(
                            np.arange(2, max_topk + 1),
                            np.zeros(max_topk - 1),
                            color="black",
                            linewidth=0.5,
                        )

                        axes[l].set_title("sharpe ratio")
                        axes[l].set_ylabel("sharpe ratio")
                        axes[l].set_ylim(0.0, ymax_sharpe_ratio)

                    else:
                        axes[l].set_title("safety violation")
                        axes[l].set_ylabel("safety violation rate")
                        axes[l].set_ylim(-0.05, 1.05)

                    axes[l].set_xlabel("# of policies deployed")

                    if legend:
                        axes[l].legend(loc="upper right")

                if legend:
                    handles, labels = axes[0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

            else:
                for l, ope_ci in enumerate(ope_cis):
                    for j, metric in enumerate(metrics):
                        for i, estimator in enumerate(compared_estimators):
                            axes[l, j].plot(
                                np.arange(1, max_topk + 1),
                                metric_dict[ope_ci][estimator][metric].mean(axis=1),
                                color=color[i % n_colors],
                                marker=markers[i],
                                label=estimator,
                            )

                            if visualize_ci:
                                lower = np.zeros(max_topk)
                                upper = np.zeros(max_topk)

                                for topk in range(max_topk):
                                    ci = self._estimate_confidence_interval[plot_ci](
                                        metric_dict[ope_ci][estimator][metric][topk],
                                        alpha=plot_alpha,
                                        n_bootstrap_samples=plot_n_bootstrap_samples,
                                        random_state=random_state,
                                    )
                                    lower[topk] = ci[
                                        f"{100 * (1. - plot_alpha)}% CI (lower)"
                                    ]
                                    upper[topk] = ci[
                                        f"{100 * (1. - plot_alpha)}% CI (upper)"
                                    ]

                                axes[l, j].fill_between(
                                    np.arange(1, max_topk + 1),
                                    lower,
                                    upper,
                                    color=color[i % n_colors],
                                    alpha=0.3,
                                )

                        if metric in ["k-th", "best", "worst", "mean"]:
                            if safety_threshold is not None:
                                axes[l, j].plot(
                                    np.arange(1, max_topk + 1),
                                    np.full(max_topk, safety_threshold),
                                    color=dkred,
                                    label="safety threshold",
                                )
                                axes[l, j].plot(
                                    np.arange(1, max_topk + 1),
                                    np.full(max_topk, max_val),
                                    color="black",
                                    linewidth=0.5,
                                )
                                axes[l, j].plot(
                                    np.arange(1, max_topk + 1),
                                    np.full(max_topk, min_val),
                                    color="black",
                                    linewidth=0.5,
                                )

                            axes[l, j].set_title(f"{metric}")
                            axes[l, j].set_ylabel(f"{metric} policy value")
                            axes[l, j].set_ylim(
                                yaxis_min_val - margin, yaxis_max_val + margin
                            )

                        elif metric == "std":
                            axes[l, j].set_title("std")
                            axes[l, j].set_ylabel("standard deviation")

                        elif metric == "sharpe_ratio":
                            axes[l, j].plot(
                                np.arange(2, max_topk + 1),
                                np.zeros(max_topk - 1),
                                color="black",
                                linewidth=0.5,
                            )

                            axes[l, j].set_title("sharpe ratio")
                            axes[l, j].set_ylabel("sharpe ratio")
                            axes[l, j].set_ylim(0.0, ymax_sharpe_ratio)

                        else:
                            axes[l, j].set_title("safety violation")
                            axes[l, j].set_ylabel("safety violation rate")
                            axes[l, j].set_ylim(-0.05, 1.05)

                        axes[l, j].set_xlabel("# of policies deployed")

                        if legend:
                            axes[l, j].legend(loc="upper right")

                if legend:
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    # n_cols shows err
                    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), n_cols=min(len(labels), 6))

        fig.subplots_adjust(hspace=0.35, wspace=0.2)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_topk_conditional_value_at_risk_selected_by_standard_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        ope_alpha: float = 0.05,
        metrics: List[str] = [
            "k-th",
            "best",
            "worst",
            "mean",
            "std",
            "safety_violation_rate",
            "sharpe_ratio",
        ],
        max_topk: Optional[int] = None,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        ymax_sharpe_ratio: Optional[float] = None,
        visualize_ci: bool = False,
        plot_ci: str = "bootstrap",
        plot_alpha: float = 0.05,
        plot_n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "topk_cvar_standard_ope.png",
    ):
        """Visualize the topk deployment result (CVaR) selected by standard OPE.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        ope_alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 1]`.

        metrics: list of {"k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"}, default=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"]
            Indicate which of the policy performance among {"best", "worst", "mean", "std"}, sharpe ratio, and safety violation rate to report.
            For "k-th", it means that the policy performance of the (estimated) k-th policy will be visualized.

            We define the sharpe ratio for OPE as :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        safety_threshold: float, default=0.0 (>= 0)
            The conditional value at risk required to be considered a safe policy.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        visualize_ci: bool, default=False
            Whether to visualize ci. (Only applicable when :class:`MultipleInputDict` is given.)

        plot_ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        plot_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        plot_n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        legend: bool, default=True
            Whether to include a legend in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="topk_cvar_standard_ope.png"
            Name of the bar figure.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            metrics=metrics,
            safety_threshold=safety_threshold,
            gamma=gamma,
        )
        self._check_basic_visualization_inputs(fig_dir=fig_dir, fig_name=fig_name)

        true_dict = self.obtain_true_selection_result(
            input_dict,
            return_conditional_value_at_risk=True,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            cvar_alpha=ope_alpha,
        )
        estimation_dict = self.select_by_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        # note: true_dict is transformed in this function, as it is passed by reference
        metric_dict = self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking_by_conditional_value_at_risk",
            true_dict_value_arg="conditional_value_at_risk",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            ope_alpha=ope_alpha,
            return_safety_violation_rate=("safety_violation_rate" in metrics),
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
        )
        # in the case with single input_dict, true_dict has not been transformed
        if not isinstance(input_dict, MultipleInputDict) or (
            behavior_policy_name is not None and dataset_id is not None
        ):
            true_dict = dict(
                zip(
                    true_dict["ranking_by_conditional_value_at_risk"],
                    true_dict["conditional_value_at_risk"],
                )
            )

        min_val, max_val = self._obtain_min_max_val_for_topk_visualization(
            true_dict=true_dict,
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        self._visualize_topk_policy_performance(
            metric_dict=metric_dict,
            min_val=min_val,
            max_val=max_val,
            compared_estimators=compared_estimators,
            metrics=metrics,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            visualize_ci=visualize_ci,
            ci=plot_ci,
            alpha=plot_alpha,
            n_bootstrap_samples=plot_n_bootstrap_samples,
            random_state=random_state,
            ymax_sharpe_ratio=ymax_sharpe_ratio,
            legend=legend,
            ylabel=f"CVaR ({ope_alpha})",
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_topk_conditional_value_at_risk_selected_by_cumulative_distribution_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        ope_alpha: float = 0.05,
        metrics: List[str] = [
            "k-th",
            "best",
            "worst",
            "mean",
            "std",
            "safety_violation_rate",
            "sharpe_ratio",
        ],
        max_topk: Optional[int] = None,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        ymax_sharpe_ratio: Optional[float] = None,
        visualize_ci: bool = False,
        plot_ci: str = "bootstrap",
        plot_alpha: float = 0.05,
        plot_n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "topk_cvar_cumulative_distribution_ope.png",
    ):
        """Visualize the topk deployment result (CVaR) selected by cumulative distribution OPE.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        ope_alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 1]`.

        metrics: list of {"k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"}, default=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"]
            Indicate which of the policy performance among {"best", "worst", "mean", "std"}, sharpe ratio, and safety violation rate to report.
            For "k-th", it means that the policy performance of the (estimated) k-th policy will be visualized.

            We define the sharpe ratio for OPE as :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        safety_threshold: float, default=0.0 (>= 0)
            The conditional value at risk required to be considered a safe policy.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        visualize_ci: bool, default=False
            Whether to visualize ci. (Only applicable when :class:`MultipleInputDict` is given.)

        plot_ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        plot_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        plot_n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        legend: bool, default=True
            Whether to include a legend in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="topk_cvar_cumulative_distribution_ope.png"
            Name of the bar figure.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            metrics=metrics,
            safety_threshold=safety_threshold,
            gamma=gamma,
        )
        self._check_basic_visualization_inputs(fig_dir=fig_dir, fig_name=fig_name)

        true_dict = self.obtain_true_selection_result(
            input_dict,
            return_conditional_value_at_risk=True,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            cvar_alpha=ope_alpha,
        )
        estimation_dict = self.select_by_conditional_value_at_risk(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            alpha=ope_alpha,
        )
        # note: true_dict is transformed in this function, as it is passed by reference
        metric_dict = self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking_by_conditional_value_at_risk",
            true_dict_value_arg="conditional_value_at_risk",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            ope_alpha=ope_alpha,
            return_safety_violation_rate=("safety_violation_rate" in metrics),
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
        )
        # in the case with single input_dict, true_dict has not been transformed
        if not isinstance(input_dict, MultipleInputDict) or (
            behavior_policy_name is not None and dataset_id is not None
        ):
            true_dict = dict(
                zip(
                    true_dict["ranking_by_conditional_value_at_risk"],
                    true_dict["conditional_value_at_risk"],
                )
            )

        min_val, max_val = self._obtain_min_max_val_for_topk_visualization(
            true_dict=true_dict,
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        self._visualize_topk_policy_performance(
            metric_dict=metric_dict,
            min_val=min_val,
            max_val=max_val,
            compared_estimators=compared_estimators,
            metrics=metrics,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            visualize_ci=visualize_ci,
            ci=plot_ci,
            alpha=plot_alpha,
            n_bootstrap_samples=plot_n_bootstrap_samples,
            random_state=random_state,
            legend=legend,
            ylabel=f"CVaR ({ope_alpha})",
            ymax_sharpe_ratio=ymax_sharpe_ratio,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_topk_lower_quartile_selected_by_standard_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        ope_alpha: float = 0.05,
        metrics: List[str] = [
            "k-th",
            "best",
            "worst",
            "mean",
            "std",
            "safety_violation_rate",
            "sharpe_ratio",
        ],
        max_topk: Optional[int] = None,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        ymax_sharpe_ratio: Optional[float] = None,
        visualize_ci: bool = False,
        plot_ci: str = "bootstrap",
        plot_alpha: float = 0.05,
        plot_n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "topk_lower_quartile_standard_ope.png",
    ):
        """Visualize the topk deployment result (lower quartile) selected by standard OPE.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 0.5]`.

        metrics: list of {"k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"}, default=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"]
            Indicate which of the policy performance among {"best", "worst", "mean", "std"}, sharpe ratio, and safety violation rate to report.
            For "k-th", it means that the policy performance of the (estimated) k-th policy will be visualized.

            We define the sharpe ratio for OPE as :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        safety_threshold: float, default=0.0 (>= 0)
            The conditional value at risk required to be considered a safe policy.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        visualize_ci: bool, default=False
            Whether to visualize ci. (Only applicable when :class:`MultipleInputDict` is given.)

        plot_ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        plot_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        plot_n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        legend: bool, default=True
            Whether to include a legend in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="topk_lower_quartile_standard_ope.png"
            Name of the bar figure.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            metrics=metrics,
            safety_threshold=safety_threshold,
            gamma=gamma,
        )
        self._check_basic_visualization_inputs(fig_dir=fig_dir, fig_name=fig_name)

        true_dict = self.obtain_true_selection_result(
            input_dict,
            return_lower_quartile=True,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            quartile_alpha=ope_alpha,
        )
        estimation_dict = self.select_by_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )
        # note: true_dict is transformed in this function, as it is passed by reference
        metric_dict = self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking_by_lower_quartile",
            true_dict_value_arg="lower_quartile",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            ope_alpha=ope_alpha,
            return_safety_violation_rate=("safety_violation_rate" in metrics),
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
        )
        # in the case with single input_dict, true_dict has not been transformed
        if not isinstance(input_dict, MultipleInputDict) or (
            behavior_policy_name is not None and dataset_id is not None
        ):
            true_dict = dict(
                zip(
                    true_dict["ranking_by_lower_quartile"],
                    true_dict["lower_quartile"],
                )
            )

        min_val, max_val = self._obtain_min_max_val_for_topk_visualization(
            true_dict=true_dict,
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        self._visualize_topk_policy_performance(
            metric_dict=metric_dict,
            min_val=min_val,
            max_val=max_val,
            compared_estimators=compared_estimators,
            metrics=metrics,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            visualize_ci=visualize_ci,
            ci=plot_ci,
            alpha=plot_alpha,
            n_bootstrap_samples=plot_n_bootstrap_samples,
            random_state=random_state,
            legend=legend,
            ylabel=f"lower quartile ({ope_alpha})",
            ymax_sharpe_ratio=ymax_sharpe_ratio,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_topk_lower_quartile_selected_by_cumulative_distribution_ope(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        ope_alpha: float = 0.05,
        metrics: List[str] = [
            "k-th",
            "best",
            "worst",
            "mean",
            "std",
            "safety_violation_rate",
            "sharpe_ratio",
        ],
        max_topk: Optional[int] = None,
        safety_threshold: Optional[float] = None,
        clip_sharpe_ratio: bool = False,
        ymax_sharpe_ratio: Optional[float] = None,
        visualize_ci: bool = False,
        plot_ci: str = "bootstrap",
        plot_alpha: float = 0.05,
        plot_n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "topk_lower_quartile_cumulative_distribution_ope.png",
    ):
        """Visualize the topk deployment result (lower quartile) selected by cumulative distribution OPE.

        Parameters
        -------
        input_dict: OPEInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.
            If `None`, the average of the result will be shown.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 0.5]`.

        metrics: list of {"k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"}, default=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"]
            Indicate which of the policy performance among {"best", "worst", "mean", "std"}, sharpe ratio, and safety violation rate to report.
            For "k-th", it means that the policy performance of the (estimated) k-th policy will be visualized.

            We define the sharpe ratio for OPE as :math:`S(\\hat{V}) := (\\mathrm{Best@k} - V(\\pi_0)) / \\mathrm{Std@k}`.

        max_topk: int, default=None
            Maximum number of policies to be deployed.
            If `None` is given, all the policies will be deployed.

        safety_threshold: float, default=0.0 (>= 0)
            The conditional value at risk required to be considered a safe policy.

        clip_sharpe_ratio: bool, default=False
            Whether to clip a large value of SharpeRatio with 1e2.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        visualize_ci: bool, default=False
            Whether to visualize ci. (Only applicable when :class:`MultipleInputDict` is given.)

        plot_ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Method to estimate the confidence interval.

        plot_alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        plot_n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        ymax_sharp_ratio: float, default=None
            Maximum value in y-axis of the plot of SharpeRatio.

        legend: bool, default=True
            Whether to include a legend in the figure.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="topk_lower_quartile_cumulative_distribution_ope.png"
            Name of the bar figure.

        """
        if isinstance(input_dict, MultipleInputDict):
            input_dict_ = input_dict.get(
                behavior_policy_name=input_dict.behavior_policy_names[0], dataset_id=0
            )
            gamma = list(input_dict_.values())[0]["gamma"]
        else:
            gamma = list(input_dict.values())[0]["gamma"]

        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        max_topk, safety_threshold = self._check_topk_inputs(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            metrics=metrics,
            safety_threshold=safety_threshold,
            gamma=gamma,
        )
        self._check_basic_visualization_inputs(fig_dir=fig_dir, fig_name=fig_name)

        true_dict = self.obtain_true_selection_result(
            input_dict,
            return_lower_quartile=True,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            quartile_alpha=ope_alpha,
        )
        estimation_dict = self.select_by_lower_quartile(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            alpha=ope_alpha,
        )
        # note: true_dict is transformed in this function, as it is passed by reference
        metric_dict = self._obtain_topk_policy_performance(
            true_dict=true_dict,
            estimation_dict=estimation_dict,
            input_dict=input_dict,
            true_dict_ranking_arg="ranking_by_lower_quartile",
            true_dict_value_arg="lower_quartile",
            estimation_dict_ranking_arg="estimated_ranking",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            max_topk=max_topk,
            ope_alpha=ope_alpha,
            return_safety_violation_rate=("safety_violation_rate" in metrics),
            safety_threshold=safety_threshold,
            clip_sharpe_ratio=clip_sharpe_ratio,
        )
        # in the case with single input_dict, true_dict has not been transformed
        if not isinstance(input_dict, MultipleInputDict) or (
            behavior_policy_name is not None and dataset_id is not None
        ):
            true_dict = dict(
                zip(
                    true_dict["ranking_by_lower_quartile"],
                    true_dict["lower_quartile"],
                )
            )

        min_val, max_val = self._obtain_min_max_val_for_topk_visualization(
            true_dict=true_dict,
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        self._visualize_topk_policy_performance(
            metric_dict=metric_dict,
            min_val=min_val,
            max_val=max_val,
            compared_estimators=compared_estimators,
            metrics=metrics,
            max_topk=max_topk,
            safety_threshold=safety_threshold,
            visualize_ci=visualize_ci,
            ci=plot_ci,
            alpha=plot_alpha,
            n_bootstrap_samples=plot_n_bootstrap_samples,
            random_state=random_state,
            legend=legend,
            ylabel=f"lower quartile ({ope_alpha})",
            ymax_sharpe_ratio=ymax_sharpe_ratio,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def _visualize_policy_performance_for_validation(
        self,
        estimation_dict: Dict,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        true_value_arg: str,
        estimated_value_arg: str,
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        legend: bool = True,
        ylabel: str = "policy performance",
        fig_dir: Optional[Path] = None,
        fig_name: Optional[str] = None,
    ):
        """Visualize the correlation between the true and estimated policy performance.

        Parameters
        -------
        estimation_dict: dict
            Dictionary containing the estimated policy performance.

        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        true_value_arg: str
            Name of the key indicating the true policy performance in estimation_dict.

        estimated_value_arg: str
            Name of the key indicating the estimated policy performance in estimation_dict.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        ylabel: str, default="policy performance"
            Label of the y-axis.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default=None
            Name of the bar figure.

        """
        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        n_figs = len(compared_estimators)
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
            for i, estimator in enumerate(compared_estimators):
                if isinstance(input_dict, MultipleInputDict):
                    if behavior_policy_name is None and dataset_id is None:
                        min_val, max_val = np.infty, -np.infty

                        for l, behavior_policy in enumerate(
                            input_dict.behavior_policy_names
                        ):
                            n_datasets = input_dict.n_datasets[behavior_policy]
                            min_vals = np.zeros(n_datasets)
                            max_vals = np.zeros(n_datasets)

                            for dataset_id_ in range(n_datasets):
                                true_policy_value = estimation_dict[behavior_policy][
                                    dataset_id_
                                ][estimator][true_value_arg]
                                estimated_policy_value = estimation_dict[
                                    behavior_policy
                                ][dataset_id_][estimator][estimated_value_arg]

                                if dataset_id_ == 0:
                                    axes[i].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[l % n_colors],
                                        label=behavior_policy,
                                    )
                                else:  # to avoid duplicated labels
                                    axes[i].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[l % n_colors],
                                    )

                                min_vals[dataset_id_] = np.minimum(
                                    np.nanmin(true_policy_value),
                                    np.nanmin(estimated_policy_value),
                                )
                                max_vals[dataset_id_] = np.maximum(
                                    np.nanmax(true_policy_value),
                                    np.nanmax(estimated_policy_value),
                                )

                            min_val = min(min_val, min_vals.min())
                            max_val = max(max_val, max_vals.max())

                    elif behavior_policy_name is None and dataset_id is not None:
                        min_val, max_val = np.infty, -np.infty

                        for l, behavior_policy in enumerate(
                            input_dict.behavior_policy_names
                        ):
                            true_policy_value = estimation_dict[behavior_policy][
                                estimator
                            ][true_value_arg]
                            estimated_policy_value = estimation_dict[behavior_policy][
                                estimator
                            ][estimated_value_arg]

                            axes[i].scatter(
                                true_policy_value,
                                estimated_policy_value,
                                color=color[l % n_colors],
                                label=behavior_policy,
                            )

                            min_val_ = np.minimum(
                                np.nanmin(true_policy_value),
                                np.nanmin(estimated_policy_value),
                            )
                            max_val_ = np.maximum(
                                np.nanmax(true_policy_value),
                                np.nanmax(estimated_policy_value),
                            )

                        min_val = min(min_val, min_val_)
                        max_val = max(max_val, max_val_)

                    elif behavior_policy_name is not None and dataset_id is None:
                        n_datasets = input_dict.n_datasets[behavior_policy_name]
                        min_vals = np.zeros(n_datasets)
                        max_vals = np.zeros(n_datasets)

                        for dataset_id_ in range(n_datasets):
                            true_policy_value = estimation_dict[dataset_id_][estimator][
                                true_value_arg
                            ]
                            estimated_policy_value = estimation_dict[dataset_id_][
                                estimator
                            ][estimated_value_arg]

                            axes[i].scatter(
                                true_policy_value,
                                estimated_policy_value,
                                color=color[0],
                            )

                            min_vals[dataset_id_] = np.minimum(
                                np.nanmin(true_policy_value),
                                np.nanmin(estimated_policy_value),
                            )
                            max_vals[dataset_id_] = np.maximum(
                                np.nanmax(true_policy_value),
                                np.nanmax(estimated_policy_value),
                            )

                        min_val = min_vals.min()
                        max_val = max_vals.max()

                    else:
                        true_policy_value = estimation_dict[estimator][true_value_arg]
                        estimated_policy_value = estimation_dict[estimator][
                            estimated_value_arg
                        ]

                        axes[i].scatter(
                            true_policy_value,
                            estimated_policy_value,
                            color=color[0],
                        )

                        min_val = np.minimum(
                            np.nanmin(true_policy_value),
                            np.nanmin(estimated_policy_value),
                        )
                        max_val = np.maximum(
                            np.nanmax(true_policy_value),
                            np.nanmax(estimated_policy_value),
                        )

                else:
                    true_policy_value = estimation_dict[estimator][true_value_arg]
                    estimated_policy_value = estimation_dict[estimator][
                        estimated_value_arg
                    ]

                    axes[i].scatter(
                        true_policy_value,
                        estimated_policy_value,
                        color=color[0],
                    )

                    min_val = np.minimum(
                        np.nanmin(true_policy_value),
                        np.nanmin(estimated_policy_value),
                    )
                    max_val = np.maximum(
                        np.nanmax(true_policy_value),
                        np.nanmax(estimated_policy_value),
                    )

                axes[i].set_title(estimator)
                axes[i].set_xlabel(f"true {ylabel}")
                axes[i].set_ylabel(f"estimated {ylabel}")

                if (
                    legend
                    and behavior_policy_name is None
                    and isinstance(input_dict, MultipleInputDict)
                ):
                    axes[i].legend(title="behavior_policy", loc="lower right")

                if not share_axes:
                    margin = (max_val - min_val) * 0.05
                    guide = np.linspace(min_val - margin, max_val + margin)
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

            if share_axes:
                margin = (guide_max - guide_min) * 0.05
                guide = np.linspace(guide_min - margin, guide_max + margin)
                for i, estimator in enumerate(compared_estimators):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(compared_estimators):
                if isinstance(input_dict, MultipleInputDict):
                    if behavior_policy_name is None and dataset_id is None:
                        min_val, max_val = np.infty, -np.infty

                        for l, behavior_policy in enumerate(
                            input_dict.behavior_policy_names
                        ):
                            n_datasets = input_dict.n_datasets[behavior_policy]
                            min_vals = np.zeros(n_datasets)
                            max_vals = np.zeros(n_datasets)

                            for dataset_id_ in range(n_datasets):
                                true_policy_value = estimation_dict[behavior_policy][
                                    dataset_id_
                                ][estimator][true_value_arg]
                                estimated_policy_value = estimation_dict[
                                    behavior_policy
                                ][dataset_id_][estimator][estimated_value_arg]

                                if dataset_id_ == 0:
                                    axes[i // n_cols, i % n_cols].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[l % n_colors],
                                        label=behavior_policy,
                                    )
                                else:  # to avoid duplicated labels
                                    axes[i // n_cols, i % n_cols].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[l % n_colors],
                                    )

                                min_vals[dataset_id_] = np.minimum(
                                    np.nanmin(true_policy_value),
                                    np.nanmin(estimated_policy_value),
                                )
                                max_vals[dataset_id_] = np.maximum(
                                    np.nanmax(true_policy_value),
                                    np.nanmax(estimated_policy_value),
                                )

                            min_val = min(min_val, min_vals.min())
                            max_val = max(max_val, max_vals.max())

                    elif behavior_policy_name is None and dataset_id is not None:
                        min_val, max_val = np.infty, -np.infty

                        for l, behavior_policy in enumerate(
                            input_dict.behavior_policy_names
                        ):
                            true_policy_value = estimation_dict[behavior_policy][
                                estimator
                            ][true_value_arg]
                            estimated_policy_value = estimation_dict[behavior_policy][
                                estimator
                            ][estimated_value_arg]

                            axes[i // n_cols, i % n_cols].scatter(
                                true_policy_value,
                                estimated_policy_value,
                                color=color[l % n_colors],
                                label=behavior_policy,
                            )

                            min_val_ = np.minimum(
                                np.nanmin(true_policy_value),
                                np.nanmin(estimated_policy_value),
                            )
                            max_val_ = np.maximum(
                                np.nanmax(true_policy_value),
                                np.nanmax(estimated_policy_value),
                            )

                        min_val = min(min_val, min_val_)
                        max_val = max(max_val, max_val_)

                    elif behavior_policy_name is not None and dataset_id is None:
                        n_datasets = input_dict.n_datasets[behavior_policy_name]
                        min_vals = np.zeros(n_datasets)
                        max_vals = np.zeros(n_datasets)

                        for dataset_id_ in range(n_datasets):
                            true_policy_value = estimation_dict[dataset_id_][estimator][
                                true_value_arg
                            ]
                            estimated_policy_value = estimation_dict[dataset_id_][
                                estimator
                            ][estimated_value_arg]

                            axes[i // n_cols, i % n_cols].scatter(
                                true_policy_value,
                                estimated_policy_value,
                                color=color[0],
                            )

                            min_vals[dataset_id_] = np.minimum(
                                np.nanmin(true_policy_value),
                                np.nanmin(estimated_policy_value),
                            )
                            max_vals[dataset_id_] = np.maximum(
                                np.nanmax(true_policy_value),
                                np.nanmax(estimated_policy_value),
                            )

                        min_val = min_vals.min()
                        max_val = max_vals.max()

                    else:
                        true_policy_value = estimation_dict[estimator][true_value_arg]
                        estimated_policy_value = estimation_dict[estimator][
                            estimated_value_arg
                        ]

                        axes[i // n_cols, i % n_cols].scatter(
                            true_policy_value,
                            estimated_policy_value,
                            color=color[0],
                        )

                        min_val = np.minimum(
                            np.nanmin(true_policy_value),
                            np.nanmin(estimated_policy_value),
                        )
                        max_val = np.maximum(
                            np.nanmax(true_policy_value),
                            np.nanmax(estimated_policy_value),
                        )

                else:
                    true_policy_value = estimation_dict[estimator][true_value_arg]
                    estimated_policy_value = estimation_dict[estimator][
                        estimated_value_arg
                    ]

                    axes[i // n_cols, i % n_cols].scatter(
                        true_policy_value,
                        estimated_policy_value,
                        color=color[0],
                    )

                    min_val = np.minimum(
                        np.nanmin(true_policy_value),
                        np.nanmin(estimated_policy_value),
                    )
                    max_val = np.maximum(
                        np.nanmax(true_policy_value),
                        np.nanmax(estimated_policy_value),
                    )

                axes[i // n_cols, i % n_cols].set_title(estimator)
                axes[i // n_cols, i % n_cols].set_xlabel(f"true {ylabel}")
                axes[i // n_cols, i % n_cols].set_ylabel(f"estimated {ylabel}")

                if (
                    legend
                    and behavior_policy_name is None
                    and isinstance(input_dict, MultipleInputDict)
                ):
                    axes[i // n_cols, i % n_cols].legend(
                        title="behavior_policy", loc="lower right"
                    )

                if not share_axes:
                    margin = (max_val - min_val) * 0.05
                    guide = np.linspace(min_val - margin, max_val + margin)
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

            if share_axes:
                margin = (guide_max - guide_min) * 0.05
                guide = np.linspace(guide_min - margin, guide_max + margin)
                for i, estimator in enumerate(compared_estimators):
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

    def visualize_policy_value_for_validation(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "validation_policy_value_standard_ope.png",
    ):
        """Visualize the true policy value and its estimate (scatter plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="validation_policy_value_standard_ope.png"
            Name of the bar figure.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        self._check_basic_visualization_inputs(
            n_cols=n_cols, fig_dir=fig_dir, fig_name=fig_name
        )

        policy_value_dict = self.select_by_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            return_true_values=True,
        )

        self._visualize_policy_performance_for_validation(
            estimation_dict=policy_value_dict,
            input_dict=input_dict,
            true_value_arg="true_policy_value",
            estimated_value_arg="estimated_policy_value",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            n_cols=n_cols,
            share_axes=share_axes,
            legend=legend,
            ylabel="policy value",
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_policy_value_of_cumulative_distribution_ope_for_validation(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "validation_policy_value_cumulative_distribution_ope.png",
    ):
        """Visualize the true policy value and its estimate obtained by cumulative distribution OPE (scatter plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="validation_policy_value_cumulative_distribution_ope.png"
            Name of the bar figure.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        self._check_basic_visualization_inputs(
            n_cols=n_cols, fig_dir=fig_dir, fig_name=fig_name
        )

        policy_value_dict = self.select_by_policy_value_via_cumulative_distribution_ope(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            return_true_values=True,
        )

        self._visualize_policy_performance_for_validation(
            estimation_dict=policy_value_dict,
            input_dict=input_dict,
            true_value_arg="true_policy_value",
            estimated_value_arg="estimated_policy_value",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            n_cols=n_cols,
            share_axes=share_axes,
            legend=legend,
            ylabel="policy value",
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_policy_value_lower_bound_for_validation(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        cis: List[str] = ["bootstrap"],
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "validation_policy_value_lower_bound.png",
    ):
        """Visualize the true policy value and its estimate lower bound (scatter plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        cis: list of {"bootstrap", "hoeffding", "bernstein", "ttest"}, default=["bootstrap"]
            Estimation methods for confidence intervals.

        alpha: float, default=0.05
            Significance level. The value should be within `[0, 1)`.

        n_bootstrap_samples: int, default=100 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="validation_policy_value_lower_bound.png"
            Name of the bar figure.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="standard_ope"
        )
        self._check_basic_visualization_inputs(
            n_cols=n_cols, fig_dir=fig_dir, fig_name=fig_name
        )

        policy_value_dict = self.select_by_policy_value_lower_bound(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            cis=cis,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
            return_true_values=True,
        )

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        n_figs = len(compared_estimators) * len(cis)
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
            if n_rows == 1:
                for ci in cis:
                    for i, estimator in enumerate(compared_estimators):
                        if isinstance(input_dict, MultipleInputDict):
                            if behavior_policy_name is None and dataset_id is None:
                                min_val, max_val = np.infty, -np.infty

                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    n_datasets = input_dict.n_datasets[behavior_policy]
                                    min_vals = np.zeros(n_datasets)
                                    max_vals = np.zeros(n_datasets)

                                    for dataset_id_ in range(n_datasets):
                                        true_policy_value = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "true_policy_value"
                                        ]
                                        estimated_policy_value = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "estimated_policy_value_lower_bound"
                                        ]

                                        if dataset_id_ == 0:
                                            axes[i].scatter(
                                                true_policy_value,
                                                estimated_policy_value,
                                                color=color[l % n_colors],
                                                label=behavior_policy,
                                            )
                                        else:  # to remove duplicated labels
                                            axes[i].scatter(
                                                true_policy_value,
                                                estimated_policy_value,
                                                color=color[l % n_colors],
                                            )

                                        min_vals[dataset_id_] = np.minimum(
                                            np.nanmin(true_policy_value),
                                            np.nanmin(estimated_policy_value),
                                        )
                                        max_vals[dataset_id_] = np.maximum(
                                            np.nanmax(true_policy_value),
                                            np.nanmax(estimated_policy_value),
                                        )

                                    min_val = min(min_val, min_vals.min())
                                    max_val = max(max_val, max_vals.max())

                            elif (
                                behavior_policy_name is None and dataset_id is not None
                            ):
                                min_val, max_val = np.infty, -np.infty

                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    true_policy_value = policy_value_dict[
                                        behavior_policy
                                    ][ci][estimator]["true_policy_value"]
                                    estimated_policy_value = policy_value_dict[
                                        behavior_policy
                                    ][ci][estimator][
                                        "estimated_policy_value_lower_bound"
                                    ]

                                    axes[i].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[l % n_colors],
                                        label=behavior_policy,
                                    )

                                    min_val_ = np.minimum(
                                        np.nanmin(true_policy_value),
                                        np.nanmin(estimated_policy_value),
                                    )
                                    max_val_ = np.maximum(
                                        np.nanmax(true_policy_value),
                                        np.nanmax(estimated_policy_value),
                                    )

                                min_val = min(min_val, min_val_)
                                max_val = max(max_val, max_val_)

                            elif (
                                behavior_policy_name is not None and dataset_id is None
                            ):
                                n_datasets = input_dict.n_datasets[behavior_policy_name]
                                min_vals = np.zeros(n_datasets)
                                max_vals = np.zeros(n_datasets)

                                for dataset_id_ in range(n_datasets):
                                    true_policy_value = policy_value_dict[dataset_id_][
                                        ci
                                    ][estimator]["true_policy_value"]
                                    estimated_policy_value = policy_value_dict[
                                        dataset_id_
                                    ][ci][estimator][
                                        "estimated_policy_value_lower_bound"
                                    ]

                                    axes[i].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[0],
                                    )

                                    min_vals[dataset_id_] = np.minimum(
                                        np.nanmin(true_policy_value),
                                        np.nanmin(estimated_policy_value),
                                    )
                                    max_vals[dataset_id_] = np.maximum(
                                        np.nanmax(true_policy_value),
                                        np.nanmax(estimated_policy_value),
                                    )

                                min_val = min_vals.min()
                                max_val = max_vals.max()

                            else:
                                true_policy_value = policy_value_dict[ci][estimator][
                                    "true_policy_value"
                                ]
                                estimated_policy_value = policy_value_dict[ci][
                                    estimator
                                ]["estimated_policy_value_lower_bound"]

                                axes[i].scatter(
                                    true_policy_value,
                                    estimated_policy_value,
                                    color=color[0],
                                )

                                min_val = np.minimum(
                                    np.nanmin(true_policy_value),
                                    np.nanmin(estimated_policy_value),
                                )
                                max_val = np.maximum(
                                    np.nanmax(true_policy_value),
                                    np.nanmax(estimated_policy_value),
                                )

                        else:
                            true_policy_value = policy_value_dict[ci][estimator][
                                "true_policy_value"
                            ]
                            estimated_policy_value = policy_value_dict[ci][estimator][
                                "estimated_policy_value_lower_bound"
                            ]

                            axes[i].scatter(
                                true_policy_value,
                                estimated_policy_value,
                                color=color[0],
                            )

                            min_val = np.minimum(
                                np.nanmin(true_policy_value),
                                np.nanmin(estimated_policy_value),
                            )
                            max_val = np.maximum(
                                np.nanmax(true_policy_value),
                                np.nanmax(estimated_policy_value),
                            )

                        axes[i].set_title(f"{ci}, {estimator}")
                        axes[i].set_xlabel("true policy value")
                        axes[i].set_ylabel("estimated policy value lower bound")

                        if (
                            legend
                            and behavior_policy_name is None
                            and isinstance(input_dict, MultipleInputDict)
                        ):
                            axes[i].legend(title="behavior_policy", loc="lower right")

                        if not share_axes:
                            margin = (max_val - min_val) * 0.05
                            guide = np.linspace(min_val - margin, max_val + margin)
                            axes[i].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

                        guide_min = min_val if guide_min > min_val else guide_min
                        guide_max = max_val if guide_max < max_val else guide_max

                    if share_axes:
                        margin = (guide_max - guide_min) * 0.05
                        guide = np.linspace(guide_min - margin, guide_max + margin)
                        for i, estimator in enumerate(compared_estimators):
                            axes[i].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

            else:
                for ci in cis:
                    for i, estimator in enumerate(compared_estimators):
                        if isinstance(input_dict, MultipleInputDict):
                            if behavior_policy_name is None and dataset_id is None:
                                min_val, max_val = np.infty, -np.infty

                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    n_datasets = input_dict.n_datasets[behavior_policy]
                                    min_vals = np.zeros(n_datasets)
                                    max_vals = np.zeros(n_datasets)

                                    for dataset_id_ in range(n_datasets):
                                        true_policy_value = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "true_policy_value"
                                        ]
                                        estimated_policy_value = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "estimated_policy_value_lower_bound"
                                        ]

                                        if dataset_id_ == 0:
                                            axes[i // n_cols, i % n_cols].scatter(
                                                true_policy_value,
                                                estimated_policy_value,
                                                color=color[l % n_colors],
                                                label=behavior_policy,
                                            )
                                        else:  # to remove duplicated labels
                                            axes[i // n_cols, i % n_cols].scatter(
                                                true_policy_value,
                                                estimated_policy_value,
                                                color=color[l % n_colors],
                                            )

                                        min_vals[dataset_id_] = np.minimum(
                                            np.nanmin(true_policy_value),
                                            np.nanmin(estimated_policy_value),
                                        )
                                        max_vals[dataset_id_] = np.maximum(
                                            np.nanmax(true_policy_value),
                                            np.nanmax(estimated_policy_value),
                                        )

                                    min_val = min(min_val, min_vals.min())
                                    max_val = max(max_val, max_vals.max())

                            elif (
                                behavior_policy_name is None and dataset_id is not None
                            ):
                                min_val, max_val = np.infty, -np.infty

                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    true_policy_value = policy_value_dict[
                                        behavior_policy
                                    ][ci][estimator]["true_policy_value"]
                                    estimated_policy_value = policy_value_dict[
                                        behavior_policy
                                    ][ci][estimator][
                                        "estimated_policy_value_lower_bound"
                                    ]

                                    axes[i // n_cols, i % n_cols].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[l % n_colors],
                                        label=behavior_policy,
                                    )

                                    min_val_ = np.minimum(
                                        np.nanmin(true_policy_value),
                                        np.nanmin(estimated_policy_value),
                                    )
                                    max_val_ = np.maximum(
                                        np.nanmax(true_policy_value),
                                        np.nanmax(estimated_policy_value),
                                    )

                                min_val = min(min_val, min_val_)
                                max_val = max(max_val, max_val_)

                            elif (
                                behavior_policy_name is not None and dataset_id is None
                            ):
                                n_datasets = input_dict.n_datasets[behavior_policy_name]
                                min_vals = np.zeros(n_datasets)
                                max_vals = np.zeros(n_datasets)

                                for dataset_id_ in range(n_datasets):
                                    true_policy_value = policy_value_dict[dataset_id_][
                                        ci
                                    ][estimator]["true_policy_value"]
                                    estimated_policy_value = policy_value_dict[
                                        dataset_id_
                                    ][ci][estimator][
                                        "estimated_policy_value_lower_bound"
                                    ]

                                    axes[i // n_cols, i % n_cols].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[0],
                                    )

                                    min_vals[dataset_id_] = np.minimum(
                                        np.nanmin(true_policy_value),
                                        np.nanmin(estimated_policy_value),
                                    )
                                    max_vals[dataset_id_] = np.maximum(
                                        np.nanmax(true_policy_value),
                                        np.nanmax(estimated_policy_value),
                                    )

                                min_val = min_vals.min()
                                max_val = max_vals.max()

                            else:
                                true_policy_value = policy_value_dict[ci][estimator][
                                    "true_policy_value"
                                ]
                                estimated_policy_value = policy_value_dict[ci][
                                    estimator
                                ]["estimated_policy_value_lower_bound"]

                                axes[i // n_cols, i % n_cols].scatter(
                                    true_policy_value,
                                    estimated_policy_value,
                                    color=color[0],
                                )

                                min_val = np.minimum(
                                    np.nanmin(true_policy_value),
                                    np.nanmin(estimated_policy_value),
                                )
                                max_val = np.maximum(
                                    np.nanmax(true_policy_value),
                                    np.nanmax(estimated_policy_value),
                                )

                        else:
                            true_policy_value = policy_value_dict[ci][estimator][
                                "true_policy_value"
                            ]
                            estimated_policy_value = policy_value_dict[ci][estimator][
                                "estimated_policy_value_lower_bound"
                            ]

                            axes[i // n_cols, i % n_cols].scatter(
                                true_policy_value,
                                estimated_policy_value,
                                color=color[0],
                            )

                            min_val = np.minimum(
                                np.nanmin(true_policy_value),
                                np.nanmin(estimated_policy_value),
                            )
                            max_val = np.maximum(
                                np.nanmax(true_policy_value),
                                np.nanmax(estimated_policy_value),
                            )

                        axes[i // n_cols, i % n_cols].set_title(f"{ci}, {estimator}")
                        axes[i // n_cols, i % n_cols].set_xlabel("true policy value")
                        axes[i // n_cols, i % n_cols].set_ylabel(
                            "estimated policy value lower bound"
                        )

                        if (
                            legend
                            and behavior_policy_name is None
                            and isinstance(input_dict, MultipleInputDict)
                        ):
                            axes[i // n_cols, i % n_cols].legend(
                                title="behavior_policy",
                                loc="lower right",
                            )

                        if not share_axes:
                            margin = (max_val - min_val) * 0.05
                            guide = np.linspace(min_val - margin, max_val + margin)
                            axes[i // n_cols, i % n_cols].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

                        guide_min = min_val if guide_min > min_val else guide_min
                        guide_max = max_val if guide_max < max_val else guide_max

                    if share_axes:
                        margin = (guide_max - guide_min) * 0.05
                        guide = np.linspace(guide_min - margin, guide_max + margin)
                        for i, estimator in enumerate(compared_estimators):
                            axes[i // n_cols, i % n_cols].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

        else:
            if n_cols == 1:
                for j, ci in enumerate(cis):
                    for i, estimator in enumerate(compared_estimators):
                        if isinstance(input_dict, MultipleInputDict):
                            if behavior_policy_name is None and dataset_id is None:
                                min_val, max_val = np.infty, -np.infty

                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    n_datasets = input_dict.n_datasets[behavior_policy]
                                    min_vals = np.zeros(n_datasets)
                                    max_vals = np.zeros(n_datasets)

                                    for dataset_id_ in range(n_datasets):
                                        true_policy_value = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "true_policy_value"
                                        ]
                                        estimated_policy_value = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "estimated_policy_value_lower_bound"
                                        ]

                                        if dataset_id_ == 0:
                                            axes[j].scatter(
                                                true_policy_value,
                                                estimated_policy_value,
                                                color=color[l % n_colors],
                                                label=behavior_policy,
                                            )
                                        else:  # to remove duplicated labels
                                            axes[j].scatter(
                                                true_policy_value,
                                                estimated_policy_value,
                                                color=color[l % n_colors],
                                            )

                                        min_vals[dataset_id_] = np.minimum(
                                            np.nanmin(true_policy_value),
                                            np.nanmin(estimated_policy_value),
                                        )
                                        max_vals[dataset_id_] = np.maximum(
                                            np.nanmax(true_policy_value),
                                            np.nanmax(estimated_policy_value),
                                        )

                                    min_val = min(min_val, min_vals.min())
                                    max_val = max(max_val, max_vals.max())

                            elif (
                                behavior_policy_name is None and dataset_id is not None
                            ):
                                min_val, max_val = np.infty, -np.infty

                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    true_policy_value = policy_value_dict[ci][
                                        behavior_policy
                                    ][ci][estimator]["true_policy_value"]
                                    estimated_policy_value = policy_value_dict[
                                        behavior_policy
                                    ][ci][estimator][
                                        "estimated_policy_value_lower_bound"
                                    ]

                                    axes[j].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[l % n_colors],
                                        label=behavior_policy,
                                    )

                                    min_val_ = np.minimum(
                                        np.nanmin(true_policy_value),
                                        np.nanmin(estimated_policy_value),
                                    )
                                    max_val_ = np.maximum(
                                        np.nanmax(true_policy_value),
                                        np.nanmax(estimated_policy_value),
                                    )

                                min_val = min(min_val, min_val_)
                                max_val = max(max_val, max_val_)

                            elif (
                                behavior_policy_name is not None and dataset_id is None
                            ):
                                n_datasets = input_dict.n_datasets[behavior_policy_name]
                                min_vals = np.zeros(n_datasets)
                                max_vals = np.zeros(n_datasets)

                                for dataset_id_ in range(n_datasets):
                                    true_policy_value = policy_value_dict[dataset_id_][
                                        ci
                                    ][estimator]["true_policy_value"]
                                    estimated_policy_value = policy_value_dict[
                                        dataset_id_
                                    ][ci][estimator][
                                        "estimated_policy_value_lower_bound"
                                    ]

                                    axes[j].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[0],
                                    )

                                    min_vals[dataset_id_] = np.minimum(
                                        np.nanmin(true_policy_value),
                                        np.nanmin(estimated_policy_value),
                                    )
                                    max_vals[dataset_id_] = np.maximum(
                                        np.nanmax(true_policy_value),
                                        np.nanmax(estimated_policy_value),
                                    )

                                min_val = min_vals.min()
                                max_val = max_vals.max()

                            else:
                                true_policy_value = policy_value_dict[ci][estimator][
                                    "true_policy_value"
                                ]
                                estimated_policy_value = policy_value_dict[ci][
                                    estimator
                                ]["estimated_policy_value_lower_bound"]

                                axes[j].scatter(
                                    true_policy_value,
                                    estimated_policy_value,
                                    color=color[0],
                                )

                                min_val = np.minimum(
                                    np.nanmin(true_policy_value),
                                    np.nanmin(estimated_policy_value),
                                )
                                max_val = np.maximum(
                                    np.nanmax(true_policy_value),
                                    np.nanmax(estimated_policy_value),
                                )

                        else:
                            true_policy_value = policy_value_dict[ci][estimator][
                                "true_policy_value"
                            ]
                            estimated_policy_value = policy_value_dict[ci][estimator][
                                "estimated_policy_value_lower_bound"
                            ]

                            axes[j].scatter(
                                true_policy_value,
                                estimated_policy_value,
                                color=color[0],
                            )

                            min_val = np.minimum(
                                np.nanmin(true_policy_value),
                                np.nanmin(estimated_policy_value),
                            )
                            max_val = np.maximum(
                                np.nanmax(true_policy_value),
                                np.nanmax(estimated_policy_value),
                            )

                        axes[j].set_title(f"{ci}, {estimator}")
                        axes[j].set_xlabel("true policy value")
                        axes[j].set_ylabel("estimated policy value lower bound")

                        if (
                            legend
                            and behavior_policy_name is None
                            and isinstance(input_dict, MultipleInputDict)
                        ):
                            axes[j].legend(title="behavior_policy", loc="lower right")

                        if not share_axes:
                            margin = (max_val - min_val) * 0.05
                            guide = np.linspace(min_val - margin, max_val + margin)
                            axes[j].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

                        guide_min = min_val if guide_min > min_val else guide_min
                        guide_max = max_val if guide_max < max_val else guide_max

                if share_axes:
                    margin = (guide_max - guide_min) * 0.05
                    guide = np.linspace(guide_min - margin, guide_max + margin)
                    for j, ci in enumerate(cis):
                        axes[j].plot(
                            guide,
                            guide,
                            color="black",
                            linewidth=1.0,
                        )

            else:
                for j, ci in enumerate(cis):
                    for i, estimator in enumerate(compared_estimators):
                        if isinstance(input_dict, MultipleInputDict):
                            if behavior_policy_name is None and dataset_id is None:
                                min_val, max_val = np.infty, -np.infty

                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    n_datasets = input_dict.n_datasets[behavior_policy]
                                    min_vals = np.zeros(n_datasets)
                                    max_vals = np.zeros(n_datasets)

                                    for dataset_id_ in range(n_datasets):
                                        true_policy_value = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "true_policy_value"
                                        ]
                                        estimated_policy_value = policy_value_dict[
                                            behavior_policy
                                        ][dataset_id_][ci][estimator][
                                            "estimated_policy_value_lower_bound"
                                        ]

                                        if dataset_id_ == 0:
                                            axes[i, j].scatter(
                                                true_policy_value,
                                                estimated_policy_value,
                                                color=color[l % n_colors],
                                                label=behavior_policy,
                                            )
                                        else:  # to remove duplicated labels
                                            axes[i, j].scatter(
                                                true_policy_value,
                                                estimated_policy_value,
                                                color=color[l % n_colors],
                                            )

                                        min_vals[dataset_id_] = np.minimum(
                                            np.nanmin(true_policy_value),
                                            np.nanmin(estimated_policy_value),
                                        )
                                        max_vals[dataset_id_] = np.maximum(
                                            np.nanmax(true_policy_value),
                                            np.nanmax(estimated_policy_value),
                                        )

                                    min_val = min(min_val, min_vals.min())
                                    max_val = max(max_val, max_vals.max())

                            elif (
                                behavior_policy_name is None and dataset_id is not None
                            ):
                                min_val, max_val = np.infty, -np.infty

                                for l, behavior_policy in enumerate(
                                    input_dict.behavior_policy_names
                                ):
                                    true_policy_value = policy_value_dict[
                                        behavior_policy
                                    ][ci][estimator]["true_policy_value"]
                                    estimated_policy_value = policy_value_dict[
                                        behavior_policy
                                    ][ci][estimator][
                                        "estimated_policy_value_lower_bound"
                                    ]

                                    axes[i, j].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[l % n_colors],
                                        label=behavior_policy,
                                    )

                                    min_val_ = np.minimum(
                                        np.nanmin(true_policy_value),
                                        np.nanmin(estimated_policy_value),
                                    )
                                    max_val_ = np.maximum(
                                        np.nanmax(true_policy_value),
                                        np.nanmax(estimated_policy_value),
                                    )

                                min_val = min(min_val, min_val_)
                                max_val = max(max_val, max_val_)

                            elif (
                                behavior_policy_name is not None and dataset_id is None
                            ):
                                n_datasets = input_dict.n_datasets[behavior_policy_name]
                                min_vals = np.zeros(n_datasets)
                                max_vals = np.zeros(n_datasets)

                                for dataset_id_ in range(n_datasets):
                                    true_policy_value = policy_value_dict[dataset_id_][
                                        ci
                                    ][estimator]["true_policy_value"]
                                    estimated_policy_value = policy_value_dict[
                                        dataset_id_
                                    ][ci][estimator][
                                        "estimated_policy_value_lower_bound"
                                    ]

                                    axes[i, j].scatter(
                                        true_policy_value,
                                        estimated_policy_value,
                                        color=color[0],
                                    )

                                    min_vals[dataset_id_] = np.minimum(
                                        np.nanmin(true_policy_value),
                                        np.nanmin(estimated_policy_value),
                                    )
                                    max_vals[dataset_id_] = np.maximum(
                                        np.nanmax(true_policy_value),
                                        np.nanmax(estimated_policy_value),
                                    )

                                min_val = min_vals.min()
                                max_val = max_vals.max()

                            else:
                                true_policy_value = policy_value_dict[ci][estimator][
                                    "true_policy_value"
                                ]
                                estimated_policy_value = policy_value_dict[ci][
                                    estimator
                                ]["estimated_policy_value_lower_bound"]

                                axes[i, j].scatter(
                                    true_policy_value,
                                    estimated_policy_value,
                                    color=color[0],
                                )

                                min_val = np.minimum(
                                    np.nanmin(true_policy_value),
                                    np.nanmin(estimated_policy_value),
                                )
                                max_val = np.maximum(
                                    np.nanmax(true_policy_value),
                                    np.nanmax(estimated_policy_value),
                                )

                        else:
                            true_policy_value = policy_value_dict[ci][estimator][
                                "true_policy_value"
                            ]
                            estimated_policy_value = policy_value_dict[ci][estimator][
                                "estimated_policy_value_lower_bound"
                            ]

                            axes[i, j].scatter(
                                true_policy_value,
                                estimated_policy_value,
                                color=color[0],
                            )

                            min_val = np.minimum(
                                np.nanmin(true_policy_value),
                                np.nanmin(estimated_policy_value),
                            )
                            max_val = np.maximum(
                                np.nanmax(true_policy_value),
                                np.nanmax(estimated_policy_value),
                            )

                        axes[i, j].set_title(f"{ci}, {estimator}")
                        axes[i, j].set_xlabel("true policy value")
                        axes[i, j].set_ylabel("estimated policy value lower bound")

                        if (
                            legend
                            and behavior_policy_name is None
                            and isinstance(input_dict, MultipleInputDict)
                        ):
                            axes[i, j].legend(
                                title="behavior_policy", loc="lower right"
                            )

                        if not share_axes:
                            margin = (max_val - min_val) * 0.05
                            guide = np.linspace(min_val - margin, max_val + margin)
                            axes[i, j].plot(
                                guide,
                                guide,
                                color="black",
                                linewidth=1.0,
                            )

                        guide_min = min_val if guide_min > min_val else guide_min
                        guide_max = max_val if guide_max < max_val else guide_max

                if share_axes:
                    margin = (guide_max - guide_min) * 0.05
                    guide = np.linspace(guide_min - margin, guide_max + margin)
                    for j, ci in enumerate(cis):
                        for i, estimator in enumerate(compared_estimators):
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
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "validation_variance.png",
    ):
        """Visualize the true variance and its estimate (scatter plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="validation_variance.png"
            Name of the bar figure.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        self._check_basic_visualization_inputs(
            n_cols=n_cols, fig_dir=fig_dir, fig_name=fig_name
        )

        ground_truth_policy_value_dict = self.obtain_true_selection_result(
            input_dict=input_dict,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            return_variance=True,
        )
        estimated_variance_dict = self.cumulative_distribution_ope.estimate_variance(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
        )

        if isinstance(input_dict, MultipleInputDict):
            if behavior_policy_name is None and dataset_id is None:
                candidate_policy_names = defaultdict(list)
                true_variance = defaultdict(list)

                for (
                    behavior_policy,
                    n_datasets,
                ) in input_dict.n_datasets.items():
                    for dataset_id_ in range(n_datasets):
                        candidate_policy_names[behavior_policy].append(
                            ground_truth_policy_value_dict[behavior_policy][
                                dataset_id_
                            ]["ranking"]
                        )
                        true_variance[behavior_policy].append(
                            ground_truth_policy_value_dict[behavior_policy][
                                dataset_id_
                            ]["variance"]
                        )

                candidate_policy_names = defaultdict_to_dict(candidate_policy_names)
                true_variance = defaultdict_to_dict(true_variance)

            elif behavior_policy_name is None and dataset_id is not None:
                candidate_policy_names = {}
                true_variance = {}

                for behavior_policy in input_dict.behavior_policy_names:
                    candidate_policy_names[
                        behavior_policy
                    ] = ground_truth_policy_value_dict[behavior_policy]["ranking"]
                    true_variance[behavior_policy] = ground_truth_policy_value_dict[
                        behavior_policy
                    ]["variance"]

            elif behavior_policy_name is not None and dataset_id is None:
                candidate_policy_names = []
                true_variance = []

                for dataset_id_ in range(input_dict.n_datasets[behavior_policy_name]):
                    candidate_policy_names.append(
                        ground_truth_policy_value_dict[dataset_id_]["ranking"]
                    )
                    true_variance.append(
                        ground_truth_policy_value_dict[dataset_id_]["variance"]
                    )

            else:
                candidate_policy_names = ground_truth_policy_value_dict["ranking"]
                true_variance = ground_truth_policy_value_dict["variance"]

        else:
            candidate_policy_names = ground_truth_policy_value_dict["ranking"]
            true_variance = ground_truth_policy_value_dict["variance"]

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        n_figs = len(compared_estimators)
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
            for i, estimator in enumerate(compared_estimators):
                if isinstance(input_dict, MultipleInputDict):
                    if behavior_policy_name is None and dataset_id is None:
                        min_val, max_val = np.infty, -np.infty

                        for l, behavior_policy in enumerate(
                            input_dict.behavior_policy_names
                        ):
                            n_datasets = input_dict.n_datasets[behavior_policy]
                            min_vals = np.zeros(n_datasets)
                            max_vals = np.zeros(n_datasets)

                            for dataset_id_ in range(n_datasets):
                                estimated_variance = np.zeros(
                                    len(
                                        candidate_policy_names[behavior_policy][
                                            dataset_id_
                                        ]
                                    )
                                )
                                for j, eval_policy in enumerate(
                                    candidate_policy_names[behavior_policy][dataset_id_]
                                ):
                                    estimated_variance[j] = estimated_variance_dict[
                                        behavior_policy
                                    ][dataset_id_][eval_policy][estimator]

                                if dataset_id_ == 0:
                                    axes[i].scatter(
                                        true_variance[behavior_policy][dataset_id_],
                                        estimated_variance,
                                        color=color[l % n_colors],
                                        label=behavior_policy,
                                    )
                                else:
                                    axes[i].scatter(
                                        true_variance[behavior_policy][dataset_id_],
                                        estimated_variance,
                                        color=color[l % n_colors],
                                    )

                                min_vals[dataset_id_] = np.minimum(
                                    np.nanmin(
                                        true_variance[behavior_policy][dataset_id_]
                                    ),
                                    np.nanmin(estimated_variance),
                                )
                                max_vals[dataset_id_] = np.maximum(
                                    np.nanmax(
                                        true_variance[behavior_policy][dataset_id_]
                                    ),
                                    np.nanmax(estimated_variance),
                                )

                            min_val = min(min_val, min_vals.min())
                            max_val = max(max_val, max_vals.max())

                    elif behavior_policy_name is None and dataset_id is not None:
                        min_val, max_val = np.infty, -np.infty

                        for l, behavior_policy in enumerate(
                            input_dict.behavior_policy_names
                        ):
                            estimated_variance = np.zeros(
                                len(candidate_policy_names[behavior_policy])
                            )
                            for j, eval_policy in enumerate(
                                candidate_policy_names[behavior_policy]
                            ):
                                estimated_variance[j] = estimated_variance_dict[
                                    behavior_policy
                                ][eval_policy][estimator]

                            axes[i].scatter(
                                true_variance[behavior_policy],
                                estimated_variance,
                                color=color[l % n_colors],
                                label=behavior_policy,
                            )

                            min_val_ = np.minimum(
                                np.nanmin(true_variance[behavior_policy]),
                                np.nanmin(estimated_variance),
                            )
                            max_val_ = np.maximum(
                                np.nanmax(true_variance[behavior_policy]),
                                np.nanmax(estimated_variance),
                            )

                        min_val = min(min_val, min_val_)
                        max_val = max(max_val, max_val_)

                    elif behavior_policy_name is not None and dataset_id is None:
                        n_datasets = input_dict.n_datasets[behavior_policy_name]
                        min_vals = np.zeros(n_datasets)
                        max_vals = np.zeros(n_datasets)

                        for dataset_id_ in range(n_datasets):
                            estimated_variance = np.zeros(
                                len(candidate_policy_names[dataset_id_])
                            )
                            for j, eval_policy in enumerate(
                                candidate_policy_names[dataset_id_]
                            ):
                                estimated_variance[j] = estimated_variance_dict[
                                    dataset_id_
                                ][eval_policy][estimator]

                            axes[i].scatter(
                                true_variance[dataset_id_],
                                estimated_variance,
                                color=color[0],
                            )

                            min_vals[dataset_id_] = np.minimum(
                                np.nanmin(true_variance[dataset_id_]),
                                np.nanmin(estimated_variance[dataset_id_]),
                            )
                            max_vals[dataset_id_] = np.maximum(
                                np.nanmax(true_variance[dataset_id_]),
                                np.nanmax(estimated_variance[dataset_id_]),
                            )

                        min_val = min_vals.min()
                        max_val = max_vals.max()

                    else:
                        estimated_variance = np.zeros(len(candidate_policy_names))
                        for j, eval_policy in enumerate(candidate_policy_names):
                            estimated_variance[j] = estimated_variance_dict[
                                eval_policy
                            ][estimator]

                        axes[i].scatter(
                            true_variance,
                            estimated_variance,
                            color=color[0],
                        )

                        min_val = np.minimum(
                            np.nanmin(true_variance),
                            np.nanmin(estimated_variance),
                        )
                        max_val = np.maximum(
                            np.nanmax(true_variance),
                            np.nanmax(estimated_variance),
                        )

                else:
                    estimated_variance = np.zeros(len(candidate_policy_names))
                    for j, eval_policy in enumerate(candidate_policy_names):
                        estimated_variance[j] = estimated_variance_dict[eval_policy][
                            estimator
                        ]

                    axes[i].scatter(
                        true_variance,
                        estimated_variance,
                        color=color[0],
                    )

                    min_val = np.minimum(
                        np.nanmin(true_variance),
                        np.nanmin(estimated_variance),
                    )
                    max_val = np.maximum(
                        np.nanmax(true_variance),
                        np.nanmax(estimated_variance),
                    )

                axes[i].set_title(estimator)
                axes[i].set_xlabel("true variance")
                axes[i].set_ylabel("estimated variance")

                if (
                    legend
                    and behavior_policy_name is None
                    and isinstance(input_dict, MultipleInputDict)
                ):
                    axes[i].legend(title="behavior_policy", loc="lower right")

                if not share_axes:
                    margin = (max_val - min_val) * 0.05
                    guide = np.linspace(min_val - margin, max_val + margin)
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

            if share_axes:
                margin = (guide_max - guide_min) * 0.05
                guide = np.linspace(guide_min - margin, guide_max + margin)
                for i, estimator in enumerate(compared_estimators):
                    axes[i].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

        else:
            for i, estimator in enumerate(compared_estimators):
                if isinstance(input_dict, MultipleInputDict):
                    if behavior_policy_name is None and dataset_id is None:
                        min_val, max_val = np.infty, -np.infty

                        for l, behavior_policy in enumerate(
                            input_dict.behavior_policy_names
                        ):
                            n_datasets = input_dict.n_datasets[behavior_policy]
                            min_vals = np.zeros(n_datasets)
                            max_vals = np.zeros(n_datasets)

                            for dataset_id_ in range(n_datasets):
                                estimated_variance = np.zeros(
                                    len(
                                        candidate_policy_names[behavior_policy][
                                            dataset_id_
                                        ]
                                    )
                                )
                                for j, eval_policy in enumerate(
                                    candidate_policy_names[behavior_policy][dataset_id_]
                                ):
                                    estimated_variance[j] = estimated_variance_dict[
                                        behavior_policy
                                    ][dataset_id_][eval_policy][estimator]

                                if dataset_id_ == 0:
                                    axes[i // n_cols, i % n_cols].scatter(
                                        true_variance[behavior_policy][dataset_id_],
                                        estimated_variance,
                                        color=color[l % n_colors],
                                        label=behavior_policy,
                                    )
                                else:
                                    axes[i // n_cols, i % n_cols].scatter(
                                        true_variance[behavior_policy][dataset_id_],
                                        estimated_variance,
                                        color=color[l % n_colors],
                                    )

                                min_vals[dataset_id_] = np.minimum(
                                    np.nanmin(
                                        true_variance[behavior_policy][dataset_id_]
                                    ),
                                    np.nanmin(estimated_variance),
                                )
                                max_vals[dataset_id_] = np.maximum(
                                    np.nanmax(
                                        true_variance[behavior_policy][dataset_id_]
                                    ),
                                    np.nanmax(estimated_variance),
                                )

                            min_val = min(min_val, min_vals.min())
                            max_val = max(max_val, max_vals.max())

                    elif behavior_policy_name is None and dataset_id is not None:
                        min_val, max_val = np.infty, -np.infty

                        for l, behavior_policy in enumerate(
                            input_dict.behavior_policy_names
                        ):
                            estimated_variance = np.zeros(
                                len(candidate_policy_names[behavior_policy])
                            )
                            for j, eval_policy in enumerate(
                                candidate_policy_names[behavior_policy]
                            ):
                                estimated_variance[j] = estimated_variance_dict[
                                    behavior_policy
                                ][eval_policy][estimator]

                            axes[i // n_cols, i % n_cols].scatter(
                                true_variance[behavior_policy],
                                estimated_variance,
                                color=color[l % n_colors],
                                label=behavior_policy,
                            )

                            min_val_ = np.minimum(
                                np.nanmin(true_variance[behavior_policy]),
                                np.nanmin(estimated_variance),
                            )
                            max_val_ = np.maximum(
                                np.nanmax(true_variance[behavior_policy]),
                                np.nanmax(estimated_variance),
                            )

                        min_val = min(min_val, min_val_)
                        max_val = max(max_val, max_val_)

                    elif behavior_policy_name is not None and dataset_id is None:
                        n_datasets = input_dict.n_datasets[behavior_policy_name]
                        min_vals = np.zeros(n_datasets)
                        max_vals = np.zeros(n_datasets)

                        for dataset_id_ in range(n_datasets):
                            estimated_variance = np.zeros(
                                len(candidate_policy_names[dataset_id_])
                            )
                            for j, eval_policy in enumerate(
                                candidate_policy_names[dataset_id_]
                            ):
                                estimated_variance[j] = estimated_variance_dict[
                                    dataset_id_
                                ][eval_policy][estimator]

                            axes[i // n_cols, i % n_cols].scatter(
                                true_variance[dataset_id_],
                                estimated_variance,
                                color=color[0],
                            )

                            min_vals[dataset_id_] = np.minimum(
                                np.nanmin(true_variance[dataset_id_]),
                                np.nanmin(estimated_variance[dataset_id_]),
                            )
                            max_vals[dataset_id_] = np.maximum(
                                np.nanmax(true_variance[dataset_id_]),
                                np.nanmax(estimated_variance[dataset_id_]),
                            )

                        min_val = min_vals.min()
                        max_val = max_vals.max()

                    else:
                        estimated_variance = np.zeros(len(candidate_policy_names))
                        for j, eval_policy in enumerate(candidate_policy_names):
                            estimated_variance[j] = estimated_variance_dict[
                                eval_policy
                            ][estimator]

                        axes[i // n_cols, i % n_cols].scatter(
                            true_variance,
                            estimated_variance,
                            color=color[0],
                        )

                        min_val = np.minimum(
                            np.nanmin(true_variance),
                            np.nanmin(estimated_variance),
                        )
                        max_val = np.maximum(
                            np.nanmax(true_variance),
                            np.nanmax(estimated_variance),
                        )

                else:
                    estimated_variance = np.zeros(len(candidate_policy_names))
                    for j, eval_policy in enumerate(candidate_policy_names):
                        estimated_variance[j] = estimated_variance_dict[eval_policy][
                            estimator
                        ]

                    axes[i // n_cols, i % n_cols].scatter(
                        true_variance,
                        estimated_variance,
                        color=color[0],
                    )

                    min_val = np.minimum(
                        np.nanmin(true_variance),
                        np.nanmin(estimated_variance),
                    )
                    max_val = np.maximum(
                        np.nanmax(true_variance),
                        np.nanmax(estimated_variance),
                    )

                axes[i // n_cols, i % n_cols].set_title(estimator)
                axes[i // n_cols, i % n_cols].set_xlabel("true variance")
                axes[i // n_cols, i % n_cols].set_ylabel("estimated variance")

                if (
                    legend
                    and behavior_policy_name is None
                    and isinstance(input_dict, MultipleInputDict)
                ):
                    axes[i // n_cols, i % n_cols].legend(
                        title="behavior_policy", loc="lower right"
                    )

                if not share_axes:
                    margin = (max_val - min_val) * 0.05
                    guide = np.linspace(min_val - margin, max_val + margin)
                    axes[i // n_cols, i % n_cols].plot(
                        guide,
                        guide,
                        color="black",
                        linewidth=1.0,
                    )

                guide_min = min_val if guide_min > min_val else guide_min
                guide_max = max_val if guide_max < max_val else guide_max

            if share_axes:
                margin = (guide_max - guide_min) * 0.05
                guide = np.linspace(guide_min - margin, guide_max + margin)
                for i, estimator in enumerate(compared_estimators):
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
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        alpha: float = 0.05,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "validation_lower_quartile.png",
    ):
        """Visualize the true lower quartile and its estimate (scatter plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 0.5]`.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="validation_lower_quartile.png"
            Name of the bar figure.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        self._check_basic_visualization_inputs(
            n_cols=n_cols, fig_dir=fig_dir, fig_name=fig_name
        )

        lower_quartile_dict = self.select_by_lower_quartile(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            alpha=alpha,
            return_true_values=True,
        )

        self._visualize_policy_performance_for_validation(
            estimation_dict=lower_quartile_dict,
            input_dict=input_dict,
            true_value_arg="true_lower_quartile",
            estimated_value_arg="estimated_lower_quartile",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            n_cols=n_cols,
            share_axes=share_axes,
            legend=legend,
            ylabel=f"lower quartile ({alpha})",
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    def visualize_conditional_value_at_risk_for_validation(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        alpha: float = 0.05,
        n_cols: Optional[int] = None,
        share_axes: bool = False,
        legend: bool = True,
        fig_dir: Optional[Path] = None,
        fig_name: str = "validation_conditional_value_at_risk.png",
    ):
        """Visualize the true conditional value at risk and its estimate (scatter plot).

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
            Dictionary of the OPE inputs for each evaluation policy.

            .. code-block:: python

                key: [evaluation_policy][
                    evaluation_policy_action,
                    evaluation_policy_action_dist,
                    state_action_value_prediction,
                    initial_state_value_prediction,
                    state_action_marginal_importance_weight,
                    state_marginal_importance_weight,
                    on_policy_policy_value,
                    gamma,
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            When `None` is given, all the estimators are compared.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        alpha: float, default=0.05
            Proportion of the shaded region. The value should be within `[0, 1]`.

        n_cols: int, default=None (> 0)
            Number of columns in the figure.

        share_axes: bool, default=False
            Whether to share x- and y-axes or not.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default="validation_conditional_value_at_risk.png"
            Name of the bar figure.

        """
        compared_estimators = self._check_compared_estimators(
            compared_estimators, ope_type="cumulative_distribution_ope"
        )
        self._check_basic_visualization_inputs(
            n_cols=n_cols, fig_dir=fig_dir, fig_name=fig_name
        )

        cvar_dict = self.select_by_conditional_value_at_risk(
            input_dict,
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            alpha=alpha,
            return_true_values=True,
        )

        self._visualize_policy_performance_for_validation(
            estimation_dict=cvar_dict,
            input_dict=input_dict,
            true_value_arg="true_conditional_value_at_risk",
            estimated_value_arg="estimated_conditional_value_at_risk",
            compared_estimators=compared_estimators,
            behavior_policy_name=behavior_policy_name,
            dataset_id=dataset_id,
            n_cols=n_cols,
            share_axes=share_axes,
            legend=legend,
            ylabel=f"CVaR ({alpha})",
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    @property
    def estimators_name(self):
        estimators_name = {
            "standard_ope": None if self.ope is None else self.ope.estimators_name,
            "cumulative_distribution_ope": None
            if self.cumulative_distribution_ope is None
            else self.cumulative_distribution_ope.estimators_name,
        }
        return estimators_name
