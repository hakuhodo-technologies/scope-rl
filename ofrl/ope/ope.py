"""Meta class to handle standard and cumulative distribution OPE."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from collections import defaultdict

import numpy as np
from scipy.stats import norm
from sklearn.utils import check_scalar

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from d3rlpy.preprocessing import ActionScaler

from .estimators_base import (
    BaseOffPolicyEstimator,
    BaseCumulativeDistributionOffPolicyEstimator,
)
from ..utils import (
    MultipleLoggedDataset,
    MultipleInputDict,
    estimate_confidence_interval_by_bootstrap,
    estimate_confidence_interval_by_hoeffding,
    estimate_confidence_interval_by_empirical_bernstein,
    estimate_confidence_interval_by_t_test,
    defaultdict_to_dict,
    check_array,
    check_logged_dataset,
    check_input_dict,
)
from ..types import LoggedDataset, OPEInputDict


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
    logged_dataset: LoggedDataset or MultipleLoggedDataset
        Logged dataset used to conduct OPE.

        .. code-block:: python

            key: [
                size,
                n_trajectories,
                step_per_trajectory,
                action_type,
                n_actions,
                action_dim,
                action_keys,
                action_meaning,
                state_dim,
                state_keys,
                state,
                action,
                reward,
                done,
                terminal,
                info,
                pscore,
            ]

            .. seealso::

                :class:`ofrl.dataset.SyntheticDataset` describes the components of :class:`logged_dataset`.

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

    logged_dataset: Union[LoggedDataset, MultipleLoggedDataset]
    ope_estimators: List[BaseOffPolicyEstimator]
    n_step_pdis: int = 0
    sigma: float = 1.0
    action_scaler: Optional[ActionScaler] = None

    def __post_init__(self) -> None:
        self.use_multiple_logged_dataset = False
        if isinstance(self.logged_dataset, MultipleLoggedDataset):
            self.multiple_logged_dataset = self.logged_dataset
            self.logged_dataset = self.multiple_logged_dataset.get(0)
            self.use_multiple_logged_dataset = True

        check_logged_dataset(self.logged_dataset)
        self.step_per_trajectory = self.logged_dataset["step_per_trajectory"]
        self.action_type = self.logged_dataset["action_type"]

        if not self.use_multiple_logged_dataset:
            self._register_logged_dataset()

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

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _register_logged_dataset(self, id: Optional[int] = None):
        if id is not None:
            self.logged_dataset = self.multiple_logged_dataset.get(id)

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

    def _estimate_policy_value(
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

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

    def _estimate_intervals(
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

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

    def _summarize_off_policy_estimates(
        self,
        policy_value_dict: Dict[str, Any],
        policy_value_interval_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]:
        """Summarize the policy value and their confidence intervals estimated by OPE estimators.

        Parameters
        -------
        policy_value_dict: dict
            Dictionary containing the policy value of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        policy_value_interval_dict: dict
            Dictionary containing the confidence intervals estimated by nonparametric bootstrap.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        Return
        -------
        policy_value_df_dict: dict
            Dictionary containing the policy value of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        policy_value_interval_df_dict: dict
            Dictionary containing the confidence intervals estimated by nonparametric bootstrap.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        policy_value_df_dict = dict()
        policy_value_interval_df_dict = dict()

        for eval_policy in policy_value_dict.keys():
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

            policy_value_df_dict[eval_policy] = policy_value_df_
            policy_value_interval_df_dict[eval_policy] = DataFrame(
                policy_value_interval_dict[eval_policy],
            ).T

        return policy_value_df_dict, policy_value_interval_df_dict

    def _evaluate_performance_of_ope_estimators(
        self,
        input_dict: OPEInputDict,
        policy_value_dict: Dict[str, Any],
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
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        policy_value_dict: dict
            Dictionary containing the policy value of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

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
        eval_metric_ope_dict = defaultdict(dict)

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

    def estimate_policy_value(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
    ) -> Dict[str, float]:
        """Estimate the policy value of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of policy_value_dict.

        Return
        -------
        policy_value_dict: dict (, list of dict)
            Dictionary containing the policy value of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )

        if dataset_id is None and self.use_multiple_logged_dataset:
            if not isinstance(input_dict, MultipleInputDict):
                raise RuntimeError(
                    "MultipleInputDict should be given for input_dict, when MultipleLoggedDataset is used and dataset_id is not specified"
                    "Please pass MultipleInputDict or specify dataset_id."
                )
            if len(input_dict) != len(self.multiple_logged_dataset):
                raise ValueError(
                    "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
                )

            policy_value_dict = []
            for i in range(len(self.multiple_logged_dataset)):
                self._register_logged_dataset(i)
                input_dict_ = input_dict.get(i)

                policy_value_dict_ = self._estimate_policy_value(
                    input_dict_,
                    compared_estimators=compared_estimators,
                )
                policy_value_dict.append(policy_value_dict_)

        else:
            if self.use_multiple_logged_dataset:
                if dataset_id is None:
                    raise ValueError(
                        "dataset_id must be given when using MultipleInputDict."
                    )
                if isinstance(input_dict, MultipleInputDict):
                    input_dict = input_dict.get(dataset_id)

                self._register_logged_dataset(dataset_id)

            policy_value_dict = self._estimate_policy_value(
                input_dict,
                compared_estimators=compared_estimators,
            )

        return policy_value_dict

    def estimate_intervals(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate the confidence intervals of the policy value by nonparametric bootstrap.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of policy_value_interval_dict.

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

        if dataset_id is None and self.use_multiple_logged_dataset:
            if not isinstance(input_dict, MultipleInputDict):
                raise RuntimeError(
                    "MultipleInputDict should be given for input_dict, when MultipleLoggedDataset is used and dataset_id is not specified"
                    "Please pass MultipleInputDict or specify dataset_id."
                )
            if len(input_dict) != len(self.multiple_logged_dataset):
                raise ValueError(
                    "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
                )

            policy_value_interval_dict = []
            for i in range(len(self.multiple_logged_dataset)):
                self._register_logged_dataset(i)
                input_dict_ = input_dict.get(i)

                policy_value_interval_dict_ = self._estimate_intervals(
                    input_dict_,
                    compared_estimators=compared_estimators,
                    alpha=alpha,
                    ci=ci,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )
                policy_value_interval_dict.append(
                    defaultdict_to_dict(policy_value_interval_dict_)
                )

        else:
            if self.use_multiple_logged_dataset:
                if dataset_id is None:
                    raise ValueError(
                        "dataset_id must be given when using MultipleInputDict."
                    )
                if isinstance(input_dict, MultipleInputDict):
                    input_dict = input_dict.get(dataset_id)

                self._register_logged_dataset(dataset_id)

            policy_value_interval_dict = self._estimate_intervals(
                input_dict,
                compared_estimators=compared_estimators,
                alpha=alpha,
                ci=ci,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]:
        """Summarize the policy value and their confidence intervals estimated by OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of policy_value_dict and policy_value_interval_dict.

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
        policy_value_df_dict: dict (, list of dict)
            Dictionary containing the policy value of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        policy_value_interval_df_dict: dict (, list of dict)
            Dictionary containing the confidence intervals estimated by nonparametric bootstrap.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        policy_value_dict = self.estimate_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            dataset_id=dataset_id,
        )
        policy_value_interval_dict = self.estimate_intervals(
            input_dict,
            compared_estimators=compared_estimators,
            dataset_id=dataset_id,
            alpha=alpha,
            ci=ci,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

        if dataset_id is None and self.use_multiple_logged_dataset:
            policy_value_df_dict = []
            policy_value_interval_df_dict = []

            for i in range(len(self.multiple_logged_dataset)):
                (
                    policy_value_df_dict_,
                    policy_value_interval_df_dict_,
                ) = self._summarize_off_policy_estimates(
                    policy_value_dict=policy_value_dict[i],
                    policy_value_interval_dict=policy_value_interval_dict[i],
                )

                policy_value_df_dict.append(policy_value_df_dict_)
                policy_value_interval_df_dict.append(policy_value_interval_df_dict_)

        else:
            (
                policy_value_df_dict,
                policy_value_interval_df_dict,
            ) = self._summarize_off_policy_estimates(
                policy_value_dict=policy_value_dict,
                policy_value_interval_dict=policy_value_interval_dict,
            )

        return policy_value_df_dict, policy_value_interval_df_dict

    def evaluate_performance_of_ope_estimators(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
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
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of dictionary or dataframe.

        metric: {"relative-ee", "se"}, default="relative-ee"
            Evaluation metric used to evaluate and compare the estimation performance/accuracy of OPE estimators.

        return_by_dataframe: bool, default=False
            Whether to return the result in a dataframe format.

        Return
        -------
        eval_metric_ope_dict/eval_metric_ope_df: dict or dataframe (, list of dict or dataframe)
            Dictionary/dataframe containing evaluation metric for evaluating the estimation performance/accuracy of OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
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

        policy_value_dict = self.estimate_policy_value(
            input_dict,
            compared_estimators=compared_estimators,
            dataset_id=dataset_id,
        )

        if dataset_id is None and self.use_multiple_logged_dataset:
            eval_metric_ope = []

            for i in range(len(self.multiple_logged_dataset)):
                input_dict_ = input_dict.get(i)
                eval_metric_ope_ = self._evaluate_performance_of_ope_estimators(
                    input_dict=input_dict_,
                    policy_value_dict=policy_value_dict[i],
                    compared_estimators=compared_estimators,
                    metric=metric,
                    return_by_dataframe=return_by_dataframe,
                )
                eval_metric_ope.append(eval_metric_ope_)

        else:
            if self.use_multiple_logged_dataset:
                input_dict = input_dict.get(dataset_id)

            eval_metric_ope = self._evaluate_performance_of_ope_estimators(
                input_dict=input_dict,
                policy_value_dict=policy_value_dict,
                compared_estimators=compared_estimators,
                metric=metric,
                return_by_dataframe=return_by_dataframe,
            )

        return eval_metric_ope

    def visualize_off_policy_estimates(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
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
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            RRequired when using :class:`MultipleLoggedDataset`.

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
            relative to the on-policy policy value of the behavior policy. (Only applicable when using a single input_dict.)

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
        if self.use_multiple_logged_dataset:
            if dataset_id is None:
                raise ValueError(
                    "dataset_id must be given when using MultipleLoggedDataset."
                )

            if isinstance(input_dict, MultipleInputDict):
                input_dict = input_dict.get(dataset_id)

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
            dataset_id=dataset_id,
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

    def visualize_policy_value_with_multiple_estimates(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        plot_type: str = "ci",
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value_multiple.png",
    ) -> None:
        """Visualize policy value estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, the method visualizes the average policy value and the confidence intervals based on the multiple estimate.
            If "scatter" is given, the method visualizes the individual estimation result.

        alpha: float, default=0.05
            Significance level. The value should be within (0, 1].

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

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
        if not isinstance(input_dict, MultipleInputDict):
            raise ValueError("input_dict must be an instance of MultipleInputDict.")
        if (
            not self.use_multiple_logged_dataset
            or not input_dict.use_same_eval_policy_across_dataset
        ):
            raise RuntimeError(
                "This function is applicable only when MultipleLoggedDataset is used "
                "and MultipleInputDict is collected by the same evaluation policy across logged datasets, "
                "but found False."
            )
        if len(self.multiple_logged_dataset) != len(input_dict):
            raise ValueError(
                "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
            )

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

        policy_value_dict_ = self.estimate_policy_value(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
        )

        policy_value_dict = defaultdict(dict)
        policy_value_interval_dict = defaultdict(dict)

        input_dict = input_dict.get(0)
        for eval_policy in input_dict:
            for estimator in compared_estimators:

                policy_value = np.zeros((len(self.multiple_logged_dataset),))
                for i in range(len(self.multiple_logged_dataset)):
                    policy_value[i] = policy_value_dict_[i][eval_policy][estimator]

                policy_value_dict[eval_policy][estimator] = policy_value
                policy_value_interval_dict[eval_policy][
                    estimator
                ] = self._estimate_confidence_interval[ci](
                    policy_value,
                    alpha=alpha,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

            on_policy = input_dict[eval_policy]["on_policy_policy_value"]

            if on_policy is not None:
                policy_value_dict[eval_policy]["on_policy"] = on_policy.mean()
            else:
                policy_value_dict[eval_policy]["on_policy"] = None

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        n_policies = len(input_dict)
        n_estimators = len(compared_estimators)

        if plot_type == "ci":
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
                        mean[j] = policy_value_interval_dict[eval_policy][estimator][
                            "mean"
                        ]
                        lower[j] = policy_value_interval_dict[eval_policy][estimator][
                            f"{100 * (1. - alpha)}% CI (lower)"
                        ]
                        upper[j] = policy_value_interval_dict[eval_policy][estimator][
                            f"{100 * (1. - alpha)}% CI (upper)"
                        ]

                    ax.bar(
                        np.arange(n_estimators),
                        mean,
                        yerr=[upper - mean, mean - lower],
                        color=color,
                        tick_label=compared_estimators,
                    )

                    on_policy = policy_value_dict[eval_policy]["on_policy"]
                    if on_policy is not None:
                        ax.scatter(
                            np.arange(n_estimators),
                            np.full((n_estimators), on_policy),
                            color="black",
                            marker="*",
                            s=150,
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

                    on_policy = np.zeros(n_policies)
                    for j, eval_policy in enumerate(input_dict.keys()):
                        on_policy[j] = policy_value_dict[eval_policy]["on_policy"]

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
                        mean[j] = policy_value_interval_dict[eval_policy][estimator][
                            "mean"
                        ]
                        lower[j] = policy_value_interval_dict[eval_policy][estimator][
                            f"{100 * (1. - alpha)}% CI (lower)"
                        ]
                        upper[j] = policy_value_interval_dict[eval_policy][estimator][
                            f"{100 * (1. - alpha)}% CI (upper)"
                        ]

                    ax.bar(
                        np.arange(n_policies),
                        mean,
                        yerr=[upper - mean, mean - lower],
                        color=color,
                        tick_label=list(input_dict.keys()),
                    )

                    if visualize_on_policy:
                        ax.scatter(
                            np.arange(n_policies),
                            on_policy,
                            color="black",
                            marker="*",
                            s=150,
                        )

                    ax.set_title(estimator, fontsize=16)
                    ax.set_ylabel(
                        f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)",
                        fontsize=12,
                    )
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)

        else:
            if hue == "estimator":
                fig = plt.figure(figsize=(2 * n_estimators, 4 * n_policies))

                for i, eval_policy in enumerate(input_dict.keys()):
                    if i == 0:
                        ax = ax0 = fig.add_subplot(n_policies, 1, i + 1)
                    elif sharey:
                        ax = fig.add_subplot(n_policies, 1, i + 1, sharey=ax0)
                    else:
                        ax = fig.add_subplot(n_policies, 1, i + 1)

                    df = DataFrame()
                    for j, estimator in enumerate(compared_estimators):
                        df[estimator] = policy_value_dict[eval_policy][estimator]

                    df["dataset_id"] = np.arange(len(self.multiple_logged_dataset))
                    df = pd.melt(
                        df,
                        id_vars=["dataset_id"],
                        var_name="estimator",
                        value_name="policy_value",
                    )

                    palette = {}
                    if plot_type == "violin":
                        for j, estimator in enumerate(compared_estimators):
                            palette[estimator] = color[j % n_colors]

                        sns.violinplot(
                            data=df,
                            x="estimator",
                            y="policy_value",
                            scale="width",
                            width=0.5,
                            palette=palette,
                            ax=ax,
                        )

                    else:
                        for j in range(len(self.multiple_logged_dataset)):
                            palette[j] = color[j % n_colors]

                        sns.swarmplot(
                            data=df,
                            x="estimator",
                            y="policy_value",
                            hue="dataset_id",
                            palette=palette,
                            ax=ax,
                        )

                    on_policy = policy_value_dict[eval_policy]["on_policy"]
                    if on_policy is not None:
                        ax.scatter(
                            np.arange(n_estimators),
                            np.full((n_estimators), on_policy),
                            color="black",
                            marker="*",
                            s=150,
                        )

                    if not legend:
                        ax.get_legend().remove()

                    ax.set_title(eval_policy, fontsize=16)
                    ax.set_xlabel("")
                    ax.set_ylabel(
                        f"Estimated Policy Value",
                        fontsize=12,
                    )
                    ax.set_xticks(np.arange(n_estimators), compared_estimators)
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)

            else:
                visualize_on_policy = True
                for eval_policy in input_dict.keys():
                    if input_dict[eval_policy]["on_policy_policy_value"] is None:
                        visualize_on_policy = False

                    on_policy = np.zeros(n_policies)
                    for j, eval_policy in enumerate(input_dict.keys()):
                        on_policy[j] = policy_value_dict[eval_policy]["on_policy"]

                fig = plt.figure(figsize=(2 * n_policies, 4 * n_estimators))

                for i, estimator in enumerate(compared_estimators):
                    if i == 0:
                        ax = ax0 = fig.add_subplot(n_estimators, 1, i + 1)
                    elif sharey:
                        ax = fig.add_subplot(n_estimators, 1, i + 1, sharey=ax0)
                    else:
                        ax = fig.add_subplot(n_estimators, 1, i + 1)

                    df = DataFrame()
                    for j, eval_policy in enumerate(input_dict.keys()):
                        df[eval_policy] = policy_value_dict[eval_policy][estimator]

                    df["dataset_id"] = np.arange(len(self.multiple_logged_dataset))
                    df = pd.melt(
                        df,
                        id_vars=["dataset_id"],
                        var_name="eval_policy",
                        value_name="policy_value",
                    )

                    palette = {}
                    if plot_type == "violin":
                        for j, estimator in enumerate(compared_estimators):
                            palette[estimator] = color[j % n_colors]

                        sns.violinplot(
                            data=df,
                            x="eval_policy",
                            y="policy_value",
                            scale="width",
                            width=0.5,
                            palette=palette,
                            ax=ax,
                        )

                    else:
                        for j in range(len(self.multiple_logged_dataset)):
                            palette[j] = color[j % n_colors]

                        sns.swarmplot(
                            data=df,
                            x="eval_policy",
                            y="policy_value",
                            hue="dataset_id",
                            palette=palette,
                            ax=ax,
                        )

                    if visualize_on_policy:
                        ax.scatter(
                            np.arange(n_policies),
                            on_policy,
                            color="black",
                            marker="*",
                            s=150,
                        )

                    ax.set_title(estimator, fontsize=16)
                    ax.set_xlabel("")
                    ax.set_ylabel(
                        f"Estimated Policy Value",
                        fontsize=12,
                    )
                    ax.set_xticks(np.arange(n_policies), list(input_dict.keys()))
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)

        fig.subplots_adjust(top=1.0)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

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
    logged_dataset: LoggedDataset or MultipleLoggedDataset
        Logged dataset used to conduct OPE.

        .. code-block:: python

            key: [
                size,
                n_trajectories,
                step_per_trajectory,
                action_type,
                n_actions,
                action_dim,
                action_keys,
                action_meaning,
                state_dim,
                state_keys,
                state,
                action,
                reward,
                done,
                terminal,
                info,
                pscore,
            ]

            .. seealso::

                :class:`ofrl.dataset.SyntheticDataset` describes the components of :class:`logged_dataset`.

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
        self.use_multiple_logged_dataset = False
        if isinstance(self.logged_dataset, MultipleLoggedDataset):
            self.multiple_logged_dataset = self.logged_dataset
            self.logged_dataset = self.multiple_logged_dataset.get(0)
            self.use_multiple_logged_dataset = True

        check_logged_dataset(self.logged_dataset)
        self.step_per_trajectory = self.logged_dataset["step_per_trajectory"]
        self.action_type = self.logged_dataset["action_type"]

        if not self.use_multiple_logged_dataset:
            self._register_logged_dataset()

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

        if self.action_type == "continuous":
            if self.action_scaler is not None and not isinstance(
                self.action_scaler, ActionScaler
            ):
                raise ValueError(
                    "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
                )
            check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)

        self._estimate_confidence_interval = {
            "bootstrap": estimate_confidence_interval_by_bootstrap,
            "hoeffding": estimate_confidence_interval_by_hoeffding,
            "bernstein": estimate_confidence_interval_by_empirical_bernstein,
            "ttest": estimate_confidence_interval_by_t_test,
        }

    def _register_logged_dataset(self, id: Optional[int] = None):
        if id is not None:
            self.logged_dataset = self.multiple_logged_dataset.get(id)

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

    def _estimate_cumulative_distribution_function(
        self,
        input_dict: OPEInputDict,
        compared_estimators: Optional[List[str]] = None,
        reward_scale: Optional[np.ndarray] = None,
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        reward_scale: array-like of shape (n_partition, ), default=None
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        Return
        -------
        cumulative_distribution_dict: dict
            Dictionary containing the cumulative distribution of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        check_input_dict(input_dict)

        cumulative_distribution_dict = defaultdict(dict)
        reward_scale = (
            self.obtain_reward_scale() if reward_scale is None else reward_scale
        )

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

    def _estimate_mean(
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

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

    def _estimate_variance(
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

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

    def _estimate_conditional_value_at_risk(
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

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

    def _estimate_interquartile_range(
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

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

    def estimate_cumulative_distribution_function(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
        reward_scale: Optional[np.ndarray] = None,
    ):
        """Estimate the cumulative distribution of the trajectory wise reward of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of cumulative_distribution_dict.

        reward_scale: array-like of shape (n_partition, ), default=None
            Scale of the trajectory wise reward used for x-axis of CDF curve.

        Return
        -------
        cumulative_distribution_dict: dict (, list of dict)
            Dictionary containing the cumulative distribution of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )

        if reward_scale is not None:
            check_array(reward_scale, name="reward_scale", expected_dim=1)
            reward_scale = np.sort(reward_scale)

        if dataset_id is None and self.use_multiple_logged_dataset:
            if not isinstance(input_dict, MultipleInputDict):
                raise RuntimeError(
                    "MultipleInputDict should be given for input_dict, when MultipleLoggedDataset is used and dataset_id is not specified"
                    "Please pass MultipleInputDict or specify dataset_id."
                )
            if len(input_dict) != len(self.multiple_logged_dataset):
                raise ValueError(
                    "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
                )

            cumulative_distribution_dict = []
            for i in range(len(self.multiple_logged_dataset)):
                self._register_logged_dataset(i)
                input_dict_ = input_dict.get(i)

                cumulative_distribution_dict_ = (
                    self._estimate_cumulative_distribution_function(
                        input_dict_,
                        compared_estimators=compared_estimators,
                        reward_scale=reward_scale,
                    )
                )
                cumulative_distribution_dict.append(cumulative_distribution_dict_)

        else:
            if self.use_multiple_logged_dataset:
                if dataset_id is None:
                    raise ValueError(
                        "dataset_id must be given when using MultipleInputDict."
                    )
                if isinstance(input_dict, MultipleInputDict):
                    input_dict = input_dict.get(dataset_id)

                self._register_logged_dataset(dataset_id)

            cumulative_distribution_dict = (
                self._estimate_cumulative_distribution_function(
                    input_dict,
                    compared_estimators=compared_estimators,
                    reward_scale=reward_scale,
                )
            )

        return cumulative_distribution_dict

    def estimate_mean(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
    ):
        """Estimate the mean of the trajectory wise reward (i.e., policy value) of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of mean_dict.

        Return
        -------
        mean_dict: dict (, list of dict)
            Dictionary containing the mean trajectory wise reward of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )

        if dataset_id is None and self.use_multiple_logged_dataset:
            if not isinstance(input_dict, MultipleInputDict):
                raise RuntimeError(
                    "MultipleInputDict should be given for input_dict, when MultipleLoggedDataset is used and dataset_id is not specified"
                    "Please pass MultipleInputDict or specify dataset_id."
                )
            if len(input_dict) != len(self.multiple_logged_dataset):
                raise ValueError(
                    "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
                )

            mean_dict = []
            for i in range(len(self.multiple_logged_dataset)):
                self._register_logged_dataset(i)
                input_dict_ = input_dict.get(i)

                mean_dict_ = self._estimate_mean(
                    input_dict_,
                    compared_estimators=compared_estimators,
                )
                mean_dict.append(mean_dict_)

        else:
            if self.use_multiple_logged_dataset:
                if dataset_id is None:
                    raise ValueError(
                        "dataset_id must be given when using MultipleInputDict."
                    )
                if isinstance(input_dict, MultipleInputDict):
                    input_dict = input_dict.get(dataset_id)

                self._register_logged_dataset(dataset_id)

            mean_dict = self._estimate_mean(
                input_dict,
                compared_estimators=compared_estimators,
            )

        return mean_dict

    def estimate_variance(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
    ):
        """Estimate the variance of the trajectory wise reward of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of variance_dict.

        Return
        -------
        variance_dict: dict (, list of dict)
            Dictionary containing the variance of trajectory wise reward of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )

        if dataset_id is None and self.use_multiple_logged_dataset:
            if not isinstance(input_dict, MultipleInputDict):
                raise RuntimeError(
                    "MultipleInputDict should be given for input_dict, when MultipleLoggedDataset is used and dataset_id is not specified"
                    "Please pass MultipleInputDict or specify dataset_id."
                )
            if len(input_dict) != len(self.multiple_logged_dataset):
                raise ValueError(
                    "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
                )

            variance_dict = []
            for i in range(len(self.multiple_logged_dataset)):
                self._register_logged_dataset(i)
                input_dict_ = input_dict.get(i)

                variance_dict_ = self._estimate_variance(
                    input_dict_,
                    compared_estimators=compared_estimators,
                )
                variance_dict.append(variance_dict_)

        else:
            if self.use_multiple_logged_dataset:
                if dataset_id is None:
                    raise ValueError(
                        "dataset_id must be given when using MultipleInputDict."
                    )
                if isinstance(input_dict, MultipleInputDict):
                    input_dict = input_dict.get(dataset_id)

                self._register_logged_dataset(dataset_id)

            variance_dict = self._estimate_variance(
                input_dict,
                compared_estimators=compared_estimators,
            )

        return variance_dict

    def estimate_conditional_value_at_risk(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
        alphas: Optional[Union[np.ndarray, float]] = None,
    ):
        """Estimate the conditional value at risk of the trajectory wise reward of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of conditional_value_at_risk_dict.

        alphas: {float, array-like of shape (n_alpha, )}, default=None
            Set of proportions of the sided region. The value(s) should be within `[0, 1)`.
            If `None` is given, :class:`np.linspace(0, 1, 21)` will be used.

        Return
        -------
        conditional_value_at_risk_dict: dict (, list of dict)
            Dictionary containing the conditional value at risk of trajectory wise reward of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name]`

        """
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

        if dataset_id is None and self.use_multiple_logged_dataset:
            if not isinstance(input_dict, MultipleInputDict):
                raise RuntimeError(
                    "MultipleInputDict should be given for input_dict, when MultipleLoggedDataset is used and dataset_id is not specified"
                    "Please pass MultipleInputDict or specify dataset_id."
                )
            if len(input_dict) != len(self.multiple_logged_dataset):
                raise ValueError(
                    "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
                )

            cvar_dict = []
            for i in range(len(self.multiple_logged_dataset)):
                self._register_logged_dataset(i)
                input_dict_ = input_dict.get(i)
                cvar_dict_ = self._estimate_conditional_value_at_risk(
                    input_dict_,
                    compared_estimators=compared_estimators,
                    alphas=alphas,
                )
                cvar_dict.append(cvar_dict_)

        else:
            if self.use_multiple_logged_dataset:
                if dataset_id is None:
                    raise ValueError(
                        "dataset_id must be given when using MultipleInputDict."
                    )
                if isinstance(input_dict, MultipleInputDict):
                    input_dict = input_dict.get(dataset_id)

                self._register_logged_dataset(dataset_id)

            cvar_dict = self._estimate_conditional_value_at_risk(
                input_dict,
                compared_estimators=compared_estimators,
                alphas=alphas,
            )

        return cvar_dict

    def estimate_interquartile_range(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
        alpha: float = 0.05,
    ):
        """Estimate the interquartile range of the trajectory wise reward of the evaluation policies.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of interquartile_range_dict.

        alpha: float, default=0.05
            Proportion of the sided region. The value should be within (0, 1].

        Return
        -------
        interquartile_range_dict: dict (, list of dict)
            Dictionary containing the interquartile range of trajectory wise reward of each evaluation policy estimated by OPE estimators.
            key: :class:`[evaluation_policy_name][OPE_estimator_name][quartile_name]`

        """
        if compared_estimators is None:
            compared_estimators = self.estimators_name
        elif not set(compared_estimators).issubset(self.estimators_name):
            raise ValueError(
                "compared_estimators must be a subset of self.estimators_name, but found False."
            )
        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=0.5)

        if dataset_id is None and self.use_multiple_logged_dataset:
            if not isinstance(input_dict, MultipleInputDict):
                raise RuntimeError(
                    "MultipleInputDict should be given for input_dict, when MultipleLoggedDataset is used and dataset_id is not specified"
                    "Please pass MultipleInputDict or specify dataset_id."
                )
            if len(input_dict) != len(self.multiple_logged_dataset):
                raise ValueError(
                    "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
                )

            interquartile_range_dict = []
            for i in range(len(self.multiple_logged_dataset)):
                self._register_logged_dataset(i)
                input_dict_ = input_dict.get(i)

                interquartile_range_dict_ = self._estimate_interquartile_range(
                    input_dict_,
                    compared_estimators=compared_estimators,
                    alpha=alpha,
                )
                interquartile_range_dict.append(interquartile_range_dict_)

        else:
            if self.use_multiple_logged_dataset:
                if dataset_id is None:
                    raise ValueError(
                        "dataset_id must be given when using MultipleInputDict."
                    )
                if isinstance(input_dict, MultipleInputDict):
                    input_dict = input_dict.get(dataset_id)

                self._register_logged_dataset(dataset_id)

            interquartile_range_dict = self._estimate_interquartile_range(
                input_dict,
                compared_estimators=compared_estimators,
                alpha=alpha,
            )

        return interquartile_range_dict

    def visualize_cumulative_distribution_function(
        self,
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_cumulative_distribution_function.png",
    ) -> None:
        """Visualize the cumulative distribution function estimated by OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of interquartile_range_dict.

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
        if self.use_multiple_logged_dataset:
            if dataset_id is None:
                raise ValueError(
                    "dataset_id must be given when using MultipleLoggedDataset."
                )

            if isinstance(input_dict, MultipleInputDict):
                input_dict = input_dict.get(dataset_id)

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

        cumulative_distribution_function_dict = (
            self.estimate_cumulative_distribution_function(
                input_dict,
                compared_estimators=compared_estimators,
                dataset_id=dataset_id,
            )
        )
        reward_scale = self.obtain_reward_scale()

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
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
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
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of interquartile_range_dict.

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
        if self.use_multiple_logged_dataset:
            if dataset_id is None:
                raise ValueError(
                    "dataset_id must be given when using MultipleLoggedDataset."
                )

            if isinstance(input_dict, MultipleInputDict):
                input_dict = input_dict.get(dataset_id)

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

        check_scalar(alpha, name="alpha", target_type=float, min_val=0.0, max_val=1.0)

        mean_dict = self.estimate_mean(
            input_dict,
            compared_estimators=compared_estimators,
            dataset_id=dataset_id,
        )
        variance_dict = self.estimate_variance(
            input_dict,
            compared_estimators=compared_estimators,
            dataset_id=dataset_id,
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
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
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
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of interquartile_range_dict.

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
        if self.use_multiple_logged_dataset:
            if dataset_id is None:
                raise ValueError(
                    "dataset_id must be given when using MultipleLoggedDataset."
                )

            if isinstance(input_dict, MultipleInputDict):
                input_dict = input_dict.get(dataset_id)

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
            dataset_id=dataset_id,
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
        input_dict: Union[OPEInputDict, MultipleInputDict],
        compared_estimators: Optional[List[str]] = None,
        dataset_id: Optional[Union[int, str]] = None,
        alpha: float = 0.05,
        hue: str = "estimator",
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_interquartile_range.png",
    ) -> None:
        """Visualize the interquartile range estimated by OPE estimators.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the method returns the list of interquartile_range_dict.

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
        if self.use_multiple_logged_dataset:
            if dataset_id is None:
                raise ValueError(
                    "dataset_id must be given when using MultipleLoggedDataset."
                )

            if isinstance(input_dict, MultipleInputDict):
                input_dict = input_dict.get(dataset_id)

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
            dataset_id=dataset_id,
        )
        interquartile_dict = self.estimate_interquartile_range(
            input_dict,
            compared_estimators=compared_estimators,
            dataset_id=dataset_id,
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

    def _visualize_off_policy_estimates_with_multiple_estimates(
        self,
        estimation_dict: Dict[str, Dict[str, np.ndarray]],
        estimation_interval_dict: Dict[str, Dict[str, Dict[str, float]]],
        compared_estimators: Optional[List[str]] = None,
        plot_type: str = "ci",
        alpha: float = 0.05,
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: Optional[str] = None,
    ) -> None:
        """Visualize values estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        estimation_dict: dict
            Dictionary containing estimation result of OPE. key: ``[eval_policy][estimator]``

        estimation_interval_dict: dict
            Dictionary containing confidence interval of multiple OPE estimate. key: ``[eval_policy][estimator]``

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, the method visualizes the average policy value and the confidence intervals based on the multiple estimate.
            If "scatter" is given, the method visualizes the individual estimation result.

        alpha: float, default=0.05
            Significance level. The value should be within (0, 1].

        hue: {"estimator", "policy"}, default="estimator"
            Hue of the plot.

        legend: bool, default=True
            Whether to include a legend in the scatter plot.

        sharey: bool, default=False
            If `True`, the y-axis will be shared among different estimators or evaluation policies.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If `None` is given, the figure will not be saved.

        fig_name: str, default=None
            Name of the bar figure.

        """
        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        n_policies = len(estimation_dict)
        n_estimators = len(compared_estimators)

        if plot_type == "ci":
            if hue == "estimator":
                fig = plt.figure(figsize=(2 * n_estimators, 4 * n_policies))

                for i, eval_policy in enumerate(estimation_dict.keys()):
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
                        mean[j] = estimation_interval_dict[eval_policy][estimator][
                            "mean"
                        ]
                        lower[j] = estimation_interval_dict[eval_policy][estimator][
                            f"{100 * (1. - alpha)}% CI (lower)"
                        ]
                        upper[j] = estimation_interval_dict[eval_policy][estimator][
                            f"{100 * (1. - alpha)}% CI (upper)"
                        ]

                    ax.bar(
                        np.arange(n_estimators),
                        mean,
                        yerr=[upper - mean, mean - lower],
                        color=color,
                        tick_label=compared_estimators,
                    )

                    on_policy = estimation_dict[eval_policy]["on_policy"]
                    if on_policy is not None:
                        ax.scatter(
                            np.arange(n_estimators),
                            np.full((n_estimators), on_policy),
                            color="black",
                            marker="*",
                            s=150,
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
                for eval_policy in estimation_dict.keys():
                    if estimation_dict[eval_policy]["on_policy"] is None:
                        visualize_on_policy = False

                    on_policy = np.zeros(n_policies)
                    for j, eval_policy in enumerate(estimation_dict.keys()):
                        on_policy[j] = estimation_dict[eval_policy]["on_policy"]

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

                    for j, eval_policy in enumerate(estimation_dict.keys()):
                        mean[j] = estimation_interval_dict[eval_policy][estimator][
                            "mean"
                        ]
                        lower[j] = estimation_interval_dict[eval_policy][estimator][
                            f"{100 * (1. - alpha)}% CI (lower)"
                        ]
                        upper[j] = estimation_interval_dict[eval_policy][estimator][
                            f"{100 * (1. - alpha)}% CI (upper)"
                        ]

                    ax.bar(
                        np.arange(n_policies),
                        mean,
                        yerr=[upper - mean, mean - lower],
                        color=color,
                        tick_label=list(estimation_dict.keys()),
                    )

                    if visualize_on_policy:
                        ax.scatter(
                            np.arange(n_policies),
                            on_policy,
                            color="black",
                            marker="*",
                            s=150,
                        )

                    ax.set_title(estimator, fontsize=16)
                    ax.set_ylabel(
                        f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)",
                        fontsize=12,
                    )
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)

        else:
            if hue == "estimator":
                fig = plt.figure(figsize=(2 * n_estimators, 4 * n_policies))

                for i, eval_policy in enumerate(estimation_dict.keys()):
                    if i == 0:
                        ax = ax0 = fig.add_subplot(n_policies, 1, i + 1)
                    elif sharey:
                        ax = fig.add_subplot(n_policies, 1, i + 1, sharey=ax0)
                    else:
                        ax = fig.add_subplot(n_policies, 1, i + 1)

                    df = DataFrame()
                    for j, estimator in enumerate(compared_estimators):
                        df[estimator] = estimation_dict[eval_policy][estimator]

                    df["dataset_id"] = np.arange(len(self.multiple_logged_dataset))
                    df = pd.melt(
                        df,
                        id_vars=["dataset_id"],
                        var_name="estimator",
                        value_name="policy_value",
                    )

                    palette = {}
                    if plot_type == "violin":
                        for j, estimator in enumerate(compared_estimators):
                            palette[estimator] = color[j % n_colors]

                        sns.violinplot(
                            data=df,
                            x="estimator",
                            y="policy_value",
                            scale="width",
                            width=0.5,
                            palette=palette,
                            ax=ax,
                        )

                    else:
                        for j in range(len(self.multiple_logged_dataset)):
                            palette[j] = color[j % n_colors]

                        sns.swarmplot(
                            data=df,
                            x="estimator",
                            y="policy_value",
                            hue="dataset_id",
                            palette=palette,
                            ax=ax,
                        )

                    on_policy = estimation_dict[eval_policy]["on_policy"]
                    if on_policy is not None:
                        ax.scatter(
                            np.arange(n_estimators),
                            np.full((n_estimators), on_policy),
                            color="black",
                            marker="*",
                            s=150,
                        )

                    if not legend:
                        ax.get_legend().remove()

                    ax.set_title(eval_policy, fontsize=16)
                    ax.set_xlabel("")
                    ax.set_ylabel(
                        f"Estimated Policy Value",
                        fontsize=12,
                    )
                    ax.set_xticks(np.arange(n_estimators), compared_estimators)
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)

            else:
                visualize_on_policy = True
                for eval_policy in estimation_dict.keys():
                    if estimation_dict[eval_policy]["on_policy"] is None:
                        visualize_on_policy = False

                    on_policy = np.zeros(n_policies)
                    for j, eval_policy in enumerate(estimation_dict.keys()):
                        on_policy[j] = estimation_dict[eval_policy]["on_policy"]

                fig = plt.figure(figsize=(2 * n_policies, 4 * n_estimators))

                for i, estimator in enumerate(compared_estimators):
                    if i == 0:
                        ax = ax0 = fig.add_subplot(n_estimators, 1, i + 1)
                    elif sharey:
                        ax = fig.add_subplot(n_estimators, 1, i + 1, sharey=ax0)
                    else:
                        ax = fig.add_subplot(n_estimators, 1, i + 1)

                    df = DataFrame()
                    for j, eval_policy in enumerate(estimation_dict.keys()):
                        df[eval_policy] = estimation_dict[eval_policy][estimator]

                    df["dataset_id"] = np.arange(len(self.multiple_logged_dataset))
                    df = pd.melt(
                        df,
                        id_vars=["dataset_id"],
                        var_name="eval_policy",
                        value_name="policy_value",
                    )

                    palette = {}
                    if plot_type == "violin":
                        for j, estimator in enumerate(compared_estimators):
                            palette[estimator] = color[j % n_colors]

                        sns.violinplot(
                            data=df,
                            x="eval_policy",
                            y="policy_value",
                            scale="width",
                            width=0.5,
                            palette=palette,
                            ax=ax,
                        )

                    else:
                        for j in range(len(self.multiple_logged_dataset)):
                            palette[j] = color[j % n_colors]

                        sns.swarmplot(
                            data=df,
                            x="eval_policy",
                            y="policy_value",
                            hue="dataset_id",
                            palette=palette,
                            ax=ax,
                        )

                    if visualize_on_policy:
                        ax.scatter(
                            np.arange(n_policies),
                            on_policy,
                            color="black",
                            marker="*",
                            s=150,
                        )

                    ax.set_title(estimator, fontsize=16)
                    ax.set_xlabel("")
                    ax.set_ylabel(
                        f"Estimated Policy Value",
                        fontsize=12,
                    )
                    ax.set_xticks(np.arange(n_policies), list(estimation_dict.keys()))
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)

        fig.subplots_adjust(top=1.0)
        plt.show()

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_cumulative_distribution_function_with_multiple_estimates(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        scale_min: Optional[float] = None,
        scale_max: Optional[float] = None,
        n_partition: Optional[int] = None,
        plot_type: str = "ci",
        hue: str = "estimator",
        legend: bool = True,
        n_cols: Optional[int] = None,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value_multiple.png",
    ) -> None:
        """Visualize policy value estimated by OPE estimators across multiple logged dataset.

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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        plot_type: {"ci", "enumerate"}, default="ci"
            Type of plot.
            If "ci" is given, the method visualizes the average policy value and its 95% confidence intervals based on the multiple estimate.
            If "enumerate" is given, the method visualizes the individual estimation result.

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
        if scale_min is None:
            if self.scale_min is None:
                raise ValueError(
                    "scale_min must be specified when self.scale_min is None"
                )
            else:
                scale_min = self.scale_min
        if scale_max is None:
            if self.scale_max is None:
                raise ValueError(
                    "scale_max must be specified when self.scale_max is None"
                )
            else:
                scale_max = self.scale_max
        if n_partition is None:
            if self.n_partition is None:
                raise ValueError(
                    "n_partition must be specified when self.n_partition is None"
                )
            else:
                n_partition = self.n_partition
        check_scalar(scale_min, name="scale_min", target_type=float)
        check_scalar(scale_max, name="scale_max", target_type=float)
        check_scalar(n_partition, name="n_partition", target_type=int, min_val=1)

        if not isinstance(input_dict, MultipleInputDict):
            raise ValueError("input_dict must be an instance of MultipleInputDict.")
        if (
            not self.use_multiple_logged_dataset
            or not input_dict.use_same_eval_policy_across_dataset
        ):
            raise RuntimeError(
                "This function is applicable only when MultipleLoggedDataset is used "
                "and MultipleInputDict is collected by the same evaluation policy across logged datasets, "
                "but found False."
            )
        if len(self.multiple_logged_dataset) != len(input_dict):
            raise ValueError(
                "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
            )

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

        reward_scale = np.linspace(scale_min, scale_max, num=n_partition)

        cdf_dict_ = self.estimate_cumulative_distribution_function(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            reward_scale=reward_scale,
        )

        cdf_dict = defaultdict(dict)
        input_dict = input_dict.get(0)
        for eval_policy in input_dict:
            for estimator in compared_estimators:

                cdf = np.zeros((len(self.multiple_logged_dataset), n_partition))
                for i in range(len(self.multiple_logged_dataset)):
                    cdf[i] = cdf_dict_[i][eval_policy][estimator]

                    if not (cdf[i][1:] - cdf[i][:-1]).all():
                        print(i, eval_policy, estimator, cdf[i])

                cdf_dict[eval_policy][estimator] = cdf

            cdf_dict[eval_policy]["on_policy"] = cdf_dict_[0][eval_policy]["on_policy"]

        plt.style.use("ggplot")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        n_colors = len(color)

        if plot_type == "ci":
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

                            df = DataFrame()
                            for l in range(len(self.multiple_logged_dataset)):
                                df["xscale"] = reward_scale
                                df[l] = cdf_dict[eval_policy][estimator][l]

                            df = pd.melt(
                                df,
                                id_vars=["xscale"],
                                var_name="dataset_id",
                                value_name="cdf",
                            )
                            sns.lineplot(
                                data=df,
                                x="xscale",
                                y="cdf",
                                ax=axes[i],
                                palette=[color[j % n_colors]],
                                label="estimator",
                            )

                        on_policy = cdf_dict[eval_policy]["on_policy"]
                        if on_policy is not None:
                            axes[i].plot(
                                reward_scale,
                                on_policy,
                                label="on_policy",
                                color="black",
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
                            df = DataFrame()
                            for l in range(len(self.multiple_logged_dataset)):
                                df["xscale"] = reward_scale
                                df[l] = cdf_dict[eval_policy][estimator][l]

                            df = pd.melt(
                                df,
                                id_vars=["xscale"],
                                var_name="dataset_id",
                                value_name="cdf",
                            )
                            sns.lineplot(
                                data=df,
                                x="xscale",
                                y="cdf",
                                ax=axes[i // n_cols, i % n_cols],
                                palette=[color[j % n_colors]],
                                label=estimator,
                            )

                        on_policy = cdf_dict[eval_policy]["on_policy"]
                        if on_policy is not None:
                            axes[i // n_cols, i % n_cols].plot(
                                reward_scale,
                                on_policy,
                                label="on_policy",
                                color="black",
                            )

                        axes[i // n_cols, i % n_cols].set_title(eval_policy)
                        axes[i // n_cols, i % n_cols].set_xlabel(
                            "trajectory wise reward"
                        )
                        axes[i // n_cols, i % n_cols].set_ylabel(
                            "cumulative probability"
                        )
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
                            df = DataFrame()
                            for l in range(len(self.multiple_logged_dataset)):
                                df["xscale"] = reward_scale
                                df[l] = cdf_dict[eval_policy][estimator][l]

                            df = pd.melt(
                                df,
                                id_vars=["xscale"],
                                var_name="dataset_id",
                                value_name="cdf",
                            )
                            sns.lineplot(
                                data=df,
                                x="xscale",
                                y="cdf",
                                ax=axes[i],
                                palette=[color[j % n_colors]],
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
                                cdf_dict[eval_policy]["on_policy"],
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
                            df = DataFrame()
                            for l in range(len(self.multiple_logged_dataset)):
                                df["xscale"] = reward_scale
                                df[l] = cdf_dict[eval_policy][estimator][l]

                            df = pd.melt(
                                df,
                                id_vars=["xscale"],
                                var_name="dataset_id",
                                value_name="cdf",
                            )
                            sns.lineplot(
                                data=df,
                                x="xscale",
                                y="cdf",
                                ax=axes[i // n_cols, i % n_cols],
                                palette=[color[j % n_colors]],
                                label=eval_policy,
                            )

                        axes[i // n_cols, i % n_cols].set_title(estimator)
                        axes[i // n_cols, i % n_cols].set_xlabel(
                            "trajectory wise reward"
                        )
                        axes[i // n_cols, i % n_cols].set_ylabel(
                            "cumulative probability"
                        )
                        if legend:
                            axes[i // n_cols, i % n_cols].legend()

                    if visualize_on_policy:
                        for j, eval_policy in enumerate(input_dict.keys()):
                            axes[(i + 1) // n_cols, (i + 1) % n_cols].plot(
                                reward_scale,
                                cdf_dict[eval_policy]["on_policy"],
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

        else:
            if hue == "estimator":
                n_cols = len(compared_estimators)
                n_rows = len(input_dict)

                fig, axes = plt.subplots(
                    nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows)
                )

                for i, eval_policy in enumerate(input_dict.keys()):
                    for j, estimator in enumerate(compared_estimators):
                        for l in range(len(self.multiple_logged_dataset)):
                            axes[i, j].plot(
                                reward_scale,
                                cdf_dict[eval_policy][estimator][l],
                                label=l,
                            )

                        on_policy = cdf_dict[eval_policy]["on_policy"]
                        if on_policy is not None:
                            axes[i, j].plot(
                                reward_scale,
                                on_policy,
                                color="black",
                            )

                        axes[i, j].set_title(f"{eval_policy}, {estimator}")
                        axes[i, j].set_xlabel("trajectory wise reward")
                        axes[i, j].set_ylabel("cumulative probability")
                        if legend:
                            axes[i, j].legend(title="dataset_id")

            else:
                n_cols = len(input_dict)
                n_rows = len(compared_estimators)

                fig, axes = plt.subplots(
                    nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows)
                )

                for i, estimator in enumerate(compared_estimators):
                    for j, eval_policy in enumerate(input_dict.keys()):
                        for l in range(len(self.multiple_logged_dataset)):
                            axes[i, j].plot(
                                reward_scale,
                                cdf_dict[eval_policy][estimator][l],
                                label=l,
                            )

                        on_policy = cdf_dict[eval_policy]["on_policy"]
                        if on_policy is not None:
                            axes[i, j].plot(
                                reward_scale,
                                on_policy,
                                color="black",
                            )

                        axes[i, j].set_title(f"{estimator}, {eval_policy}")
                        axes[i, j].set_xlabel("trajectory wise reward")
                        axes[i, j].set_ylabel("cumulative probability")
                        if legend:
                            axes[i, j].legend(title="dataset_id")

            fig.subplots_adjust(hspace=0.35, wspace=0.2)
            plt.show()

            if fig_dir:
                fig.savefig(str(fig_dir / fig_name), dpi=300, bbox_inches="tight")

    def visualize_policy_value_with_multiple_estimates(
        self,
        input_dict: MultipleInputDict,
        compared_estimators: Optional[List[str]] = None,
        plot_type: str = "ci",
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value_multiple.png",
    ) -> None:
        """Visualize policy value estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, the method visualizes the average policy value and the confidence intervals based on the multiple estimate.
            If "scatter" is given, the method visualizes the individual estimation result.

        alpha: float, default=0.05
            Significance level. The value should be within (0, 1].

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

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
        if not isinstance(input_dict, MultipleInputDict):
            raise ValueError("input_dict must be an instance of MultipleInputDict.")
        if (
            not self.use_multiple_logged_dataset
            or not input_dict.use_same_eval_policy_across_dataset
        ):
            raise RuntimeError(
                "This function is applicable only when MultipleLoggedDataset is used "
                "and MultipleInputDict is collected by the same evaluation policy across logged datasets, "
                "but found False."
            )
        if len(self.multiple_logged_dataset) != len(input_dict):
            raise ValueError(
                "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
            )

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

        policy_value_dict_ = self.estimate_mean(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
        )

        policy_value_dict = defaultdict(dict)
        policy_value_interval_dict = defaultdict(dict)

        input_dict = input_dict.get(0)
        for eval_policy in input_dict:
            for estimator in compared_estimators:

                policy_value = np.zeros((len(self.multiple_logged_dataset),))
                for i in range(len(self.multiple_logged_dataset)):
                    policy_value[i] = policy_value_dict_[i][eval_policy][estimator]

                policy_value_dict[eval_policy][estimator] = policy_value
                policy_value_interval_dict[eval_policy][
                    estimator
                ] = self._estimate_confidence_interval[ci](
                    policy_value,
                    alpha=alpha,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

            on_policy = input_dict[eval_policy]["on_policy_policy_value"]

            if on_policy is not None:
                policy_value_dict[eval_policy]["on_policy"] = on_policy.mean()
            else:
                policy_value_dict[eval_policy]["on_policy"] = None

        self._visualize_off_policy_estimates_with_multiple_estimates(
            estimation_dict=policy_value_dict,
            estimation_interval_dict=policy_value_interval_dict,
            compared_estimators=compared_estimators,
            plot_type=plot_type,
            alpha=alpha,
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
        plot_type: str = "ci",
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_variance_multiple.png",
    ) -> None:
        """Visualize variance estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, the method visualizes the average policy value and the confidence intervals based on the multiple estimate.
            If "scatter" is given, the method visualizes the individual estimation result.

        alpha: float, default=0.05
            Significance level. The value should be within (0, 1].

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

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
        if not isinstance(input_dict, MultipleInputDict):
            raise ValueError("input_dict must be an instance of MultipleInputDict.")
        if (
            not self.use_multiple_logged_dataset
            or not input_dict.use_same_eval_policy_across_dataset
        ):
            raise RuntimeError(
                "This function is applicable only when MultipleLoggedDataset is used "
                "and MultipleInputDict is collected by the same evaluation policy across logged datasets, "
                "but found False."
            )
        if len(self.multiple_logged_dataset) != len(input_dict):
            raise ValueError(
                "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
            )

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

        variance_dict_ = self.estimate_variance(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
        )

        variance_dict = defaultdict(dict)
        variance_interval_dict = defaultdict(dict)

        input_dict = input_dict.get(0)
        for eval_policy in input_dict:
            for estimator in compared_estimators:

                variance = np.zeros((len(self.multiple_logged_dataset),))
                for i in range(len(self.multiple_logged_dataset)):
                    variance[i] = variance_dict_[i][eval_policy][estimator]

                variance_dict[eval_policy][estimator] = variance
                variance_interval_dict[eval_policy][
                    estimator
                ] = self._estimate_confidence_interval[ci](
                    variance,
                    alpha=alpha,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

            on_policy = input_dict[eval_policy]["on_policy_policy_value"]

            if on_policy is not None:
                variance_dict[eval_policy]["on_policy"] = on_policy.var(ddof=1)
            else:
                variance_dict[eval_policy]["on_policy"] = None

        self._visualize_off_policy_estimates_with_multiple_estimates(
            estimation_dict=variance_dict,
            estimation_interval_dict=variance_interval_dict,
            compared_estimators=compared_estimators,
            plot_type=plot_type,
            alpha=alpha,
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
        ope_alpha: float = 0.05,
        plot_type: str = "ci",
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_conditional_value_at_risk_multiple.png",
    ) -> None:
        """Visualize conditional value at risk estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        ope_alpha: float = 0.05.
            Proportion of the sided region in CVaR estimate. The value should be within `[0, 1)`.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, the method visualizes the average policy value and the confidence intervals based on the multiple estimate.
            If "scatter" is given, the method visualizes the individual estimation result.

        alpha: float, default=0.05
            Significance level. The value should be within (0, 1].

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

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
        if not isinstance(input_dict, MultipleInputDict):
            raise ValueError("input_dict must be an instance of MultipleInputDict.")
        if (
            not self.use_multiple_logged_dataset
            or not input_dict.use_same_eval_policy_across_dataset
        ):
            raise RuntimeError(
                "This function is applicable only when MultipleLoggedDataset is used "
                "and MultipleInputDict is collected by the same evaluation policy across logged datasets, "
                "but found False."
            )
        if len(self.multiple_logged_dataset) != len(input_dict):
            raise ValueError(
                "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
            )

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

        cvar_dict_ = self.estimate_conditional_value_at_risk(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            alphas=ope_alpha,
        )

        cvar_dict = defaultdict(dict)
        cvar_interval_dict = defaultdict(dict)

        input_dict = input_dict.get(0)
        for eval_policy in input_dict:
            for estimator in compared_estimators:

                cvar = np.zeros((len(self.multiple_logged_dataset),))
                for i in range(len(self.multiple_logged_dataset)):
                    cvar[i] = cvar_dict_[i][eval_policy][estimator]

                cvar_dict[eval_policy][estimator] = cvar
                cvar_interval_dict[eval_policy][
                    estimator
                ] = self._estimate_confidence_interval[ci](
                    cvar,
                    alpha=alpha,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

            on_policy = input_dict[eval_policy]["on_policy_policy_value"]

            if on_policy is not None:
                cvar_idx = int(ope_alpha * len(on_policy))
                cvar_dict[eval_policy]["on_policy"] = np.partition(on_policy, cvar_idx)[
                    :cvar_idx
                ].mean()
            else:
                cvar_dict[eval_policy]["on_policy"] = None

        self._visualize_off_policy_estimates_with_multiple_estimates(
            estimation_dict=cvar_dict,
            estimation_interval_dict=cvar_interval_dict,
            compared_estimators=compared_estimators,
            plot_type=plot_type,
            alpha=alpha,
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
        ope_alpha: float = 0.05,
        plot_type: str = "ci",
        alpha: float = 0.05,
        ci: str = "bootstrap",
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        hue: str = "estimator",
        legend: bool = True,
        sharey: bool = False,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_conditional_value_at_risk_multiple.png",
    ) -> None:
        """Visualize lower quartile estimated by OPE estimators across multiple logged dataset.

        Note
        -------
        This function is applicable only when MultipleLoggedDataset is used and
        MultipleInputDict is collected by the same evaluation policy across logged datasets.

        Parameters
        -------
        input_dict: OPEInputDict or MultipleInputDict
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

                :class:`ofrl.ope.input.CreateOPEInput` describes the components of :class:`input_dict`.

        compared_estimators: list of str, default=None
            Name of compared estimators.
            If `None` is given, all the estimators are compared.

        ope_alpha: float = 0.05.
            Proportion of the sided region in CVaR estimate. The value should be within `[0, 1)`.

        plot_type: {"ci", "scatter", "violin"}, default="ci"
            Type of plot.
            If "ci" is given, the method visualizes the average policy value and the confidence intervals based on the multiple estimate.
            If "scatter" is given, the method visualizes the individual estimation result.

        alpha: float, default=0.05
            Significance level. The value should be within (0, 0.5].

        ci: {"bootstrap", "hoeffding", "bernstein", "ttest"}, default="bootstrap"
            Estimation method for confidence intervals.

        n_bootstrap_samples: int, default=10000 (> 0)
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None (>= 0)
            Random state.

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
        if not isinstance(input_dict, MultipleInputDict):
            raise ValueError("input_dict must be an instance of MultipleInputDict.")
        if (
            not self.use_multiple_logged_dataset
            or not input_dict.use_same_eval_policy_across_dataset
        ):
            raise RuntimeError(
                "This function is applicable only when MultipleLoggedDataset is used "
                "and MultipleInputDict is collected by the same evaluation policy across logged datasets, "
                "but found False."
            )
        if len(self.multiple_logged_dataset) != len(input_dict):
            raise ValueError(
                "Expected `len(input_dict) == len(self.multiple_logged_dataset)`, but found False."
            )

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

        lower_quartile_dict_ = self.estimate_interquartile_range(
            input_dict=input_dict,
            compared_estimators=compared_estimators,
            alpha=ope_alpha,
        )

        lower_quartile_dict = defaultdict(dict)
        lower_quartile_interval_dict = defaultdict(dict)

        input_dict = input_dict.get(0)
        for eval_policy in input_dict:
            for estimator in compared_estimators:

                lower_quartile = np.zeros((len(self.multiple_logged_dataset),))
                for i in range(len(self.multiple_logged_dataset)):
                    lower_quartile[i] = lower_quartile_dict_[i][eval_policy][estimator][
                        f"{100 * (1. - ope_alpha)}% quartile (lower)"
                    ]

                lower_quartile_dict[eval_policy][estimator] = lower_quartile
                lower_quartile_interval_dict[eval_policy][
                    estimator
                ] = self._estimate_confidence_interval[ci](
                    lower_quartile,
                    alpha=alpha,
                    n_bootstrap_samples=n_bootstrap_samples,
                    random_state=random_state,
                )

            on_policy = input_dict[eval_policy]["on_policy_policy_value"]

            if on_policy is not None:
                lower_idx = int(ope_alpha * len(on_policy))
                lower_quartile_dict[eval_policy]["on_policy"] = np.partition(
                    on_policy, lower_idx
                )[lower_idx]
            else:
                lower_quartile_dict[eval_policy]["on_policy"] = None

        self._visualize_off_policy_estimates_with_multiple_estimates(
            estimation_dict=lower_quartile_dict,
            estimation_interval_dict=lower_quartile_interval_dict,
            compared_estimators=compared_estimators,
            plot_type=plot_type,
            alpha=alpha,
            hue=hue,
            legend=legend,
            sharey=sharey,
            fig_dir=fig_dir,
            fig_name=fig_name,
        )

    @property
    def estimators_name(self):
        return list(self.ope_estimators_.keys())
