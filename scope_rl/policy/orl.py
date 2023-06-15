# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Meta class to handle Offline Learning (ORL)."""
from dataclasses import dataclass
from collections import defaultdict
from typing import Union, Optional, Any, Dict, List, Tuple
from tqdm.auto import tqdm

import numpy as np

from d3rlpy.algos import AlgoBase
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split

from .head import BaseHead
from ..utils import MultipleLoggedDataset, defaultdict_to_dict
from ..types import LoggedDataset


HeadDict = Dict[str, Tuple[BaseHead, Dict[str, Any]]]


@dataclass
class TrainCandidatePolicies:
    """Class to handle ORL by multiple algorithms simultaneously. (applicable to both discrete/continuous action cases)

    Imported as: :class:`scope_rl.policy.TrainCandidatePolicies`

    Parameters
    -------
    fitting_args: dict, default=None
        Arguments of fitting function to learn model.

    Examples
    ----------

    Preparation:

    .. code-block:: python

        # import necessary module from SCOPE-RL
        from scope_rl.dataset import SyntheticDataset
        from scope_rl.policy import TrainCandidatePolicies
        from scope_rl.policy import EpsilonGreedyHead, SoftmaxHead

        # import necessary module from other libraries
        import gym
        import rtbgym
        from d3rlpy.algos import DoubleDQN
        from d3rlpy.online.buffers import ReplayBuffer
        from d3rlpy.online.explorers import ConstantEpsilonGreedy
        from d3rlpy.algos import DiscreteBCQ, DiscreteCQL

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

        # convert ddqn policy to a stochastic data collection policy
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
            behavior_policy=behavior_policy,
            random_state=12345,
        )

        # data collection
        logged_dataset = dataset.obtain_episodes(n_trajectories=100)

    **Learning Evaluation Policies**:

    .. code-block:: python

        # base algorithms
        bcq = DiscreteBCQ()
        cql = DiscreteCQL()
        algorithms = [bcq, cql]
        algorithms_name = ["bcq", "cql"]

        # policy wrappers
        policy_wrappers = {
            "eps_01": (
                EpsilonGreedyHead,
                {
                    "epsilon": 0.1,
                    "n_actions": env.action_space.n,
                }
            ),
            "eps_03": (
                EpsilonGreedyHead,
                {
                    "epsilon": 0.3,
                    "n_actions": env.action_space.n,
                }
            ),
            "softmax": (
                SoftmaxHead,
                {
                    "tau": 1.0,
                    "n_actions": env.action_space.n,
                }
            ),
        }

        # off-policy learning
        orl = TrainCandidatePolicies()
        eval_policies = orl.obtain_evaluation_policy(
            algorithms=algorithms,
            algorithms_name=algorithms_name,
            policy_wrappers=policy_wrappers,
            random_state=12345,
        )

    **Output**:

    .. code-block:: python

        >>> [eval_policy.name for eval_policy in eval_policies[0]]

        ['bcq_eps_01', 'bcq_eps_03', 'bcq_softmax', 'cql_eps_01', 'cql_eps_03', 'cql_softmax']


    """

    fitting_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.fitting_args is None:
            self.fitting_args = {
                "n_steps": 10000,
                "scorer": {},
            }

    def _learn_base_policy(
        self,
        logged_dataset: LoggedDataset,
        algorithms: List[AlgoBase],
        random_state: Optional[int] = None,
    ):
        """Learn base policy.

        Parameters
        -------
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

                :class:`scope_rl.dataset.SyntheticDataset` describes the components of :class:`logged_dataset`.

        algorithms: list of AlgoBase
            List of algorithms to fit.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        base_policies: List of AlgoBase
            List of learned policies.

        """
        offlinerl_dataset = MDPDataset(
            observations=logged_dataset["state"],
            actions=logged_dataset["action"],
            rewards=logged_dataset["reward"],
            terminals=logged_dataset["done"],
            episode_terminals=logged_dataset["done"],
            discrete_action=(logged_dataset["action_type"] == "discrete"),
        )
        train_episodes, test_episodes = train_test_split(
            offlinerl_dataset,
            test_size=0.2,
            random_state=random_state,
        )

        for i in tqdm(
            np.arange(len(algorithms)),
            desc="[learn_policies: algorithms]",
            total=len(algorithms),
        ):
            algorithms[i].fit(
                train_episodes,
                eval_episodes=test_episodes,
                **self.fitting_args,
            )

        return algorithms

    def _apply_head(
        self,
        base_policies: List[AlgoBase],
        base_policies_name: List[str],
        policy_wrappers: HeadDict,
        random_state: Optional[int] = None,
    ):
        """Apply policy wrappers to the (deterministic) base policies.

        Parameters
        -------
        base_policies: list of AlgoBase
            List of base (learned) policies.

        base_policies_name: list of str
            List of the name of each base policy.

        policy_wrappers: HeadDict.
            Dictionary containing information about policy wrappers.
            The HeadDict should follow the following format.

            .. code-block:: python

                key: wrapper_name

                value: (BaseHead, params_dict)

            (Example of ``HeadDict``)

            .. code-block:: python

                {
                    "eps_01":  # wrapper_name
                        (
                            EpsilonGreedyHead,  # BaseHead
                            {
                                "epsilon": 0.1,         # params_dict
                                "n_actions": 5,
                            },
                        )
                }

            .. note::

                ``random_state``, ``name``, and ``base_policy`` should be omitted from the ``params_dict``.

            .. seealso::

                :doc:`/documentation/_autosummary/scope_rl.policy.head` described various policy wrappers and their parameters.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        evaluation_policies: list of BaseHead
            List of (stochastic) evaluation policies.

        """
        eval_policies = []
        for i in range(len(base_policies)):
            for head_name in policy_wrappers:
                Head, kwargs = policy_wrappers[head_name]

                eval_policy = Head(
                    base_policy=base_policies[i],
                    name=base_policies_name[i] + f"_{head_name}",
                    random_state=random_state,
                    **kwargs,
                )
                eval_policies.append(eval_policy)

        return eval_policies

    def learn_base_policy(
        self,
        logged_dataset: Union[LoggedDataset, MultipleLoggedDataset],
        algorithms: List[AlgoBase],
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Learn base policy.

        Parameters
        -------
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
                    behavior_policy,
                    dataset_id,
                ]

            .. seealso::

                :class:`scope_rl.dataset.SyntheticDataset` describes the components of :class:`logged_dataset`.

        algorithms: list of AlgoBase
            List of algorithms to fit.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        base_policies: AlgoBase
            List of learned policies.

        """
        if isinstance(logged_dataset, MultipleLoggedDataset):
            if behavior_policy_name is None and dataset_id is None:
                base_policies = defaultdict(list)

                for behavior_policy, n_datasets in logged_dataset.n_datasets.items():
                    for dataset_id_ in range(n_datasets):
                        logged_dataset_ = logged_dataset.get(
                            behavior_policy_name=behavior_policy, dataset_id=dataset_id_
                        )
                        base_policies_ = self._learn_base_policy(
                            logged_dataset=logged_dataset_,
                            algorithms=algorithms,
                            random_state=random_state,
                        )
                        base_policies[behavior_policy].append(base_policies_)

                base_policies = defaultdict_to_dict(base_policies)

            elif behavior_policy_name is None and dataset_id is not None:
                base_policies = {}

                for behavior_policy in logged_dataset.behavior_policy_names:
                    logged_dataset_ = logged_dataset.get(
                        behavior_policy_name=behavior_policy, dataset_id=dataset_id
                    )
                    base_policies_ = self._learn_base_policy(
                        logged_dataset=logged_dataset_,
                        algorithms=algorithms,
                        random_state=random_state,
                    )
                    base_policies[behavior_policy] = base_policies_

            elif behavior_policy_name is not None and dataset_id is None:
                base_policies = []

                for dataset_id_ in range(
                    logged_dataset.n_datasets[behavior_policy_name]
                ):
                    logged_dataset_ = logged_dataset.get(
                        behavior_policy_name=behavior_policy_name,
                        dataset_id=dataset_id_,
                    )
                    base_policies_ = self._learn_base_policy(
                        logged_dataset=logged_dataset_,
                        algorithms=algorithms,
                        random_state=random_state,
                    )
                    base_policies.append(base_policies_)

            else:
                logged_dataset = logged_dataset.get(
                    behavior_policy_name=behavior_policy_name, dataset_id=dataset_id_
                )
                base_policies = self._learn_base_policy(
                    logged_dataset=logged_dataset,
                    algorithms=algorithms,
                    random_state=random_state,
                )

        else:
            base_policies = self._learn_base_policy(
                logged_dataset=logged_dataset,
                algorithms=algorithms,
                random_state=random_state,
            )

        return base_policies

    def apply_head(
        self,
        base_policies: Union[List[AlgoBase], Dict[str, List[AlgoBase]]],
        base_policies_name: List[str],
        policy_wrappers: HeadDict,
        random_state: Optional[int] = None,
    ):
        """Apply policy wrappers to the (deterministic) base policies.

        Parameters
        -------
        base_policies: list of AlgoBase
            List of base (learned) policies.

        base_policies_name: list of str
            List of the name of each base policy.

        policy_wrappers: HeadDict.
            Dictionary containing information about policy wrappers.
            The HeadDict should follow the following format.

            .. code-block:: python

                key: wrapper_name

                value: (BaseHead, params_dict)

            (Example of ``HeadDict``)

            .. code-block:: python

                {
                    "eps_01":  # wrapper_name
                        (
                            EpsilonGreedyHead,  # BaseHead
                            {
                                "epsilon": 0.1,         # params_dict
                                "n_actions": 5,
                            },
                        )
                }

            .. note::

                ``random_state``, ``name``, and ``base_policy`` should be omitted from the ``params_dict``.

            .. seealso::

                :doc:`/documentation/_autosummary/scope_rl.policy.head` described various policy wrappers and their parameters.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        evaluation_policies: list of BaseHead
            List of (stochastic) evaluation policies.

        """
        if isinstance(base_policies, dict):
            evaluation_policies = {}
            for behavior_policy in base_policies.keys():
                if isinstance(base_policies[behavior_policy][0], AlgoBase):
                    if len(base_policies[behavior_policy]) != len(base_policies_name):
                        raise ValueError(
                            "Expected `len(base_policies[behavior_policy]) == len(base_policies_name)`, but found False"
                        )
                    evaluation_policies[behavior_policy] = self._apply_head(
                        base_policies=base_policies[behavior_policy],
                        base_policies_name=base_policies_name,
                        policy_wrappers=policy_wrappers,
                        random_state=random_state,
                    )

                else:
                    evaluation_policies[behavior_policy] = []
                    for dataset_id_ in range(len(base_policies[behavior_policy])):
                        if len(base_policies[behavior_policy][dataset_id_]) != len(
                            base_policies_name
                        ):
                            raise ValueError(
                                "Expected `len(base_policies[behavior_policy][dataset_id_]) == len(base_policies_name)`, but found False"
                            )
                        evaluation_policies_ = self._apply_head(
                            base_policies=base_policies[behavior_policy][dataset_id_],
                            base_policies_name=base_policies_name,
                            policy_wrappers=policy_wrappers,
                            random_state=random_state,
                        )
                        evaluation_policies[behavior_policy].append(
                            evaluation_policies_
                        )

        else:
            if isinstance(base_policies[0], AlgoBase):
                if len(base_policies) != len(base_policies_name):
                    raise ValueError(
                        "Expected `len(base_policies) == len(base_policies_name)`, but found False"
                    )
                evaluation_policies = self._apply_head(
                    base_policies=base_policies,
                    base_policies_name=base_policies_name,
                    policy_wrappers=policy_wrappers,
                    random_state=random_state,
                )
            else:
                evaluation_policies = []
                for dataset_id_ in range(len(base_policies)):
                    evaluation_policies_ = self._apply_head(
                        base_policies=base_policies[dataset_id_],
                        base_policies_name=base_policies_name,
                        policy_wrappers=policy_wrappers,
                        random_state=random_state,
                    )
                    evaluation_policies.append(evaluation_policies_)

        return evaluation_policies

    def obtain_evaluation_policy(
        self,
        logged_dataset: Union[LoggedDataset, MultipleLoggedDataset],
        algorithms: List[AlgoBase],
        algorithms_name: List[str],
        policy_wrappers: HeadDict,
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Obtain evaluation policies given base algorithms and policy wrappers.

        Parameters
        -------
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

                :class:`scope_rl.dataset.SyntheticDataset` describes the components of :class:`logged_dataset`.

        algorithms: list of AlgoBase
            List of algorithms to fit.

        algorithms_name: list of str
            List of the name of each base policy.

        policy_wrappers: HeadDict
            Dictionary containing information about policy wrappers.
            The HeadDict should follow the following format.

            .. code-block:: python

                key: wrapper_name

                value: (BaseHead, params_dict)

            (Example of ``HeadDict``)

            .. code-block:: python

                {
                    "eps_01":  # wrapper_name
                        (
                            EpsilonGreedyHead,  # BaseHead
                            {
                                "epsilon": 0.1,         # params_dict
                                "n_actions": 5,
                            },
                        )
                }

            .. note::

                ``random_state``, ``name``, and ``base_policy`` should be omitted from the ``params_dict``.

            .. seealso::

                :doc:`/documentation/_autosummary/scope_rl.policy.head` described various policy wrappers and their parameters.

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        evaluation_policies: list of BaseHead
            List of (stochastic) evaluation policies.

        """
        if not len(algorithms) != len(algorithms_name):
            raise ValueError(
                "algorithms and alogirthms_name must have the same length, but found False"
            )
        if isinstance(logged_dataset, MultipleLoggedDataset):
            if behavior_policy_name is None and dataset_id is None:
                eval_policies = defaultdict(list)

                for behavior_policy, n_datasets in logged_dataset.n_datasets.items():
                    for dataset_id_ in range(n_datasets):
                        logged_dataset_ = logged_dataset.get(
                            behavior_policy_name=behavior_policy, dataset_id=dataset_id_
                        )
                        base_policies_ = self._learn_base_policy(
                            logged_dataset=logged_dataset_,
                            algorithms=algorithms,
                            random_state=random_state,
                        )
                        eval_policies_ = self._apply_head(
                            base_policies=base_policies_,
                            base_policies_name=algorithms_name,
                            policy_wrappers=policy_wrappers,
                            random_state=random_state,
                        )
                        eval_policies[behavior_policy].append(eval_policies_)

                eval_policies = defaultdict_to_dict(eval_policies)

            elif behavior_policy_name is None and dataset_id is not None:
                eval_policies = {}

                for behavior_policy in logged_dataset.behavior_policy_names:
                    logged_dataset_ = logged_dataset.get(
                        behavior_policy_name=behavior_policy, dataset_id=dataset_id
                    )
                    base_policies_ = self._learn_base_policy(
                        logged_dataset=logged_dataset_,
                        algorithms=algorithms,
                        random_state=random_state,
                    )
                    eval_policies_ = self._apply_head(
                        base_policies=base_policies_,
                        base_policies_name=algorithms_name,
                        policy_wrappers=policy_wrappers,
                        random_state=random_state,
                    )
                    eval_policies[behavior_policy] = eval_policies_

            elif behavior_policy_name is not None and dataset_id is None:
                eval_policies = []

                for dataset_id_ in range(
                    logged_dataset.n_datasets[behavior_policy_name]
                ):
                    logged_dataset_ = logged_dataset.get(
                        behavior_policy_name=behavior_policy_name,
                        dataset_id=dataset_id_,
                    )
                    base_policies_ = self._learn_base_policy(
                        logged_dataset=logged_dataset_,
                        algorithms=algorithms,
                        random_state=random_state,
                    )
                    eval_policies_ = self._apply_head(
                        base_policies=base_policies_,
                        base_policies_name=algorithms_name,
                        policy_wrappers=policy_wrappers,
                        random_state=random_state,
                    )
                    eval_policies.append(eval_policies_)

            else:
                logged_dataset = logged_dataset.get(
                    behavior_policy_name=behavior_policy_name, dataset_id=dataset_id_
                )
                base_policies = self._learn_base_policy(
                    logged_dataset=logged_dataset,
                    algorithms=algorithms,
                    random_state=random_state,
                )
                eval_policies = self._apply_head(
                    base_policies=base_policies,
                    base_policies_name=algorithms_name,
                    policy_wrappers=policy_wrappers,
                    random_state=random_state,
                )

        else:
            base_policies = self._learn_base_policy(
                logged_dataset=logged_dataset,
                algorithms=algorithms,
                random_state=random_state,
            )
            eval_policies = self._apply_head(
                base_policies=base_policies,
                base_policies_name=algorithms_name,
                policy_wrappers=policy_wrappers,
                random_state=random_state,
            )

        return eval_policies
