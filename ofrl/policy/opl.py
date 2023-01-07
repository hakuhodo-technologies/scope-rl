"""Meta class to handle Off-Policy Learning (OPL)."""
from dataclasses import dataclass
from typing import Union, Optional, Any, Dict, List, Tuple
from tqdm.auto import tqdm

import numpy as np

from d3rlpy.algos import AlgoBase
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split

from .head import BaseHead
from ..utils import MultipleLoggedDataset
from ..types import LoggedDataset


HeadDict = Dict[str, Tuple[BaseHead, Dict[str, Any]]]


@dataclass
class OffPolicyLearning:
    """Class to handle OPL by multiple algorithms simultaneously. (applicable to both discrete/continuous action cases)

    Imported as: :class:`ofrl.policy.OffPolicyLearning`

    Parameters
    -------
    fitting_args: dict, default=None
        Arguments of fitting function to learn model.

    Examples
    ----------

    Preparation:

    .. code-block:: python

        # import necessary module from OFRL
        from ofrl.dataset import SyntheticDataset
        from ofrl.policy import OffPolicyLearning
        from ofrl.policy import DiscreteEpsilonGreedyHead, DiscreteSoftmaxHead

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
                DiscreteEpsilonGreedyHead,
                {
                    "epsilon": 0.1,
                    "n_actions": env.action_space.n,
                }
            ),
            "eps_03": (
                DiscreteEpsilonGreedyHead,
                {
                    "epsilon": 0.3,
                    "n_actions": env.action_space.n,
                }
            ),
            "softmax": (
                DiscreteSoftmaxHead,
                {
                    "tau": 1.0,
                    "n_actions": env.action_space.n,
                }
            ),
        }

        # off-policy learning
        opl = OffPolicyLearning()
        eval_policies = opl.obtain_evaluation_policy(
            algorithms=algorithms,
            algorithms_name=algorithms_name,
            policy_wrappers=policy_wrappers,
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

                :class:`ofrl.dataset.SyntheticDataset` describes the components of :class:`logged_dataset`.

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
                            DiscreteEpsilonGreedyHead,  # BaseHead
                            {
                                "epsilon": 0.1,         # params_dict
                                "n_actions": 5,
                            },
                        )
                }

            .. note::

                ``random_state``, ``name``, and ``base_policy`` should be omitted from the ``params_dict``.

            .. seealso::

                :doc:`/documentation/_autosummary/ofrl.policy.head` described various policy wrappers and their parameters.

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
        algorithms: Union[List[AlgoBase], List[List[AlgoBase]]],
        dataset_id: Optional[Union[int, str]] = None,
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

                :class:`ofrl.dataset.SyntheticDataset` describes the components of :class:`logged_dataset`.

        algorithms: list of AlgoBase
            List of algorithms to fit.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the algorithms are trained on multiple logged datasets.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        base_policies: List of AlgoBase
            List of learned policies.

        """
        if isinstance(logged_dataset, MultipleLoggedDataset):
            if dataset_id is None:
                if isinstance(algorithms[0], AlgoBase):
                    algorithms = [algorithms for _ in range(len(logged_dataset))]

                base_policies = []
                for i in tqdm(
                    np.arange(len(logged_dataset)),
                    desc="[learn_policies: logged_datasets]",
                    total=len(logged_dataset),
                ):
                    logged_dataset_ = logged_dataset.get(i)
                    base_policies_ = self._learn_base_policy(
                        logged_dataset=logged_dataset_,
                        algorithms=algorithms[i],
                        random_state=random_state,
                    )
                    base_policies.append(base_policies_)

            else:
                logged_dataset = logged_dataset.get(dataset_id)
                base_policies = self._learn_base_policy(
                    logged_dataset=logged_dataset_,
                    algorithms=algorithms,
                    random_state=random_state,
                )

        else:
            base_policies = self._learn_base_policy(
                logged_dataset=logged_dataset,
                base_policies=algorithms,
                random_state=random_state,
            )

        return base_policies

    def apply_head(
        self,
        base_policies: Union[List[AlgoBase], List[List[AlgoBase]]],
        base_policies_name: Union[List[str], List[List[str]]],
        policy_wrappers: Union[HeadDict, List[HeadDict]],
        random_state: Optional[int] = None,
    ):
        """Apply policy wrappers to the (deterministic) base policies.

        Parameters
        -------
        base_policies: list of AlgoBase
            List of base (learned) policies.

        base_policies_name: list of str
            List of the name of each base policy.

        policy_wrappers: HeadDict or list of HeadDict.
            List of dictionary containing information about policy wrappers.
            The HeadDict should follow the following format.

            .. code-block:: python

                key: wrapper_name

                value: (BaseHead, params_dict)

            (Example of ``HeadDict``)

            .. code-block:: python

                {
                    "eps_01":  # wrapper_name
                        (
                            DiscreteEpsilonGreedyHead,  # BaseHead
                            {
                                "epsilon": 0.1,         # params_dict
                                "n_actions": 5,
                            },
                        )
                }

            .. note::

                ``random_state``, ``name``, and ``base_policy`` should be omitted from the ``params_dict``.

            .. seealso::

                :doc:`/documentation/_autosummary/ofrl.policy.head` described various policy wrappers and their parameters.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        evaluation_policies: list of BaseHead
            List of (stochastic) evaluation policies.

        """
        if isinstance(base_policies[0], list):
            if not isinstance(base_policies_name[0], list):
                base_policies_name = [
                    base_policies_name for _ in range(len(base_policies))
                ]
            if not isinstance(policy_wrappers, list):
                policy_wrappers = [policy_wrappers for _ in range(len(base_policies))]

                eval_policies = []
                for i in range(len(base_policies)):
                    eval_policies_ = self._apply_head(
                        base_policies=base_policies[i],
                        base_policies_name=base_policies_name[i],
                        policy_wrappers=policy_wrappers[i],
                        random_state=random_state,
                    )
                    eval_policies.append(eval_policies_)

        else:
            eval_policies_ = self._apply_head(
                base_policies=base_policies,
                base_policies_name=base_policies_name,
                policy_wrappers=policy_wrappers,
                random_state=random_state,
            )

        return eval_policies

    def obtain_evaluation_policy(
        self,
        logged_dataset: Union[LoggedDataset, MultipleLoggedDataset],
        algorithms: Union[List[AlgoBase], List[List[AlgoBase]]],
        algorithms_name: Union[List[str], List[List[str]]],
        policy_wrappers: Union[HeadDict, List[HeadDict]],
        dataset_id: Optional[Union[int, str]] = None,
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

                :class:`ofrl.dataset.SyntheticDataset` describes the components of :class:`logged_dataset`.

        algorithms: list of AlgoBase
            List of algorithms to fit.

        algorithms_name: list of str
            List of the name of each base policy.

        policy_wrappers: HeadDict or list of HeadDict.
            List of dictionary containing information about policy wrappers.
            The HeadDict should follow the following format.

            .. code-block:: python

                key: wrapper_name

                value: (BaseHead, params_dict)

            (Example of ``HeadDict``)

            .. code-block:: python

                {
                    "eps_01":  # wrapper_name
                        (
                            DiscreteEpsilonGreedyHead,  # BaseHead
                            {
                                "epsilon": 0.1,         # params_dict
                                "n_actions": 5,
                            },
                        )
                }

            .. note::

                ``random_state``, ``name``, and ``base_policy`` should be omitted from the ``params_dict``.

            .. seealso::

                :doc:`/documentation/_autosummary/ofrl.policy.head` described various policy wrappers and their parameters.

        dataset_id: int or str, default=None
            Id (or name) of the logged dataset.
            If `None`, the algorithms are trained on multiple logged datasets.

        random_state: int, default=None (>= 0)
            Random state.

        Returns
        -------
        evaluation_policies: list of BaseHead
            List of (stochastic) evaluation policies.

        """
        base_policies = self.learn_base_policy(
            logged_dataset=logged_dataset,
            algorithms=algorithms,
            dataset_id=dataset_id,
            random_state=random_state,
        )
        eval_policies = self.apply_head(
            base_policies=base_policies,
            base_policies_name=algorithms_name,
            policy_wrappers=policy_wrappers,
            random_state=random_state,
        )
        return eval_policies
