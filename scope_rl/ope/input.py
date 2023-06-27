# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Meta class to create input for Off-Policy Evaluation (OPE)."""
from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Optional, Any, Union

from collections import defaultdict
from tqdm.auto import tqdm

import torch
import numpy as np
from sklearn.utils import check_scalar, check_random_state

import gym
from gym.spaces import Discrete

from d3rlpy.dataset import MDPDataset
from d3rlpy.ope import DiscreteFQE
from d3rlpy.ope import FQE as ContinuousFQE
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.preprocessing import Scaler, ActionScaler

from .weight_value_learning import (
    DiscreteDiceStateActionWightValueLearning,
    DiscreteDiceStateWightValueLearning,
    ContinuousDiceStateActionWightValueLearning,
    ContinuousDiceStateWightValueLearning,
    DiscreteMinimaxStateActionValueLearning,
    DiscreteMinimaxStateValueLearning,
    ContinuousMinimaxStateActionValueLearning,
    ContinuousMinimaxStateValueLearning,
    DiscreteMinimaxStateActionWeightLearning,
    DiscreteMinimaxStateWeightLearning,
    ContinuousMinimaxStateActionWeightLearning,
    ContinuousMinimaxStateWeightLearning,
)
from .weight_value_learning.function import (
    DiscreteQFunction,
    ContinuousQFunction,
    VFunction,
    DiscreteStateActionWeightFunction,
    ContinuousStateActionWeightFunction,
    StateWeightFunction,
)
from .online import rollout_policy_online
from ..policy.head import BaseHead
from ..utils import (
    MultipleLoggedDataset,
    MultipleInputDict,
    defaultdict_to_dict,
    check_logged_dataset,
    check_array,
)
from ..types import LoggedDataset, OPEInputDict


@dataclass
class CreateOPEInput:
    """Class to prepare OPE inputs.

    Imported as: :class:`scope_rl.ope.CreateOPEInput`

    Parameters
    -------
    env: gym.Env, default=None
        Reinforcement learning (RL) environment.

    model_args: dict of dict, default=None
        Arguments of the models.

        .. code-block:: python

            key: [
                "fqe",
                "state_action_dual",
                "state_action_value",
                "state_action_weight",
                "state_dual",
                "state_value",
                "state_weight",
                "hidden_dim",  # hidden dim of value/weight function, except FQE
            ]

        .. note::

            Please specify :class:`scaler` and :class:`action_scaler` when calling :class:`.obtain_whole_inputs()`
            (, as we will overwrite those specified by :class:`model_args[model]["scaler/action_scaler"]`).

        .. seealso::

            The followings describe the parameters of each model.

            * (external) `d3rlpy's documentation about FQE <https://d3rlpy.readthedocs.io/en/latest/references/off_policy_evaluation.html>`_
            * (API reference) :class:`scope_rl.ope.weight_value_learning`

    gamma: float, default=1.0
        Discount factor. The value should be within (0, 1].

    bandwidth: float, default=1.0 (> 0)
        Bandwidth hyperparameter of the kernel.

    state_scaler: d3rlpy.preprocessing.Scaler, default=None
        Scaling factor of state.

    action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
        Scaling factor of action. Only applicable in the continuous action case.

    device: Optional[str] = None
        Specifies device used for torch.

    Examples
    ----------
    Preparation:

    .. code-block:: python

        # import necessary module from SCOPE-RL
        from scope_rl.dataset import SyntheticDataset
        from scope_rl.policy import EpsilonGreedyHead
        from scope_rl.ope import CreateOPEInput
        from scope_rl.ope import OffPolicyEvaluation as OPE
        from scope_rl.ope.discrete import TrajectoryWiseImportanceSampling as TIS
        from scope_rl.ope.discrete import PerDecisionImportanceSampling as PDIS

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

    **Create Input**:

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
            require_value_prediction=True,
            n_trajectories_on_policy_evaluation=100,
            random_state=12345,
        )

    **Output**:

    .. code-block:: python

        >>> input_dict

        {'ddqn':
            {'evaluation_policy_action_dist':
                array([[0., 0., 0., ..., 0., 1., 0.],
                      [0., 0., 0., ..., 0., 0., 0.],
                      [0., 0., 0., ..., 0., 1., 0.],
                      ...,
                      [0., 0., 0., ..., 0., 0., 0.],
                      [0., 0., 0., ..., 0., 0., 0.],
                      [0., 0., 0., ..., 0., 1., 0.]]),
            'evaluation_policy_action': None,
            'state_action_value_prediction':
                array([[11.64699173, 10.1278677 , 10.09877205, ..., 10.16476822, 15.13939476,  8.95065594],
                      [10.42242146,  7.73790789,  7.27790451, ...,  3.51157165, 12.0761919 ,  3.75301909],
                      [ 7.22864819,  6.88499546,  5.68951464, ...,  6.10659647, 7.05469513,  4.81715965],
                      ...,
                      [ 7.28475332,  3.91264176,  4.6845212 , ..., -0.02834684, 7.94454432,  2.59267783],
                      [13.44723797,  3.08360171,  5.99188185, ..., -2.16886044, 7.13434792,  5.72265959],
                      [ 2.27913332,  3.07881427,  1.8636421 , ...,  3.37803316, 3.20135021,  2.68845224]]),
            'initial_state_value_prediction': array([15.13939476, 14.83423805, 13.82990742, ..., 15.49367523, 15.49053097, 14.88922691]),
            'state_action_marginal_importance_weight': None,
            'state_marginal_importance_weight': None,
            'on_policy_policy_value': array([ 8., 10.,  9., ...,  13., 18.,  4.]),
            'gamma': 1.0,
            'behavior_policy': 'ddqn_epsilon_0.3',
            'evaluation_policy': 'ddqn',
            'dataset_id': 0},},
        'random':
            {'evaluation_policy_action_dist':
                array([[0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1],
                      ...,
                      [0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1]]),
            'evaluation_policy_action': None,
            'state_action_value_prediction':
                array([[10.63342857, 10.61063576, 11.16767025, ..., 15.32427979, 15.08568764, 10.50707436],
                      [ 4.02995491,  4.80947208,  7.07302999, ...,  9.928442  , 10.78198528,  9.04977417],
                      [ 6.21145582,  6.08772421,  6.5972681 , ...,  9.68579388, 7.2353406 ,  6.17404699],
                      ...,
                      [ 1.2350018 ,  1.37531543,  3.48139453, ...,  3.44862366, 5.41990328, -0.20314722],
                      [ 0.81208032, -0.28935188,  2.62608957, ...,  6.6619091 , -2.18710518, -2.34665537],
                      [ 2.36533523,  2.24474525,  2.31729817, ...,  4.7845993 , 2.83752441,  3.00596046]]),
            'initial_state_value_prediction': array([12.5472518 , 12.56364899, 12.30248432, ..., 12.62372198, 12.6544138 , 12.54314356]),
            'state_action_marginal_importance_weight': None,
            'state_marginal_importance_weight': None,
            'on_policy_policy_value': array([ 9.,  7.,  4., ..., 15.,  8.,  5.]),
            'gamma': 1.0,
            'behavior_policy': 'ddqn_epsilon_0.3',
            'evaluation_policy': 'random',
            'dataset_id': 0}}

    .. seealso::

        * :doc:`Quickstart </documentation/quickstart>`

    """

    env: Optional[gym.Env] = None
    model_args: Optional[Dict[str, Any]] = None
    gamma: float = 1.0
    bandwidth: float = 1.0
    state_scaler: Optional[Scaler] = None
    action_scaler: Optional[ActionScaler] = None
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model_args is None:
            self.model_args = {
                "fqe": None,
                "state_action_dual": None,
                "state_action_value": None,
                "state_action_weight": None,
                "state_dual": None,
                "state_value": None,
                "state_weight": None,
                "hidden_dim": None,
            }

        for key in [
            "fqe",
            "state_action_dual",
            "state_action_value",
            "state_action_weight",
            "state_dual",
            "state_value",
            "state_weight",
            "hidden_dim",
        ]:
            if key not in self.model_args:
                self.model_args[key] = None

        if self.model_args["hidden_dim"] is None:
            self.model_args["hidden_dim"] = 100

        if self.model_args["fqe"] is None:
            self.model_args["fqe"] = {
                "encoder_factory": VectorEncoderFactory(
                    hidden_units=[self.model_args["hidden_dim"]]
                ),
                "q_func_factory": MeanQFunctionFactory(),
                "learning_rate": 1e-4,
                "use_gpu": torch.cuda.is_available(),
            }

        if self.model_args["state_action_dual"] is None:
            self.model_args["state_action_dual"] = {}

        if self.model_args["state_action_value"] is None:
            self.model_args["state_action_value"] = {}

        if self.model_args["state_action_weight"] is None:
            self.model_args["state_action_weight"] = {}

        if self.model_args["state_dual"] is None:
            self.model_args["state_dual"] = {}

        if self.model_args["state_value"] is None:
            self.model_args["state_value"] = {}

        if self.model_args["state_weight"] is None:
            self.model_args["state_weight"] = {}

        check_scalar(
            self.gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0
        )
        check_scalar(self.bandwidth, name="bandwidth", target_type=float, min_val=0.0)
        if self.state_scaler is not None:
            if not isinstance(self.state_scaler, Scaler):
                raise ValueError(
                    "state_scaler must be an instance of d3rlpy.preprocessing.Scaler, but found False"
                )
        if self.action_scaler is not None:
            if not isinstance(self.action_scaler, ActionScaler):
                raise ValueError(
                    "action_scaler must be an instance of d3rlpy.preprocessing.ActionScaler, but found False"
                )

        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _register_logged_dataset(self, logged_dataset: LoggedDataset):
        self.fqe = {}
        self.state_action_dual_function = {}
        self.state_action_value_function = {}
        self.state_action_weight_function = {}
        self.state_dual_function = {}
        self.state_value_function = {}
        self.state_weight_function = {}
        self.initial_state_dict = {}

        self.logged_dataset = logged_dataset
        self.behavior_policy_name = logged_dataset["behavior_policy"]
        self.dataset_id = logged_dataset["dataset_id"]

        self.action_type = self.logged_dataset["action_type"]
        self.n_actions = self.logged_dataset["n_actions"]
        self.action_dim = self.logged_dataset["action_dim"]
        self.state_dim = self.logged_dataset["state_dim"]
        self.n_trajectories = self.logged_dataset["n_trajectories"]
        self.step_per_trajectory = self.logged_dataset["step_per_trajectory"]
        self.n_samples = self.n_trajectories * self.step_per_trajectory

        self.state = self.logged_dataset["state"]
        self.action = self.logged_dataset["action"]
        self.reward = self.logged_dataset["reward"]
        self.pscore = self.logged_dataset["pscore"]
        self.done = self.logged_dataset["done"]
        self.terminal = self.logged_dataset["terminal"]

        self.state_2d = self.state.reshape(
            (-1, self.step_per_trajectory, self.state_dim)
        )
        self.reward_2d = self.reward.reshape((-1, self.step_per_trajectory))
        self.pscore_2d = self.pscore.reshape((-1, self.step_per_trajectory))
        self.done_2d = self.done.reshape((-1, self.step_per_trajectory))
        self.terminal_2d = self.terminal.reshape((-1, self.step_per_trajectory))
        if self.action_type == "discrete":
            self.action_2d = self.action.reshape((-1, self.step_per_trajectory))
        else:
            self.action_2d = self.action.reshape(
                (-1, self.step_per_trajectory, self.action_dim)
            )

        self.mdp_dataset = MDPDataset(
            observations=self.state,
            actions=self.action,
            rewards=self.reward,
            terminals=self.done,
            episode_terminals=self.terminal,
            discrete_action=(self.action_type == "discrete"),
        )

    def build_and_fit_FQE(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_steps: int = 10000,
    ) -> None:
        """Perform Fitted Q Evaluation (FQE).

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)

        if evaluation_policy.name in self.fqe:
            pass

        else:
            self.fqe[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.fqe[evaluation_policy.name].append(
                        DiscreteFQE(
                            algo=evaluation_policy,
                            scaler=self.state_scaler,
                            gamma=self.gamma,
                            **self.model_args["fqe"],
                        )
                    )
                else:
                    self.fqe[evaluation_policy.name].append(
                        ContinuousFQE(
                            algo=evaluation_policy,
                            scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            gamma=self.gamma,
                            **self.model_args["fqe"],
                        )
                    )

            if k_fold == 1:
                self.fqe[evaluation_policy.name][0].fit(
                    self.mdp_dataset.episodes,
                    eval_episodes=self.mdp_dataset.episodes,
                    n_steps=n_steps,
                    scorers={},
                )
            else:
                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [(self.n_trajectories + k) // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)
                idx = idx.cumsum()

                for k in tqdm(
                    np.arange(k_fold),
                    desc="[cross-fitting]",
                    total=k_fold,
                ):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    subset_idx_ = np.setdiff1d(all_idx, idx_)

                    if self.action_type == "discrete":
                        action_ = self.action_2d[subset_idx_].flatten()
                    else:
                        action_ = self.action_2d[subset_idx_].reshape(
                            (-1, self.action_dim)
                        )

                    mdp_dataset_ = MDPDataset(
                        observations=self.state_2d[subset_idx_].reshape(
                            (-1, self.state_dim)
                        ),
                        actions=action_,
                        rewards=self.reward_2d[subset_idx_].flatten(),
                        terminals=self.done_2d[subset_idx_].flatten(),
                        episode_terminals=self.terminal_2d[subset_idx_].flatten(),
                        discrete_action=(self.action_type == "discrete"),
                    )
                    self.fqe[evaluation_policy.name][k].fit(
                        mdp_dataset_.episodes,
                        eval_episodes=mdp_dataset_.episodes,
                        n_steps=n_steps,
                        scorers={},
                    )

    def build_and_fit_state_action_dual_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_steps: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Perform Augmented Lagrangian Method (ALM) to estimate the state-action value weight function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        random_state: int, default=None (>= 0)
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)

        if evaluation_policy.name in self.state_action_dual_function:
            pass

        else:
            self.state_action_dual_function[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.state_action_dual_function[evaluation_policy.name].append(
                        DiscreteDiceStateActionWightValueLearning(
                            q_function=DiscreteQFunction(
                                n_actions=self.n_actions,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                device=self.device,
                            ),
                            w_function=DiscreteStateActionWeightFunction(
                                n_actions=self.n_actions,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                enable_gradient_reversal=True,
                                device=self.device,
                            ),
                            state_scaler=self.state_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_action_dual"],
                        )
                    )
                else:
                    self.state_action_dual_function[evaluation_policy.name].append(
                        ContinuousDiceStateActionWightValueLearning(
                            q_function=ContinuousQFunction(
                                action_dim=self.action_dim,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            w_function=ContinuousStateActionWeightFunction(
                                action_dim=self.action_dim,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                enable_gradient_reversal=True,
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_action_dual"],
                        )
                    )

            if self.action_type == "discrete":
                evaluation_policy_action_dist = (
                    self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    )
                )
            else:
                evaluation_policy_action = self.obtain_evaluation_policy_action(
                    evaluation_policy=evaluation_policy,
                )

            if k_fold == 1:
                if self.action_type == "discrete":
                    self.state_action_dual_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        evaluation_policy_action_dist=evaluation_policy_action_dist,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
                else:
                    self.state_action_dual_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        evaluation_policy_action=evaluation_policy_action,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
            else:
                if self.action_type == "discrete":
                    evaluation_policy_action_dist = (
                        evaluation_policy_action_dist.reshape(
                            (-1, self.step_per_trajectory, self.n_actions)
                        )
                    )
                else:
                    evaluation_policy_action = evaluation_policy_action.reshape(
                        (-1, self.step_per_trajectory, self.action_dim)
                    )

                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [(self.n_trajectories + k) // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)
                idx = idx.cumsum()

                for k in tqdm(
                    np.arange(k_fold),
                    desc="[cross-fitting]",
                    total=k_fold,
                ):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    subset_idx_ = np.setdiff1d(all_idx, idx_)

                    if self.action_type == "discrete":
                        action_ = self.action_2d[subset_idx_].flatten()
                        evaluation_policy_action_dist_ = evaluation_policy_action_dist[
                            subset_idx_
                        ].reshape((-1, self.n_actions))
                    else:
                        action_ = self.action_2d[subset_idx_].reshape(
                            (-1, self.action_dim)
                        )
                        evaluation_policy_action_ = evaluation_policy_action[
                            subset_idx_
                        ].reshape((-1, self.action_dim))

                    if self.action_type == "discrete":
                        self.state_action_dual_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            evaluation_policy_action_dist=evaluation_policy_action_dist_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )
                    else:
                        self.state_action_dual_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            evaluation_policy_action=evaluation_policy_action_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )

    def build_and_fit_state_action_value_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_steps: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Perform Minimax Q Learning (MQL) to estimate the state-action value function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        random_state: int, default=None (>= 0)
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)

        if evaluation_policy.name in self.state_action_value_function:
            pass

        else:
            self.state_action_value_function[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.state_action_value_function[evaluation_policy.name].append(
                        DiscreteMinimaxStateActionValueLearning(
                            q_function=DiscreteQFunction(
                                n_actions=self.n_actions,
                                state_dim=self.state_dim,
                                device=self.device,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_action_value"],
                        )
                    )
                else:
                    self.state_action_value_function[evaluation_policy.name].append(
                        ContinuousMinimaxStateActionValueLearning(
                            q_function=ContinuousQFunction(
                                action_dim=self.action_dim,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_action_value"],
                        )
                    )

            if self.action_type == "discrete":
                evaluation_policy_action_dist = (
                    self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    )
                )
            else:
                evaluation_policy_action = self.obtain_evaluation_policy_action(
                    evaluation_policy=evaluation_policy,
                )

            if k_fold == 1:
                if self.action_type == "discrete":
                    self.state_action_value_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action_dist=evaluation_policy_action_dist,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
                else:
                    self.state_action_value_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action=evaluation_policy_action,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
            else:
                if self.action_type == "discrete":
                    evaluation_policy_action_dist = (
                        evaluation_policy_action_dist.reshape(
                            (-1, self.step_per_trajectory, self.n_actions)
                        )
                    )
                else:
                    evaluation_policy_action = evaluation_policy_action.reshape(
                        (-1, self.step_per_trajectory, self.action_dim)
                    )

                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [(self.n_trajectories + k) // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)
                idx = idx.cumsum()

                for k in tqdm(
                    np.arange(k_fold),
                    desc="[cross-fitting]",
                    total=k_fold,
                ):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    subset_idx_ = np.setdiff1d(all_idx, idx_)

                    if self.action_type == "discrete":
                        action_ = self.action_2d[subset_idx_].flatten()
                        evaluation_policy_action_dist_ = evaluation_policy_action_dist[
                            subset_idx_
                        ].reshape((-1, self.n_actions))
                    else:
                        action_ = self.action_2d[subset_idx_].reshape(
                            (-1, self.action_dim)
                        )
                        evaluation_policy_action_ = evaluation_policy_action[
                            subset_idx_
                        ].reshape((-1, self.action_dim))

                    if self.action_type == "discrete":
                        self.state_action_value_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].flatten(),
                            evaluation_policy_action_dist=evaluation_policy_action_dist_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )
                    else:
                        self.state_action_value_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].reshape(
                                (-1, self.action_dim)
                            ),
                            evaluation_policy_action=evaluation_policy_action_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )

    def build_and_fit_state_action_weight_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_steps: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Perform Minimax Weight Learning (MWL) to estimate the state-action weight function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        random_state: int, default=None (>= 0)
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)

        if evaluation_policy.name in self.state_action_weight_function:
            pass

        else:
            self.state_action_weight_function[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.state_action_weight_function[evaluation_policy.name].append(
                        DiscreteMinimaxStateActionWeightLearning(
                            w_function=DiscreteStateActionWeightFunction(
                                n_actions=self.n_actions,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                enable_gradient_reversal=True,
                                device=self.device,
                            ),
                            state_scaler=self.state_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_action_weight"],
                        )
                    )
                else:
                    self.state_action_weight_function[evaluation_policy.name].append(
                        ContinuousMinimaxStateActionWeightLearning(
                            w_function=ContinuousStateActionWeightFunction(
                                action_dim=self.action_dim,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                enable_gradient_reversal=True,
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_action_weight"],
                        )
                    )

            if self.action_type == "discrete":
                evaluation_policy_action_dist = (
                    self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    )
                )
            else:
                evaluation_policy_action = self.obtain_evaluation_policy_action(
                    evaluation_policy=evaluation_policy,
                )

            if k_fold == 1:
                if self.action_type == "discrete":
                    self.state_action_weight_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        evaluation_policy_action_dist=evaluation_policy_action_dist,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
                else:
                    self.state_action_weight_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        evaluation_policy_action=evaluation_policy_action,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
            else:
                if self.action_type == "discrete":
                    evaluation_policy_action_dist = (
                        evaluation_policy_action_dist.reshape(
                            (-1, self.step_per_trajectory, self.n_actions)
                        )
                    )
                else:
                    evaluation_policy_action = evaluation_policy_action.reshape(
                        (-1, self.step_per_trajectory, self.action_dim)
                    )

                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [(self.n_trajectories + k) // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)
                idx = idx.cumsum()

                for k in tqdm(
                    np.arange(k_fold),
                    desc="[cross-fitting]",
                    total=k_fold,
                ):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    subset_idx_ = np.setdiff1d(all_idx, idx_)

                    if self.action_type == "discrete":
                        action_ = self.action_2d[subset_idx_].flatten()
                        evaluation_policy_action_dist_ = evaluation_policy_action_dist[
                            subset_idx_
                        ].reshape((-1, self.n_actions))
                    else:
                        action_ = self.action_2d[subset_idx_].reshape(
                            (-1, self.action_dim)
                        )
                        evaluation_policy_action_ = evaluation_policy_action[
                            subset_idx_
                        ].reshape((-1, self.action_dim))

                    if self.action_type == "discrete":
                        self.state_action_weight_function[evaluation_policy.name][
                            k
                        ].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            evaluation_policy_action_dist=evaluation_policy_action_dist_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )
                    else:
                        self.state_action_weight_function[evaluation_policy.name][
                            k
                        ].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            evaluation_policy_action=evaluation_policy_action_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )

    def build_and_fit_state_dual_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_steps: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Perform Augmented Lagrangian Method (ALM) to estimate the state value weight function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        random_state: int, default=None (>= 0)
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)

        if evaluation_policy.name in self.state_dual_function:
            pass

        else:
            self.state_dual_function[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.state_dual_function[evaluation_policy.name].append(
                        DiscreteDiceStateWightValueLearning(
                            v_function=VFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            w_function=StateWeightFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                enable_gradient_reversal=True,
                            ),
                            state_scaler=self.state_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_dual"],
                        )
                    )
                else:
                    self.state_dual_function[evaluation_policy.name].append(
                        ContinuousDiceStateWightValueLearning(
                            v_function=VFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            w_function=StateWeightFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                enable_gradient_reversal=True,
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_dual"],
                        )
                    )

            if self.action_type == "discrete":
                evaluation_policy_action_dist = (
                    self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    )
                )
            else:
                evaluation_policy_action = self.obtain_evaluation_policy_action(
                    evaluation_policy=evaluation_policy,
                )

            if k_fold == 1:
                if self.action_type == "discrete":
                    self.state_dual_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action_dist=evaluation_policy_action_dist,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
                else:
                    self.state_dual_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action=evaluation_policy_action,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
            else:
                if self.action_type == "discrete":
                    evaluation_policy_action_dist = (
                        evaluation_policy_action_dist.reshape(
                            (-1, self.step_per_trajectory, self.n_actions)
                        )
                    )
                else:
                    evaluation_policy_action = evaluation_policy_action.reshape(
                        (-1, self.step_per_trajectory, self.action_dim)
                    )

                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [(self.n_trajectories + k) // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)
                idx = idx.cumsum()

                for k in tqdm(
                    np.arange(k_fold),
                    desc="[cross-fitting]",
                    total=k_fold,
                ):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    subset_idx_ = np.setdiff1d(all_idx, idx_)

                    if self.action_type == "discrete":
                        action_ = self.action_2d[subset_idx_].flatten()
                        evaluation_policy_action_dist_ = evaluation_policy_action_dist[
                            subset_idx_
                        ].reshape((-1, self.n_actions))
                    else:
                        action_ = self.action_2d[subset_idx_].reshape(
                            (-1, self.action_dim)
                        )
                        evaluation_policy_action_ = evaluation_policy_action[
                            subset_idx_
                        ].reshape((-1, self.action_dim))

                    if self.action_type == "discrete":
                        self.state_dual_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].flatten(),
                            evaluation_policy_action_dist=evaluation_policy_action_dist_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )
                    else:
                        self.state_dual_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].reshape(
                                (-1, self.action_dim)
                            ),
                            evaluation_policy_action=evaluation_policy_action_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )

    def build_and_fit_state_value_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_steps: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Perform Minimax V Learning (MVL) to estimate the state value function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        random_state: int, default=None (>= 0)
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)

        if evaluation_policy.name in self.state_value_function:
            pass

        else:
            self.state_value_function[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.state_value_function[evaluation_policy.name].append(
                        DiscreteMinimaxStateValueLearning(
                            v_function=VFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_value"],
                        )
                    )
                else:
                    self.state_value_function[evaluation_policy.name].append(
                        ContinuousMinimaxStateValueLearning(
                            v_function=VFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_value"],
                        )
                    )

            if self.action_type == "discrete":
                evaluation_policy_action_dist = (
                    self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    )
                )
            else:
                evaluation_policy_action = self.obtain_evaluation_policy_action(
                    evaluation_policy=evaluation_policy,
                )

            if k_fold == 1:
                if self.action_type == "discrete":
                    self.state_value_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action_dist=evaluation_policy_action_dist,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
                else:
                    self.state_value_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action=evaluation_policy_action,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
            else:
                if self.action_type == "discrete":
                    evaluation_policy_action_dist = (
                        evaluation_policy_action_dist.reshape(
                            (-1, self.step_per_trajectory, self.n_actions)
                        )
                    )
                else:
                    evaluation_policy_action = evaluation_policy_action.reshape(
                        (-1, self.step_per_trajectory, self.action_dim)
                    )

                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [(self.n_trajectories + k) // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)
                idx = idx.cumsum()

                for k in tqdm(
                    np.arange(k_fold),
                    desc="[cross-fitting]",
                    total=k_fold,
                ):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    subset_idx_ = np.setdiff1d(all_idx, idx_)

                    if self.action_type == "discrete":
                        action_ = self.action_2d[subset_idx_].flatten()
                        evaluation_policy_action_dist_ = evaluation_policy_action_dist[
                            subset_idx_
                        ].reshape((-1, self.n_actions))
                    else:
                        action_ = self.action_2d[subset_idx_].reshape(
                            (-1, self.action_dim)
                        )
                        evaluation_policy_action_ = evaluation_policy_action[
                            subset_idx_
                        ].reshape((-1, self.action_dim))

                    if self.action_type == "discrete":
                        self.state_value_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].flatten(),
                            evaluation_policy_action_dist=evaluation_policy_action_dist_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )
                    else:
                        self.state_value_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].reshape(
                                (-1, self.action_dim)
                            ),
                            evaluation_policy_action=evaluation_policy_action_,
                            n_steps=n_steps,
                            random_state=random_state,
                        )

    def build_and_fit_state_weight_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_steps: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Perform Minimax Weight Learning (MWL) to estimate state weight function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps.

        random_state: int, default=None (>= 0)
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)

        if evaluation_policy.name in self.state_weight_function:
            pass

        else:
            self.state_weight_function[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.state_weight_function[evaluation_policy.name].append(
                        DiscreteMinimaxStateWeightLearning(
                            w_function=StateWeightFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                enable_gradient_reversal=True,
                            ),
                            state_scaler=self.state_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_weight"],
                        )
                    )
                else:
                    self.state_weight_function[evaluation_policy.name].append(
                        ContinuousMinimaxStateWeightLearning(
                            w_function=StateWeightFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                                enable_gradient_reversal=True,
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            bandwidth=self.bandwidth,
                            device=self.device,
                            **self.model_args["state_weight"],
                        )
                    )

            if self.action_type == "discrete":
                evaluation_policy_action_dist = (
                    self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    )
                )
            else:
                evaluation_policy_action = self.obtain_evaluation_policy_action(
                    evaluation_policy=evaluation_policy,
                )

            if k_fold == 1:
                if self.action_type == "discrete":
                    self.state_weight_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action_dist=evaluation_policy_action_dist,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
                else:
                    self.state_weight_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action=evaluation_policy_action,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
            else:
                if self.action_type == "discrete":
                    evaluation_policy_action_dist = (
                        evaluation_policy_action_dist.reshape(
                            (-1, self.step_per_trajectory, self.n_actions)
                        )
                    )
                else:
                    evaluation_policy_action = evaluation_policy_action.reshape(
                        (-1, self.step_per_trajectory, self.action_dim)
                    )

                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [(self.n_trajectories + k) // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)
                idx = idx.cumsum()

                for k in tqdm(
                    np.arange(k_fold),
                    desc="[cross-fitting]",
                    total=k_fold,
                ):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    subset_idx_ = np.setdiff1d(all_idx, idx_)

                    if self.action_type == "discrete":
                        action_ = self.action_2d[subset_idx_].flatten()
                        evaluation_policy_action_dist_ = evaluation_policy_action_dist[
                            subset_idx_
                        ].reshape((-1, self.n_actions))
                    else:
                        action_ = self.action_2d[subset_idx_].reshape(
                            (-1, self.action_dim)
                        )
                        evaluation_policy_action_ = evaluation_policy_action[
                            subset_idx_
                        ].reshape((-1, self.action_dim))

                        if self.action_type == "discrete":
                            self.state_weight_function[evaluation_policy.name][k].fit(
                                step_per_trajectory=self.step_per_trajectory,
                                state=self.state_2d[subset_idx_].reshape(
                                    (-1, self.state_dim)
                                ),
                                action=action_,
                                reward=self.reward_2d[subset_idx_].flatten(),
                                pscore=self.pscore_2d[subset_idx_].flatten(),
                                evaluation_policy_action_dist=evaluation_policy_action_dist_,
                                n_steps=n_steps,
                                random_state=random_state,
                            )
                        else:
                            self.state_weight_function[evaluation_policy.name][k].fit(
                                step_per_trajectory=self.step_per_trajectory,
                                state=self.state_2d[subset_idx_].reshape(
                                    (-1, self.state_dim)
                                ),
                                action=action_,
                                reward=self.reward_2d[subset_idx_].flatten(),
                                pscore=self.pscore_2d[subset_idx_].reshape(
                                    (-1, self.action_dim)
                                ),
                                evaluation_policy_action=evaluation_policy_action_,
                                n_steps=n_steps,
                                random_state=random_state,
                            )

    def obtain_initial_state(
        self,
        evaluation_policy: BaseHead,
        resample_initial_state: bool = False,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        random_state: Optional[int] = None,
    ):
        """Obtain initial state distribution (stationary distribution) of the evaluation policy.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        resample_initial_state: bool, default=False
            Whether to resample from initial state distribution using the given evaluation policy.
            If `False`, the initial state distribution of the behavior policy is used instead.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout by the behavior policy before generating the logged dataset
            when working on the infinite horizon setting.
            This argument is irrelevant when working on the finite horizon setting.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout by the behavior policy before generating the logged dataset
            when working on the infinite horizon setting.
            This argument is irrelevant when working on the finite horizon setting.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        evaluation_policy_initial_state: ndarray of shape (n_trajectories, )
            Initial state distribution of the evaluation policy.
            This is intended to be used in marginal OPE estimators.

        """
        check_scalar(
            minimum_rollout_length,
            name=minimum_rollout_length,
            target_type=int,
            min_val=0,
        )
        check_scalar(
            maximum_rollout_length,
            name=maximum_rollout_length,
            target_type=int,
            min_val=0,
        )
        if maximum_rollout_length < minimum_rollout_length:
            raise ValueError(
                "maximum_rollout_length must be larger than minimum_rollout_length, but found False."
            )
        if random_state is None:
            raise ValueError("random_state must be given")
        random_ = check_random_state(random_state)

        if not resample_initial_state:
            initial_state = self.state

        else:
            if evaluation_policy.name in self.initial_state_dict:
                initial_state = self.initial_state_dict[evaluation_policy.name]

            else:
                initial_state = np.zeros((self.n_trajectories), self.state_dim)
                rollout_lengths = random_.choice(
                    np.arange(minimum_rollout_length, maximum_rollout_length),
                    size=self.n_trajectories,
                )

                self.env.reset(random_state)
                done = True

                for i in tqdm(
                    np.arange(self.n_trajectories),
                    desc="[obtain_trajectories]",
                    total=self.n_trajectories,
                ):
                    for rollout_step in rollout_lengths[i]:
                        if done:
                            state, _ = self.env.reset()

                        action = evaluation_policy.sample_action_online(state)
                        state, _, done, _, _ = self.env.step(action)

                    initial_state[i] = state

        return initial_state

    def obtain_evaluation_policy_action(
        self,
        evaluation_policy: BaseHead,
        state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Obtain evaluation policy action.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        state: np.ndarray, default=None
            Sample an action from the evaluation_policy at this state.
            If None is given, state observed in the logged data will be used.

        Return
        -------
        evaluation_policy_action: ndarray of shape (n_trajectories * step_per_trajectory, )
            Evaluation policy action :math:`a_t \\sim \\pi(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        if state is not None:
            check_array(state, name="state", expected_dim=2)
            if state.shape[1] != self.state_dim:
                raise ValueError(
                    "Expected state.shape[1] == self.logged_dataset['state_dim'], but found False"
                )

        state = self.state if state is None else state
        return evaluation_policy.sample_action(state)

    def obtain_evaluation_policy_action_dist(
        self,
        evaluation_policy: BaseHead,
        state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Obtain action choice probability of the evaluation policy and its an estimated Q-function of the observed state.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        state: np.ndarray, default=None
            Sample an action from the evaluation_policy at this state..
            If None is given, state observed in the logged data will be used.

        Return
        -------
        evaluation_policy_action_dist: ndarray of shape (n_trajectories * step_per_trajectory, n_actions)
            Evaluation policy pscore :math:`\\pi(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        if state is not None:
            check_array(state, name="state", expected_dim=2)
            if state.shape[1] != self.state_dim:
                raise ValueError(
                    "Expected state.shape[1] == self.logged_dataset['state_dim'], but found False"
                )

        state = self.state if state is None else state
        return evaluation_policy.calc_action_choice_probability(state)

    def obtain_evaluation_policy_action_prob_for_observed_state_action(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain the pscore of an observed state action pair.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_pscore: ndarray of shape (n_trajectories * step_per_trajectory, )
            Evaluation policy pscore :math:`\\pi(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        return evaluation_policy.calc_pscore_given_action(self.state, self.action)

    def obtain_state_action_value_prediction(
        self,
        method: str,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
    ) -> np.ndarray:
        """Obtain an estimated Q-function of the observed state and all actions (discrete) or that of the actions chosen by behavior and (deterministic) evaluation policies (continuous).

        Parameters
        -------
        method: {"fqe", "dice_q", "mql"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        Return
        -------
        state_action_value_prediction: ndarray of shape (n_trajectories * step_per_trajectory, n_actions) or (n_trajectories * step_per_trajectory, 2)
            If action_type is "discrete", output is state action value for observed state and all actions,
            i.e., :math:`\\hat{Q}(s, a) \\forall a \\in \\mathcal{A}`.

            If action_type is "continuous", output is state action value for the observed action and that chosen by the evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(s_t))`.

        """
        if method not in ["fqe", "dice_q", "mql"]:
            raise ValueError(
                f"method must be either 'fqe', 'dice_q', or 'mql', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        idx = np.array([(self.n_trajectories + k) // k_fold for k in range(k_fold)])
        idx = np.insert(idx, 0, 0)
        idx = idx.cumsum()

        if self.action_type == "discrete":
            state_action_value_prediction = np.zeros(
                (self.n_trajectories, self.step_per_trajectory, self.n_actions)
            )

            for k in range(k_fold):
                idx_ = np.arange(idx[k], idx[k + 1])
                x = self.state_2d[idx_].reshape((-1, self.state_dim))

                x_ = []
                for i in range(x.shape[0]):
                    x_.append(np.tile(x[i], (self.n_actions, 1)))
                x_ = np.array(x_).reshape((-1, self.state_dim))
                a_ = np.tile(np.arange(self.n_actions), x.shape[0])

                if method == "fqe":
                    state_action_value_prediction_ = self.fqe[evaluation_policy.name][
                        k
                    ].predict_value(x_, a_)

                elif method == "dice_q":
                    state_action_value_prediction_ = self.state_action_dual_function[
                        evaluation_policy.name
                    ][k].predict_value(x_, a_)

                elif method == "mql":
                    state_action_value_prediction_ = self.state_action_value_function[
                        evaluation_policy.name
                    ][k].predict_value(x_, a_)

                state_action_value_prediction[
                    idx_
                ] = state_action_value_prediction_.reshape(
                    (-1, self.step_per_trajectory, self.n_actions)
                )

            state_action_value_prediction = state_action_value_prediction.reshape(
                (-1, self.n_actions)
            )

        else:
            state_action_value_prediction = np.zeros(
                (self.n_trajectories, self.step_per_trajectory, 2)
            )

            evaluation_policy_action = self.obtain_evaluation_policy_action(
                evaluation_policy
            ).reshape((-1, self.step_per_trajectory, self.action_dim))

            for k in range(k_fold):
                idx_ = np.arange(idx[k], idx[k + 1])
                state_ = self.state_2d[idx_].reshape((-1, self.state_dim))
                action_ = self.action_2d[idx_].reshape((-1, self.action_dim))
                evaluation_policy_action_ = evaluation_policy_action[idx_].reshape(
                    (-1, self.action_dim)
                )

                if method == "fqe":
                    state_action_value_prediction_behavior_ = self.fqe[
                        evaluation_policy.name
                    ][k].predict_value(state_, action_)
                    state_action_value_prediction_eval_ = self.fqe[
                        evaluation_policy.name
                    ][k].predict_value(state_, evaluation_policy_action_)

                elif method == "dice_q":
                    state_action_value_prediction_behavior_ = (
                        self.state_action_dual_function[evaluation_policy.name][
                            k
                        ].predict_value(state_, action_)
                    )
                    state_action_value_prediction_eval_ = (
                        self.state_action_dual_function[evaluation_policy.name][
                            k
                        ].predict_value(state_, evaluation_policy_action_)
                    )

                elif method == "mql":
                    state_action_value_prediction_behavior_ = (
                        self.state_action_value_function[evaluation_policy.name][
                            k
                        ].predict_value(state_, action_)
                    )
                    state_action_value_prediction_eval_ = (
                        self.state_action_value_function[evaluation_policy.name][
                            k
                        ].predict_value(state_, evaluation_policy_action_)
                    )

                state_action_value_prediction[
                    idx_, :, 0
                ] = state_action_value_prediction_behavior_.reshape(
                    (-1, self.step_per_trajectory)
                )
                state_action_value_prediction[
                    idx_, :, 1
                ] = state_action_value_prediction_eval_.reshape(
                    (-1, self.step_per_trajectory)
                )

            state_action_value_prediction = state_action_value_prediction.reshape(
                (-1, 2)
            )

        return state_action_value_prediction

    def obtain_initial_state_value_prediction(
        self,
        method: str,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        resample_initial_state: bool = False,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Obtain the initial state value of the evaluation policy in the case of discrete action spaces.

        Parameters
        -------
        method: {"fqe", "dice_q", "dice_v", "mql", "mvl"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        resample_initial_state: bool, default=False
            Whether to resample initial state distribution using the given evaluation policy.
            If `False`, the initial state distribution of the behavior policy is used instead.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout by the behavior policy before generating the logged dataset
            when working on the infinite horizon setting.
            This argument is irrelevant when working on the finite horizon setting.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout by the behavior policy before generating the logged dataset
            when working on the infinite horizon setting.
            This argument is irrelevant when working on the finite horizon setting.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        initial_state_value_prediction: ndarray of shape (n_trajectories, )
            State action value of the observed initial state.

        """
        if method not in ["fqe", "dice_q", "dice_v", "mql", "mvl"]:
            raise ValueError(
                f"method must be either 'fqe', 'dice_q', 'dice_v', 'mql', or 'mvl', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        if resample_initial_state:
            initial_state = self.obtain_initial_state(
                evaluation_policy=evaluation_policy,
                resample_initial_state=resample_initial_state,
                minimum_rollout_length=minimum_rollout_length,
                maximum_rollout_length=maximum_rollout_length,
                random_state=random_state,
            )

            if method in ["fqe", "dice_q", "mql"]:
                if self.action_type == "discrete":
                    x_ = []
                    for i in range(self.n_trajectories):
                        x_.append(np.tile(initial_state[i], (self.n_actions, 1)))
                    x_ = np.array(x_).reshape((-1, self.state_dim))
                    a_ = np.tile(np.arange(self.n_actions), self.n_trajectories)
                else:
                    x_ = initial_state
                    a_ = self.obtain_evaluation_policy_action(
                        evaluation_policy=evaluation_policy,
                        state=initial_state,
                    )

                if method == "fqe":
                    state_action_value = self.fqe[evaluation_policy.name][
                        0
                    ].predict_value(x_, a_)

                elif method == "dice_q":
                    state_action_value = self.state_action_dual_function[
                        evaluation_policy.name
                    ][0].predict_value(x_, a_)

                elif method == "mql":
                    state_action_value = self.state_action_value_function[
                        evaluation_policy.name
                    ][0].predict_value(x_, a_)

                if self.action_type == "discrete":
                    evaluation_policy_action_dist = (
                        self.obtain_evaluation_policy_action_dist(
                            evaluation_policy=evaluation_policy,
                            state=initial_state,
                        )
                    )
                    initial_state_value = (
                        state_action_value * evaluation_policy_action_dist
                    ).sum(axis=1)
                else:
                    initial_state_value = state_action_value

            else:
                initial_state_value = self.state_value_function[evaluation_policy.name][
                    0
                ].predict(state=initial_state)

        else:
            if method in ["fqe", "dice_q", "mql"]:
                state_action_value = self.obtain_state_action_value_prediction(
                    method=method,
                    evaluation_policy=evaluation_policy,
                    k_fold=k_fold,
                )

                if self.action_type == "discrete":
                    action_dist = self.obtain_evaluation_policy_action_dist(
                        evaluation_policy
                    )
                    state_value = np.sum(state_action_value * action_dist, axis=1)

                else:
                    state_value = self.obtain_state_action_value_prediction(
                        method=method,
                        evaluation_policy=evaluation_policy,
                        k_fold=k_fold,
                    )[:, 1]

                state_value = state_value.reshape((-1, self.step_per_trajectory))

            else:
                idx = np.array(
                    [(self.n_trajectories + k) // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)
                idx = idx.cumsum()

                state_value = np.zeros((self.n_trajectories, self.step_per_trajectory))

                for k in range(k_fold):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    state_ = self.state_2d[idx_].reshape((-1, self.state_dim))

                    state_value_ = self.state_value_function[evaluation_policy.name][
                        k
                    ].predict(state_)
                    state_value[idx_] = state_value_.reshape(
                        (-1, self.step_per_trajectory)
                    )

            initial_state_value = state_value[:, 0]

        return initial_state_value  # (n_trajectories, )

    def obtain_state_action_marginal_importance_weight(
        self,
        method: str,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
    ) -> np.ndarray:
        """Predict state-action marginal importance weight.

        Parameters
        -------
        method: {"dice", "mwl"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        Return
        -------
        state_action_weight_prediction: ndarray of shape (n_trajectories * step_per_trajectory, )
            State-action marginal importance weight for the observed state and action.

        """
        if method not in ["dice", "mwl"]:
            raise ValueError(
                f"method must be either 'dice' or 'mwl', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        idx = np.array([(self.n_trajectories + k) // k_fold for k in range(k_fold)])
        idx = np.insert(idx, 0, 0)
        idx = idx.cumsum()

        state_action_weight_prediction = np.zeros(
            (self.n_trajectories, self.step_per_trajectory)
        )

        for k in range(k_fold):
            idx_ = np.arange(idx[k], idx[k + 1])
            state_ = self.state_2d[idx_].reshape((-1, self.state_dim))

            if self.action_type == "discrete":
                action_ = self.action_2d[idx_].flatten()
            else:
                action_ = self.action_2d[idx_].reshape((-1, self.action_dim))

            if method == "dice":
                state_action_weight_prediction_ = self.state_action_dual_function[
                    evaluation_policy.name
                ][k].predict_weight(state_, action_)

            elif method == "mwl":
                state_action_weight_prediction_ = self.state_action_weight_function[
                    evaluation_policy.name
                ][k].predict_weight(state_, action_)

            state_action_weight_prediction[
                idx_
            ] = state_action_weight_prediction_.reshape((-1, self.step_per_trajectory))

        return state_action_weight_prediction.flatten()

    def obtain_state_marginal_importance_weight(
        self,
        method: str,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
    ) -> np.ndarray:
        """Predict state marginal importance weight.

        Parameters
        -------
        method: {"dice", "mwl"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

        Return
        -------
        state_weight_prediction: ndarray of shape (n_trajectories * step_per_trajectory, )
            State marginal importance weight for observed state.

        """
        if method not in ["dice", "mwl"]:
            raise ValueError(
                f"method must be either 'dice' or 'mwl', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        idx = np.array([(self.n_trajectories + k) // k_fold for k in range(k_fold)])
        idx = np.insert(idx, 0, 0)
        idx = idx.cumsum()

        state_weight_prediction = np.zeros(
            (self.n_trajectories, self.step_per_trajectory)
        )

        for k in range(k_fold):
            idx_ = np.arange(idx[k], idx[k + 1])
            state_ = self.state_2d[idx_].reshape((-1, self.state_dim))

            if method == "dice":
                state_weight_prediction_ = self.state_dual_function[
                    evaluation_policy.name
                ][k].predict_weight(state_)

            elif method == "mwl":
                state_weight_prediction_ = self.state_weight_function[
                    evaluation_policy.name
                ][k].predict_weight(state_)

            state_weight_prediction[idx_] = state_weight_prediction_.reshape(
                (-1, self.step_per_trajectory)
            )

        return state_weight_prediction.flatten()

    def _obtain_whole_inputs(
        self,
        logged_dataset: LoggedDataset,
        evaluation_policies: List[BaseHead],
        require_value_prediction: bool = False,
        require_weight_prediction: bool = False,
        resample_initial_state: bool = False,
        q_function_method: str = "fqe",
        v_function_method: str = "fqe",
        w_function_method: str = "dice",
        k_fold: int = 1,
        n_steps: int = 10000,
        n_trajectories_on_policy_evaluation: int = 100,
        use_stationary_distribution_on_policy_evaluation: bool = False,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        random_state: Optional[int] = None,
    ) -> OPEInputDict:
        """Obtain input as a dictionary.

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

        evaluation_policies: list of BaseHead
            Evaluation policies.

        require_value_prediction: bool, default=False
            Whether to obtain an estimated value function.

        require_weight_prediction: bool, default=False
            Whether to obtain an estimated weight function.

        resample_initial_state: bool, default=False
            Whether to resample initial state distribution using the given evaluation policy.
            If `False`, the initial state distribution of the behavior policy is used instead.

            Note that this parameter is applicable only when self.env is given.

        q_function_method: {"fqe", "dice_q", "mql"}, default="fqe"
            Method to estimate :math:`Q(s, a)`.

        v_function_method: {"fqe", "dice_q", "dice_v", "mql", "mvl"}, default="fqe"
            Method to estimate :math:`V(s)`.

        w_function_method: {"dice", "mwl"}, default="dice"
            Method to estimate :math:`w(s, a)` and :math:`w(s)`.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

            If :math:`K>1`, we split the logged dataset into :math:`K` folds.
            :math:`\\mathcal{D}_j` is the :math:`j`-th split of logged data consisting of :math:`n_k` samples.
            Then, the value and weight functions (:math:`w^j` and :math:`Q^j`) are trained on the subset of data used for OPE,
            i.e., :math:`\\mathcal{D} \\setminus \\mathcal{D}_j`.

            If :math:`K=1`, the value and weight functions are trained on the entire data.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps to train weight and value learning models.

        n_trajectories_on_policy_evaluation: int, default=None (> 0)
            Number of episodes to perform on-policy evaluation.

        use_stationary_distribution_on_policy_evaluation: bool, default=False
            Whether to evaluate a policy based on stationary distribution.
            If `True`, evaluation policy is evaluated by rollout without resetting environment at each episode.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout to collect initial state.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout to collect initial state.

        random_state: int, default=None (>= 0)
            Random state.

        Return
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
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            evaluation_policy_action: ndarray of shape (n_trajectories * step_per_trajectories, action_dim)
                Action chosen by the deterministic evaluation policy.
                If action_type is "discrete", `None` is recorded.

            evaluation_policy_action_dist: ndarray of shape (n_trajectories * step_per_trajectory, n_actions)
                Conditional action distribution induced by the evaluation policy,
                i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`.
                If action_type is "continuous", `None` is recorded.

            state_action_value_prediction: ndarray
                If action_type is "discrete", :math:`\\hat{Q}` for all actions,
                i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.
                shape (n_trajectories * step_per_trajectory, n_actions)

                If action_type is "continuous", :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
                i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.
                shape (n_trajectories * step_per_trajectory, 2)

                If require_value_prediction is `False`, `None` is recorded.

            initial_state_value_prediction: ndarray of shape (n_trajectories, )
                Estimated initial state value.

                If use_base_model is `False`, `None` is recorded.

            state_action_marginal_importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
                Estimated state-action marginal importance weight,
                i.e., :math:`\\hat{w}(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_b}(s_t, a_t)`.

                If require_weight_prediction is `False`, `None` is recorded.

            state_marginal_importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
                Estimated state marginal importance weight,
                i.e., :math:`\\hat{w}(s_t) \\approx d^{\\pi}(s_t) / d^{\\pi_b}(s_t)`.

                If require_weight_prediction is `False`, `None` is recorded.

            on_policy_policy_value: ndarray of shape (n_trajectories_on_policy_evaluation, )
                On-policy policy value.
                If self.env is `None`, `None` is recorded.

            gamma: float
                Discount factor.

            behavior_policy: str
                Name of the behavior policy.

            evaluation_policy: str
                Name of the evaluation policy.

            dataset_id: int
                Id of the logged dataset.

        """
        check_logged_dataset(logged_dataset)
        self._register_logged_dataset(logged_dataset)

        if self.env is not None:
            if isinstance(self.env.action_space, Discrete):
                if logged_dataset["action_type"] != "discrete":
                    raise RuntimeError(
                        f"Detected mismatch between action_type of logged_dataset and env.action_space."
                    )
                elif logged_dataset["n_actions"] != self.env.action_space.n:
                    raise RuntimeError(
                        f"Detected mismatch between n_actions of logged_dataset and env.action_space.n."
                    )
            else:
                if logged_dataset["action_type"] != "continuous":
                    raise RuntimeError(
                        f"Detected mismatch between action_type of logged_dataset and env.action_space."
                    )
                elif logged_dataset["action_dim"] != self.env.action_space.shape[0]:
                    raise RuntimeError(
                        f"Detected mismatch between action_dim of logged_dataset and env.action_space.shape[0]."
                    )

        for eval_policy in evaluation_policies:
            if eval_policy.action_type != self.action_type:
                raise RuntimeError(
                    f"One of the evaluation_policies, {eval_policy.name} does not match action_type in logged_dataset."
                    " Please use {self.action_type} action_type instead."
                )

        if require_value_prediction:
            if q_function_method == "fqe" or v_function_method == "fqe":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit FQE model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_FQE(
                        evaluation_policies[i], k_fold=k_fold, n_steps=n_steps
                    )

            if q_function_method == "dice_q" or v_function_method == "dice_q":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit Augmented Lagrangian model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_action_dual_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_steps=n_steps,
                        random_state=random_state,
                    )

            if v_function_method == "dice_v":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit Augmented Lagrangian model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_dual_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_steps=n_steps,
                        random_state=random_state,
                    )

            if q_function_method == "mql" or v_function_method == "mql":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit MQL model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_action_value_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_steps=n_steps,
                        random_state=random_state,
                    )

            if v_function_method == "mvl":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit MVL model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_value_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_steps=n_steps,
                        random_state=random_state,
                    )

        if require_weight_prediction:
            if w_function_method == "dice":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit Augmented Lagrangian model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_action_dual_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
                    self.build_and_fit_state_dual_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_steps=n_steps,
                        random_state=random_state,
                    )

            elif w_function_method == "mwl":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit MWL model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_action_weight_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_steps=n_steps,
                        random_state=random_state,
                    )
                    self.build_and_fit_state_weight_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_steps=n_steps,
                        random_state=random_state,
                    )

        input_dict = defaultdict(dict)

        for i in tqdm(
            range(len(evaluation_policies)),
            desc="[collect input data: eval_policy]",
            total=len(evaluation_policies),
        ):
            # input for IPW, DR
            if self.action_type == "discrete":
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action_dist"
                ] = self.obtain_evaluation_policy_action_dist(evaluation_policies[i])
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action"
                ] = None
            else:
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action_dist"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action"
                ] = self.obtain_evaluation_policy_action(evaluation_policies[i])

            # input for DM, DR
            if require_value_prediction:
                input_dict[evaluation_policies[i].name][
                    "state_action_value_prediction"
                ] = self.obtain_state_action_value_prediction(
                    method=q_function_method,
                    evaluation_policy=evaluation_policies[i],
                    k_fold=k_fold,
                )
                input_dict[evaluation_policies[i].name][
                    "initial_state_value_prediction"
                ] = self.obtain_initial_state_value_prediction(
                    method=v_function_method,
                    evaluation_policy=evaluation_policies[i],
                    k_fold=k_fold,
                    resample_initial_state=resample_initial_state,
                    minimum_rollout_length=minimum_rollout_length,
                    maximum_rollout_length=maximum_rollout_length,
                    random_state=random_state,
                )
            else:
                input_dict[evaluation_policies[i].name][
                    "state_action_value_prediction"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "initial_state_value_prediction"
                ] = None

            # input for marginal OPE
            if require_weight_prediction:
                input_dict[evaluation_policies[i].name][
                    "state_action_marginal_importance_weight"
                ] = self.obtain_state_action_marginal_importance_weight(
                    method=w_function_method,
                    evaluation_policy=evaluation_policies[i],
                    k_fold=k_fold,
                )
                input_dict[evaluation_policies[i].name][
                    "state_marginal_importance_weight"
                ] = self.obtain_state_marginal_importance_weight(
                    method=w_function_method,
                    evaluation_policy=evaluation_policies[i],
                    k_fold=k_fold,
                )
            else:
                input_dict[evaluation_policies[i].name][
                    "state_action_marginal_importance_weight"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "state_marginal_importance_weight"
                ] = None

            # input for the evaluation of OPE estimators
            if self.env is not None:
                if n_trajectories_on_policy_evaluation is None:
                    n_trajectories_on_policy_evaluation = self.n_trajectories

                input_dict[evaluation_policies[i].name][
                    "on_policy_policy_value"
                ] = rollout_policy_online(
                    self.env,
                    evaluation_policies[i],
                    n_trajectories=n_trajectories_on_policy_evaluation,
                    evaluate_on_stationary_distribution=use_stationary_distribution_on_policy_evaluation,
                    step_per_trajectory=self.step_per_trajectory,
                    gamma=self.gamma,
                    random_state=random_state,
                )
            else:
                input_dict[evaluation_policies[i].name]["on_policy_policy_value"] = None

            input_dict[evaluation_policies[i].name]["gamma"] = self.gamma
            input_dict[evaluation_policies[i].name][
                "behavior_policy"
            ] = self.behavior_policy_name
            input_dict[evaluation_policies[i].name][
                "evaluation_policy"
            ] = evaluation_policies[i].name
            input_dict[evaluation_policies[i].name]["dataset_id"] = self.dataset_id

        return defaultdict_to_dict(input_dict)

    def obtain_whole_inputs(
        self,
        logged_dataset: Union[LoggedDataset, MultipleLoggedDataset],
        evaluation_policies: List[BaseHead],
        behavior_policy_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        require_value_prediction: bool = False,
        require_weight_prediction: bool = False,
        resample_initial_state: bool = False,
        q_function_method: str = "fqe",
        v_function_method: str = "fqe",
        w_function_method: str = "dice",
        k_fold: int = 1,
        n_steps: int = 10000,
        n_trajectories_on_policy_evaluation: int = 100,
        use_stationary_distribution_on_policy_evaluation: bool = False,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        random_state: Optional[int] = None,
        path: str = "input_dict/",
        save_relative_path: bool = False,
    ) -> OPEInputDict:
        """Obtain input as a dictionary.

        Parameters
        -------
        logged_dataset: LoggedDataset or MultipleLoggedDataset
            Logged dataset containing the following.

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

        evaluation_policies: list of BaseHead or BaseHead
            Evaluation policies.

            .. tip::

                1. When using LoggedDataset, evaluation_policies should be ``List[BaseHead]`` (``[BaseHead, BaseHead, ..]``).

                2. When using MultipleLoggedDataset and apply the same evaluation policies across behavior_policies and dataset_ids,
                evaluation_policies should be ``List[BaseHead]``.

                3. When using MultipleLoggedDataset and apply the same evaluation policies across dataset_ids but different evaluation_policies across behavior policies,
                evaluation_policies should be ``Dict[str, List[BaseHead]]``. (key: ``[behavior_policy_name]``).

                4. When using MultipleLoggedDataset and apply different evaluation policies across dataset_ids and behavior policies,
                evaluation_policies should be ``Dict[str, List[BaseHead]]``. (key: ``[behavior_policy_name][dataset_id]``)

        behavior_policy_name: str, default=None
            Name of the behavior policy.

        dataset_id: int, default=None
            Id of the logged dataset.

        require_value_prediction: bool, default=False
            Whether to obtain an estimated value function.

        require_weight_prediction: bool, default=False
            Whether to obtain an estimated weight function.

        resample_initial_state: bool, default=False
            Whether to resample initial state distribution using the given evaluation policy.
            If `False`, the initial state distribution of the behavior policy is used instead.

            Note that this parameter is applicable only when self.env is given.

        q_function_method: {"fqe", "dice_q", "mql"}, default="fqe"
            Method to estimate :math:`Q(s, a)`.

        v_function_method: {"fqe", "dice_q", "dice_v", "mql", "mvl"}, default="fqe"
            Method to estimate :math:`V(s)`.

        w_function_method: {"dice", "mwl"}, default="dice"
            Method to estimate :math:`w(s, a)` and :math:`w(s)`.

        k_fold: int, default=1 (> 0)
            Number of folds to perform cross-fitting.

            If :math:`K>1`, we split the logged dataset into :math:`K` folds.
            :math:`\\mathcal{D}_j` is the :math:`j`-th split of logged data consisting of :math:`n_k` samples.
            Then, the value and weight functions (:math:`w^j` and :math:`Q^j`) are trained on the subset of data used for OPE,
            i.e., :math:`\\mathcal{D} \\setminus \\mathcal{D}_j`.

            If :math:`K=1`, the value and weight functions are trained on the entire data.

        n_steps: int, default=10000 (> 0)
            Number of gradient steps to fit weight and value learning methods.

        n_trajectories_on_policy_evaluation: int, default=None (> 0)
            Number of episodes to perform on-policy evaluation.

        use_stationary_distribution_on_policy_evaluation: bool, default=False
            Whether to evaluate a policy based on stationary distribution.
            If `True`, evaluation policy is evaluated by rollout without resetting environment at each episode.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout to collect initial state.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout to collect initial state.

        random_state: int, default=None (>= 0)
            Random state.

        path: str
            Path to the directory. Either absolute or relative path is acceptable.

        save_relative_path: bool, default=False.
            Whether to save a relative path.
            If `True`, a path relative to the scope-rl directory will be saved.
            If `False`, the absolute path will be saved.

            Note that this option was added in order to run examples in the documentation properly.
            Otherwise, the default setting (`False`) is recommended.

        Return
        -------
        input_dicts: OPEInputDict or MultipleInputDict
            MultipleInputDict is an instance containing (multiple) input dictionary for OPE.

            Each input dict is accessible by the following command.

            .. code-block:: python

                input_dict_ = input_dict.get(behavior_policy_name=behavior_policy.name, dataset_id=0)

            Each input dict consists of the following.

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
                    behavior_policy,
                    evaluation_policy,
                    dataset_id,
                ]

            evaluation_policy_action: ndarray of shape (n_trajectories * step_per_trajectories, action_dim)
                Action chosen by the deterministic evaluation policy.
                If action_type is "discrete", `None` is recorded.

            evaluation_policy_action_dist: ndarray of shape (n_trajectories * step_per_trajectory, n_actions)
                Conditional action distribution induced by the evaluation policy,
                i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`.
                If action_type is "continuous", `None` is recorded.

            state_action_value_prediction: ndarray
                If action_type is "discrete", :math:`\\hat{Q}` for all actions,
                i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.
                shape (n_trajectories * step_per_trajectory, n_actions)

                If action_type is "continuous", :math:`\\hat{Q}` for the observed action and that chosen by the evaluation policy,
                i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.
                shape (n_trajectories * step_per_trajectory, 2)

                If require_value_prediction is `False`, `None` is recorded.

            initial_state_value_prediction: ndarray of shape (n_trajectories, )
                Estimated initial state value.

                If use_base_model is `False`, `None` is recorded.

            state_action_marginal_importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
                Estimated state-action marginal importance weight,
                i.e., :math:`\\hat{w}(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_b}(s_t, a_t)`.

                If require_weight_prediction is `False`, `None` is recorded.

            state_marginal_importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
                Estimated state marginal importance weight,
                i.e., :math:`\\hat{w}(s_t) \\approx d^{\\pi}(s_t) / d^{\\pi_b}(s_t)`.

                If require_weight_prediction is `False`, `None` is recorded.

            on_policy_policy_value: ndarray of shape (n_trajectories_on_policy_evaluation, )
                On-policy policy value.
                If self.env is `None`, `None` is recorded.

            gamma: float
                Discount factor.

            behavior_policy: str
                Name of the behavior policy.

            evaluation_policy: str
                Name of the evaluation policy.

            dataset_id: int
                Id of the logged dataset.

        """
        apply_different_eval_policies_across_all_datasets = False
        apply_different_eval_policies_across_behavior_policies = False
        if isinstance(evaluation_policies, dict):
            if isinstance(list(evaluation_policies.values())[0], BaseHead):
                apply_different_eval_policies_across_behavior_policies = True
            else:
                apply_different_eval_policies_across_all_datasets = True

        evaluation_policies_ = deepcopy(evaluation_policies)
        if isinstance(logged_dataset, MultipleLoggedDataset):
            if behavior_policy_name is None and dataset_id is None:
                evaluation_policies = defaultdict(list)

                for behavior_policy, n_datasets in logged_dataset.n_datasets.items():
                    for dataset_id_ in range(n_datasets):
                        if apply_different_eval_policies_across_all_datasets:
                            evaluation_policies[behavior_policy].append(
                                evaluation_policies_[behavior_policy][dataset_id_]
                            )

                        elif apply_different_eval_policies_across_behavior_policies:
                            evaluation_policies[behavior_policy].append(
                                evaluation_policies_[behavior_policy]
                            )

                        else:
                            evaluation_policies[behavior_policy].append(
                                evaluation_policies_
                            )

            elif behavior_policy_name is None and dataset_id is not None:
                evaluation_policies = {}

                for behavior_policy, n_datasets in logged_dataset.n_datasets.items():
                    if apply_different_eval_policies_across_all_datasets:
                        evaluation_policies[behavior_policy] = evaluation_policies_[
                            behavior_policy
                        ][dataset_id]

                    elif apply_different_eval_policies_across_behavior_policies:
                        evaluation_policies[behavior_policy] = evaluation_policies_[
                            behavior_policy
                        ]

                    else:
                        evaluation_policies[behavior_policy] = evaluation_policies_

            elif behavior_policy_name is not None and dataset_id is None:
                evaluation_policies = []

                for dataset_id_ in range(
                    logged_dataset.n_datasets[behavior_policy_name]
                ):
                    if apply_different_eval_policies_across_all_datasets:
                        evaluation_policies.append(
                            evaluation_policies_[behavior_policy_name][dataset_id_]
                        )

                    elif apply_different_eval_policies_across_behavior_policies:
                        evaluation_policies[behavior_policy].append(
                            evaluation_policies_[behavior_policy_name]
                        )

                    else:
                        evaluation_policies[behavior_policy].append(
                            evaluation_policies_
                        )

            else:
                if apply_different_eval_policies_across_all_datasets:
                    evaluation_policies = evaluation_policies_[behavior_policy_name][
                        dataset_id
                    ]

                elif apply_different_eval_policies_across_behavior_policies:
                    evaluation_policies = evaluation_policies_[behavior_policy_name]

                else:
                    evaluation_policies = evaluation_policies_

        else:
            behavior_policy_name = logged_dataset["behavior_policy"]
            dataset_id = logged_dataset["dataset_id"]

            if apply_different_eval_policies_across_all_datasets:
                evaluation_policies = evaluation_policies_[behavior_policy_name][
                    dataset_id
                ]

            elif apply_different_eval_policies_across_behavior_policies:
                evaluation_policies = evaluation_policies_[behavior_policy_name]

            else:
                evaluation_policies = evaluation_policies_

        if isinstance(logged_dataset, MultipleLoggedDataset):
            input_dict = MultipleInputDict(
                action_type=logged_dataset.action_type,
                path=path,
                save_relative_path=save_relative_path,
            )

            if behavior_policy_name is None and dataset_id is None:
                behavior_policies = logged_dataset.behavior_policy_names

                for i in tqdm(
                    np.arange(len(behavior_policies)),
                    desc="[collect input data: behavior_policy]",
                    total=len(behavior_policies),
                ):
                    n_datasets = logged_dataset.n_datasets[behavior_policies[i]]

                    for dataset_id_ in tqdm(
                        np.arange(n_datasets),
                        desc="[collect input data: dataset_id]",
                        total=n_datasets,
                    ):
                        logged_dataset_ = logged_dataset.get(
                            behavior_policy_name=behavior_policies[i],
                            dataset_id=dataset_id_,
                        )
                        input_dict_ = self._obtain_whole_inputs(
                            logged_dataset=logged_dataset_,
                            evaluation_policies=evaluation_policies[
                                behavior_policies[i]
                            ][dataset_id_],
                            require_value_prediction=require_value_prediction,
                            require_weight_prediction=require_weight_prediction,
                            resample_initial_state=resample_initial_state,
                            q_function_method=q_function_method,
                            v_function_method=v_function_method,
                            w_function_method=w_function_method,
                            k_fold=k_fold,
                            n_steps=n_steps,
                            n_trajectories_on_policy_evaluation=n_trajectories_on_policy_evaluation,
                            use_stationary_distribution_on_policy_evaluation=use_stationary_distribution_on_policy_evaluation,
                            minimum_rollout_length=minimum_rollout_length,
                            maximum_rollout_length=maximum_rollout_length,
                            random_state=random_state,
                        )
                        input_dict.add(
                            input_dict_,
                            behavior_policy_name=behavior_policies[i],
                            dataset_id=dataset_id_,
                        )

            elif behavior_policy_name is None and dataset_id is not None:
                behavior_policies = logged_dataset.behavior_policy_names

                for i in tqdm(
                    np.arange(len(behavior_policies)),
                    desc="[collect input data: behavior_policy]",
                    total=len(behavior_policies),
                ):
                    logged_dataset_ = logged_dataset.get(
                        behavior_policy_name=behavior_policies[i], dataset_id=dataset_id
                    )
                    input_dict_ = self._obtain_whole_inputs(
                        logged_dataset=logged_dataset_,
                        evaluation_policies=evaluation_policies[behavior_policies[i]],
                        require_value_prediction=require_value_prediction,
                        require_weight_prediction=require_weight_prediction,
                        resample_initial_state=resample_initial_state,
                        q_function_method=q_function_method,
                        v_function_method=v_function_method,
                        w_function_method=w_function_method,
                        k_fold=k_fold,
                        n_steps=n_steps,
                        n_trajectories_on_policy_evaluation=n_trajectories_on_policy_evaluation,
                        use_stationary_distribution_on_policy_evaluation=use_stationary_distribution_on_policy_evaluation,
                        minimum_rollout_length=minimum_rollout_length,
                        maximum_rollout_length=maximum_rollout_length,
                        random_state=random_state,
                    )
                    input_dict.add(
                        input_dict_,
                        behavior_policy_name=behavior_policies[i],
                        dataset_id=dataset_id,
                    )

            elif behavior_policy_name is not None and dataset_id is None:
                n_datasets = logged_dataset.n_datasets[behavior_policy_name]

                for dataset_id_ in tqdm(
                    np.arange(n_datasets),
                    desc="[collect input data: dataset_id]",
                    total=n_datasets,
                ):
                    logged_dataset_ = logged_dataset.get(
                        behavior_policy_name=behavior_policy_name,
                        dataset_id=dataset_id_,
                    )
                    input_dict_ = self._obtain_whole_inputs(
                        logged_dataset=logged_dataset_,
                        evaluation_policies=evaluation_policies[dataset_id_],
                        require_value_prediction=require_value_prediction,
                        require_weight_prediction=require_weight_prediction,
                        resample_initial_state=resample_initial_state,
                        q_function_method=q_function_method,
                        v_function_method=v_function_method,
                        w_function_method=w_function_method,
                        k_fold=k_fold,
                        n_steps=n_steps,
                        n_trajectories_on_policy_evaluation=n_trajectories_on_policy_evaluation,
                        use_stationary_distribution_on_policy_evaluation=use_stationary_distribution_on_policy_evaluation,
                        minimum_rollout_length=minimum_rollout_length,
                        maximum_rollout_length=maximum_rollout_length,
                        random_state=random_state,
                    )
                    input_dict.add(
                        input_dict_,
                        behavior_policy_name=behavior_policy_name,
                        dataset_id=dataset_id_,
                    )

            else:
                logged_dataset = logged_dataset.get(
                    behavior_policy_name=behavior_policy_name, dataset_id=dataset_id
                )
                input_dict = self._obtain_whole_inputs(
                    logged_dataset=logged_dataset,
                    evaluation_policies=evaluation_policies,
                    require_value_prediction=require_value_prediction,
                    require_weight_prediction=require_weight_prediction,
                    resample_initial_state=resample_initial_state,
                    q_function_method=q_function_method,
                    v_function_method=v_function_method,
                    w_function_method=w_function_method,
                    k_fold=k_fold,
                    n_steps=n_steps,
                    n_trajectories_on_policy_evaluation=n_trajectories_on_policy_evaluation,
                    use_stationary_distribution_on_policy_evaluation=use_stationary_distribution_on_policy_evaluation,
                    minimum_rollout_length=minimum_rollout_length,
                    maximum_rollout_length=maximum_rollout_length,
                    random_state=random_state,
                )

        else:
            input_dict = self._obtain_whole_inputs(
                logged_dataset=logged_dataset,
                evaluation_policies=evaluation_policies,
                require_value_prediction=require_value_prediction,
                require_weight_prediction=require_weight_prediction,
                resample_initial_state=resample_initial_state,
                q_function_method=q_function_method,
                v_function_method=v_function_method,
                w_function_method=w_function_method,
                k_fold=k_fold,
                n_steps=n_steps,
                n_trajectories_on_policy_evaluation=n_trajectories_on_policy_evaluation,
                use_stationary_distribution_on_policy_evaluation=use_stationary_distribution_on_policy_evaluation,
                minimum_rollout_length=minimum_rollout_length,
                maximum_rollout_length=maximum_rollout_length,
                random_state=random_state,
            )

        return input_dict
