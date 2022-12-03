"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from warnings import warn

from collections import defaultdict
from tqdm.auto import tqdm

import torch
import numpy as np
from sklearn.utils import check_scalar, check_random_state

import gym
from gym.spaces import Box, Discrete

from d3rlpy.dataset import MDPDataset
from d3rlpy.ope import DiscreteFQE
from d3rlpy.ope import FQE as ContinuousFQE
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.preprocessing import Scaler, ActionScaler

from .weight_value_learning import (
    DiscreteAugmentedLagrangianStateActionWightValueLearning,
    DiscreteAugmentedLagrangianStateWightValueLearning,
    ContinuousAugmentedLagrangianStateActionWightValueLearning,
    ContinuousAugmentedLagrangianStateWightValueLearning,
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
from ..types import LoggedDataset, OPEInputDict
from ..utils import (
    defaultdict_to_dict,
    check_logged_dataset,
    check_array,
)


@dataclass
class CreateOPEInput:
    """Class to prepare OPE inputs.

    Parameters
    -------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    env: gym.Env, default=None
        Reinforcement learning (RL) environment.

    model_args: dict[str, dict], default=None
        Arguments of the base model.
        The key should be: [
            "fqe",
            "state_action_dual",
            "state_action_value",
            "state_action_weight",
            "state_dual",
            "state_value",
            "state_weight",
            "hidden_dim",  # hidden dim of value/weight function, except FQE
        ]

        Please refer to initialization arguments for FQE, and refer to arguments of fit method for other models.
        (FQE is implemented in d3rlpy. The other models are implemented in ofrl/ope/weight_value_learning/.)

        Note that, please "scaler" and "action_scaler" in the following argument
        (, as we will overwrite those specified by model_args[model]["scaler/action_scaler"]).

    gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

    sigma: float, default=1.0 (> 0.0)
        Bandwidth hyperparameter of gaussian kernel.

    state_scaler: d3rlpy.preprocessing.Scaler, default=None
        Scaling factor of state.

    action_scaler: d3rlpy.preprocessing.ActionScaler, default=None
        Scaling factor of action.

    device: str, default="cuda:0"
        Specifies device used for torch.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Takuma Seno and Michita Imai.
    "d3rlpy: An Offline Deep Reinforcement Library.", 2021.

    Masatoshi Uehara, Jiawei Huang, and Nan Jiang.
    "Minimax Weight and Q-Function Learning for Off-Policy Evaluation.", 2020.

    Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans.
    "Off-Policy Evaluation via the Regularized Lagrangian.", 2020.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    """

    logged_dataset: LoggedDataset
    env: Optional[gym.Env] = None,
    model_args: Optional[Dict[str, Any]] = None
    gamma: float = 1.0
    sigma: float = 1.0
    state_scaler: Optional[Scaler] = None
    action_scaler: Optional[ActionScaler] = None
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
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

        if self.env is not None:
            if isinstance(self.env.action_space, Box) and self.action_type == "discrete":
                raise RuntimeError(
                    "Found mismatch in action_type between env and logged_dataset"
                )
            elif (
                isinstance(self.env.action_space, Discrete)
                and self.action_type == "continuous"
            ):
                raise RuntimeError(
                    "Found mismatch in action_type between env and logged_dataset"
                )

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

        self.fqe = {}
        if self.model_args["fqe"] is None:
            self.model_args["fqe"] = {
                "encoder_factory": VectorEncoderFactory(
                    hidden_units=[self.model_args["hidden_dim"]]
                ),
                "q_func_factory": MeanQFunctionFactory(),
                "learning_rate": 1e-4,
                "use_gpu": torch.cuda.is_available(),
            }

        self.state_action_dual_function = {}
        if self.model_args["state_action_dual"] is None:
            self.model_args["state_action_dual"] = {}

        self.state_action_value_function = {}
        if self.model_args["state_action_value"] is None:
            self.model_args["state_action_value"] = {}

        self.state_action_weight_function = {}
        if self.model_args["state_action_weight"] is None:
            self.model_args["state_action_weight"] = {}

        self.state_dual_function = {}
        if self.model_args["state_dual"] is None:
            self.model_args["state_dual"] = {}

        self.state_value_function = {}
        if self.model_args["state_value"] is None:
            self.model_args["state_value"] = {}

        self.state_weight_function = {}
        if self.model_args["state_weight"] is None:
            self.model_args["state_weight"] = {}

        self.initial_state_dict = {}

        check_scalar(self.gamma, name="gamma", target_type=float, min_val=0.0, max_val=1.0)
        check_scalar(self.sigma, name="sigma", target_type=float, min_val=0.0)
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

    def build_and_fit_FQE(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
    ) -> None:
        """Fit Fitted Q Evaluation (FQE).

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        n_epochs: int, default=1 (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

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
                    n_epochs=n_epochs,
                    n_steps_per_epoch=n_steps_per_epoch,
                    scorers={},
                )
            else:
                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [self.n_trajectories + k // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)

                for k in range(k_fold):
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
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                        scorers={},
                    )

    def build_and_fit_state_action_dual_model(
        self,
        method: str,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Augmented Lagrangian Method (ALM) for state-action value weight function.

        Parameters
        -------
         method: {"dual_dice", "gen_dice", "algae_dice", "best_dice", "mql", "mwl", "custom"}, default="best_dice"
            Indicates which parameter set should be used. When, "custom" users can specify their own parameter.

        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        n_epochs: int, default=1 (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        random_state: int, default=None
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

        if evaluation_policy.name in self.state_action_dual_function:
            pass

        else:
            self.state_action_dual_function[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.state_action_dual_function[evaluation_policy.name].append(
                        DiscreteAugmentedLagrangianStateActionWightValueLearning(
                            method=method,
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
                            ),
                            state_scaler=self.state_scaler,
                            device=self.device,
                        )
                    )
                else:
                    self.state_action_dual_function[evaluation_policy.name].append(
                        ContinuousAugmentedLagrangianStateActionWightValueLearning(
                            method=method,
                            q_function=ContinuousQFunction(
                                action_dim=self.action_dim,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            w_function=ContinuousStateActionWeightFunction(
                                action_dim=self.action_dim,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            device=self.device,
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
                        random_state=random_state,
                        **self.model_args["state_action_dual"],
                    )
                else:
                    self.state_action_dual_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        evaluation_policy_action=evaluation_policy_action,
                        random_state=random_state,
                        **self.model_args["state_action_dual"],
                    )
            else:
                if self.action_type == "discrete":
                    evaluation_policy_action = evaluation_policy_action.reshape(
                        (-1, self.step_per_trajectory)
                    )
                else:
                    evaluation_policy_action = evaluation_policy_action.reshape(
                        (-1, self.step_per_trajectory, self.action_dim)
                    )

                all_idx = np.arange(self.n_trajectories)
                idx = np.array(
                    [self.n_trajectories + k // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)

                for k in range(k_fold):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    subset_idx_ = np.setdiff1d(all_idx, idx_)

                    if self.action_type == "discrete":
                        action_ = self.action_2d[subset_idx_].flatten()
                        evaluation_policy_action_dist_ = evaluation_policy_action_dist_[
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
                            random_state=random_state,
                            **self.model_args["state_action_dual"],
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
                            random_state=random_state,
                            **self.model_args["state_action_dual"],
                        )

    def build_and_fit_state_action_value_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Minimax Q Learning (MQL) for state-action value function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        n_epochs: int, default=1 (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        random_state: int, default=None
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

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
                                hidden_dim=self.model_args["hidden_dim"],
                                device=self.device,
                            ),
                            state_scaler=self.state_scaler,
                            device=self.device,
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
                            device=self.device,
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
                        random_state=random_state,
                        **self.model_args["state_action_value"],
                    )
                else:
                    self.state_action_value_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action=evaluation_policy_action,
                        random_state=random_state,
                        **self.model_args["state_action_value"],
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
                    [self.n_trajectories + k // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)

                for k in range(k_fold):
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
                            random_state=random_state,
                            **self.model_args["state_action_value"],
                        )
                    else:
                        self.state_action_value_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].flatten(),
                            evaluation_policy_action=evaluation_policy_action_,
                            random_state=random_state,
                            **self.model_args["state_action_value"],
                        )

    def build_and_fit_state_action_weight_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Minimax Weight Learning (MWL) for state-action weight function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        n_epochs: int, default=1 (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        random_state: int, default=None
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

        if n_epochs is None:
            n_epochs = 1

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
                                device=self.device,
                            ),
                            state_scaler=self.state_scaler,
                            device=self.device,
                        )
                    )
                else:
                    self.state_action_weight_function[evaluation_policy.name].append(
                        ContinuousMinimaxStateActionWeightLearning(
                            w_function=ContinuousStateActionWeightFunction(
                                action_dim=self.action_dim,
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            device=self.device,
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
                        random_state=random_state,
                        **self.model_args["state_action_weight"],
                    )
                else:
                    self.state_action_weight_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        evaluation_policy_action=evaluation_policy_action,
                        random_state=random_state,
                        **self.model_args["state_action_weight"],
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
                    [self.n_trajectories + k // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)

                for k in range(k_fold):
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
                            random_state=random_state,
                            **self.model_args["state_action_weight"],
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
                            random_state=random_state,
                            **self.model_args["state_action_weight"],
                        )

    def build_and_fit_state_dual_model(
        self,
        method: str,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Augmented Lagrangian Method (ALM) for state value weight function.

        Parameters
        -------
        method: {"dual_dice", "gen_dice", "algae_dice", "best_dice", "mql", "mwl", "custom"}, default="best_dice"
            Indicates which parameter set should be used. When, "custom" users can specify their own parameter.

        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        n_epochs: int, default=1 (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        random_state: int, default=None
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

        if evaluation_policy.name in self.state_dual_function:
            pass

        else:
            self.state_dual_function[evaluation_policy.name] = []

            for k in range(k_fold):
                if self.action_type == "discrete":
                    self.state_dual_function[evaluation_policy.name].append(
                        DiscreteAugmentedLagrangianStateWightValueLearning(
                            method=method,
                            v_function=VFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            w_function=StateWeightFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            device=self.device,
                        )
                    )
                else:
                    self.state_dual_function[evaluation_policy.name].append(
                        ContinuousAugmentedLagrangianStateWightValueLearning(
                            method=method,
                            v_function=VFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            w_function=StateWeightFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            device=self.device,
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
                        random_state=random_state,
                        **self.model_args["state_dual"],
                    )
                else:
                    self.state_dual_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action=evaluation_policy_action,
                        random_state=random_state,
                        **self.model_args["state_dual"],
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
                    [self.n_trajectories + k // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)

                for k in range(k_fold):
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
                            random_state=random_state,
                            **self.model_args["state_dual"],
                        )
                    else:
                        self.state_dual_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].flatten(),
                            evaluation_policy_action=evaluation_policy_action_,
                            random_state=random_state,
                            **self.model_args["state_dual"],
                        )

    def build_and_fit_state_value_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Minimax V Learning (MVL) for state value function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        n_epochs: int, default=1 (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        random_state: int, default=None
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

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
                            device=self.device,
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
                            device=self.device,
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
                        random_state=random_state,
                        **self.model_args["state_value"],
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
                    [self.n_trajectories + k // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)

                for k in range(k_fold):
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
                            random_state=random_state,
                            **self.model_args["state_value"],
                        )
                    else:
                        self.state_value_function[evaluation_policy.name][k].fit(
                            step_per_trajectory=self.step_per_trajectory,
                            state=self.state_2d[subset_idx_].reshape(
                                (-1, self.state_dim)
                            ),
                            action=action_,
                            reward=self.reward_2d[subset_idx_].flatten(),
                            pscore=self.pscore_2d[subset_idx_].flatten(),
                            evaluation_policy_action=evaluation_policy_action_,
                            random_state=random_state,
                            **self.model_args["state_value"],
                        )

    def build_and_fit_state_weight_model(
        self,
        evaluation_policy: BaseHead,
        k_fold: int = 1,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Minimax Weight Learning (MWL) for state weight function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        n_epochs: int, default=1 (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        random_state: int, default=None
            Random state.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

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
                            ),
                            state_scaler=self.state_scaler,
                            device=self.device,
                        )
                    )
                else:
                    self.state_weight_function[evaluation_policy.name].append(
                        ContinuousMinimaxStateWeightLearning(
                            w_function=StateWeightFunction(
                                state_dim=self.state_dim,
                                hidden_dim=self.model_args["hidden_dim"],
                            ),
                            state_scaler=self.state_scaler,
                            action_scaler=self.action_scaler,
                            device=self.device,
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
                        random_state=random_state,
                        **self.model_args["state_weight"],
                    )
                else:
                    self.state_weight_function[evaluation_policy.name][0].fit(
                        step_per_trajectory=self.step_per_trajectory,
                        state=self.state,
                        action=self.action,
                        reward=self.reward,
                        pscore=self.pscore,
                        evaluation_policy_action=evaluation_policy_action,
                        random_state=random_state,
                        **self.model_args["state_weight"],
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
                    [self.n_trajectories + k // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)

                for k in range(k_fold):
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
                                random_state=random_state,
                                **self.model_args["state_weight"],
                            )
                        else:
                            self.state_weight_function[evaluation_policy.name][k].fit(
                                step_per_trajectory=self.step_per_trajectory,
                                state=self.state_2d[subset_idx_].reshape(
                                    (-1, self.state_dim)
                                ),
                                action=action_,
                                reward=self.reward_2d[subset_idx_].flatten(),
                                pscore=self.pscore_2d[subset_idx_].flatten(),
                                evaluation_policy_action=evaluation_policy_action_,
                                random_state=random_state,
                                **self.model_args["state_weight"],
                            )

    def obtain_initial_state(
        self,
        evaluation_policy: BaseHead,
        resample_initial_state: bool = False,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        random_state: Optional[int] = None,
    ):
        """Obtain initial state distribution (stationary distribution) of evaluation policy.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        resample_initial_state: bool, default=False
            Whether to resample initial state distribution using the given evaluation policy.
            When False, the initial state distribution of the behavior policy is used instead.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout before collecting dataset.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout before collecting dataset.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        evaluation_policy_initial_state: ndarray of shape (n_trajectories, )
            Initial state distribution of evaluation policy.
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
            State to obtain evaluation policy action.
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
        """Obtain action choice probability of the discrete evaluation policy and its Q hat of the observed state.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        state: np.ndarray, default=None
            State to obtain evaluation policy action.
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
        """Obtain Q hat of the observed state and all actions (discrete) or that actions chosen by behavior and (deterministic) evaluation policies (continuous).

        Parameters
        -------
        method: {"fqe", "dice", "mql"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        Return
        -------
        state_action_value_prediction: ndarray of shape (n_trajectories * step_per_trajectory, n_actions) or (n_trajectories * step_per_trajectory, 2)
            When `action_type == "discrete"`, output is state action value for observed state and all actions,
            i.e., math`\\hat{Q}(s, a) \\forall a \\in \\mathcal{A}`.

            When `action_type == "continuous"`, output is state action value for the observed action and action chosen evaluation policy,
            i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.

        """
        if method not in ["fqe", "dice", "mql"]:
            raise ValueError(
                f"method must be either 'fqe', 'dice', or 'mql', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        idx = np.array([self.n_trajectories + k // k_fold for k in range(k_fold)])
        idx = np.insert(idx, 0, 0)

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

                elif method == "dice":
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

                elif method == "dice":
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
        """Obtain the initial state value of the discrete evaluation policy.

        Parameters
        -------
        method: {"fqe", "dice_q", "dice_v", "mql", "mvl"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

        resample_initial_state: bool, default=False
            Whether to resample initial state distribution using the given evaluation policy.
            When False, the initial state distribution of the behavior policy is used instead.

        minimum_rollout_length: int, default=0 (>= 0)
            Minimum length of rollout before collecting dataset.

        maximum_rollout_length: int, default=100 (>= minimum_rollout_length)
            Maximum length of rollout before collecting dataset.

        random_state: int, default=None (>= 0)
            Random state.

        Return
        -------
        initial_state_value_prediction: ndarray of shape (n_trajectories, )
            State action value of the observed state.

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

                elif method == "dice":
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
                    [self.n_trajectories + k // k_fold for k in range(k_fold)]
                )
                idx = np.insert(idx, 0, 0)

                state_value = np.zeros((self.n_trajectories, self.step_per_trajectory))

                for k in range(k_fold):
                    idx_ = np.arange(idx[k], idx[k + 1])
                    state_ = self.state_2d[idx_].reshape((-1, self.state_dim))

                    state_value_ = self.state_value_function[evaluation_policy.name][
                        k
                    ].predict(state_)
                    state_value[idx_] = state_value_

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
            Number of folds for cross-fitting.

        Return
        -------
        state_action_weight_prediction: ndarray of shape (n_trajectories * step_per_trajectory, )
            State-action marginal importance weight for observed state and the action chosen by the behavior policy.

        """
        if method not in ["dice", "mwl"]:
            raise ValueError(
                f"method must be either 'dice' or 'mwl', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        idx = np.array([self.n_trajectories + k // k_fold for k in range(k_fold)])
        idx = np.insert(idx, 0, 0)

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
            Number of folds for cross-fitting.

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

        idx = np.array([self.n_trajectories + k // k_fold for k in range(k_fold)])
        idx = np.insert(idx, 0, 0)

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

    def obtain_whole_inputs(
        self,
        evaluation_policies: List[BaseHead],
        require_value_prediction: bool = False,
        require_weight_prediction: bool = False,
        resample_initial_state: bool = False,
        q_function_method: str = "fqe",
        v_function_method: str = "fqe",
        w_function_method: str = "mwl",
        k_fold: int = 1,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        n_trajectories_on_policy_evaluation: int = 100,
        use_stationary_distribution_on_policy_evaluation: bool = False,
        minimum_rollout_length: int = 0,
        maximum_rollout_length: int = 100,
        random_state: Optional[int] = None,
    ) -> OPEInputDict:
        """Obtain input as a dictionary.

        Parameters
        -------
        evaluation_policies: list of BaseHead
            Evaluation policies.

        require_value_prediction: bool, default=False
            Whether to obtain value prediction.

        require_weight_prediction: bool, default=False
            Whether to obtain weight prediction.

        resample_initial_state: bool, default=False
            Whether to resample initial state distribution using the given evaluation policy.
            When False, the initial state distribution of the behavior policy is used instead.

            Note that, this parameter is applicable only when self.env is given.

        q_function_method: {"fqe", "dice", "mql"}
            Estimation method of :math:`Q(s, a)`.

        v_function_method: {"fqe", "dice_q", "dice_v", "mql", "mvl"}
            Estimation method of :math:`V(s)`.

        w_function_method: {"dice", "mwl"}
            Estimation method of :math:`w(s, a)` and :math:`w(s)`.

        k_fold: int, default=1 (> 0)
            Number of folds for cross-fitting.

            If :math:`K>1`, we split the logged dataset into :matk:`K` folds.
            :math:`\\mathcal{D}_j` is the :math:`j`-th split of logged data consisting of :math:`n_k` samples.
            Then, the value and weight functions (:math:`w^j` and :math:`Q^j`) are trained on the subset of data used for OPE,
            i.e., :math:`\\mathcal{D} \\setminus \\mathcal{D}_j`.

            If :math:`K=1`, the value and weight functions are trained on the entire data.

        n_epochs: int, default=None (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=None (> 0)
            Number of steps in an epoch.

        n_trajectories_on_policy_evaluation: int, default=None (> 0)
            Number of episodes to perform on-policy evaluation.

        use_stationary_distribution_on_policy_evaluation: bool, default=False
            Whether to evaluate policy on stationary distribution.
            When True, evaluation policy is evaluated by rollout without resetting environment at each episode.

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

            evaluation_policy_action: ndarray of shape (n_trajectories * step_per_trajectorys, action_dim)
                Action chosen by the deterministic evaluation policy.
                If `action_type == "discrete"`, `None` is recorded.

            evaluation_policy_action_dist: ndarray of shape (n_trajectories * step_per_trajectory, n_actions)
                Conditional action distribution induced by the evaluation policy,
                i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`
                If `action_type == "continuous"`, `None` is recorded.

            state_action_value_prediction: ndarray
                If `action_type == "discrete"`, :math:`\\hat{Q}` for all actions,
                i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.
                shape (n_trajectories * step_per_trajectory, n_actions)

                If `action_type == "continuous"`, :math:`\\hat{Q}` for the observed action and action chosen evaluation policy,
                i.e., (row 0) :math:`\\hat{Q}(s_t, a_t)` and (row 2) :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.
                shape (n_trajectories * step_per_trajectory, 2)

                If `require_value_prediction == False`, `None` is recorded.

            initial_state_value_prediction: ndarray of shape (n_trajectories, )
                Estimated initial state value.

                If `use_base_model == False`, `None` is recorded.

            state_action_marginal_importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
                Estimated state-action marginal importance weight,
                i.e., :math:`\\hat{w}(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_0}(s_t, a_t)`.

                If `require_weight_prediction == False`, `None` is recorded.

            state_marginal_importance_weight: ndarray of shape (n_trajectories * step_per_trajectory, )
                Estimated state marginal importance weight,
                i.e., :math:`\\hat{w}(s_t) \\approx d^{\\pi}(s_t) / d^{\\pi_0}(s_t)`.

                If `require_weight_prediction == False`, `None` is recorded.

            on_policy_policy_value: ndarray of shape (n_trajectories_on_policy_evaluation, )
                On-policy policy value.
                If `self.env is None`, `None` is recorded.

            gamma: float
                Discount factor.

        """
        if resample_initial_state and self.env is None:
            warn(
                "resample_initial_state is True, but self.env is not given. Thus, initial_state will not be resampled."
                "Please initialize CreateInput class with self.env to resample initial state."
            )

        for eval_policy in evaluation_policies:
            if eval_policy.action_type != self.action_type:
                raise RuntimeError(
                    f"One of the evaluation_policies, {eval_policy.name} does not match action_type in logged_dataset."
                    " Please use {self.action_type} action type instead."
                )

        if n_trajectories_on_policy_evaluation is not None:
            check_scalar(
                n_trajectories_on_policy_evaluation,
                name="n_trajectories_on_policy_evaluation",
                target_type=int,
                min_val=1,
            )

        if require_value_prediction:

            if q_function_method == "fqe" or v_function_method == "fqe":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit FQE model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_FQE(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                    )

            if q_function_method == "dice" or v_function_method == "dice_q":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit Augmented Lagrangian model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_action_dual_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
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
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
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
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
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
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                        random_state=random_state,
                    )

        if require_weight_prediction:

            if w_function_method == "dual":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit Augmented Lagrangian model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_action_dual_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                        random_state=random_state,
                    )
                    self.build_and_fit_state_dual_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
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
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                        random_state=random_state,
                    )
                    self.build_and_fit_state_weight_model(
                        evaluation_policies[i],
                        k_fold=k_fold,
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                        random_state=random_state,
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

        return defaultdict_to_dict(input_dict)
