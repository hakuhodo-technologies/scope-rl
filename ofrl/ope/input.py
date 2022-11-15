"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from collections import defaultdict
from tqdm.auto import tqdm

import torch
import numpy as np
from sklearn.utils import check_scalar

import gym
from gym.spaces import Box, Discrete
from d3rlpy.dataset import MDPDataset
from d3rlpy.ope import DiscreteFQE
from d3rlpy.ope import FQE as ContinuousFQE
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory

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
)


@dataclass
class CreateOPEInput:
    """Class to prepare OPE inputs.

    Parameters
    -------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

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
    model_args: Optional[Dict[str, Any]] = None
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        "Initialize class."
        check_logged_dataset(self.logged_dataset)
        self.n_episodes = self.logged_dataset["n_episodes"]
        self.action_type = self.logged_dataset["action_type"]
        self.n_actions = self.logged_dataset["n_actions"]
        self.action_dim = self.logged_dataset["action_dim"]
        self.state_dim = self.logged_dataset["state_dim"]
        self.step_per_episode = self.logged_dataset["step_per_episode"]

        if self.logged_dataset["action_type"] == "discrete":
            self.mdp_dataset = MDPDataset(
                observations=self.logged_dataset["state"],
                actions=self.logged_dataset["action"],
                rewards=self.logged_dataset["reward"],
                terminals=self.logged_dataset["done"],
                episode_terminals=self.logged_dataset["terminal"],
                discrete_action=True,
            )
        else:
            self.mdp_dataset = MDPDataset(
                observations=self.logged_dataset["state"],
                actions=self.logged_dataset["action"],
                rewards=self.logged_dataset["reward"],
                terminals=self.logged_dataset["done"],
                episode_terminals=self.logged_dataset["terminal"],
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
        if self.model_args["state_action_dual"]:
            self.model_args["state_action_dual"] = {}

        self.state_action_value_function = {}
        if self.model_args["state_action_value"]:
            self.model_args["state_action_value"] = {}

        self.state_action_weight_function = {}
        if self.model_args["state_action_weight"]:
            self.model_args["state_action_weight"] = {}

        self.state_dual_function = {}
        if self.model_args["state_dual"]:
            self.model_args["state_dual"] = {}

        self.state_value_function = {}
        if self.model_args["state_value"]:
            self.model_args["state_value"] = {}

        self.state_weight_function = {}
        if self.model_args["state_weight"]:
            self.model_args["state_weight"] = {}

    def build_and_fit_FQE(
        self,
        evaluation_policy: BaseHead,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
    ) -> None:
        """Fit Fitted Q Evaluation (FQE).

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

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
                n_steps_per_epoch=n_steps_per_epoch,
                scorers={},
            )

    def build_and_fit_state_action_dual_model(
        self,
        method: str,
        evaluation_policy: BaseHead,
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
            if self.action_type == "discrete":
                self.state_action_dual_function[
                    evaluation_policy.name
                ] = DiscreteAugmentedLagrangianStateActionWightValueLearning(
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
                    device=self.device,
                )
                self.state_action_dual_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    evaluation_policy_action_dist=self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_action_dual"],
                )
            else:
                self.state_action_dual_function[
                    evaluation_policy.name
                ] = ContinuousAugmentedLagrangianStateActionWightValueLearning(
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
                    device=self.device,
                )
                self.state_action_dual_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    evaluation_policy_action=self.obtain_evaluation_policy_action(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_action_dual"],
                )

    def build_and_fit_state_action_value_model(
        self,
        evaluation_policy: BaseHead,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Minimax Q Learning (MQL) for state-action value function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

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
            if self.action_type == "discrete":
                self.state_action_value_function[
                    evaluation_policy.name
                ] = DiscreteMinimaxStateActionValueLearning(
                    q_function=DiscreteQFunction(
                        n_actions=self.n_actions,
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                        device=self.device,
                    ),
                    device=self.device,
                )
                self.state_action_value_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    pscore=self.logged_dataset["pscore"],
                    evaluation_policy_action_dist=self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_action_value"],
                )
            else:
                self.state_action_value_function[
                    evaluation_policy.name
                ] = ContinuousMinimaxStateActionValueLearning(
                    q_function=ContinuousQFunction(
                        action_dim=self.action_dim,
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    device=self.device,
                )
                self.state_action_value_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    pscore=self.logged_dataset["pscore"],
                    evaluation_policy_action=self.obtain_evaluation_policy_action(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_action_value"],
                )

    def build_and_fit_state_action_weight_model(
        self,
        evaluation_policy: BaseHead,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Minimax Weight Learning (MWL) for state-action weight function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

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
            if self.action_type == "discrete":
                self.state_action_weight_function[
                    evaluation_policy.name
                ] = DiscreteMinimaxStateActionWeightLearning(
                    w_function=DiscreteStateActionWeightFunction(
                        n_actions=self.n_actions,
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                        device=self.device,
                    ),
                    device=self.device,
                )
                self.state_action_weight_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    evaluation_policy_action_dist=self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_action_weight"],
                )
            else:
                self.state_action_weight_function[
                    evaluation_policy.name
                ] = ContinuousMinimaxStateActionWeightLearning(
                    w_function=ContinuousStateActionWeightFunction(
                        action_dim=self.action_dim,
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    device=self.device,
                )
                self.state_action_weight_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    evaluation_policy_action=self.obtain_evaluation_policy_action(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_action_weight"],
                )

    def build_and_fit_state_dual_model(
        self,
        method: str,
        evaluation_policy: BaseHead,
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
            if self.action_type == "discrete":
                self.state_dual_function[
                    evaluation_policy.name
                ] = DiscreteAugmentedLagrangianStateWightValueLearning(
                    method=method,
                    v_function=VFunction(
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    w_function=StateWeightFunction(
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    device=self.device,
                )
                self.state_dual_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    pscore=self.logged_dataset["pscore"],
                    evaluation_policy_action_dist=self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_dual"],
                )
            else:
                self.state_dual_function[
                    evaluation_policy.name
                ] = ContinuousAugmentedLagrangianStateWightValueLearning(
                    method=method,
                    v_function=VFunction(
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    w_function=StateWeightFunction(
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    device=self.device,
                )
                self.state_dual_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    pscore=self.logged_dataset["pscore"],
                    evaluation_policy_action=self.obtain_evaluation_policy_action(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_dual"],
                )

    def build_and_fit_state_value_model(
        self,
        evaluation_policy: BaseHead,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Minimax V Learning (MVL) for state value function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

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
            if self.action_type == "discrete":
                self.state_value_function[
                    evaluation_policy.name
                ] = DiscreteMinimaxStateValueLearning(
                    v_function=VFunction(
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    device=self.device,
                )
                self.state_value_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    pscore=self.logged_dataset["pscore"],
                    evaluation_policy_action_dist=self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_value"],
                )
            else:
                self.state_value_function[
                    evaluation_policy.name
                ] = ContinuousMinimaxStateValueLearning(
                    v_function=VFunction(
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    device=self.device,
                )
                self.state_value_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    pscore=self.logged_dataset["pscore"],
                    evaluation_policy_action=self.obtain_evaluation_policy_action(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_value"],
                )

    def build_and_fit_state_weight_model(
        self,
        evaluation_policy: BaseHead,
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit Minimax Weight Learning (MWL) for state weight function.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

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
            if self.action_type == "discrete":
                self.state_weight_function[
                    evaluation_policy.name
                ] = DiscreteMinimaxStateWeightLearning(
                    w_function=StateWeightFunction(
                        n_actions=self.n_actions,
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    device=self.device,
                )
                self.state_weight_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    pscore=self.logged_dataset["pscore"],
                    evaluation_policy_action_dist=self.obtain_evaluation_policy_action_dist(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_weight"],
                )
            else:
                self.state_weight_function[
                    evaluation_policy.name
                ] = ContinuousMinimaxStateWeightLearning(
                    w_function=StateWeightFunction(
                        action_dim=self.action_dim,
                        state_dim=self.state_dim,
                        hidden_dim=self.model_args["hidden_dim"],
                    ),
                    device=self.device,
                )
                self.state_weight_function[evaluation_policy.name].fit(
                    state=self.logged_dataset["state"],
                    action=self.logged_dataset["action"],
                    reward=self.logged_dataset["reward"],
                    pscore=self.logged_dataset["pscore"],
                    evaluation_policy_action=self.obtain_evaluation_policy_action(
                        evaluation_policy=evaluation_policy,
                    ),
                    random_state=random_state,
                    **self.model_args["state_weight"],
                )

    def obtain_evaluation_policy_action(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain evaluation policy action.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_action: ndarray of shape (n_episodes * step_per_episode, )
            Evaluation policy action :math:`a_t \\sim \\pi(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        return evaluation_policy.predict(x=self.logged_dataset["state"])

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
        evaluation_policy_pscore: ndarray of shape (n_episodes * step_per_episode, )
            Evaluation policy pscore :math:`\\pi(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        return evaluation_policy.calc_pscore_given_action(
            x=self.logged_dataset["state"],
            action=self.logged_dataset["action"],
        )

    def obtain_evaluation_policy_action_dist(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain action choice probability of the discrete evaluation policy and its Q hat of the observed state.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_action_dist: ndarray of shape (n_episodes * step_per_episode, n_actions)
            Evaluation policy pscore :math:`\\pi(a_t \\mid s_t)`.

        state_action_value_prediction: ndarray of shape (n_episodes * step_per_episode, n_actions)
            State action value for all observed state and possible action.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        action_dist = evaluation_policy.calc_action_choice_probability(
            self.logged_dataset["state"]
        )

        return action_dist

    def obtain_state_action_value_prediction_discrete(
        self,
        method: str,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain Q hat of the observed state and all actions (discrete).

        Parameters
        -------
        method: {"fqe", "dice", "mql"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        state_action_value_prediction: ndarray of shape (n_episodes * step_per_episode, n_actions)
            State action value for observed state and all actions,
            i.e., math`\\hat{Q}(s, a) \\forall a \\in \\mathcal{A}`.

        """
        if method not in ["fqe", "dice", "mql"]:
            raise ValueError(
                f"method must be either 'fqe', 'dice', or 'mql', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        x = self.logged_dataset["state"]

        x_ = []
        for i in range(x.shape[0]):
            x_.append(np.tile(x[i], (self.n_actions, 1)))
        x_ = np.array(x_).reshape((-1, x.shape[1]))
        a_ = np.tile(np.arange(self.n_actions), x.shape[0])

        if method == "fqe":
            state_action_value_prediction = self.fqe[
                evaluation_policy.name
            ].predict_value(x_, a_)
        elif method == "dice":
            state_action_value_prediction = self.state_action_dual_function[
                evaluation_policy.name
            ].predict_value(x_, a_)
        elif method == "mql":
            state_action_value_prediction = self.state_action_value_function[
                evaluation_policy.name
            ].predict_value(x_, a_)

        return state_action_value_prediction  # (n_samples, n_actions)

    def obtain_state_action_value_prediction_continuous(
        self,
        method: str,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain Q hat for the continuous (deterministic) evaluation policy.

        Parameters
        -------
        method: {"fqe", "dice", "mql"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        state_action_value_prediction: ndarray of shape (n_episodes * step_per_episode, )
            State action value of the observed state and action chosen by the evaluation policy.

        """
        if method not in ["fqe", "dice", "minimax"]:
            raise ValueError(
                f"method must be either 'fqe', 'dice', or 'mql', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        state = self.logged_dataset["state"]
        action = evaluation_policy.predict(state)

        if method == "fqe":
            state_action_value_prediction = self.fqe[
                evaluation_policy.name
            ].predict_value(state, action)
        elif method == "dice":
            state_action_value_prediction = self.state_action_dual_function[
                evaluation_policy.name
            ].predict_value(state, action)
        elif method == "mql":
            state_action_value_prediction = self.state_action_value_function[
                evaluation_policy.name
            ].predict_value(state, action)

        return state_action_value_prediction  # (n_samples, n_actions)

    def obtain_initial_state_value_prediction_discrete(
        self,
        method: str,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain the initial state value of the discrete evaluation policy.

        Parameters
        -------
        method: {"fqe", "dice_q", "dice_v", "mql", "mvl"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        initial_state_value_prediction: ndarray of shape (n_episodes, )
            State action value of the observed state.

        """
        if method not in ["fqe", "dice_q", "dice_v", "mql", "mvl"]:
            raise ValueError(
                f"method must be either 'fqe', 'dice_q', 'dice_v', 'mql', or 'mvl', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        if method in ["fqe", "dice_q", "mql"]:
            action_dist = self.obtain_evaluation_policy_action_dist(evaluation_policy)
            state_action_value = self.obtain_state_action_value_prediction_discrete(
                method=method,
                evaluation_policy=evaluation_policy,
            )
            state_value = np.sum(state_action_value * action_dist, axis=1)
        else:
            state_value = self.state_value_function[evaluation_policy.name].predict(
                state=self.logged_dataset["state"],
            )

        return state_value.reshape((-1, self.step_per_episode))[:, 0]  # (n_episodes, )

    def obtain_initial_state_value_prediction_continuous(
        self,
        method: str,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain the initial state value of the (deterministic) continuous evaluation policy.

        Parameters
        -------
        method: {"fqe", "dice_q", "dice_v", "mql", "mvl"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        initial_state_value_prediction: ndarray of shape (n_episodes, )
            State action value of the observed state.

        """
        if method not in ["fqe", "dice_q", "dice_v", "mql", "mvl"]:
            raise ValueError(
                f"method must be either 'fqe', 'dice_q', 'dice_v', 'mql', or 'mvl', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        if method in ["fqe", "dice_q", "mql"]:
            state_value = self.obtain_state_action_value_prediction_continuous(
                method=method,
                evaluation_policy=evaluation_policy,
            )
        else:
            state_value = self.state_value_function[evaluation_policy.name].predict(
                state=self.logged_dataset["state"],
            )

        return state_value.reshape((-1, self.step_per_episode))[:, 0]

    def obtain_state_action_marginal_importance_weight(
        self,
        method: str,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Predict state-action marginal importance weight.

        Parameters
        -------
        method: {"dice", "mwl"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        state_action_weight_prediction: ndarray of shape (n_episodes * step_per_episode, )
            State-action marginal importance weight for observed state and the action chosen by the behavior policy.

        """
        if method not in ["dice", "mwl"]:
            raise ValueError(
                f"method must be either 'dice' or 'mwl', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        state = self.logged_dataset["state"]
        action = self.logged_dataset["action"]

        if method == "dice":
            state_action_weight_prediction = self.state_action_dual_function[
                evaluation_policy.name
            ].predict_weight(state, action)
        elif method == "mwl":
            state_action_weight_prediction = self.state_action_weight_function[
                evaluation_policy.name
            ].predict_weight(state, action)

        return state_action_weight_prediction

    def obtain_state_marginal_importance_weight(
        self,
        method: str,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Predict state marginal importance weight.

        Parameters
        -------
        method: {"dice", "mwl"}
            Estimation method.

        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        state_weight_prediction: ndarray of shape (n_episodes * step_per_episode, )
            State marginal importance weight for observed state.

        """
        if method not in ["dice", "mwl"]:
            raise ValueError(
                f"method must be either 'dice' or 'mwl', but {method} is given"
            )
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        state = self.logged_dataset["state"]

        if method == "dice":
            state_weight_prediction = self.state_dual_function[
                evaluation_policy.name
            ].predict_weight(state)
        elif method == "mwl":
            state_weight_prediction = self.state_weight_function[
                evaluation_policy.name
            ].predict_weight(state)

        return state_weight_prediction

    def obtain_whole_inputs(
        self,
        evaluation_policies: List[BaseHead],
        env: Optional[gym.Env] = None,
        require_value_prediction: bool = False,
        require_weight_prediction: bool = False,
        q_function_method: str = "fqe",
        v_function_method: str = "fqe",
        w_function_method: str = "mwl",
        n_epochs: int = 1,
        n_steps_per_epoch: int = 10000,
        n_episodes_on_policy_evaluation: int = 100,
        gamma: float = 1.0,
        random_state: Optional[int] = None,
    ) -> OPEInputDict:
        """Obtain input as a dictionary.

        Parameters
        -------
        evaluation_policies: list of BaseHead
            Evaluation policies.

        env: gym.Env
            Reinforcement learning (RL) environment.

        require_value_prediction: bool, default=False
            Whether to obtain value prediction.

        require_weight_prediction:bool, default=False
            Whether to obtain weight prediction.

        q_function_method: {"fqe", "dice", "mql"}
            Estimation method of :math:`Q(s, a)`.

        v_function_method: {"fqe", "dice_q", "dice_v", "mql", "mvl"}
            Estimation method of :math:`V(s)`.

        w_function_method: {"dice", "mwl"}
            Estimation method of :math:`w(s, a)` and :math:`w(s)`.

        n_epochs: int, default=None (> 0)
            Number of epochs to fit FQE.

        n_steps_per_epoch: int, default=None (> 0)
            Number of steps in an epoch.

        n_episodes_on_policy_evaluation: int, default=None (> 0)
            Number of episodes to perform on-policy evaluation.

        gamma: float, default=1.0
            Discount factor. The value should be within `(0, 1]`.

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

            evaluation_policy_action: ndarray of shape (n_episodes * step_per_episodes, action_dim)
                Action chosen by the deterministic evaluation policy.
                If `action_type == "discrete"`, `None` is recorded.

            evaluation_policy_action_dist: ndarray of shape (n_episodes * step_per_episode, n_actions)
                Conditional action distribution induced by the evaluation policy,
                i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`
                If `action_type == "continuous"`, `None` is recorded.

            state_action_value_prediction: ndarray
                If `action_type == "discrete"`, :math:`\\hat{Q}` for all actions,
                i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.
                shape (n_episodes * step_per_episode, n_actions)

                If `action_type == "continuous"`, :math:`\\hat{Q}` for the action chosen by the evaluation policy,
                i.e., :math:`\\hat{Q}(s_t, \\pi(s_t))`.
                shape (n_episodes * step_per_episode, )

                If `require_value_prediction == False`, `None` is recorded.

            initial_state_value_prediction: ndarray of shape (n_episodes, )
                Estimated initial state value.

                If `use_base_model == False`, `None` is recorded.

            state_action_marginal_importance_weight: ndarray of shape (n_episodes * step_per_episode, )
                Estimated state-action marginal importance weight,
                i.e., :math:`\\hat{w}(s_t, a_t) \\approx d^{\\pi}(s_t, a_t) / d^{\\pi_0}(s_t, a_t)`.

                If `require_weight_prediction == False`, `None` is recorded.

            state_marginal_importance_weight: ndarray of shape (n_episodes * step_per_episode, )
                Estimated state marginal importance weight,
                i.e., :math:`\\hat{w}(s_t) \\approx d^{\\pi}(s_t) / d^{\\pi_0}(s_t)`.

                If `require_weight_prediction == False`, `None` is recorded.

            on_policy_policy_value: ndarray of shape (n_episodes_on_policy_evaluation, )
                On-policy policy value.
                If `env is None`, `None` is recorded.

            gamma: float
                Discount factor.

        """
        if env is not None:
            if isinstance(env.action_space, Box) and self.action_type == "discrete":
                raise RuntimeError(
                    "Found mismatch in action_type between env and logged_dataset"
                )
            elif (
                isinstance(env.action_space, Discrete)
                and self.action_type == "continuous"
            ):
                raise RuntimeError(
                    "Found mismatch in action_type between env and logged_dataset"
                )

        for eval_policy in evaluation_policies:
            if eval_policy.action_type != self.action_type:
                raise RuntimeError(
                    f"One of the evaluation_policies, {eval_policy.name} does not match action_type in logged_dataset."
                    " Please use {self.action_type} action type instead."
                )

        if n_episodes_on_policy_evaluation is not None:
            check_scalar(
                n_episodes_on_policy_evaluation,
                name="n_episodes_on_policy_evaluation",
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
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                    )

            if v_function_method == "dice_v":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit Augmented Lagrangian model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_dual_model(
                        evaluation_policies[i],
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                    )

            if q_function_method == "mql" or v_function_method == "mql":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit MQL model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_action_value_model(
                        evaluation_policies[i],
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                    )

            if v_function_method == "mvl":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit MVL model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_value_model(
                        evaluation_policies[i],
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
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
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                    )
                    self.build_and_fit_state_dual_model(
                        evaluation_policies[i],
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                    )

            elif w_function_method == "mwl":
                for i in tqdm(
                    range(len(evaluation_policies)),
                    desc="[fit MWL model]",
                    total=len(evaluation_policies),
                ):
                    self.build_and_fit_state_action_dual_model(
                        evaluation_policies[i],
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
                    )
                    self.build_and_fit_state_dual_model(
                        evaluation_policies[i],
                        n_epochs=n_epochs,
                        n_steps_per_epoch=n_steps_per_epoch,
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
            if self.action_type == "discrete":
                if require_value_prediction:
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = self.obtain_state_action_value_prediction_discrete(
                        method=q_function_method,
                        evaluation_policy=evaluation_policies[i],
                    )
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = self.obtain_initial_state_value_prediction_discrete(
                        method=v_function_method,
                        evaluation_policy=evaluation_policies[i],
                    )
                else:
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = None
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = None
            else:
                if require_value_prediction:
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = self.obtain_state_action_value_prediction_continuous(
                        method=q_function_method,
                        evaluation_policy=evaluation_policies[i],
                    )
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = self.obtain_initial_state_value_prediction_continuous(
                        method=v_function_method,
                        evaluation_policy=evaluation_policies[i],
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
                )
                input_dict[evaluation_policies[i].name][
                    "state_marginal_importance_weight"
                ] = self.obtain_state_marginal_importance_weight(
                    method=w_function_method,
                    evaluation_policy=evaluation_policies[i],
                )
            else:
                input_dict[evaluation_policies[i].name][
                    "state_action_marginal_importance_weight"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "state_marginal_importance_weight"
                ] = None

            # input for the evaluation of OPE estimators
            if env is not None:
                if n_episodes_on_policy_evaluation is None:
                    n_episodes_on_policy_evaluation = self.n_episodes

                input_dict[evaluation_policies[i].name][
                    "on_policy_policy_value"
                ] = rollout_policy_online(
                    env,
                    evaluation_policies[i],
                    n_episodes=n_episodes_on_policy_evaluation,
                    gamma=gamma,
                    random_state=random_state,
                )

            else:
                input_dict[evaluation_policies[i].name]["on_policy_policy_value"] = None

            input_dict[evaluation_policies[i].name]["gamma"] = gamma

        return defaultdict_to_dict(input_dict)
