"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from collections import defaultdict
from tqdm.autonotebook import tqdm

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

    base_model_args: dict, default=None
        Arguments of baseline Fitted Q Evaluation (FQE) model.

    use_base_model: bool, default=False
        Whether to use FQE and obtain :math:`\\hat{Q}`.

    References
    -------
    Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita.
    "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.", 2021.

    Takuma Seno and Michita Imai.
    "d3rlpy: An Offline Deep Reinforcement Library.", 2021.

    Hoang Le, Cameron Voloshin, and Yisong Yue.
    "Batch Policy Learning under Constraints.", 2019.

    """

    logged_dataset: LoggedDataset
    base_model_args: Optional[Dict[str, Any]] = None
    use_base_model: bool = False

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

        if self.use_base_model:
            self.fqe = {}
            if self.base_model_args is None:
                self.base_model_args = {
                    "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
                    "q_func_factory": MeanQFunctionFactory(),
                    "learning_rate": 1e-4,
                    "use_gpu": torch.cuda.is_available(),
                }

    def build_and_fit_FQE(
        self,
        evaluation_policy: BaseHead,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,  # should be more than n_steps_per_epoch
        n_steps_per_epoch: int = 10000,
    ) -> None:
        """Fit Fitted Q Evaluation (FQE).

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        n_epochs: int, default=None (> 0)
            Number of epochs to fit FQE.

        n_steps: int, default=None (> 0)
            Total number pf steps to fit FQE.

        n_steps_per_epoch: int, default=10000 (> 0)
            Number of steps in an epoch.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")

        if n_epochs is not None:
            check_scalar(n_epochs, name="n_epochs", target_type=int, min_val=1)
        if n_steps is not None:
            check_scalar(n_steps, name="n_steps", target_type=int, min_val=1)
        check_scalar(
            n_steps_per_epoch, name="n_steps_per_epoch", target_type=int, min_val=1
        )

        if n_epochs is None and n_steps is None:
            n_steps = n_steps_per_epoch

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
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                scorers={},
            )

    def predict_state_action_value(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Predict state action value for all actions.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        state_action_value_prediction: ndarray of shape (n_episodes * step_per_episode, n_actions)
            State action value for observed state and all actions,
            i.e., math`\\hat{Q}(s, a) \\forall a \\in \\mathcal{A}`.

        """
        x = self.logged_dataset["state"]
        x_ = []
        for i in range(x.shape[0]):
            x_.append(np.tile(x[i], (self.n_actions, 1)))
        x_ = np.array(x_).reshape((-1, x.shape[1]))
        a_ = np.tile(np.arange(self.n_actions), x.shape[0])
        return self.fqe[evaluation_policy.name].predict_value(
            x_, a_
        )  # (n_samples, n_actions)

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

    def obtain_pscore_for_observed_state_action(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain pscore for observed state action pair.

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

    def obtain_step_wise_pscore(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain step-wise pscore for the observed state action pair.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_step_wise_pscore: ndarray of shape (n_episodes * step_per_episode, )
            Evaluation policy's step-wise pscore :math:`\\prod_{t'=1}^t \\pi(a_{t'} \\mid s_{t'})`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        base_pscore = self.obtain_pscore_for_observed_state_action(
            evaluation_policy
        ).reshape((-1, self.step_per_episode))
        return np.cumprod(base_pscore, axis=1).flatten()

    def obtain_trajectory_wise_pscore(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain trajectory-wise pscore for the observed state action pair.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        evaluation_policy_trajectory_wise_pscore: ndarray of shape (n_episodes * step_per_episode, )
            Evaluation policy's trajectory-wise pscore :math:`\\prod_{t=1}^T \\pi(a_t \\mid s_t)`.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        base_pscore = self.obtain_step_wise_pscore(evaluation_policy).reshape(
            (-1, self.step_per_episode)
        )[:, -1]
        return np.tile(base_pscore, (self.step_per_episode, 1)).T.flatten()

    def obtain_action_dist_with_state_action_value_prediction_discrete(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain action choice probability of the discrete evaluation policy and its Q hat for the observed state.

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
        state_action_value_prediction = (
            self.predict_state_action_value(evaluation_policy)
        ).reshape((-1, self.n_actions))
        return action_dist, state_action_value_prediction  # (n_samples, n_actions)

    def obtain_state_action_value_prediction_continuous(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain Q hat for the continuous (deterministic) evaluation policy.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        state_action_value_prediction: ndarray of shape (n_episodes * step_per_episode, )
            State action value for the observed state and action chosen by evaluation policy.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        state = self.logged_dataset["state"]
        action = evaluation_policy.predict(state)
        return self.fqe[evaluation_policy.name].predict_value(state, action)

    def obtain_initial_state_value_prediction_discrete(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain initial state value for the discrete evaluation policy.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        initial_state_value_prediction: ndarray of shape (n_episodes, n_actions)
            State action value for the observed state and action chosen by evaluation policy.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        (
            state_action_value,
            pscore,
        ) = self.obtain_action_dist_with_state_action_value_prediction_discrete(
            evaluation_policy
        )
        state_action_value = state_action_value.reshape((-1, self.n_actions))
        state_value = np.sum(state_action_value * pscore, axis=1)
        return state_value.reshape((-1, self.step_per_episode))[:, 0]  # (n_samples, )

    def obtain_initial_state_value_prediction_continuous(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain initial state value for the continuous evaluation policy.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        initial_state_value_prediction: ndarray of shape (n_episodes, )
            State action value for the observed state and action chosen by evaluation policy.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        state_value = self.obtain_state_action_value_prediction_continuous(
            evaluation_policy
        )
        return state_value.reshape((-1, self.step_per_episode))[:, 0]

    def obtain_initial_state_action_distribution(
        self,
        evaluation_policy: BaseHead,
    ) -> np.ndarray:
        """Obtain Evaluation policy pscore of discrete actions at the initial state of each episode.

        Parameters
        -------
        evaluation_policy: BaseHead
            Evaluation policy.

        Return
        -------
        initial_state_action_distribution: ndarray of shape (n_episodes, n_actions)
            Evaluation policy pscore at the initial state of each episode.

        """
        if not isinstance(evaluation_policy, BaseHead):
            raise ValueError("evaluation_policy must be a child class of BaseHead")
        state = self.logged_dataset["state"].reshape(
            (-1, self.step_per_episode, self.state_dim)
        )
        action_dist = evaluation_policy.calc_action_choice_probability(state[:, 0, :])
        return action_dist

    def obtain_whole_inputs(
        self,
        evaluation_policies: List[BaseHead],
        env: Optional[gym.Env] = None,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,  # should be more than n_steps_per_epoch
        n_steps_per_epoch: Optional[int] = None,
        n_episodes_on_policy_evaluation: Optional[int] = None,
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

        n_epochs: int, default=None (> 0)
            Number of epochs to fit FQE.

        n_steps: int, default=None (> 0)
            Total number pf steps to fit FQE.

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

            evaluation_policy_step_wise_pscore: ndarray of shape (n_episodes * step_per_episodes, )
                Step-wise action choice probability of evaluation policy,
                i.e., :math:`\\prod_{t'=0}^t \\pi(a_{t'} \\mid s_{t'})`
                If `action_type == "continuous"`, `None` is recorded.

            evaluation_policy_trajectory_wise_pscore: ndarray of shape (n_episodes * step_per_episodes, )
                Trajectory-wise action choice probability of evaluation policy,
                i.e., :math:`\\prod_{t=0}^T \\pi(a_t \\mid s_t)`
                If `action_type == "continuous"`, `None` is recorded.

            evaluation_policy_action: ndarray of shape (n_episodes * step_per_episodes, action_dim)
                Action chosen by the deterministic evaluation policy.
                If `action_type == "discrete"`, `None` is recorded.

            evaluation_policy_action_dist: ndarray of shape (n_episodes * step_per_episode, n_actions)
                Action choice probability of evaluation policy for all actions,
                i.e., :math:`\\pi(a \\mid s_t) \\forall a \\in \\mathcal{A}`
                If `action_type == "continuous"`, `None` is recorded.

            state_action_value_prediction: ndarray of shape (n_episodes * step_per_episode, n_actions) or (n_episodes * step_per_episode, )
                If `action_type == "discrete"`, :math:`\\hat{Q}` for all actions,
                i.e., :math:`\\hat{Q}(s_t, a) \\forall a \\in \\mathcal{A}`.
                shape (n_episodes * step_per_episode, n_actions)

                If `action_type == "continuous"`, :math:`\\hat{Q}` for the action chosen by evaluation policy,
                i.e., :math:`\\hat{Q}(s_t, \\pi(a \\mid s_t))`.
                shape (n_episodes * step_per_episode, )

                If `use_base_model == False`, `None` is recorded.

            initial_state_value_prediction: ndarray of shape (n_episodes, )
                Estimated initial state value.
                If `use_base_model == False`, `None` is recorded.

            initial_state_action_distribution: ndarray of shape (n_episodes, n_actions)
                Evaluation policy pscore at the initial state of each episode.
                If `action_type == "continuous"`, `None` is recorded.

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

        if self.use_base_model:
            if n_steps_per_epoch is None:
                n_steps_per_epoch = 10000

            for i in tqdm(
                range(len(evaluation_policies)),
                desc="[fit FQE model]",
                total=len(evaluation_policies),
            ):
                self.build_and_fit_FQE(
                    evaluation_policies[i],
                    n_epochs=n_epochs,
                    n_steps=n_steps,
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
                    "evaluation_policy_step_wise_pscore"
                ] = self.obtain_step_wise_pscore(evaluation_policies[i])
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_trajectory_wise_pscore"
                ] = self.obtain_trajectory_wise_pscore(evaluation_policies[i])
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action"
                ] = None
            else:
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_step_wise_pscore"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_trajectory_wise_pscore"
                ] = None
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action"
                ] = self.obtain_evaluation_policy_action(evaluation_policies[i])

            # input for DM, DR
            if self.action_type == "discrete":
                if self.use_base_model:
                    (
                        action_dist,
                        state_action_value_prediction,
                    ) = self.obtain_action_dist_with_state_action_value_prediction_discrete(
                        evaluation_policies[i]
                    )
                    input_dict[evaluation_policies[i].name][
                        "evaluation_policy_action_dist"
                    ] = action_dist
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = state_action_value_prediction
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = self.obtain_initial_state_value_prediction_discrete(
                        evaluation_policies[i]
                    )
                else:
                    input_dict[evaluation_policies[i].name][
                        "evaluation_policy_action_dist"
                    ] = None
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = None
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = None
            else:
                input_dict[evaluation_policies[i].name][
                    "evaluation_policy_action_dist"
                ] = None

                if self.use_base_model:
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = self.obtain_state_action_value_prediction_continuous(
                        evaluation_policies[i]
                    )
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = self.obtain_initial_state_value_prediction_continuous(
                        evaluation_policies[i]
                    )
                else:
                    input_dict[evaluation_policies[i].name][
                        "state_action_value_prediction"
                    ] = None
                    input_dict[evaluation_policies[i].name][
                        "initial_state_value_prediction"
                    ] = None

            # input for the distributionally robust OPE estimators
            if self.action_type == "discrete":
                input_dict[evaluation_policies[i].name][
                    "initial_state_action_distribution"
                ] = self.obtain_initial_state_action_distribution(
                    evaluation_policies[i]
                )
            else:
                input_dict[evaluation_policies[i].name][
                    "initial_state_action_distribution"
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
