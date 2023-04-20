import time
from typing import List
from pathlib import Path
import pickle

import hydra
from omegaconf import DictConfig

import gym
from gym.spaces import Box

import basicgym
from basicgym import BasicEnv

import numpy as np
import torch

from d3rlpy.algos import SAC
from d3rlpy.algos import DoubleDQN as DDQN
from d3rlpy.algos import CQL
from d3rlpy.algos import IQL
from d3rlpy.algos import DiscreteCQL
from d3rlpy.algos import DiscreteBCQ
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.online.buffers import ReplayBuffer

from ofrl.dataset import SyntheticDataset
from ofrl.policy import BaseHead
from ofrl.policy import ContinuousTruncatedGaussianHead as TruncatedGaussianHead
from ofrl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from ofrl.policy import DiscreteSoftmaxHead as SoftmaxHead
from ofrl.policy import OffPolicyLearning

from ofrl.ope import OffPolicyEvaluation
from ofrl.ope import OffPolicySelection
from ofrl.ope import ContinuousDirectMethod as C_DM
from ofrl.ope import ContinuousTrajectoryWiseImportanceSampling as C_TIS
from ofrl.ope import ContinuousSelfNormalizedPerDecisionImportanceSampling as C_PDIS
from ofrl.ope import ContinuousSelfNormalizedDoublyRobust as C_DR
from ofrl.ope import (
    ContinuousStateActionMarginalSelfNormalizedImportanceSampling as C_MIS,
)
from ofrl.ope import DiscreteDirectMethod as D_DM
from ofrl.ope import DiscreteSelfNormalizedPerDecisionImportanceSampling as D_PDIS
from ofrl.ope import DiscreteSelfNormalizedDoublyRobust as D_DR
from ofrl.ope import (
    DiscreteStateActionMarginalSelfNormalizedImportanceSampling as D_MIS,
)
from ofrl.ope import CreateOPEInput

from ofrl.utils import MinMaxScaler
from ofrl.utils import MinMaxActionScaler
from ofrl.utils import OldGymAPIWrapper
from ofrl.utils import MultipleLoggedDataset
from ofrl.types import LoggedDataset

from experiments.utils import torch_seed, format_runtime


def train_behavior_policy(
    env_name: str,
    env: gym.Env,
    behavior_sigma: float,
    behavior_tau: float,
    device: str,
    base_random_state: int,
    log_dir: str,
):
    env_ = OldGymAPIWrapper(env)
    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
    if action_type == "continuous":
        action_dim = env.action_space.shape[0]

    path_ = Path(log_dir + f"/behavior_policy")
    path_.mkdir(exist_ok=True, parents=True)
    path_behavior_policy = Path(path_ / f"behavior_policy_{env_name}.pt")

    torch_seed(base_random_state, device=device)

    if action_type == "continuous":
        model = SAC(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            use_gpu=(device == "cuda:0"),
            action_scaler=MinMaxActionScaler(
                minimum=env.action_space.low,
                maximum=env.action_space.high,
            ),
        )
    else:
        model = DDQN(
            encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            target_update_interval=100,
            use_gpu=(device == "cuda:0"),
        )

    if path_behavior_policy.exists():
        model.build_with_env(env_)
        model.load_model(path_behavior_policy)

    else:
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env_,
        )
        explorer = LinearDecayEpsilonGreedy(
            start_epsilon=1.0,
            end_epsilon=0.1,
            duration=1000,
        )
        if action_type == "continuous":
            model.fit_online(
                env_,
                buffer,
                eval_env=env_,
                n_steps=1000,
                n_steps_per_epoch=100,
                update_start_step=100,
            )

        else:
            model.fit_online(
                env_,
                buffer,
                explorer=explorer,
                eval_env=env_,
                n_steps=100000,
                n_steps_per_epoch=100,
                update_start_step=100,
            )
            behavior_policy = SoftmaxHead(
                model,
                n_actions=env.action_space.n,
                tau=behavior_tau,
                name=f"ddqn_softmax_{behavior_tau}",
                random_state=base_random_state,
            )

        model.save_model(path_behavior_policy)

    if action_type == "continuous":
        behavior_policy = TruncatedGaussianHead(
            model,
            minimum=env.action_space.low,
            maximum=env.action_space.high,
            sigma=np.full((action_dim,), behavior_sigma),
            name=f"sac_gauss_{behavior_sigma}",
            random_state=base_random_state,
        )
    else:
        behavior_policy = SoftmaxHead(
            model,
            n_actions=env.action_space.n,
            tau=behavior_tau,
            name=f"ddqn_softmax_{behavior_tau}",
            random_state=base_random_state,
        )

    return behavior_policy


def obtain_logged_dataset(
    env_name: str,
    env: gym.Env,
    behavior_policy: BaseHead,
    n_trajectories: int,
    n_random_state: int,
    base_random_state: int,
    log_dir: str,
):
    behavior_policy_name = behavior_policy.name

    path_ = Path(log_dir + f"/logged_dataset")
    path_.mkdir(exist_ok=True, parents=True)
    path_train_logged_dataset = Path(
        path_
        / f"train_logged_dataset_{env_name}_{behavior_policy_name}_{n_trajectories}.pkl"
    )
    path_test_logged_dataset = Path(
        path_
        / f"test_logged_dataset_{env_name}_{behavior_policy_name}_{n_trajectories}.pkl"
    )

    if path_train_logged_dataset.exists():
        with open(path_train_logged_dataset, "rb") as f:
            train_logged_dataset = pickle.load(f)
        with open(path_test_logged_dataset, "rb") as f:
            test_logged_dataset = pickle.load(f)

    else:
        if env_name in ["BasicEnv-discrete-v0", "BasicEnv-continuous-v0"]:
            max_episode_steps = env.step_per_episode
        else:
            max_episode_steps = env.spec.max_episode_steps

        dataset = SyntheticDataset(
            env=env,
            max_episode_steps=max_episode_steps,
        )

        train_logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policy,
            n_datasets=1,
            n_trajectories=n_trajectories,
            obtain_info=False,
            random_state=base_random_state,
        )
        test_logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policy,
            n_datasets=n_random_state,
            n_trajectories=n_trajectories,
            obtain_info=False,
            path=log_dir + f"/logged_dataset/multiple",
            random_state=base_random_state + 1,
        )

        with open(path_train_logged_dataset, "wb") as f:
            pickle.dump(train_logged_dataset, f)
        with open(path_test_logged_dataset, "wb") as f:
            pickle.dump(test_logged_dataset, f)

    return train_logged_dataset, test_logged_dataset


def train_candidate_policies(
    env_name: str,
    env: gym.Env,
    n_trajectories: int,
    train_logged_dataset: LoggedDataset,
    candidate_sigmas: List[float],
    candidate_epsilons: List[float],
    device: str,
    base_random_state: int,
    log_dir: str,
):
    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
    if action_type == "continuous":
        action_dim = env.action_space.shape[0]

    path_ = Path(log_dir + f"/candidate_policies")
    path_.mkdir(exist_ok=True, parents=True)
    path_candidate_policy = Path(
        path_ / f"candidate_policy_{env_name}_{n_trajectories}.pkl"
    )

    opl = OffPolicyLearning(
        fitting_args={
            "n_steps": 10000,
            "scorers": {},
        }
    )

    if path_candidate_policy.exists():
        with open(path_candidate_policy, "rb") as f:
            base_policies = pickle.load(f)

    else:
        torch_seed(base_random_state, device=device)

        if action_type == "continuous":
            cql_b1 = CQL(
                actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
            )
            cql_b2 = CQL(
                actor_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
                critic_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
            )
            cql_b3 = CQL(
                actor_encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
                critic_encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
            )
            iql_b1 = IQL(
                actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                use_gpu=(device == "cuda:0"),
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
            )
            iql_b2 = IQL(
                actor_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
                critic_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
                use_gpu=(device == "cuda:0"),
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
            )
            iql_b3 = IQL(
                actor_encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
                critic_encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
                use_gpu=(device == "cuda:0"),
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
            )
            # algorithms = [cql_b1, cql_b2, cql_b3, iql_b1, iql_b2, iql_b3]
            algorithms = [cql_b1, cql_b2]

        else:
            cql_b1 = DiscreteCQL(
                encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
            )
            cql_b2 = DiscreteCQL(
                encoder_factory=VectorEncoderFactory(hidden_units=[100]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
            )
            cql_b3 = DiscreteCQL(
                encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
            )
            bcq_b1 = DiscreteBCQ(
                encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
            )
            bcq_b2 = DiscreteBCQ(
                encoder_factory=VectorEncoderFactory(hidden_units=[100]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
            )
            bcq_b3 = DiscreteBCQ(
                encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
            )
            algorithms = [cql_b1, cql_b2, cql_b3, bcq_b1, bcq_b2, bcq_b3]

        base_policies = opl.learn_base_policy(
            logged_dataset=train_logged_dataset,
            algorithms=algorithms,
            random_state=base_random_state,
        )
        with open(path_candidate_policy, "wb") as f:
            pickle.dump(base_policies, f)

    if action_type == "continuous":
        # algorithms_name = ["cql_b1", "cql_b2", "cql_b3", "iql_b1", "iql_b2", "iql_b3"]
        algorithms_name = ["cql_b1", "cql_b2"]

        policy_wrappers = {}
        for sigma in candidate_sigmas:
            policy_wrappers[f"gauss_{sigma}"] = (
                TruncatedGaussianHead,
                {
                    "sigma": np.full((action_dim,), sigma),
                    "minimum": env.action_space.low,
                    "maximum": env.action_space.high,
                },
            )

    else:
        algorithms_name = ["cql_b1", "cql_b2", "cql_b3", "bcq_b1", "bcq_b2", "bcq_b3"]

        policy_wrappers = {}
        for epsilon in candidate_epsilons:
            policy_wrappers[f"eps_{epsilon}"] = (
                EpsilonGreedyHead,
                {
                    "epsilon": epsilon,
                    "n_actions": env.action_space.n,
                },
            )

    candidate_policies = opl.apply_head(
        base_policies=base_policies,
        base_policies_name=algorithms_name,
        policy_wrappers=policy_wrappers,
        random_state=base_random_state,
    )
    return candidate_policies


def off_policy_evaluation(
    env_name: str,
    env: gym.Env,
    n_trajectories: int,
    test_logged_dataset: MultipleLoggedDataset,
    candidate_policies: List[BaseHead],
    device: str,
    base_random_state: int,
    log_dir: str,
):
    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"

    path_ = Path(log_dir + f"/input_dict")
    path_.mkdir(exist_ok=True, parents=True)
    path_input_dict = Path(path_ / f"input_dict_{env_name}_{n_trajectories}.pkl")

    if path_input_dict.exists():
        with open(path_input_dict, "rb") as f:
            input_dict = pickle.load(f)
    else:
        if action_type == "continuous":
            prep = CreateOPEInput(
                env=env,
                model_args={
                    "fqe": {
                        "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
                        "q_func_factory": MeanQFunctionFactory(),
                        "learning_rate": 1e-4,
                        "use_gpu": (device == "cuda:0"),
                    }
                },
                state_scaler=MinMaxScaler(
                    minimum=test_logged_dataset.get(
                        behavior_policy_name=test_logged_dataset.behavior_policy_names[
                            0
                        ],
                        dataset_id=0,
                    )["state"].min(axis=0),
                    maximum=test_logged_dataset.get(
                        behavior_policy_name=test_logged_dataset.behavior_policy_names[
                            0
                        ],
                        dataset_id=0,
                    )["state"].max(axis=0),
                ),
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
                sigma=0.1,
                device=device,
            )

        else:
            prep = CreateOPEInput(
                env=env,
                model_args={
                    "fqe": {
                        "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
                        "q_func_factory": MeanQFunctionFactory(),
                        "learning_rate": 1e-4,
                        "use_gpu": (device == "cuda:0"),
                    }
                },
                state_scaler=MinMaxScaler(
                    minimum=test_logged_dataset.get(
                        behavior_policy_name=test_logged_dataset.behavior_policy_names[
                            0
                        ],
                        dataset_id=0,
                    )["state"].min(axis=0),
                    maximum=test_logged_dataset.get(
                        behavior_policy_name=test_logged_dataset.behavior_policy_names[
                            0
                        ],
                        dataset_id=0,
                    )["state"].max(axis=0),
                ),
            )

        input_dict = prep.obtain_whole_inputs(
            logged_dataset=test_logged_dataset,
            evaluation_policies=candidate_policies,
            # require_value_prediction=True,
            # require_weight_prediction=True,
            n_trajectories_on_policy_evaluation=100,
            path=log_dir + f"/input_dict/multiple",
            random_state=base_random_state,
        )
        with open(path_input_dict, "wb") as f:
            pickle.dump(input_dict, f)

    if action_type == "continuous":
        ope_estimators = [C_TIS(), C_PDIS()]
        # ope_estimators = [C_DM(), C_PDIS(), C_DR(), C_MIS()]
    else:
        ope_estimators = [C_DM(), C_TIS(), C_PDIS(), C_DR()]
        # ope_estimators = [D_DM(), D_PDIS(), D_DR(), D_MIS()]

    ope = OffPolicyEvaluation(
        logged_dataset=test_logged_dataset,
        ope_estimators=ope_estimators,
    )

    policy_value_dict = ope.estimate_policy_value(
        input_dict=input_dict,
    )

    return input_dict, policy_value_dict
