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
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import BaseHead
from scope_rl.policy import ContinuousGaussianHead as GaussianHead
from scope_rl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from scope_rl.policy import DiscreteSoftmaxHead as SoftmaxHead
from scope_rl.policy import OffPolicyLearning


from scope_rl.utils import MinMaxActionScaler
from scope_rl.utils import OldGymAPIWrapper
from scope_rl.types import LoggedDataset

# ope 
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import BaseHead
from scope_rl.policy import ContinuousGaussianHead as GaussianHead
from scope_rl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from scope_rl.policy import DiscreteSoftmaxHead as SoftmaxHead
from scope_rl.policy import OffPolicyLearning

from scope_rl.ope import OffPolicyEvaluation
from scope_rl.ope import OffPolicySelection
# continuous
# basic estimators
from scope_rl.ope import ContinuousDirectMethod as C_DM
from scope_rl.ope import ContinuousTrajectoryWiseImportanceSampling as C_TIS
from scope_rl.ope import ContinuousPerDecisionImportanceSampling as C_PDIS
from scope_rl.ope import ContinuousDoublyRobust as C_DR
# self normalized estimators
from scope_rl.ope import ContinuousSelfNormalizedTIS as C_SNTIS
from scope_rl.ope import ContinuousSelfNormalizedPDIS as C_SNPDIS
from scope_rl.ope import ContinuousSelfNormalizedDR as C_SNDR
from scope_rl.ope import ContinuousStateActionMarginalSNIS as C_MIS
from scope_rl.ope import ContinuousStateActionMarginalSNDR as C_MDR
# marginal estimators
from scope_rl.ope import ContinuousStateActionMarginalIS as C_SAMIS
from scope_rl.ope import ContinuousStateActionMarginalDR as C_SAMDR
from scope_rl.ope import ContinuousStateMarginalIS as C_SMIS
from scope_rl.ope import ContinuousStateMarginalDR as C_SMDR
from scope_rl.ope import ContinuousStateActionMarginalSNIS as C_SAMSNIS
from scope_rl.ope import ContinuousStateActionMarginalSNDR as C_SAMSNDR
from scope_rl.ope import ContinuousStateMarginalSNIS as C_SMSNIS
from scope_rl.ope import ContinuousStateMarginalSNDR as C_SMSNDR
# double reinforcement learning estimators
from scope_rl.ope import ContinuousDoubleReinforcementLearning as C_DRL
# discrete
# basic estimators
from scope_rl.ope import DiscreteDirectMethod as D_DM
from scope_rl.ope import DiscreteTrajectoryWiseImportanceSampling as D_TIS
from scope_rl.ope import DiscretePerDecisionImportanceSampling as D_PDIS
from scope_rl.ope import DiscreteDoublyRobust as D_DR
# self normalized estimators
from scope_rl.ope import DiscreteSelfNormalizedTIS as D_SNTIS
from scope_rl.ope import DiscreteSelfNormalizedPDIS as D_SNPDIS
from scope_rl.ope import DiscreteSelfNormalizedDR as D_SNDR
from scope_rl.ope import DiscreteStateActionMarginalSNIS as D_MIS
from scope_rl.ope import DiscreteStateActionMarginalSNDR as D_MDR
# marginal estimators
from scope_rl.ope import DiscreteStateActionMarginalIS as D_SAMIS
from scope_rl.ope import DiscreteStateActionMarginalDR as D_SAMDR
from scope_rl.ope import DiscreteStateMarginalIS as D_SMIS
from scope_rl.ope import DiscreteStateMarginalDR as D_SMDR
from scope_rl.ope import DiscreteStateActionMarginalSNIS as D_SAMSNIS
from scope_rl.ope import DiscreteStateActionMarginalSNDR as D_SAMSNDR
from scope_rl.ope import DiscreteStateMarginalSNIS as D_SMSNIS
from scope_rl.ope import DiscreteStateMarginalSNDR as D_SMSNDR
# double reinforcement learning estimators
from scope_rl.ope import DiscreteDoubleReinforcementLearning as D_DRL

from scope_rl.ope import CreateOPEInput

from scope_rl.utils import MinMaxScaler
from scope_rl.utils import MinMaxActionScaler
from scope_rl.utils import OldGymAPIWrapper
from scope_rl.utils import MultipleLoggedDataset

from experiments.utils import torch_seed, format_runtime


def train_behavior_policy(
    env_name: str,
    env: gym.Env,
    behavior_sigma: float,
    behavior_tau: float,
    device: str,
    base_random_state: int,
    log_dir: str,
    variable,
    variable_name,
):
    env_ = OldGymAPIWrapper(env)
    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
    if action_type == "continuous":
        action_dim = env.action_space.shape[0]

    path_ = Path(log_dir + f"/behavior_policy")
    path_.mkdir(exist_ok=True, parents=True)
    path_behavior_policy = Path(path_ / f"behavior_policy_{env_name}_{variable_name}_{variable}.pt")

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
        behavior_policy = GaussianHead(
            model,
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
    variable,
    variable_name,
):
    behavior_policy_name = behavior_policy.name

    path_ = Path(log_dir + f"/logged_dataset")
    path_.mkdir(exist_ok=True, parents=True)
    path_train_logged_dataset = Path(
        path_
        / f"train_logged_dataset_{env_name}_{behavior_policy_name}_{variable_name}_{variable}.pkl"
    )
    path_test_logged_dataset = Path(
        path_
        / f"test_logged_dataset_{env_name}_{behavior_policy_name}_{variable_name}_{variable}.pkl"
    )

    if path_train_logged_dataset.exists():
        with open(path_train_logged_dataset, "rb") as f:
            train_logged_dataset = pickle.load(f)
    if path_test_logged_dataset.exists():
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
            path=log_dir + f"/logged_dataset/multiple/{variable_name}/{variable}",
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
    variable,
    variable_name,
):
    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
    if action_type == "continuous":
        action_dim = env.action_space.shape[0]

    path_ = Path(log_dir + f"/candidate_policies")
    path_.mkdir(exist_ok=True, parents=True)
    path_candidate_policy = Path(
        path_ / f"candidate_policy_{env_name}_{variable_name}_{variable}.pkl"
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
            cql = CQL(
                actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
            )
            algorithms = [cql]

        else:
            cql = DiscreteCQL(
                encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                q_func_factory=MeanQFunctionFactory(),
                use_gpu=(device == "cuda:0"),
            )
            
            algorithms = [cql]

        base_policies = opl.learn_base_policy(
            logged_dataset=train_logged_dataset,
            algorithms=algorithms,
            random_state=base_random_state,
        )
        with open(path_candidate_policy, "wb") as f:
            pickle.dump(base_policies, f)

    if action_type == "continuous":
        algorithms_name = ["cql"]

        policy_wrappers = {}
        for sigma in candidate_sigmas:
            policy_wrappers[f"gauss_{sigma}"] = (
                GaussianHead,
                {
                    "sigma": np.full((action_dim,), sigma),
                },
            )

    else:
        algorithms_name = ["cql"]

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
    variable,
    variable_name,
):
    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"

    path_ = Path(log_dir + f"/input_dict")
    path_.mkdir(exist_ok=True, parents=True)
    path_input_dict = Path(path_ / f"input_dict_{env_name}_{variable_name}_{variable}.pkl")

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
                device=device,
            )

        input_dict = prep.obtain_whole_inputs(
            logged_dataset=test_logged_dataset,
            evaluation_policies=candidate_policies,
            require_value_prediction=True,
            require_weight_prediction=True,
            n_trajectories_on_policy_evaluation=100,
            path=log_dir + f"/input_dict/multiple/{variable_name}/{variable}",
            random_state=base_random_state,
        )
        with open(path_input_dict, "wb") as f:
            pickle.dump(input_dict, f)

    if action_type == "continuous":

        basic_estimators = [C_DM(), C_TIS(), C_PDIS(), C_DR(), C_SNTIS(), C_SNPDIS(), C_SNDR()]
        state_marginal_estimators = [C_SMIS(), C_SMDR(), C_SMSNIS(), C_SMSNDR()]
        state_action_marginal_estimators = [C_SAMIS(), C_SAMDR(), C_SAMSNIS(), C_SAMSNDR()]
        drl_estimators = [C_DRL()]
    else:
        basic_estimators = [D_DM(), D_TIS(), D_PDIS(), D_DR(), D_SNTIS(), D_SNPDIS(), D_SNDR()]
        state_marginal_estimators = [D_SMIS(), D_SMDR(), D_SMSNIS(), D_SMSNDR()]
        state_action_marginal_estimators = [D_SAMIS(), D_SAMDR(), D_SAMSNIS(), D_SAMSNDR()]
        drl_estimators = [D_DRL()]
    all_estimators = basic_estimators + state_marginal_estimators + state_action_marginal_estimators + drl_estimators

    ope = OffPolicyEvaluation(
        logged_dataset=test_logged_dataset,
        ope_estimators=all_estimators,
    )

    policy_value_dict = ope.estimate_policy_value(
        input_dict=input_dict,
    )

    return input_dict, policy_value_dict
