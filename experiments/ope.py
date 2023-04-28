import time
from typing import List
from pathlib import Path
import pickle

import hydra
from omegaconf import DictConfig

import gym
from gym.spaces import Box

import numpy as np
import torch

from sklearn.utils import check_random_state

from d3rlpy.algos import SAC
from d3rlpy.algos import DoubleDQN as DDQN
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory

from ofrl.dataset import SyntheticDataset
from ofrl.policy import BaseHead
from ofrl.policy import ContinuousTruncatedGaussianHead as TruncatedGaussianHead
from ofrl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from ofrl.policy import DiscreteSoftmaxHead as SoftmaxHead
from ofrl.policy import OffPolicyLearning

from ofrl.ope import OffPolicyEvaluation
from ofrl.ope import OffPolicySelection
from ofrl.ope import ContinuousDirectMethod as C_DM
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


def load_behavior_policy(
    env_name: str,
    env: gym.Env,
    behavior_sigma: float,
    behavior_tau: float,
    device: str,
    base_random_state: int,
    base_model_config: DictConfig,
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
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
            q_func_factory=MeanQFunctionFactory(),
            actor_learning_rate=base_model_config.sac.actor_lr,
            critic_learning_rate=base_model_config.sac.critic_lr,
            temp_learning_rate=base_model_config.sac.temp_lr,
            bstch_size=base_model_config.sac.batch_size,
            use_gpu=(device == "cuda:0"),
            action_scaler=MinMaxActionScaler(
                minimum=env.action_space.low,
                maximum=env.action_space.high,
            ),
        )
    else:
        model = DDQN(
            encoder_factory=VectorEncoderFactory(hidden_units=[100]),
            q_func_factory=MeanQFunctionFactory(),
            target_update_interval=base_model_config.ddqn.target_update_interval,
            batch_size=base_model_config.ddqn.batch_size,
            learning_rate=base_model_config.ddqn.lr,
            use_gpu=(device == "cuda:0"),
        )

    if path_behavior_policy.exists():
        model.build_with_env(env_)
        model.load_model(path_behavior_policy)

    else:
        raise ValueError(
            "behavior_policy not found. Please run policy_learning.py in advance."
        )

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


def obtain_test_logged_dataset(
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
    path_test_logged_dataset = Path(
        path_ / f"test_logged_dataset_{env_name}_{behavior_policy_name}.pkl"
    )

    if path_test_logged_dataset.exists():
        with open(path_test_logged_dataset, "rb") as f:
            test_logged_dataset = pickle.load(f)

    else:
        dataset = SyntheticDataset(
            env=env,
            max_episode_steps=env.spec.max_episode_steps,
        )
        test_logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policy,
            n_datasets=n_random_state,
            n_trajectories=n_trajectories,
            obtain_info=False,
            path=log_dir + f"/logged_dataset/multiple",
            random_state=base_random_state + 1,
        )
        with open(path_test_logged_dataset, "wb") as f:
            pickle.dump(test_logged_dataset, f)

    return test_logged_dataset


def load_candidate_policies(
    env_name: str,
    env: gym.Env,
    behavior_policy_name: str,
    candidate_sigmas: List[float],
    candidate_epsilons: List[float],
    base_random_state: int,
    base_model_config: DictConfig,
    log_dir: str,
):
    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
    if action_type == "continuous":
        action_dim = env.action_space.shape[0]

    path_ = Path(log_dir + f"/candidate_policies")
    path_.mkdir(exist_ok=True, parents=True)
    path_candidate_policy = Path(
        path_ / f"candidate_policy_{env_name}_{behavior_policy_name}.pkl"
    )

    opl = OffPolicyLearning(
        fitting_args={
            "n_steps": base_model_config.opl.fitting_steps,
            "scorers": {},
        }
    )

    if path_candidate_policy.exists():
        with open(path_candidate_policy, "rb") as f:
            base_policies = pickle.load(f)

    else:
        raise ValueError(
            "base_policies not found. Please run policy_learning.py in advance."
        )

    if action_type == "continuous":
        algorithms_name = ["cql_b1", "cql_b2", "cql_b3", "iql_b1", "iql_b2", "iql_b3"]

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
    test_logged_dataset: MultipleLoggedDataset,
    candidate_policies: List[BaseHead],
    device: str,
    base_random_state: int,
    base_model_config: DictConfig,
    log_dir: str,
):
    behavior_policy_name = test_logged_dataset.behavior_policy_names[0]
    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
    n_candidate_policies = len(candidate_policies)

    path_ = Path(log_dir + f"/input_dict")
    path_.mkdir(exist_ok=True, parents=True)
    path_input_dict = Path(
        path_
        / f"input_dict_{env_name}_{behavior_policy_name}_{n_candidate_policies}.pkl"
    )

    if path_input_dict.exists():
        with open(path_input_dict, "rb") as f:
            input_dict = pickle.load(f)
    else:
        if action_type == "continuous":
            prep = CreateOPEInput(
                env=env,
                model_args={
                    "fqe": {
                        "encoder_factory": VectorEncoderFactory(hidden_units=[100]),
                        "q_func_factory": MeanQFunctionFactory(),
                        "learning_rate": base_model_config.fqe.lr,
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
                sigma=base_model_config.contiuous_ope.sigma,
                device=device,
            )

        else:
            prep = CreateOPEInput(
                env=env,
                model_args={
                    "fqe": {
                        "encoder_factory": VectorEncoderFactory(hidden_units=[100]),
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
            require_value_prediction=True,
            require_weight_prediction=True,
            n_trajectories_on_policy_evaluation=100,
            path=log_dir + f"/input_dict/multiple",
            random_state=base_random_state,
        )
        with open(path_input_dict, "wb") as f:
            pickle.dump(input_dict, f)

    if action_type == "continuous":
        ope_estimators = [C_DM(), C_PDIS(), C_DR(), C_MIS()]
    else:
        ope_estimators = [D_DM(), D_PDIS(), D_DR(), D_MIS()]

    ope = OffPolicyEvaluation(
        logged_dataset=test_logged_dataset,
        ope_estimators=ope_estimators,
    )
    ops = OffPolicySelection(ope=ope)

    path_ = Path(log_dir + f"/results/raw")
    path_.mkdir(exist_ok=True, parents=True)

    ops_dict = ops.select_by_policy_value(
        input_dict=input_dict,
        return_metrics=True,
    )
    path_metrics = Path(
        path_
        / f"conventional_metrics_dict_{env_name}_{behavior_policy_name}_{n_candidate_policies}.pkl"
    )
    with open(path_metrics, "wb") as f:
        pickle.dump(ops_dict, f)

    topk_metric_dict = ops.obtain_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        return_safety_violation_rate=True,
        relative_safety_criteria=1.0,
    )
    path_topk_metrics = Path(
        path_
        / f"topk_metrics_dict_{env_name}_{behavior_policy_name}_{n_candidate_policies}.pkl"
    )
    with open(path_topk_metrics, "wb") as f:
        pickle.dump(topk_metric_dict, f)

    path_ = Path(log_dir + f"/results/figs")
    path_.mkdir(exist_ok=True, parents=True)

    ops.visualize_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        metrics=["best", "worst", "mean", "std", "sharpe_ratio"],
        visualize_ci=True,
        relative_safety_criteria=1.0,
        legend=False,
        random_state=base_random_state,
        fig_dir=path_,
        fig_name=f"topk_metrics_visualization_{env_name}_{behavior_policy_name}_{n_candidate_policies}.png",
    )


def process(
    env_name: str,
    behavior_sigma: float,
    behavior_tau: float,
    candidate_sigmas: List[float],
    candidate_epsilons: List[float],
    n_trajectories: int,
    n_random_state: int,
    n_candidate_policies: int,
    device: str,
    base_random_state: int,
    log_dir: str,
):
    env = gym.make(env_name)

    behavior_policy = load_behavior_policy(
        env_name=env_name,
        env=env,
        behavior_sigma=behavior_sigma,
        behavior_tau=behavior_tau,
        device=device,
        base_random_state=base_random_state,
        log_dir=log_dir,
    )

    test_logged_dataset = obtain_test_logged_dataset(
        env_name=env_name,
        env=env,
        behavior_policy=behavior_policy,
        n_trajectories=n_trajectories,
        n_random_state=n_random_state,
        base_random_state=base_random_state,
        log_dir=log_dir,
    )

    candidate_policies = load_candidate_policies(
        env_name=env_name,
        env=env,
        bahavior_policy_name=behavior_policy.name,
        candidate_sigmas=candidate_sigmas,
        candidate_epsilons=candidate_epsilons,
        device=device,
        base_random_state=base_random_state,
        log_dir=log_dir,
    )

    random_ = check_random_state(base_random_state)
    candidate_policy_idx = random_.choice(
        len(candidate_policies), size=n_candidate_policies, replace=False
    )
    candidate_policies = [candidate_policies[idx] for idx in candidate_policy_idx]

    off_policy_evaluation(
        env_name=env_name,
        env=env,
        test_logged_dataset=test_logged_dataset,
        candidate_policies=candidate_policies,
        device=device,
        base_random_state=base_random_state,
        log_dir=log_dir,
    )


def register_small_envs(
    max_episode_steps: int,
):
    # continuous control
    gym.envs.register(
        id="HalfCheetah",
        entry_point="gym.envs.mojoco:HalfCheetahEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="Hopper",
        entry_point="gym.envs.mojoco:HopperEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="InvertedPendulum",
        entry_point="gym.envs.mojoco:InvertedPendulumEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="Reacher",
        entry_point="gym.envs.mojoco:ReacherEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="Swimmer",
        entry_point="gym.envs.mujoco:SwimmerEnv",
        max_episode_steps=max_episode_steps,
    )
    # discrete control
    gym.envs.register(
        id="CartPole",
        entry_point="gym.envs.classic_control:CartPoleEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="Pendulum",
        entry_point="gym.envs.classic_control:PendulumEnv",
        max_episode_steps=max_episode_steps,
    )


def assert_configuration(cfg: DictConfig):
    env_name = cfg.setting.env_name
    assert env_name in [
        "HalfCheetah",
        "Hopper",
        "InvertedPendulum",
        "Reacher",
        "Swimmer",
        "CartPole",
        "Acrobot",
    ]

    behavior_sigma = cfg.setting.behavior_sigma
    assert isinstance(behavior_sigma, float) and behavior_sigma > 0.0

    behavior_tau = cfg.setting.behavior_tau
    assert isinstance(behavior_tau, float) and behavior_tau != 0.0

    candidate_sigmas = cfg.setting.candidate_sigmas
    for value in candidate_sigmas:
        assert isinstance(value, float) and value >= 0.0

    candidate_epsilons = cfg.setting.candidate_epsilons
    for value in candidate_epsilons:
        assert isinstance(value, float) and 0.0 <= value <= 1.0

    n_trajectories = cfg.setting.n_trajectories
    assert isinstance(n_trajectories, int) and n_trajectories > 0

    max_episode_steps = cfg.setting.max_episode_steps
    assert isinstance(max_episode_steps, int) and max_episode_steps > 0

    n_random_state = cfg.setting.n_random_state
    assert isinstance(n_random_state, int) and n_random_state > 0


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    assert_configuration(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    conf = {
        "env_name": cfg.setting.env_name,
        "behavior_sigma": cfg.setting.behavior_sigma,
        "behavior_tau": cfg.setting.behavior_tau,
        "candidate_sigmas": cfg.setting.candidate_sigmas,
        "candidate_epsilons": cfg.setting.candidate_epsilons,
        "n_trajectories": cfg.setting.n_trajectories,
        "n_random_state": cfg.setting.n_random_state,
        "base_random_state": cfg.setting.base_random_state,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "log_dir": "logs/",
    }
    register_small_envs(cfg.setting.max_episode_steps)
    process(**conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
