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

from d3rlpy.algos import SAC
from d3rlpy.algos import DoubleDQN as DDQN
from d3rlpy.algos import CQL
from d3rlpy.algos import IQL
from d3rlpy.algos import BCQ
from d3rlpy.algos import DiscreteCQL
from d3rlpy.algos import DiscreteBCQ
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy, ConstantEpsilonGreedy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.online.buffers import ReplayBuffer

from ofrl.dataset import SyntheticDataset
from ofrl.policy import BaseHead
from ofrl.policy import ContinuousGaussianHead as GaussianHead
from ofrl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from ofrl.policy import DiscreteSoftmaxHead as SoftmaxHead
from ofrl.policy import OffPolicyLearning

from ofrl.ope.online import visualize_on_policy_policy_value
from ofrl.ope.online import calc_on_policy_policy_value

from ofrl.utils import MinMaxActionScaler
from ofrl.utils import OldGymAPIWrapper
from ofrl.types import LoggedDataset

from experiments.utils import torch_seed, format_runtime


def train_behavior_policy(
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
    path_behavior_policy = Path(
        path_ / f"behavior_policy_{env_name}.pt"
    )

    torch_seed(base_random_state, device=device)

    if action_type == "continuous":
        model = SAC(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
            q_func_factory=MeanQFunctionFactory(),
            actor_learning_rate=base_model_config.sac.actor_lr,
            critic_learning_rate=base_model_config.sac.critic_lr,
            temp_learning_rate=base_model_config.sac.temp_lr,
            batch_size=base_model_config.sac.batch_size,
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
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env_,
        )

        if base_model_config.ddqn.explorer.type == "linear":
            explorer = LinearDecayEpsilonGreedy(
                start_epsilon=1.0,
                end_epsilon=base_model_config.ddqn.explorer.epsilon,
                duration=base_model_config.ddqn.explorer.duration,
            )
        else:
            explorer = ConstantEpsilonGreedy(
                epsilon=base_model_config.ddqn.explorer.epsilon,
            )

        if action_type == "continuous":
            model.fit_online(
                env_,
                buffer,
                eval_env=env_,
                n_steps=base_model_config.sac.n_steps,
                n_steps_per_epoch=base_model_config.sac.n_steps // 100,
                update_start_step=base_model_config.sac.update_start_step,
            )

        else:
            model.fit_online(
                env_,
                buffer,
                explorer=explorer,
                eval_env=env_,
                n_steps=base_model_config.ddqn.n_steps,
                n_steps_per_epoch=base_model_config.ddqn.n_steps // 100,
                update_start_step=base_model_config.ddqn.update_start_step,
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


def obtain_train_logged_dataset(
    env_name: str,
    env: gym.Env,
    behavior_policy: BaseHead,
    n_trajectories: int,
    base_random_state: int,
    log_dir: str,
):
    behavior_policy_name = behavior_policy.name

    path_ = Path(log_dir + f"/logged_dataset")
    path_.mkdir(exist_ok=True, parents=True)
    path_train_logged_dataset = Path(
        path_
        / f"train_logged_dataset_{env_name}_{behavior_policy_name}.pkl"
    )

    if path_train_logged_dataset.exists():
        with open(path_train_logged_dataset, "rb") as f:
            train_logged_dataset = pickle.load(f)

    else:
        dataset = SyntheticDataset(
            env=env,
            max_episode_steps=env.spec.max_episode_steps,
        )

        train_logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policy,
            n_datasets=1,
            n_trajectories=n_trajectories,
            obtain_info=False,
            random_state=base_random_state,
        )

        with open(path_train_logged_dataset, "wb") as f:
            pickle.dump(train_logged_dataset, f)

    return train_logged_dataset


def train_candidate_policies(
    env_name: str,
    env: gym.Env,
    train_logged_dataset: LoggedDataset,
    candidate_sigmas: List[float],
    candidate_epsilons: List[float],
    device: str,
    base_random_state: int,
    base_model_config: DictConfig,
    log_dir: str,
):
    behavior_policy_name = train_logged_dataset["behavior_policy"]

    action_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
    if action_type == "continuous":
        action_dim = env.action_space.shape[0]

    path_ = Path(log_dir + f"/candidate_policies")
    path_.mkdir(exist_ok=True, parents=True)
    path_candidate_policy = Path(
        path_
        / f"candidate_policy_{env_name}_{behavior_policy_name}.pkl"
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
        torch_seed(base_random_state, device=device)

        if action_type == "continuous":
            cql_b1 = CQL(
                actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                q_func_factory=MeanQFunctionFactory(),
                action_learning_rate=base_model_config.cql.continuous.actor_lr,
                critic_learning_rate=base_model_config.cql.continuous.critic_lr,
                temp_learning_rate=base_model_config.cql.continuous.temp_lr,
                alpha_learning_rate=base_model_config.cql.continuous.alpha_lr,
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
                action_learning_rate=base_model_config.cql.continuous.actor_lr,
                critic_learning_rate=base_model_config.cql.continuous.critic_lr,
                temp_learning_rate=base_model_config.cql.continuous.temp_lr,
                alpha_learning_rate=base_model_config.cql.continuous.alpha_lr,
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
                action_learning_rate=base_model_config.cql.continuous.actor_lr,
                critic_learning_rate=base_model_config.cql.continuous.critic_lr,
                temp_learning_rate=base_model_config.cql.continuous.temp_lr,
                alpha_learning_rate=base_model_config.cql.continuous.alpha_lr,
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
            algorithms = [cql_b1, cql_b2, cql_b3, iql_b1, iql_b2, iql_b3]

        else:
            cql_b1 = DiscreteCQL(
                encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                q_func_factory=MeanQFunctionFactory(),
                learning_rate=base_model_config.cql.discrete.lr,
                alpha=base_model_config.cql.discrete.alpha,
                use_gpu=(device == "cuda:0"),
            )
            cql_b2 = DiscreteCQL(
                encoder_factory=VectorEncoderFactory(hidden_units=[100]),
                q_func_factory=MeanQFunctionFactory(),
                learning_rate=base_model_config.cql.discrete.lr,
                alpha=base_model_config.cql.discrete.alpha,
                use_gpu=(device == "cuda:0"),
            )
            cql_b3 = DiscreteCQL(
                encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
                q_func_factory=MeanQFunctionFactory(),
                learning_rate=base_model_config.cql.discrete.lr,
                alpha=base_model_config.cql.discrete.alpha,
                use_gpu=(device == "cuda:0"),
            )
            bcq_b1 = DiscreteBCQ(
                encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
                q_func_factory=MeanQFunctionFactory(),
                batch_size=base_model_config.bcq.batch_size,
                learning_rate=base_model_config.bcq.lr,
                target_update_interval=base_model_config.bcq.target_update_interval,
                action_flexibility=base_model_config.bcq.action_flexibility,
                beta=base_model_config.bcq.beta,
                use_gpu=(device == "cuda:0"),
            )
            bcq_b2 = DiscreteBCQ(
                encoder_factory=VectorEncoderFactory(hidden_units=[100]),
                q_func_factory=MeanQFunctionFactory(),
                batch_size=base_model_config.bcq.batch_size,
                learning_rate=base_model_config.bcq.lr,
                target_update_interval=base_model_config.bcq.target_update_interval,
                action_flexibility=base_model_config.bcq.action_flexibility,
                beta=base_model_config.bcq.beta,
                use_gpu=(device == "cuda:0"),
            )
            bcq_b3 = DiscreteBCQ(
                encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
                q_func_factory=MeanQFunctionFactory(),
                batch_size=base_model_config.bcq.batch_size,
                learning_rate=base_model_config.bcq.lr,
                target_update_interval=base_model_config.bcq.target_update_interval,
                action_flexibility=base_model_config.bcq.action_flexibility,
                beta=base_model_config.bcq.beta,
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
        algorithms_name = ["cql_b1", "cql_b2", "cql_b3", "iql_b1", "iql_b2", "iql_b3"]

        policy_wrappers = {}
        for sigma in candidate_sigmas:
            policy_wrappers[f"gauss_{sigma}"] = (
                GaussianHead,
                {
                    "sigma": np.full((action_dim,), sigma),
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


def process(
    env_name: str,
    behavior_sigma: float,
    behavior_tau: float,
    candidate_sigmas: List[float],
    candidate_epsilons: List[float],
    n_trajectories: int,
    use_small_env: bool,
    device: str,
    base_random_state: int,
    base_model_config: DictConfig,
    log_dir: str,
):
    if use_small_env:
        env = gym.make(env_name + '-v4')
        # env = gym.make(env_name + "Env")
    else:
        env = gym.make(env_name)

    behavior_policy = train_behavior_policy(
        env_name=env_name,
        env=env,
        behavior_sigma=behavior_sigma,
        behavior_tau=behavior_tau,
        device=device,
        base_random_state=base_random_state,
        base_model_config=base_model_config,
        log_dir=log_dir,
    )

    train_logged_dataset = obtain_train_logged_dataset(
        env_name=env_name,
        env=env,
        behavior_policy=behavior_policy,
        n_trajectories=n_trajectories,
        base_random_state=base_random_state,
        log_dir=log_dir,
    )

    candidate_policies = train_candidate_policies(
        env_name=env_name,
        env=env,
        train_logged_dataset=train_logged_dataset,
        candidate_sigmas=candidate_sigmas,
        candidate_epsilons=candidate_epsilons,
        device=device,
        base_random_state=base_random_state,
        base_model_config=base_model_config,
        log_dir=log_dir,
    )

    path_ = Path(log_dir + f"/results/on_policy")
    path_.mkdir(exist_ok=True, parents=True)

    visualize_on_policy_policy_value(
        env=env,
        policies=[behavior_policy] + candidate_policies,
        # policies=[behavior_policy],
        policy_names=[behavior_policy.name]
        + [candidate_policy.name for candidate_policy in candidate_policies],
        # policy_names=[behavior_policy.name],
        random_state=base_random_state,
        step_per_trajectory=env.spec.max_episode_steps,
        fig_dir=path_,
        fig_name=f"on_policy_policy_value_{env_name}.png",
    )

    path_ = Path(log_dir + f"/results/raw")
    path_.mkdir(exist_ok=True, parents=True)
    path_behavior_on_policy = Path(
        path_ / f"behavior_on_policy_{env_name}_{behavior_policy.name}.pkl"
    )

    behavior_on_policy = calc_on_policy_policy_value(
        env=env,
        policy=behavior_policy,
    )
    with open(path_behavior_on_policy, "wb") as f:
        pickle.dump(behavior_on_policy, f)


def register_small_envs(
    max_episode_steps: int,
):
    # continuous control
    gym.envs.register(
        id="HalfCheetah-v4",
        entry_point="gym.envs.mujoco:HalfCheetahEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="Hopper-v4",
        entry_point="gym.envs.mujoco:HopperEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="InvertedPendulum-v4",
        entry_point="gym.envs.mujoco:InvertedPendulumEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="Reacher-v4",
        entry_point="gym.envs.mujoco:ReacherEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="Swimmer-v4",
        entry_point="gym.envs.mujoco:SwimmerEnv",
        max_episode_steps=max_episode_steps,
    )
    # discrete control
    gym.envs.register(
        id="CartPole-v0",
        entry_point="gym.envs.classic_control:CartPoleEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="MountainCar-v0",
        entry_point="gym.envs.classic_control:MountainCarEnv",
        max_episode_steps=max_episode_steps,
    )
    gym.envs.register(
        id="Acrobot-v0",
        entry_point="gym.envs.classic_control:AcrobotEnv",
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
        "MountainCar",
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
        "base_random_state": cfg.setting.base_random_state,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "base_model_config": cfg.base_model_config,
        "use_small_env": cfg.setting.max_episode_steps != "None",
        "log_dir": "logs/",
    }
    if cfg.setting.max_episode_steps != "None":
        register_small_envs(cfg.setting.max_episode_steps)
    process(**conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
