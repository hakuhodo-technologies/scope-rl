import time
from typing import Dict, List, Any
from pathlib import Path

import hydra
from omegaconf import DictConfig


from conf.behavior_policy import behavior_policy_model_confs
from data_collection import data_collection
from policy_learning_behavior import policy_learning_behavior
from policy_learning_candidates import policy_learning_candidates
from ops_topk_evaluation import ops_topk_evaluation

from utils import format_runtime
from ofrl.utils import OldGymAPIWrapper


def process(
    env: str,
    behavior_policy_params: Dict[str, float],
    behavior_epsilon: float,
    candidate_policy_params: Dict[str, List[float]],
    candidate_epsilons: List[float], 
    behavior_policy_model_confs: Dict[str, Any],
    n_trajectories: int,
    random_state: int,
):
    env = OldGymAPIWrapper(env)
    behavior_policy = policy_learning_behavior(
        env, 
        behavior_policy_params,
        behavior_policy_model_confs, 
        random_state,
    )

    train_logged_dataset, test_logged_dataset = data_collection(
        env, 
        behavior_policy,
        n_trajectories,
        random_state,
    )

    eval_policy = policy_learning_candidates(
        env, 
        candidate_policy_params, 
        train_logged_dataset, 
        random_state,
    )

    ops_topk_evaluation(
        env,
        test_logged_dataset,
        behavior_policy,
        eval_policy,
        random_state,
    )



def assert_configuration(cfg: DictConfig):

    env = cfg.setting.env
    assert env in ["BasicEnv-discrete-v0"]


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    # configurations
    assert_configuration(cfg)
    conf = {
        "env": cfg.setting.env,  # str
        "behavior_policy_params": cfg.setting.behavior_policy_params,  # dict of float
        "behavior_epsilon": cfg.setting.behavior_policy_params.epsilon,  # float
        "candidate_policy_params": cfg.setting.behavior_policy_params,  # dict of list
        "candidate_epsilons": cfg.setting.candidate_policy_params.epsilon,  # list of float
        "behavior_policy_model_confs": behavior_policy_model_confs,  # dict of d3rlpy policy
        "n_trajectories": cfg.setting.n_trajectories, # int
        "random_state": cfg.random_state, # int
    }

    process(**conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))