import time
from typing import Dict, List, Any
from typing import Tuple, Optional
from pathlib import Path
import gym

import hydra
from omegaconf import DictConfig

from data_collection import data_collection
from policy_learning_behavior import policy_learning_behavior
from policy_learning_candidates import policy_learning_candidates
from ops_topk_evaluation import ops_topk_evaluation

from utils import format_runtime


def process(
    env: str,
    behavior_policy_params: Dict[str, float],
    candidate_policy_params: Dict[str, List[float]],
    n_trajectories: int = 10000,
    n_random_state: Optional[int] = None,
    random_state: Optional[int] = None,
):

    env = gym.make(env)

    behavior_policy = policy_learning_behavior(
        env,
        behavior_policy_params,
        random_state,
    )

    train_logged_dataset, test_logged_dataset = data_collection(
        env,
        behavior_policy,
        n_trajectories,
        n_random_state,
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

    return


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
    # assert_configuration(cfg)
    conf = {
        "env": cfg.setting.env,  # str
        "behavior_policy_params": cfg.setting.behavior_policy_params,  # dict of float
        "candidate_policy_params": cfg.setting.behavior_policy_params,  # dict of list
        "n_trajectories": cfg.setting.n_trajectories,  
        "n_random_state": cfg.setting.n_random_state,  
        "random_state": cfg.setting.random_state,  
    }

    process(**conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
