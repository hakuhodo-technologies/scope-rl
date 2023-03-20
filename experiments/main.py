import time
from typing import Dict, List, Any
from pathlib import Path

import hydra
from omegaconf import DictConfig


from conf.behavior_policy import behavior_policy_model_confs
from utils import format_runtime


def process(
    env: str,
    behavior_policy_params: Dict[str, float],
    behavior_epsilon: float,
    candidate_policy_params: Dict[str, List[float]],
    candidate_epsilons: List[float], 
    behavior_policy_model_confs: Dict[str, Any],
):
    pass

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
    }
    process(**conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))