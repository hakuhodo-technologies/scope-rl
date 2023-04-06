import sys

from ofrl.dataset import SyntheticDataset
from typing import Dict, Any
import pickle
import gym
from pathlib import Path


def data_collection(
    env: gym.Env,
    behavior_policies,
    n_trajectories,
    n_random_state,
    random_state,
) -> Dict[str, Any]:
    
    path_train_dataset = Path(
        f"train_dataset_{env}_{behavior_policies}_{n_trajectories}_{n_random_state}_{random_state}.pickle"
    )

    path_test_dataset = Path(
        f"test_dataset_{env}_{behavior_policies}_{n_trajectories}_{n_random_state}_{random_state}.pickle"
    ) 

    if path_train_dataset.exists():
        with open(path_train_dataset, "rb") as f:
            train_dataset = pickle.load(f)

    if path_test_dataset.exists():
        with open(path_test_dataset, "rb") as f:
            test_dataset = pickle.load(f)
    
    else:
        # initialize dataset class
        dataset = SyntheticDataset(
            env=env,
            max_episode_steps=env.step_per_episode,
        )

        # collect logged data by a behavior policy
        # skip if there is a preserved logged dataset
        train_logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policies,
            n_datasets=n_random_state,
            n_trajectories=n_trajectories,
            obtain_info=False,
            random_state=random_state,
        )
        test_logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policies,
            n_datasets=n_random_state,
            n_trajectories=n_trajectories,
            obtain_info=False,
            random_state=random_state + 1,
        )

    with open(path_train_dataset, "wb") as f:
        pickle.dump(train_logged_dataset, f)
    with open(path_test_dataset, "wb") as f:
        pickle.dump(test_logged_dataset, f)

    with open(path_train_dataset, "rb") as f:
        train_logged_dataset = pickle.load(f)
    with open(path_test_dataset, "rb") as f:
        test_logged_dataset = pickle.load(f)

    return train_logged_dataset, test_logged_dataset
