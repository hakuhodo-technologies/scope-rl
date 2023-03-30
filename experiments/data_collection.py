import sys
sys.path.append("../")
sys.path.append("../../")

from ofrl.dataset import SyntheticDataset
from typing import Dict, Any
import pickle
import gym


def data_collection(
    env: gym.Env,
    behavior_policies,
    n_trajectories,
    n_random_state,
    random_state,
) -> Dict[str, Any]:

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

    with open("logs/train_dataset.pkl", "wb") as f:
        pickle.dump(train_logged_dataset, f)
    with open("logs/test_dataset.pkl", "wb") as f:
        pickle.dump(test_logged_dataset, f)

    with open("logs/train_dataset.pkl", "rb") as f:
        train_logged_dataset = pickle.load(f)
    with open("logs/test_dataset.pkl", "rb") as f:
        test_logged_dataset = pickle.load(f)

    return train_logged_dataset, test_logged_dataset
