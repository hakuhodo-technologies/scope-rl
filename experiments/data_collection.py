from ofrl.dataset import SyntheticDataset
from typing import Dict, List, Any

def data_collection(
    env: str,
    behavior_policy_params: Dict[str, float],
    behavior_epsilon: float,
    behavior_policy_model_confs: Dict[str, Any],
):
    # initialize dataset class
    dataset = SyntheticDataset(
        env=env,
        max_episode_steps=env.step_per_episode,
    )

    # collect logged data by a behavior policy
    # skip if there is a preserved logged dataset
    train_logged_dataset = dataset.obtain_episodes(
        behavior_policies=behavior_policies, 
        n_datasets=2, 
        n_trajectories=10000, 
        obtain_info=False,
        random_state=random_state,
    )
    test_logged_dataset = dataset.obtain_episodes(
        behavior_policies=behavior_policies, 
        n_datasets=2, 
        n_trajectories=10000, 
        obtain_info=False,
        random_state=random_state + 1,
    )

    with open("logs/train_dataset_continuous_sac_5.pkl", "wb") as f:
        pickle.dump(train_logged_dataset, f)
    with open("logs/test_dataset_continuous_sac_5.pkl", "wb") as f:
        pickle.dump(test_logged_dataset, f)

    with open("logs/train_dataset_continuous_sac_5.pkl", "rb") as f:
        train_logged_dataset = pickle.load(f)
    with open("logs/test_dataset_continuous_sac_5.pkl", "rb") as f:
        test_logged_dataset = pickle.load(f)

    return train_logged_dataset, test_logged_dataset
