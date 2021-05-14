import pytest
from nptyping import NDArray

import numpy as np
from sklearn.linear_model import LogisticRegression

from _gym.dataset import SyntheticDataset
from _gym.env import RTBEnv
from _gym.policy import RandomPolicy


@pytest.mark.parametrize(
    "n_samples_pretrain_reward_predictor", [(-1), (0), ("1"), (1.5)]
)
def test_init_pretrain_samples_failure_case(n_samples_pretrain_reward_predictor):
    env = RTBEnv(
        use_reward_predictor=True,
        reward_predictor=LogisticRegression(),
    )
    agent = RandomPolicy(env)

    with pytest.raises(ValueError):
        SyntheticDataset(
            env=env,
            behavior_policy=agent,
            n_samples_pretrain_reward_predictor=n_samples_pretrain_reward_predictor,
        )


@pytest.mark.parametrize("random_state", [(-1), ("1"), (1.5)])
def test_init_random_state_failure_case(random_state):
    env = RTBEnv()
    agent = RandomPolicy(env)

    with pytest.raises(ValueError):
        SyntheticDataset(
            env=env,
            behavior_policy=agent,
            random_state=random_state,
        )


@pytest.mark.parametrize("n_episodes", [(-1), (0), ("1"), (1.5)])
def test_functions_failure_case(n_episodes):
    env = RTBEnv()
    agent = RandomPolicy(env)
    dataset = SyntheticDataset(env=env, behavior_policy=agent)

    with pytest.raises(ValueError):
        dataset.obtain_trajectories(n_episodes)

    with pytest.raises(ValueError):
        dataset.calc_ground_truth_policy_value(n_episodes)

    with pytest.raises(ValueError):
        dataset.pretrain_behavior_policy(n_episodes)


@pytest.mark.parametrize(
    "objective, action_type",
    [
        ("click", "discrete"),
        ("click", "continuous"),
        ("conversion", "discrete"),
        ("conversion", "continuous"),
    ],
)
def test_obtain_trajectories_random_policy_value_check(objective, action_type):
    env = RTBEnv(
        objective=objective,
        action_type=action_type,
    )
    agent = RandomPolicy(env)
    dataset = SyntheticDataset(env=env, behavior_policy=agent)

    logged_dataset = dataset.obtain_trajectories(n_episodes=100)
    assert logged_dataset["size"] == 24 * 100
    assert logged_dataset["n_episodes"] == 100
    assert logged_dataset["step_per_episode"] == 24
    assert logged_dataset["action_type"] == action_type
    assert (
        logged_dataset["action_dim"] == env.action_dim
        if action_type == "discrete"
        else None
    )
    assert logged_dataset["state_keys"] == [
        "timestep",
        "remaining_budget",
        "budget comsumption rate",
        "cost per mille of impression",
        "winning rate",
        "reward",
        "adjust rate",
    ]
    assert len(logged_dataset["state_keys"]) == 7

    state = logged_dataset["state"]
    action = logged_dataset["action"]
    reward = logged_dataset["reward"]
    done = logged_dataset["done"]
    pscore = logged_dataset["pscore"]
    info = logged_dataset["info"]

    assert 0.1 <= action.all() <= 10
    assert 0 <= reward.all()
    assert np.array_equal(done, done.astype(bool))
    assert np.allclose(pscore.unique(), 1 / env.action_dim)
    assert isinstance(state, NDArray[float])
    assert isinstance(action, (NDArray[int], NDArray([float])))
    assert isinstance(reward, NDArray[int])
    assert isinstance(done, NDArray[int])
    assert state.shape == (24 * 100, 7)
    assert action.shape == reward.shape == done.shape == (24 * 100,)
    assert action.shape == pscore.shape

    timestep = state[:, 0]
    remaining_budget = state[:, 1]
    budget_comsumption_rate = state[:, 2]
    cost_per_mille_of_impression = state[:, 3]
    winning_rate = state[:, 4]
    reward_ = state[:, 5]
    adjust_rate = state[:, 6]

    assert 0 <= timestep.all() < env.step_per_episode
    assert 0 <= remaining_budget.all() <= env.initial_budget
    assert 0 <= budget_comsumption_rate.all() <= 1
    assert 0 <= cost_per_mille_of_impression.all()
    assert 0 <= winning_rate.all() <= 1
    assert (reward_ == reward).all()
    assert (adjust_rate == action).all()

    impression = info["impression"]
    click = info["click"]
    conversion = info["conversion"]
    average_bid_price = info["average_bid_price"]

    assert (0 <= conversion <= click <= impression).all()
    assert 0 <= average_bid_price.all()
    assert isinstance(impression, NDArray[int])
    assert isinstance(click, NDArray[int])
    assert isinstance(conversion, NDArray[int])

    if objective == "click":
        assert (reward == click).all()
    else:  # "conversion"
        assert (reward == conversion).all()
