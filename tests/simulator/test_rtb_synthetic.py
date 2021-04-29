import pytest
from nptyping import NDArray

import numpy as np
from sklearn.linear_model import LogisticRegression

from _gym.simulator import RTBSyntheticSimulator


def test_init():
    simulator = RTBSyntheticSimulator()
    assert simulator.objective == "conversion"
    assert not simulator.use_reward_predictor
    assert simulator.step_per_episode == simulator.trend_interval

    simulator = RTBSyntheticSimulator(objective="conversion")
    assert simulator.objective == "conversion"

    simulator = RTBSyntheticSimulator(objective="click")
    assert simulator.objective == "click"

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(objective="impression")

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(use_reward_predictor=True)

    RTBSyntheticSimulator(
        use_reward_predictor=True, reward_predictor=LogisticRegression()
    )

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(
            use_reward_predictor=True, reward_predictor=LogisticRegression
        )

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(step_per_episode=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(step_per_episode=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(step_per_episode=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(step_per_episode="1")

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(n_ads=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(n_ads=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(n_ads=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(n_ads="1")

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(n_users=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(n_users=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(n_users=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(n_users="1")

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(ad_feature_dim=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(ad_feature_dim=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(ad_feature_dim=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(ad_feature_dim="1")

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(user_feature_dim=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(user_feature_dim=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(user_feature_dim=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(user_feature_dim="1")

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(standard_bid_price=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(standard_bid_price=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(user_feature_dim="1")

    RTBSyntheticSimulator(standard_bid_price=1.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(trend_interval=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(trend_interval=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(trend_interval=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(trend_interval="1")

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(random_state=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(random_state=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(random_state="1")


@pytest.mark.parametrize(
    "timestep, adjust_rate, ad_ids, user_ids",
    [
        (-1, 1, np.array([0, 1]), np.array([0, 1])),
        (1.5, 1, np.array([0, 1]), np.array([0, 1])),
        (24, 1, np.array([0, 1]), np.array([0, 1])),
        ("1", 1, np.array([0, 1]), np.array([0, 1])),
        (1, 0.01, np.array([0, 1]), np.array([0, 1])),
        (1, 10.1, np.array([0, 1]), np.array([0, 1])),
        (1, -1, np.array([0, 1]), np.array([0, 1])),
        (1, "1", np.array([0, 1]), np.array([0, 1])),
        (1, 1, np.array([-1, 1]), np.array([0, 1])),
        (1, 1, np.array([1.5, 1]), np.array([0, 1])),
        (1, 1, np.array([100, 1]), np.array([0, 1])),
        (1, 1, np.array([0, 1]), np.array([-1, 1])),
        (1, 1, np.array([0, 1]), np.array([1.5, 1])),
        (1, 1, np.array([0, 1]), np.array([100, 1])),
        (1, 1, np.array([]), np.array([])),
        (1, 1, np.array([0]), np.array([0, 1])),
        (1, 1, np.array([[0], [1]]), np.array([[0], [1]])),
    ],
)
def test_simulate_auction(timestep, adjust_rate, ad_ids, user_ids):
    simulator = RTBSyntheticSimulator()

    with pytest.raises(ValueError):
        simulator.simulate_auction(timestep, adjust_rate, ad_ids, user_ids)


@pytest.mark.parametrize(
    "timestep, adjust_rate, ad_ids, user_ids",
    [
        (1, 0.1, np.array([0, 1]), np.array([0, 1])),
        (1, 10, np.array([0, 1]), np.array([0, 1])),
    ],
)
def test_simulate_auction_(timestep, adjust_rate, ad_ids, user_ids):
    simulator = RTBSyntheticSimulator()
    bid_prices, costs, impressions, clicks, conversions = simulator.simulate_auction(
        timestep, adjust_rate, ad_ids, user_ids
    )

    assert bid_prices >= costs
    assert impressions >= clicks >= conversions
    assert np.array_equal(impressions, impressions.astype(bool))
    assert np.array_equal(clicks, clicks.astype(bool))
    assert np.array_equal(conversions, conversions.astype(bool))
    assert isinstance(bid_prices, NDArray[int])
    assert isinstance(costs, NDArray[int])
    assert isinstance(impressions, NDArray[int])
    assert isinstance(clicks, NDArray[int])
    assert isinstance(conversions, NDArray[int])
    assert ad_ids.shape == bid_prices.shape == costs.Shape
    assert ad_ids.shape == impressions.shape == clicks.shape == conversions.Shape


@pytest.mark.parametrize("n_samples", [(-1), (0), (1.5), ("1")])
def test_fit_reward_estimator(n_samples):
    simulator = RTBSyntheticSimulator(
        use_reward_predictor=True,
        reward_predictor=LogisticRegression(),
    )

    with pytest.raises(ValueError):
        simulator.fit_reward_predictor(n_samples)


def test_predict_reward_and_calc_ground_truth_reward():
    simulator_A = RTBSyntheticSimulator(
        objective="conversion",
        use_reward_predictor=True,
        reward_predictor=LogisticRegression(),
        ad_feature_dim=1,
        user_feature_dim=1,
    )
    simulator_B = RTBSyntheticSimulator(
        objective="click",
        use_reward_predictor=True,
        reward_predictor=LogisticRegression(),
        ad_feature_dim=1,
        user_feature_dim=1,
    )
    simulator_A.fit_reward_predictor(n_samples=100)
    simulator_B.fit_reward_predictor(n_samples=100)

    contexts = np.array([[1.1, 2.2], [3.3, 4.4]])
    predicted_rewards = simulator_A._predict_reward(timestep=0, contexts=contexts)
    ground_truth_rewards_A = simulator_A._calc_ground_truth_reward(
        timestep=0, contexts=contexts
    )
    ground_truth_rewards_B = simulator_B._calc_ground_truth_reward(
        timestep=0, contexts=contexts
    )

    assert 0 <= predicted_rewards.all() <= 1
    assert 0 <= ground_truth_rewards_A.all() <= 1
    assert ground_truth_rewards_A <= ground_truth_rewards_B
    assert isinstance(predicted_rewards, NDArray[float])
    assert isinstance(ground_truth_rewards_A, NDArray[float])
    assert len(contexts) == len(predicted_rewards) == len(ground_truth_rewards_A)


def test_determine_bid_price():
    simulator_A = RTBSyntheticSimulator(
        objective="conversion",
        ad_feature_dim=1,
        user_feature_dim=1,
    )
    simulator_B = RTBSyntheticSimulator(
        objective="click",
        ad_feature_dim=1,
        user_feature_dim=1,
    )
    simulator_C = RTBSyntheticSimulator(
        objective="conversion",
        use_reward_predictor=True,
        reward_predictor=LogisticRegression(),
        ad_feature_dim=1,
        user_feature_dim=1,
    )
    simulator_D = RTBSyntheticSimulator(
        objective="click",
        use_reward_predictor=True,
        reward_predictor=LogisticRegression(),
        ad_feature_dim=1,
        user_feature_dim=1,
    )

    contexts = np.array([[1.1, 2.2], [3.3, 4.4]])
    bid_prices_A = simulator_A._determine_bid_price(
        timestep=0, adjust_rate=1, contexts=contexts
    )
    bid_prices_B = simulator_B._determine_bid_price(
        timestep=0, adjust_rate=1, contexts=contexts
    )
    bid_prices_C = simulator_C._determine_bid_price(
        timestep=0, adjust_rate=1, contexts=contexts
    )
    bid_prices_D = simulator_D._determine_bid_price(
        timestep=0, adjust_rate=1, contexts=contexts
    )

    assert bid_prices_A >= 0
    assert not np.array_equal(bid_prices_A, bid_prices_B)
    assert not np.array_equal(bid_prices_B, bid_prices_C)
    assert not np.array_equal(bid_prices_C, bid_prices_D)
    assert not np.array_equal(bid_prices_D, bid_prices_A)
    assert not np.array_equal(bid_prices_A, bid_prices_C)
    assert not np.array_equal(bid_prices_B, bid_prices_D)
    assert isinstance(bid_prices_A, NDArray[float])
    assert len(contexts) == len(bid_prices)


def test_map_idx_to_contexts():
    simulator = RTBSyntheticSimulator()
    contexts = simulator._map_idx_to_contexts(
        ad_ids=np.array([0, 1]),
        user_ids=np.array([0, 1]),
    )

    assert isinstance(contexts, NDArray[float])
    assert contexts.shape == (2, simulator.ad_feature_dim + simulator.user_feature_dim)
