import pytest

import numpy as np
from sklearn.linear_model import LogisticRegression

from _gym.utils import NormalDistribution
from _gym.simulator import RTBSyntheticSimulator


def test_init():
    # default value check
    simulator = RTBSyntheticSimulator()
    assert simulator.objective == "conversion"
    assert not simulator.use_reward_predictor
    assert simulator.step_per_episode == simulator.trend_interval

    # objective -- failure case
    with pytest.raises(ValueError):
        RTBSyntheticSimulator(objective="impression")

    # objective -- success case
    simulator = RTBSyntheticSimulator(objective="conversion")
    assert simulator.objective == "conversion"

    simulator = RTBSyntheticSimulator(objective="click")
    assert simulator.objective == "click"

    # use_reward_predictor -- failure case
    with pytest.raises(ValueError):
        RTBSyntheticSimulator(reward_predictor=LogisticRegression)

    # use_reward_predictor -- success case
    RTBSyntheticSimulator(reward_predictor=LogisticRegression())

    # step_per_episode -- failure case
    with pytest.raises(ValueError):
        RTBSyntheticSimulator(step_per_episode=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(step_per_episode=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(step_per_episode=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(step_per_episode="1")

    # n_ads, n_users -- failure case
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

    # ad_feature_dim, user_feature_dim -- failure case
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

    # standard_bid_price_distribution -- failure case
    with pytest.raises(ValueError):
        RTBSyntheticSimulator(standard_bid_price_distribution=10)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(
            standard_bid_price_distribution=NormalDistribution(mean=0, std=0)
        )

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(
            standard_bid_price_distribution=NormalDistribution(mean=-1, std=0)
        )

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(
            standard_bid_price_distribution=NormalDistribution(mean="1", std=0)
        )

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(
            standard_bid_price_distribution=NormalDistribution(mean=1, std=-1)
        )

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(
            standard_bid_price_distribution=NormalDistribution(mean=1, std="1")
        )

    # standard_bid_price_distribution -- success case
    RTBSyntheticSimulator(
        standard_bid_price_distribution=NormalDistribution(mean=1, std=0)
    )

    RTBSyntheticSimulator(
        standard_bid_price_distribution=NormalDistribution(mean=1.5, std=0)
    )

    RTBSyntheticSimulator(
        standard_bid_price_distribution=NormalDistribution(mean=1, std=1.5)
    )

    # minimum_standard_bid_price -- failure case
    with pytest.raises(ValueError):
        RTBSyntheticSimulator(minimum_standard_bid_price=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(minimum_standard_bid_price=101)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(minimum_standard_bid_price="1")

    # minimum_standard_bid_price -- success case
    RTBSyntheticSimulator(minimum_standard_bid_price=0)

    RTBSyntheticSimulator(minimum_standard_bid_price=100)

    RTBSyntheticSimulator(minimum_standard_bid_price=1.5)

    # trend_interval -- failure case
    with pytest.raises(ValueError):
        RTBSyntheticSimulator(trend_interval=0)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(trend_interval=-1)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(trend_interval=0.5)

    with pytest.raises(ValueError):
        RTBSyntheticSimulator(trend_interval="1")

    # random_state -- failure case
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
        (1, 1, np.array([100, 1]), np.array([0, 1])),
        (1, 1, np.array([0, 1]), np.array([-1, 1])),
        (1, 1, np.array([0, 1]), np.array([100, 1])),
        (1, 1, np.array([]), np.array([])),
        (1, 1, np.array([0]), np.array([0, 1])),
        (1, 1, np.array([[0], [1]]), np.array([[0], [1]])),
    ],
)
def test_simulate_auction_failure_case(timestep, adjust_rate, ad_ids, user_ids):
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
def test_simulate_auction_success_case(timestep, adjust_rate, ad_ids, user_ids):
    simulator = RTBSyntheticSimulator()
    bid_prices, costs, impressions, clicks, conversions = simulator.simulate_auction(
        timestep, adjust_rate, ad_ids, user_ids
    )

    assert (bid_prices >= costs).all()
    assert (impressions >= clicks).all()
    assert (clicks >= conversions).all()
    assert np.array_equal(impressions, impressions.astype(bool).astype(int))
    assert np.array_equal(clicks, clicks.astype(bool).astype(int))
    assert np.array_equal(conversions, conversions.astype(bool).astype(int))
    assert np.allclose(np.mod(bid_prices, 1), 0)
    assert np.allclose(np.mod(costs, 1), 0)
    assert np.allclose(np.mod(impressions, 1), 0)
    assert np.allclose(np.mod(clicks, 1), 0)
    assert np.allclose(np.mod(conversions, 1), 0)
    assert ad_ids.shape == bid_prices.shape == costs.shape
    assert ad_ids.shape == impressions.shape == clicks.shape == conversions.shape


def test_simulate_auction_bid_prices_value_check():
    simulator = RTBSyntheticSimulator()

    bid_prices_01, _, _, _, _ = simulator.simulate_auction(
        timestep=0,
        adjust_rate=0.1,
        ad_ids=np.arange(10),
        user_ids=np.arange(10),
    )
    bid_prices_1, _, _, _, _ = simulator.simulate_auction(
        timestep=0,
        adjust_rate=1,
        ad_ids=np.arange(10),
        user_ids=np.arange(10),
    )
    bid_prices_5, _, _, _, _ = simulator.simulate_auction(
        timestep=0,
        adjust_rate=5,
        ad_ids=np.arange(10),
        user_ids=np.arange(10),
    )
    bid_prices_10, _, _, _, _ = simulator.simulate_auction(
        timestep=0,
        adjust_rate=10,
        ad_ids=np.arange(10),
        user_ids=np.arange(10),
    )
    assert (bid_prices_01 < bid_prices_1).all()
    assert (bid_prices_1 < bid_prices_5).all()
    assert (bid_prices_5 < bid_prices_10).all()


@pytest.mark.parametrize("n_samples", [(-1), (0), (1.5), ("1")])
def test_fit_reward_estimator_failure_case(n_samples):
    simulator = RTBSyntheticSimulator(
        reward_predictor=LogisticRegression(),
    )

    with pytest.raises(ValueError):
        simulator.fit_reward_predictor(n_samples)


def test_predict_reward_and_calc_ground_truth_reward_value_check():
    simulator_A = RTBSyntheticSimulator(
        objective="conversion",
        reward_predictor=LogisticRegression(),
        ad_feature_dim=1,
        user_feature_dim=1,
    )
    simulator_B = RTBSyntheticSimulator(
        objective="click",
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

    assert 0 <= predicted_rewards.min() and predicted_rewards.max() <= 1
    assert 0 <= ground_truth_rewards_A.min() and ground_truth_rewards_A.max() <= 1
    assert (ground_truth_rewards_A <= ground_truth_rewards_B).all()
    assert len(contexts) == len(predicted_rewards) == len(ground_truth_rewards_A)


def test_determine_bid_price_value_check():
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
        reward_predictor=LogisticRegression(),
        ad_feature_dim=1,
        user_feature_dim=1,
    )
    simulator_D = RTBSyntheticSimulator(
        objective="click",
        reward_predictor=LogisticRegression(),
        ad_feature_dim=1,
        user_feature_dim=1,
    )

    simulator_C.fit_reward_predictor(n_samples=100)
    simulator_D.fit_reward_predictor(n_samples=100)

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

    assert (bid_prices_A >= 0).all()
    assert (bid_prices_B >= 0).all()
    assert (bid_prices_C >= 0).all()
    assert (bid_prices_D >= 0).all()
    assert not np.array_equal(bid_prices_A, bid_prices_B)
    assert not np.array_equal(bid_prices_B, bid_prices_C)
    assert not np.array_equal(bid_prices_C, bid_prices_D)
    assert not np.array_equal(bid_prices_D, bid_prices_A)
    assert not np.array_equal(bid_prices_A, bid_prices_C)
    assert not np.array_equal(bid_prices_B, bid_prices_D)
    assert np.allclose(np.mod(bid_prices_A, 1), 0)
    assert np.allclose(np.mod(bid_prices_B, 1), 0)
    assert np.allclose(np.mod(bid_prices_C, 1), 0)
    assert np.allclose(np.mod(bid_prices_D, 1), 0)
    assert (
        len(contexts)
        == len(bid_prices_A)
        == len(bid_prices_B)
        == len(bid_prices_C)
        == len(bid_prices_D)
    )


def test_map_idx_to_contexts_value_check():
    simulator = RTBSyntheticSimulator()
    contexts = simulator._map_idx_to_contexts(
        ad_ids=np.array([0, 1]),
        user_ids=np.array([0, 1]),
    )

    assert contexts.shape == (2, simulator.ad_feature_dim + simulator.user_feature_dim)
