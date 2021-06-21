import pytest

import numpy as np

from _gym.utils import NormalDistribution
from _gym.env.simulator.rtb_synthetic import RTBSyntheticSimulator


# n_ads, n_users, ad_feature_dim, user_feature_dim, ad_sampling_rate, user_sampling_rate, standard_bid_price_distribution, minimum_standard_bid_price, trend_interval, err, description
invalid_input_of_init = [
    (
        -1,  #
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "n_ads must be a positive",
    ),
    (
        0,  #
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "n_ads must be a positive",
    ),
    (
        1.5,  #
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "n_ads must be a positive interger",
    ),
    (
        "1",  #
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "n_ads must be a positive interger",
    ),
    (
        5,
        -1,  #
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "n_users must be a positive",
    ),
    (
        5,
        0,  #
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "n_users must be a positive",
    ),
    (
        5,
        1.5,  #
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "n_users must be a positive interger",
    ),
    (
        5,
        "1",  #
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "n_users must be a positive interger",
    ),
    (
        5,
        5,
        -1,  #
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_feature_dim must be a positive",
    ),
    (
        5,
        5,
        0,  #
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_feature_dim must be a positive",
    ),
    (
        5,
        5,
        1.5,  #
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_feature_dim must be a positive interger",
    ),
    (
        5,
        5,
        "1",  #
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_feature_dim must be a positive interger",
    ),
    (
        5,
        5,
        2,
        -1,  #
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_feature_dim must be a positive",
    ),
    (
        5,
        5,
        2,
        0,  #
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_feature_dim must be a positive",
    ),
    (
        5,
        5,
        2,
        1.5,  #
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_feature_dim must be a positive interger",
    ),
    (
        5,
        5,
        2,
        "1",  #
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_feature_dim must be a positive interger",
    ),
    (
        5,
        5,
        2,
        2,
        -np.ones(5),  #
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_sampling_rate must be an 1-dimensional NDArray of non-negative",
    ),
    (
        5,
        5,
        2,
        2,
        np.zeros(5),  #
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_sampling_rate must be an 1-dimensional NDArray of non-negative",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones((2, 5)),  #
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "ad_sampling_rate must be an 1-dimensional NDArray of non-negative",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        -np.ones(5),  #
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "user_sampling_rate must be an 1-dimensional NDArray of non-negative",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.zeros(5),  #
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "user_sampling_rate must be an 1-dimensional NDArray of non-negative",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones((2, 5)),  #
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "user_sampling_rate must be an 1-dimensional NDArray of non-negative",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(4),  #
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "length of ad_sampling_rate must be equal to n_ads",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(4),  #
        NormalDistribution(mean=100, std=20),
        50,
        24,
        ValueError,
        "length of user_sampling_rate must be equal to n_users",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        "xxx",  #
        50,
        24,
        ValueError,
        "standard_bid_price_distribution must be a NormalDistribution",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=np.array([100, 100]), std=np.array([20, 20])),  #
        50,
        24,
        ValueError,
        "standard_bid_price_distribution must have a single parameter",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        -1,  #
        24,
        ValueError,
        "minimum_standard_bid_price must be a float value within [0, standard_bid_price_distribution.mean]",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        101,  #
        24,
        ValueError,
        "minimum_standard_bid_price must be a float value within [0, standard_bid_price_distribution.mean]",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        "1",  #
        24,
        ValueError,
        "minimum_standard_bid_price must be a float value within [0, standard_bid_price_distribution.mean]",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        -1,  #
        ValueError,
        "trend_interval must be a positive",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        0,  #
        ValueError,
        "trend_interval must be a positive",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        1.5,  #
        ValueError,
        "trend_interval must be a positive interger",
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        "1",  #
        ValueError,
        "trend_interval must be a positive interger",
    ),
]


@pytest.mark.parametrize(
    "n_ads, n_users, ad_feature_dim, user_feature_dim, ad_sampling_rate, user_sampling_rate, standard_bid_price_distribution, minimum_standard_bid_price, trend_interval, err, description",
    invalid_input_of_init,
)
def test_init_using_invalid_input(
    n_ads,
    n_users,
    ad_feature_dim,
    user_feature_dim,
    ad_sampling_rate,
    user_sampling_rate,
    standard_bid_price_distribution,
    minimum_standard_bid_price,
    trend_interval,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        RTBSyntheticSimulator(
            n_ads,
            n_users,
            ad_feature_dim,
            user_feature_dim,
            ad_sampling_rate,
            user_sampling_rate,
            standard_bid_price_distribution,
            minimum_standard_bid_price,
            trend_interval,
        )


# n_ads, n_users, ad_feature_dim, user_feature_dim, ad_sampling_rate, user_sampling_rate, standard_bid_price_distribution, minimum_standard_bid_price, trend_interval
valid_input_of_init = [
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
    ),
    (
        5,
        5,
        2,
        2,
        np.array([0, 0, 0, 0, 0.01]),  #
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        24,
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.array([0, 0, 0, 0, 0.01]),  #
        NormalDistribution(mean=100, std=20),
        50,
        24,
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        0,  #
        24,
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        100,  #
        24,
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        2,  #
    ),
    (
        5,
        5,
        2,
        2,
        np.ones(5),
        np.ones(5),
        NormalDistribution(mean=100, std=20),
        50,
        100,  #
    ),
]


@pytest.mark.parametrize(
    "n_ads, n_users, ad_feature_dim, user_feature_dim, ad_sampling_rate, user_sampling_rate, standard_bid_price_distribution, minimum_standard_bid_price, trend_interval",
    valid_input_of_init,
)
def test_init_using_invalid_input(
    n_ads,
    n_users,
    ad_feature_dim,
    user_feature_dim,
    ad_sampling_rate,
    user_sampling_rate,
    standard_bid_price_distribution,
    minimum_standard_bid_price,
    trend_interval,
):
    simulator = RTBSyntheticSimulator(
        n_ads=n_ads,
        n_users=n_users,
        ad_feature_dim=ad_feature_dim,
        user_feature_dim=user_feature_dim,
        ad_sampling_rate=ad_sampling_rate,
        user_sampling_rate=user_sampling_rate,
        standard_bid_price_distribution=standard_bid_price_distribution,
        minimum_standard_bid_price=minimum_standard_bid_price,
        trend_interval=trend_interval,
    )
    assert simulator.standard_bid_price == standard_bid_price_distribution.mean


@pytest.mark.parametrize("volume", [(-1), (0), (1.5), ("1")])
def test_generate_auction_using_invalid_input(volume):
    simulator = RTBSyntheticSimulator()
    with pytest.raises(ValueError):
        simulator.generate_auction(volume)


def test_generate_auction_using_valid_input():
    volume = 10
    simulator = RTBSyntheticSimulator()
    ad_ids, user_ids = simulator.generate_auction(volume)

    assert ad_ids.ndim == 1
    assert user_ids.ndim == 1
    assert len(ad_ids) == len(user_ids) == volume
    assert 0 < ad_ids.min() and ad_ids.max() < simulator.n_ads
    assert 0 < user_ids.min() and user_ids.max() < simulator.n_users
    assert np.allclose(np.mod(ad_ids, 1), 0)
    assert np.allclose(np.mod(user_ids, 1), 0)


# n_ads, n_users, ad_ids, user_ids, err, description
invalid_input_of_map_idx_to_contexts = [
    (
        2,
        2,
        -np.ones(3, dtype=int),  #
        np.ones(3, dtype=int),
        ValueError,
        "ad_ids must be 1-dimensional NDArray with integers within \[0, n_ads\)",
    ),
    (
        2,
        2,
        np.ones(3, dtype=float) / 2,  #
        np.ones(3, dtype=int),
        IndexError,
        "arrays used as indices must be of integer",
    ),
    (
        2,
        2,
        np.arange(3, dtype=int),  #
        np.ones(3, dtype=int),
        ValueError,
        "ad_ids must be 1-dimensional NDArray with integers within \[0, n_ads\)",
    ),
    (
        2,
        2,
        np.ones((2, 3), dtype=int),  #
        np.ones(3, dtype=int),
        ValueError,
        "ad_ids must be 1-dimensional NDArray with integers within \[0, n_ads\)",
    ),
    (
        2,
        2,
        np.ones(3, dtype=int),
        -np.ones(3, dtype=int),  #
        ValueError,
        "user_ids must be 1-dimensional NDArray with integers within \[0, n_users\)",
    ),
    (
        2,
        2,
        np.ones(3, dtype=int),
        np.ones(3, dtype=float) / 2,  #
        IndexError,
        "arrays used as indices must be of integer",
    ),
    (
        2,
        2,
        np.ones(3, dtype=int),
        np.arange(3, dtype=int),  #
        ValueError,
        "user_ids must be 1-dimensional NDArray with integers within \[0, n_users\)",
    ),
    (
        2,
        2,
        np.ones(3, dtype=int),
        np.ones((2, 3), dtype=int),  #
        ValueError,
        "user_ids must be 1-dimensional NDArray with integers within \[0, n_users\)",
    ),
    (
        2,
        2,
        np.ones(3, dtype=int),  #
        np.ones(2, dtype=int),  #
        ValueError,
        "ad_ids and user_ids must have same",
    ),
]


@pytest.mark.parametrize(
    "n_ads, n_users, ad_ids, user_ids, err, description",
    invalid_input_of_map_idx_to_contexts,
)
def test_map_idx_to_contexts_using_invalid_input(
    n_ads,
    n_users,
    ad_ids,
    user_ids,
    err,
    description,
):
    simulator = RTBSyntheticSimulator(n_ads=n_ads, n_users=n_users)
    with pytest.raises(err, match=f"{description}*"):
        simulator.map_idx_to_contexts(ad_ids, user_ids)


# n_ads, n_users, ad_ids, user_ids
valid_input_of_map_idx_to_contexts = [
    (
        2,
        2,
        np.arange(2, dtype=int),
        np.arange(2, dtype=int),
    ),
    (
        2,
        2,
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
    ),
    (
        2,
        2,
        np.zeros(3, dtype=int),
        np.zeros(3, dtype=int),
    ),
]


@pytest.mark.parametrize(
    "n_ads, n_users, ad_ids, user_ids",
    valid_input_of_map_idx_to_contexts,
)
def test_map_idx_to_contexts(n_ads, n_users, ad_ids, user_ids):
    simulator = RTBSyntheticSimulator(n_ads=n_ads, n_users=n_users)
    contexts = simulator.map_idx_to_contexts(ad_ids, user_ids)

    assert contexts.ndim == 2
    assert len(ad_ids) == len(contexts)
    assert contexts.shape[1] == simulator.ad_feature_dim + simulator.user_feature_dim


# n_ads, n_users, timestep, ad_ids, user_ids, bid_prices, err, description
invalid_input_of_calc_and_sample_outcome = [
    (
        2,
        2,
        -1,  #
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        ValueError,
        "timestep must be a non-negative",
    ),
    (
        2,
        2,
        1.5,  #
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        ValueError,
        "timestep must be a non-negative interger",
    ),
    (
        2,
        2,
        "1",  #
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        ValueError,
        "timestep must be a non-negative interger",
    ),
    (
        2,
        2,
        0,
        -np.ones(3, dtype=int),  #
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        ValueError,
        "ad_ids must be 1-dimensional NDArray with integers within \[0, n_ads\)",
    ),
    (
        2,
        2,
        0,
        np.ones(3, dtype=float) / 2,  #
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        IndexError,
        "arrays used as indices must be of integer",
    ),
    (
        2,
        2,
        0,
        np.arange(3, dtype=int),  #
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        ValueError,
        "ad_ids must be 1-dimensional NDArray with integers within \[0, n_ads\)",
    ),
    (
        2,
        2,
        0,
        np.ones(3, dtype=int),
        -np.ones(3, dtype=int),  #
        np.ones(3, dtype=int),
        ValueError,
        "user_ids must be 1-dimensional NDArray with integers within \[0, n_users\)",
    ),
    (
        2,
        2,
        0,
        np.ones(3, dtype=int),
        np.ones(3, dtype=float) / 2,  #
        np.ones(3, dtype=int),
        IndexError,
        "arrays used as indices must be of integer",
    ),
    (
        2,
        2,
        0,
        np.ones(3, dtype=int),
        np.arange(3, dtype=int),  #
        np.ones(3, dtype=int),
        ValueError,
        "user_ids must be 1-dimensional NDArray with integers within \[0, n_users\)",
    ),
    (
        2,
        2,
        0,
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        -np.ones(3, dtype=int),  #
        ValueError,
        "bid_prices must be 1-dimensional NDArray with non-negative",
    ),
    (
        2,
        2,
        0,
        np.ones((2, 3), dtype=int),  #
        np.ones((2, 3), dtype=int),
        np.ones((2, 3), dtype=int),
        ValueError,
        "bid_prices must be 1-dimensional NDArray",
    ),
    (
        2,
        2,
        0,
        np.ones(2, dtype=int),
        np.ones(3, dtype=int),  #
        np.ones(3, dtype=int),
        ValueError,
        "ad_ids, user_ids, and bid_prices must have same",
    ),
    (
        2,
        2,
        0,
        np.ones(3, dtype=int),
        np.ones(2, dtype=int),  #
        np.ones(3, dtype=int),
        ValueError,
        "ad_ids, user_ids, and bid_prices must have same",
    ),
    (
        2,
        2,
        0,
        np.ones(3, dtype=int),
        np.ones(3, dtype=int),
        np.ones(2, dtype=int),  #
        ValueError,
        "ad_ids, user_ids, and bid_prices must have same",
    ),
]


@pytest.mark.parametrize(
    "n_ads, n_users, timestep, ad_ids, user_ids, bid_prices, err, description",
    invalid_input_of_calc_and_sample_outcome,
)
def test_calc_and_sample_outcome_using_invalid_input(
    n_ads,
    n_users,
    timestep,
    ad_ids,
    user_ids,
    bid_prices,
    err,
    description,
):
    simulator = RTBSyntheticSimulator(n_ads=n_ads, n_users=n_users)
    with pytest.raises(err, match=f"{description}*"):
        simulator.calc_and_sample_outcome(timestep, ad_ids, user_ids, bid_prices)


# n_ads, n_users, ad_ids, user_ids
valid_input_of_calc_and_sample_outcome = [
    (
        20,
        20,
        np.arange(20, dtype=int),
        np.arange(20, dtype=int),
    ),
    (
        20,
        20,
        np.ones(30, dtype=int),
        np.ones(30, dtype=int),
    ),
    (
        20,
        20,
        np.zeros(30, dtype=int),
        np.zeros(30, dtype=int),
    ),
]


@pytest.mark.parametrize(
    "n_ads, n_users, ad_ids, user_ids",
    valid_input_of_calc_and_sample_outcome,
)
def test_calc_and_sample_outcome(n_ads, n_users, ad_ids, user_ids):
    timestep = 0
    search_volume = len(ad_ids)
    bid_zero = np.zeros(search_volume)
    bid_medium = np.arange(search_volume) * 10
    bid_large = np.ones(search_volume) * 1000

    simulator = RTBSyntheticSimulator(n_ads=n_ads, n_users=n_users)
    (
        costs_zero,
        impressions_zero,
        clicks_zero,
        conversions_zero,
    ) = simulator.calc_and_sample_outcome(timestep, ad_ids, user_ids, bid_zero)
    (
        costs_medium,
        impressions_medium,
        clicks_medium,
        conversions_medium,
    ) = simulator.calc_and_sample_outcome(timestep, ad_ids, user_ids, bid_medium)
    (
        _,
        impressions_large,
        _,
        _,
    ) = simulator.calc_and_sample_outcome(timestep, ad_ids, user_ids, bid_large)

    assert np.allclose(costs_zero, 0)
    assert np.allclose(impressions_zero, 0)
    assert np.allclose(clicks_zero, 0)
    assert np.allclose(conversions_zero, 0)

    assert np.allclose(np.mod(costs_medium, 1), 0)
    assert np.array_equal(
        impressions_medium, impressions_medium.astype(bool).astype(int)
    )
    assert np.array_equal(clicks_medium, clicks_medium.astype(bool).astype(int))
    assert np.array_equal(
        conversions_medium, conversions_medium.astype(bool).astype(int)
    )

    assert np.sum(impressions_medium > impressions_large) == 0
