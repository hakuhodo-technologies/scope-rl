import pytest

import numpy as np

from _gym.env.simulator.function import WinningFunction, CTR, CVR


# ks, thetas, bid_prices, err, description
invalid_input_of_winning_function_sample_outcome = [
    (
        np.array([[1], [2]]),
        np.array([[1], [2]]),
        np.array([[1], [2]]),
        ValueError,
        "ks must be an 1-dimensional NDArray",
    ),
    (
        np.array([-1, 2]),
        np.array([1, 2]),
        np.array([1, 2]),
        ValueError,
        "ks must be an 1-dimensional NDArray of positive",
    ),
    (
        np.array([0, 2]),
        np.array([1, 2]),
        np.array([1, 2]),
        ValueError,
        "ks must be an 1-dimensional NDArray of positive",
    ),
    (
        np.array([1, 2]),
        np.array([-1, 2]),
        np.array([1, 2]),
        ValueError,
        "thetas must be an 1-dimensional NDArray of positive",
    ),
    (
        np.array([1, 2]),
        np.array([0, 2]),
        np.array([1, 2]),
        ValueError,
        "thetas must be an 1-dimensional NDArray of positive",
    ),
    (
        np.array([1, 2]),
        np.array([1, 2]),
        np.array([-1, 2]),
        ValueError,
        "bid_prices must be an 1-dimensional NDArray of non-negative",
    ),
    (
        np.array([1, 2]),
        np.array([1, 2]),
        np.array([1]),
        ValueError,
        "ks, thetas, and bid_prices must have same length",
    ),
    (
        np.array([1, 2]),
        np.array([1]),
        np.array([1, 2]),
        ValueError,
        "ks, thetas, and bid_prices must have same length",
    ),
    (
        np.array([1]),
        np.array([1, 2]),
        np.array([1, 2]),
        ValueError,
        "ks, thetas, and bid_prices must have same length",
    ),
]


@pytest.mark.parametrize(
    "ks, thetas, bid_prices, err, description",
    invalid_input_of_winning_function_sample_outcome,
)
def test_winning_price_sample_outcome_using_invalid_input(
    ks,
    thetas,
    bid_prices,
    err,
    description,
):
    winning_function = WinningFunction()
    with pytest.raises(err, match=f"{description}*"):
        winning_function.sample_outcome(ks, thetas, bid_prices)


# ks, thetas, bid_prices
valid_input_of_winning_function_sample_outcome = [
    (np.array([1, 2]), np.array([1, 2]), np.array([1, 2])),
    (np.array([0.5, 2]), np.array([1, 2]), np.array([1, 2])),
    (np.array([1, 2]), np.array([0.5, 2]), np.array([1, 2])),
    (np.array([1, 2]), np.array([1, 2]), np.array([1, 1.1])),
    (np.array([1]), np.array([1]), np.array([1])),
]


@pytest.mark.parametrize(
    "ks, thetas, bid_prices",
    valid_input_of_winning_function_sample_outcome,
)
def test_winning_price_sample_outcome_using_valid_input(
    ks,
    thetas,
    bid_prices,
):
    winning_function = WinningFunction()
    impressions, winning_prices = winning_function.sample_outcome(
        ks, thetas, bid_prices
    )
    assert np.array_equal(impressions, impressions.astype(bool).astype(int))
    assert np.array_equal(impressions, winning_prices < bid_prices.astype(int))
    assert np.allclose(np.mod(impressions, 1), 0)
    assert np.allclose(np.mod(winning_prices, 1), 0)
    assert bid_prices.shape == impressions.shape == winning_prices.shape


# ad_feature_dim, user_feature_dim, trend_interval, err, description
invalid_input_of_ctr_cvr_init = [
    (
        -1,
        1,
        1,
        ValueError,
        "ad_feature_dim must be a positive",
    ),
    (
        0,
        1,
        1,
        ValueError,
        "ad_feature_dim must be a positive",
    ),
    (
        1.5,
        1,
        1,
        ValueError,
        "ad_feature_dim must be a positive interger",
    ),
    (
        "1",
        1,
        1,
        ValueError,
        "ad_feature_dim must be a positive interger",
    ),
    (
        1,
        -1,
        1,
        ValueError,
        "user_feature_dim must be a positive",
    ),
    (
        1,
        0,
        1,
        ValueError,
        "user_feature_dim must be a positive",
    ),
    (
        1,
        1.5,
        1,
        ValueError,
        "user_feature_dim must be a positive interger",
    ),
    (
        1,
        "1",
        1,
        ValueError,
        "user_feature_dim must be a positive interger",
    ),
    (
        1,
        1,
        -1,
        ValueError,
        "trend_interval must be a positive",
    ),
    (
        1,
        1,
        0,
        ValueError,
        "trend_interval must be a positive",
    ),
    (
        1,
        1,
        1.5,
        ValueError,
        "trend_interval must be a positive interger",
    ),
    (
        1,
        1,
        "1",
        ValueError,
        "trend_interval must be a positive interger",
    ),
]


@pytest.mark.parametrize(
    "ad_feature_dim, user_feature_dim, trend_interval, err, description",
    invalid_input_of_ctr_cvr_init,
)
def test_ctr_cvr_init_using_invalid_input(
    ad_feature_dim,
    user_feature_dim,
    trend_interval,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        CTR(
            ad_feature_dim=ad_feature_dim,
            user_feature_dim=user_feature_dim,
            trend_interval=trend_interval,
        )

    with pytest.raises(err, match=f"{description}*"):
        CVR(
            ad_feature_dim=ad_feature_dim,
            user_feature_dim=user_feature_dim,
            trend_interval=trend_interval,
        )


# ad_feature_dim, user_feature_dim, timestep, contexts, err, description
invalid_input_of_ctr_cvr_functions = [
    (
        1,
        1,
        -1,
        np.array([[1.1, 2.2], [3.3, 4.4]]),
        ValueError,
        "timestep must be an non-negative",
    ),
    (
        1,
        1,
        1.5,
        np.array([[1.1, 2.2], [3.3, 4.4]]),
        ValueError,
        "timestep must be an non-negative integer",
    ),
    (
        1,
        1,
        "0",
        np.array([[1.1, 2.2], [3.3, 4.4]]),
        ValueError,
        "timestep must be an non-negative integer",
    ),
    (
        1,
        1,
        np.array([-1, 0]),
        np.array([[1.1, 2.2], [3.3, 4.4]]),
        ValueError,
        "timestep must be an non-negative",
    ),
    (
        1,
        1,
        np.array([1.5, 0]),
        np.array([[1.1, 2.2], [3.3, 4.4]]),
        IndexError,
        "arrays used as indices must be of integer",
    ),
    (
        1,
        1,
        0,
        np.array([1.1, 2.2]),
        ValueError,
        "contexts must be 2-dimensional",
    ),
    (
        1,
        1,
        0,
        np.array([]),
        ValueError,
        "contexts must be 2-dimensional",
    ),
    (
        1,
        1,
        0,
        np.array([[1.1], [2.2]]),
        ValueError,
        "contexts must be 2-dimensional",
    ),
    (1, 1, 0, np.array([[[1.1, 2.2]]]), ValueError, "contexts must be 2-dimensional"),
    (
        1,
        1,
        np.array([0]),
        np.array([[1.1, 2.2], [3.3, 4.4]]),
        ValueError,
        "timestep and contexts must have same length",
    ),
]


@pytest.mark.parametrize(
    "ad_feature_dim, user_feature_dim, timestep, contexts, err, description",
    invalid_input_of_ctr_cvr_functions,
)
def test_ctr_csv_functions_using_invalid_input(
    ad_feature_dim,
    user_feature_dim,
    timestep,
    contexts,
    err,
    description,
):
    ctr = CTR(
        ad_feature_dim=ad_feature_dim,
        user_feature_dim=user_feature_dim,
        trend_interval=2,
    )
    cvr = CVR(
        ad_feature_dim=ad_feature_dim,
        user_feature_dim=user_feature_dim,
        trend_interval=2,
    )
    with pytest.raises(err, match=f"{description}*"):
        ctr.calc_prob(timestep, contexts)

    with pytest.raises(err, match=f"{description}*"):
        ctr.sample_outcome(timestep, contexts)

    with pytest.raises(err, match=f"{description}*"):
        cvr.calc_prob(timestep, contexts)

    with pytest.raises(err, match=f"{description}*"):
        cvr.sample_outcome(timestep, contexts)


# ad_feature_dim, user_feature_dim, trend_interval, timestep, contexts
valid_input_of_ctr_cvr_functions = [
    (1, 1, 2, 0, np.array([[1.1, 2.2], [3.3, 4.4]])),
    (1, 1, 2, 3, np.array([[1.1, 2.2], [3.3, 4.4]])),
    (1, 1, 2, np.array([0, 3]), np.array([[1.1, 2.2], [3.3, 4.4]])),
    (1, 1, 2, 0, np.array([[-1.1, -2.2], [3.3, 4.4]])),
]


@pytest.mark.parametrize(
    "ad_feature_dim, user_feature_dim, trend_interval, timestep, contexts",
    valid_input_of_ctr_cvr_functions,
)
def test_ctr_csv_functions_using_valid_input(
    ad_feature_dim,
    user_feature_dim,
    trend_interval,
    timestep,
    contexts,
):
    ctr = CTR(
        ad_feature_dim=ad_feature_dim,
        user_feature_dim=user_feature_dim,
        trend_interval=trend_interval,
    )
    cvr = CVR(
        ad_feature_dim=ad_feature_dim,
        user_feature_dim=user_feature_dim,
        trend_interval=trend_interval,
    )
    ctrs = ctr.calc_prob(timestep, contexts)
    clicks = ctr.sample_outcome(timestep, contexts)
    cvrs = cvr.calc_prob(timestep, contexts)
    conversions = cvr.sample_outcome(timestep, contexts)

    assert 0 <= ctrs.all() <= 1
    assert 0 <= cvrs.all() <= 1
    assert np.array_equal(clicks, clicks.astype(bool).astype(int))
    assert np.array_equal(conversions, conversions.astype(bool).astype(int))
    assert np.allclose(np.mod(clicks, 1), 0)
    assert np.allclose(np.mod(conversions, 1), 0)
    assert len(contexts) == len(ctrs) == len(cvrs)
    assert len(contexts) == len(clicks) == len(conversions)
