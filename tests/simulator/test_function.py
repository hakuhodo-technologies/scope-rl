import pytest
from nptyping import NDArray

import numpy as np

from _gym.simulator.function import WinningFunction, CTR, CVR


@pytest.mark.parametrize("random_state", [(-1), (1.5), ("1")])
def test_init_random_state_failure_case(random_state):
    with pytest.raises(ValueError):
        WinningFunction(random_state=random_state)

    with pytest.raises(ValueError):
        CTR(
            ad_feature_dim=1,
            user_feature_dim=1,
            trend_interval=1,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "ad_feature_dim, user_feature_dim, trend_interval",
    [
        (-1, 1, 1),
        (0, 1, 1),
        (1.5, 1, 1),
        ("1", 1, 1),
        (1, -1, 1),
        (1, 0, 1),
        (1.5, 1, 1),
        ("1", 1, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1.5),
        (1, 1, "1"),
    ],
)
def test_ctr_init_failure_case(ad_feature_dim, user_feature_dim, trend_interval):
    with pytest.raises(ValueError):
        CTR(
            ad_feature_dim=ad_feature_dim,
            user_feature_dim=user_feature_dim,
            trend_interval=trend_interval,
        )


@pytest.mark.parametrize(
    "ks, thetas, bid_prices",
    [
        (np.array([]), np.array([]), np.array([])),
        (np.array([1, 2]), np.array([1, 2]), np.array([1, 1.1])),
        (np.array([1, 2]), np.array([1, 2]), np.array([1])),
        (np.array([1, 2]), np.array([1]), np.array([1, 2])),
        (np.array([1]), np.array([1, 2]), np.array([1, 2])),
        (np.array([-1, 2]), np.array([1, 2]), np.array([1, 2])),
        (np.array([0, 2]), np.array([1, 2]), np.array([1, 2])),
        (np.array([1, 2]), np.array([-1, 2]), np.array([1, 2])),
        (np.array([-1, 2]), np.array([0, 2]), np.array([1, 2])),
        (np.array([1, 2]), np.array([1, 2]), np.array([-1, 2])),
        (np.array([[1], [2]]), np.array([1], [2]), np.array([1], [2])),
    ],
)
def test_wf_sample_outcome_failure_case(ks, thetas, bid_prices):
    winning_function = WinningFunction()

    with pytest.raises(ValueError):
        winning_function.sample_outcome(ks, thetas, bid_prices)


@pytest.mark.parametrize(
    "ks, thetas, bid_prices",
    [
        (np.array([1, 2]), np.array([1, 2]), np.array([1, 2])),
        (np.array([0.5, 2]), np.array([1, 2]), np.array([1, 2])),
        (np.array([1, 2]), np.array([0.5, 2]), np.array([1, 2])),
        (np.array([1]), np.array([1]), np.array([1])),
    ],
)
def test_wf_sample_outcome_success_case(ks, thetas, bid_prices):
    winning_function = WinningFunction()

    impressions, winning_prices = winning_function.sample_outcome(
        ks, thetas, bid_prices
    )

    assert np.array_equal(impressions, impressions.astype(bool))
    assert np.array_equal(impressions, winning_prices < bid_prices)
    assert isinstance(impressions, NDArray[int])
    assert isinstance(winning_prices, NDArray[int])
    assert bid_prices.shape == impressions.shape == winning_prices.shape


@pytest.mark.parametrize(
    "timestep, contexts",
    [
        (-1, np.array([[1.1, 2.2], [3.3, 4.4]])),
        ("0", np.array([[1.1, 2.2], [3.3, 4.4]])),
        (1.5, np.array([[1.1, 2.2], [3.3, 4.4]])),
        (np.array([-1, 0]), np.array([1.1, 2.2])),
        (np.array([0]), np.array([1.1, 2, 2])),
        (0, np.array([])),
        (0, np.array([[1.1], [2.2]])),
        (0, np.array([[[1.1, 2.2]]])),
    ],
)
def test_ctr_cvr_functions_failure_case(timestep, contexts):
    ctr = CTR(
        ad_feature_dim=1,
        user_feature_dim=1,
        trend_interval=2,
    )
    cvr = CVR(ctr)

    with pytest.raises(ValueError):
        ctr.calc_prob(timestep, contexts)

    with pytest.raises(ValueError):
        ctr.sample_outcome(timestep, contexts)

    with pytest.raises(ValueError):
        cvr.calc_prob(timestep, contexts)

    with pytest.raises(ValueError):
        cvr.sample_outcome(timestep, contexts)


@pytest.mark.parametrize(
    "timestep, contexts",
    [
        (0, np.array([[1.1, 2.2], [3.3, 4.4]])),
        (3, np.array([[1.1, 2.2], [3.3, 4.4]])),
        (np.array([0, 1]), np.array([[1.1, 2.2], [3.3, 4.4]])),
    ],
)
def test_ctr_cvr_function_success_case(timestep, contexts):
    ctr = CTR(
        ad_feature_dim=1,
        user_feature_dim=1,
        trend_interval=2,
    )
    cvr = CVR(ctr)

    ctrs = ctr.calc_prob(timestep, contexts)
    clicks = ctr.sample_outcome(timestep, contexts)
    cvrs = cvr.calc_prob(timestep, contexts)
    conversions = cvr.sample_outcome(timestep, contexts)

    assert 0 <= ctrs.all() <= 1
    assert 0 <= cvrs.all() <= 1
    assert np.array_equal(clicks, ctrs.astype(bool))
    assert np.array_equal(conversions, cvrs.astype(bool))
    assert isinstance(ctrs, NDArray[float])
    assert isinstance(cvrs, NDArray[float])
    assert isinstance(clicks, NDArray[int])
    assert isinstance(conversions, NDArray[int])
    assert len(contexts) == len(ctrs) == len(cvrs)
    assert len(contexts) == len(clicks) == len(conversions)
