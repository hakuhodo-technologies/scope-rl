import pytest

import numpy as np
from sklearn.linear_model import LogisticRegression

from _gym.env.bidder import Bidder
from _gym.env.simulator.rtb_synthetic import RTBSyntheticSimulator


# simulator, objective, reward_predictor, scaler, err, description
invalid_input_of_init = [
    (
        "xxx",  #
        "conversion",
        LogisticRegression(),
        0.1,
        ValueError,
        "simulator must be BaseSimulator",
    ),
    (
        RTBSyntheticSimulator(),
        "xxx",  #
        LogisticRegression(),
        0.1,
        ValueError,
        "objective must be either",
    ),
    (
        RTBSyntheticSimulator(),
        "conversion",
        "xxx",  #
        0.1,
        ValueError,
        "reward_predictor must be BaseEstimator",
    ),
    (
        RTBSyntheticSimulator(),
        "conversion",
        LogisticRegression(),
        -1,  #
        ValueError,
        "scaler must be a positive",
    ),
    (
        RTBSyntheticSimulator(),
        "conversion",
        LogisticRegression(),
        0,  #
        ValueError,
        "scaler must be a positive",
    ),
    (
        RTBSyntheticSimulator(),
        "conversion",
        LogisticRegression(),
        "1",  #
        ValueError,
        "scaler must be a positive float value",
    ),
]


@pytest.mark.parametrize(
    "simulator, objective, reward_predictor, scaler, err, description",
    invalid_input_of_init,
)
def test_init_using_invalid_input(
    simulator,
    objective,
    reward_predictor,
    scaler,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        Bidder(
            simulator=simulator,
            objective=objective,
            reward_predictor=reward_predictor,
            scaler=scaler,
        )


# simulator, objective, reward_predictor, scaler
valid_input_of_init = [
    (
        RTBSyntheticSimulator(),
        "click",
        LogisticRegression(),
        0.1,
    ),
    (
        RTBSyntheticSimulator(),
        "conversion",
        None,
        0.1,
    ),
    (
        RTBSyntheticSimulator(),
        "click",
        LogisticRegression(),
        1,
    ),
    (
        RTBSyntheticSimulator(),
        "conversion",
        LogisticRegression(),
        None,
    ),
    (
        RTBSyntheticSimulator(),
        "conversion",
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    "simulator, objective, reward_predictor, scaler",
    valid_input_of_init,
)
def test_init_using_valid_input(simulator, objective, reward_predictor, scaler):
    bidder = Bidder(
        simulator=simulator,
        objective=objective,
        reward_predictor=reward_predictor,
        scaler=scaler,
    )

    assert bidder.standard_bid_price == simulator.standard_bid_price


# n_ads, n_users, timestep, adjust_rate, ad_ids, user_ids, err, description
invalid_input_of_determine_bid_price = [
    (
        2,
        2,
        -1,  #
        1.0,
        np.arange(2, dtype=int),
        np.arange(2, dtype=int),
        ValueError,
        "timestep must be a non-negative",
    ),
    (
        2,
        2,
        1.5,  #
        1.0,
        np.arange(2, dtype=int),
        np.arange(2, dtype=int),
        ValueError,
        "timestep must be a non-negative interger",
    ),
    (
        2,
        2,
        "1",  #
        1.0,
        np.arange(2, dtype=int),
        np.arange(2, dtype=int),
        ValueError,
        "timestep must be a non-negative interger",
    ),
    (
        2,
        2,
        1,
        1,  #
        np.arange(2, dtype=int),
        np.arange(2, dtype=int),
        ValueError,
        "adjust_rate must be a non-negative float value",
    ),
    (
        2,
        2,
        1,
        -0.5,  #
        np.arange(2, dtype=int),
        np.arange(2, dtype=int),
        ValueError,
        "adjust_rate must be a non-negative float value",
    ),
    (
        2,
        2,
        1,
        "0.5",  #
        np.arange(2, dtype=int),
        np.arange(2, dtype=int),
        ValueError,
        "adjust_rate must be a non-negative float value",
    ),
    (
        2,
        2,
        1,
        0.5,
        np.array([0.1, 0.1]),  #
        np.arange(2, dtype=int),
        IndexError,
        "arrays used as indices must be of integer",
    ),
    (
        2,
        2,
        1,
        0.5,
        -np.ones(2, dtype=int),  #
        np.arange(2, dtype=int),
        ValueError,
        "ad_ids must be 1-dimensional NDArray with integers",
    ),
    (
        2,
        2,
        1,
        0.5,
        2 * np.ones(2, dtype=int),  #
        np.arange(2, dtype=int),
        ValueError,
        "ad_ids must be 1-dimensional NDArray with integers within \[0, n_ads\)",
    ),
    (
        2,
        2,
        1,
        0.5,
        np.ones((2, 3), dtype=int),  #
        np.arange(2, dtype=int),
        ValueError,
        "ad_ids must be 1-dimensional NDArray",
    ),
    (
        2,
        2,
        1,
        0.5,
        np.arange(2, dtype=int),
        np.array([0.1, 0.1]),  #
        IndexError,
        "arrays used as indices must be of integer",
    ),
    (
        2,
        2,
        1,
        0.5,
        np.arange(2, dtype=int),
        -np.ones(2, dtype=int),  #
        ValueError,
        "user_ids must be 1-dimensional NDArray with integers within \[0, n_users\)",
    ),
    (
        2,
        2,
        1,
        0.5,
        np.arange(2),
        2 * np.ones(2, dtype=int),  #
        ValueError,
        "user_ids must be 1-dimensional NDArray with integers within \[0, n_users\)",
    ),
    (
        2,
        2,
        1,
        0.5,
        np.arange(2, dtype=int),
        np.ones((2, 3), dtype=int),  #
        ValueError,
        "user_ids must be 1-dimensional NDArray",
    ),
    (
        2,
        2,
        1,
        0.5,
        np.arange(1, dtype=int),  #
        np.arange(2, dtype=int),  #
        ValueError,
        "ad_ids and user_ids must have same length",
    ),
]


@pytest.mark.parametrize(
    "n_ads, n_users, timestep, adjust_rate, ad_ids, user_ids, err, description",
    invalid_input_of_determine_bid_price,
)
def test_determine_bid_price_using_invalid_input(
    n_ads, n_users, timestep, adjust_rate, ad_ids, user_ids, err, description
):
    simulator = RTBSyntheticSimulator(n_ads=n_ads, n_users=n_users)
    bidder = Bidder(simulator=simulator)
    bidder.auto_fit_scaler(step_per_episode=5)

    with pytest.raises(err, match=f"{description}*"):
        bidder.determine_bid_price(
            timestep=timestep,
            adjust_rate=adjust_rate,
            ad_ids=ad_ids,
            user_ids=user_ids,
        )


def test_determine_bid_price_runtimeerror():
    bidder = Bidder(simulator=RTBSyntheticSimulator())

    with pytest.raises(RuntimeError):
        bidder.determine_bid_price(
            timestep=1,
            adjust_rate=1.0,
            ad_ids=np.arange(5),
            user_ids=np.arange(5),
        )


def test_determine_bid_price_using_valid_input():
    timestep = 0
    adjust_rate = 1.0
    ad_ids = np.arange(5)
    user_ids = np.arange(5)

    step_per_episode = 5
    n_samples = 100

    bidder_A = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective="conversion",
        reward_predictor=LogisticRegression(),
    )
    bidder_B = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective="click",
        reward_predictor=LogisticRegression(),
    )
    bidder_C = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective="conversion",
        reward_predictor=None,
    )
    bidder_D = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective="click",
        reward_predictor=None,
    )

    bidder_A.fit_reward_predictor(
        step_per_episode=step_per_episode, n_samples=n_samples
    )
    bidder_B.fit_reward_predictor(
        step_per_episode=step_per_episode, n_samples=n_samples
    )

    bidder_A.auto_fit_scaler(step_per_episode=step_per_episode, n_samples=n_samples)
    bidder_B.auto_fit_scaler(step_per_episode=step_per_episode, n_samples=n_samples)
    bidder_C.auto_fit_scaler(step_per_episode=step_per_episode, n_samples=n_samples)
    bidder_D.auto_fit_scaler(step_per_episode=step_per_episode, n_samples=n_samples)

    bid_prices_A = bidder_A.determine_bid_price(
        timestep=timestep,
        adjust_rate=adjust_rate,
        ad_ids=ad_ids,
        user_ids=user_ids,
    )
    bid_prices_B = bidder_B.determine_bid_price(
        timestep=timestep,
        adjust_rate=adjust_rate,
        ad_ids=ad_ids,
        user_ids=user_ids,
    )
    bid_prices_C = bidder_C.determine_bid_price(
        timestep=timestep,
        adjust_rate=adjust_rate,
        ad_ids=ad_ids,
        user_ids=user_ids,
    )
    bid_prices_D = bidder_D.determine_bid_price(
        timestep=timestep,
        adjust_rate=adjust_rate,
        ad_ids=ad_ids,
        user_ids=user_ids,
    )

    assert np.allclose(np.mod(bid_prices_A, 1), 0)
    assert not np.allclose(bid_prices_A, bid_prices_B)
    assert not np.allclose(bid_prices_C, bid_prices_D)
    assert not np.allclose(bid_prices_A, bid_prices_C)
    assert not np.allclose(bid_prices_B, bid_prices_D)


def test_determine_bid_price_using_different_adjust_rate():
    timestep = 0
    ad_ids = np.arange(5)
    user_ids = np.arange(5)

    adjust_rate_low = 0.1
    adjust_rate_medium = 1.0
    adjust_rate_high = 5.0

    bidder = Bidder(simulator=RTBSyntheticSimulator())
    bidder.auto_fit_scaler(step_per_episode=5)

    bid_prices_low = bidder.determine_bid_price(
        timestep=timestep,
        adjust_rate=adjust_rate_low,
        ad_ids=ad_ids,
        user_ids=user_ids,
    )
    bid_prices_medium = bidder.determine_bid_price(
        timestep=timestep,
        adjust_rate=adjust_rate_medium,
        ad_ids=ad_ids,
        user_ids=user_ids,
    )
    bid_prices_high = bidder.determine_bid_price(
        timestep=timestep,
        adjust_rate=adjust_rate_high,
        ad_ids=ad_ids,
        user_ids=user_ids,
    )

    assert (bid_prices_low > bid_prices_medium).sum() == 0
    assert (bid_prices_medium > bid_prices_high).sum() == 0


@pytest.mark.parametrize("scaler", [(0), (-1), ("1")])
def test_custom_set_scaler_using_invalid_input(scaler):
    bidder = Bidder(simulator=RTBSyntheticSimulator())

    with pytest.raises(ValueError):
        bidder.custom_set_scaler(scaler)


# step_per_episode, n_samples, err, description
invalid_input_of_fitting_functions = [
    (
        -1,  #
        100,
        ValueError,
        "step_per_episode must be a non-negative",
    ),
    (
        1.5,  #
        100,
        ValueError,
        "step_per_episode must be a non-negative interger",
    ),
    ("1", 100, ValueError, "step_per_episode must be a non-negative interger"),  #
    (
        1,
        0,  #
        ValueError,
        "n_samples must be a positive",
    ),
    (
        1,
        -1,  #
        ValueError,
        "n_samples must be a positive",
    ),
    (
        1,
        1.5,  #
        ValueError,
        "n_samples must be a positive interger",
    ),
    (
        1,
        "1",  #
        ValueError,
        "n_samples must be a positive interger",
    ),
]


@pytest.mark.parametrize(
    "step_per_episode, n_samples, err, description",
    invalid_input_of_fitting_functions,
)
def test_auto_fit_scaler_using_invalid_input(
    step_per_episode, n_samples, err, description
):
    bidder = Bidder(simulator=RTBSyntheticSimulator())

    with pytest.raises(err, match=f"{description}*"):
        bidder.auto_fit_scaler(
            step_per_episode=step_per_episode,
            n_samples=n_samples,
        )


def test_auto_fit_scaler_using_valid_input():
    step_per_episode = 5
    n_samples = 100

    bidder_A = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective="click",
        reward_predictor=LogisticRegression(),
    )
    bidder_B = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective="conversion",
        reward_predictor=LogisticRegression(),
    )
    bidder_C = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective="click",
        reward_predictor=None,
    )
    bidder_D = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective="conversion",
        reward_predictor=None,
    )

    bidder_A.fit_reward_predictor(
        step_per_episode=step_per_episode, n_samples=n_samples
    )
    bidder_B.fit_reward_predictor(
        step_per_episode=step_per_episode, n_samples=n_samples
    )

    bidder_A.auto_fit_scaler(step_per_episode=step_per_episode, n_samples=n_samples)
    bidder_B.auto_fit_scaler(step_per_episode=step_per_episode, n_samples=n_samples)
    bidder_C.auto_fit_scaler(step_per_episode=step_per_episode, n_samples=n_samples)
    bidder_D.auto_fit_scaler(step_per_episode=step_per_episode, n_samples=n_samples)

    assert isinstance(bidder_A.scaler, (int, float))
    assert bidder_A.scaler != bidder_B.scaler
    assert bidder_C.scaler != bidder_D.scaler
    assert bidder_A.scaler != bidder_C.scaler
    assert bidder_B.scaler != bidder_D.scaler


def test_custom_set_reward_predictor_using_invalid_input():
    bidder = Bidder(simulator=RTBSyntheticSimulator())

    with pytest.raises(ValueError):
        bidder.custom_set_reward_predictor("xxx")


def test_custom_set_reward_predictor_using_valid_input():
    bidder = Bidder(simulator=RTBSyntheticSimulator())
    bidder.custom_set_reward_predictor(LogisticRegression())

    assert bidder.use_reward_predictor


@pytest.mark.parametrize(
    "step_per_episode, n_samples, err, description",
    invalid_input_of_fitting_functions,
)
def test_fit_reward_predictor_using_invalid_input(
    step_per_episode, n_samples, err, description
):
    bidder = Bidder(
        simulator=RTBSyntheticSimulator(), reward_predictor=LogisticRegression()
    )

    with pytest.raises(err, match=f"{description}*"):
        bidder.fit_reward_predictor(
            step_per_episode=step_per_episode,
            n_samples=n_samples,
        )


# objective, reward_predictor
valid_input_of_fit_reward_predictor = [
    (
        "click",
        LogisticRegression(),
    ),
    (
        "conversion",
        LogisticRegression(),
    ),
    (
        "click",
        None,
    ),
    (
        "conversion",
        None,
    ),
]


@pytest.mark.parametrize(
    "objective, reward_predictor",
    valid_input_of_fit_reward_predictor,
)
def test_fit_reward_predictor_using_valid_input(objective, reward_predictor):
    bidder = Bidder(
        simulator=RTBSyntheticSimulator(),
        objective=objective,
        reward_predictor=reward_predictor,
    )
    bidder.fit_reward_predictor(step_per_episode=5, n_samples=100)
