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
