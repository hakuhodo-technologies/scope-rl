import pytest
from nptyping import NDArray

import numpy as np
from sklearn.linear_model import LogisticRegression

from _gym.utils import NormalDistribution
from _gym.env import RTBEnv
from _gym.policy import RandomPolicy


def test_init():
    # default value check
    env = RTBEnv()
    assert not env.use_reward_predictor
    assert env.objective == "conversion"
    assert env.action_type == "discrete"

    # semi_synthetic -- not implemented
    with pytest.raises(ValueError):
        RTBEnv(semi_synthetic=True)

    # objective -- failure case
    with pytest.raises(ValueError):
        RTBEnv(objective="impression")

    # objective -- success case
    env = RTBEnv(objective="conversion")
    assert env.objective == "conversion"

    env = RTBEnv(objective="click")
    assert env.objective == "click"

    # action_type -- failure case
    with pytest.raises(ValueError):
        RTBEnv(action_type="none")

    # action_type -- success case
    env = RTBEnv(action_type="discrete")
    assert env.action_type == "discrete"

    env = RTBEnv(action_type="continuous")
    assert env.action_type == "continuous"

    # action_dim -- failure case
    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim=2.5)

    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim=1)

    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim=0)

    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim=-1)

    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim="2")

    # action_meaning -- failure case
    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim=2, action_meaning=np.array([1]))

    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim=2, action_meaning=np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim=2, action_meaning=np.array([0.01, 2]))

    with pytest.raises(ValueError):
        RTBEnv(action_type="discrete", action_dim=2, action_meaning=np.array([10.1, 2]))

    # action_meaning -- success case
    RTBEnv(action_type="discrete", action_dim=2, action_meaning=np.array([0.1, 2]))

    RTBEnv(action_type="discrete", action_dim=2, action_meaning=np.array([10, 2]))

    # use_reward_predictor -- failure case
    with pytest.raises(ValueError):
        RTBEnv(use_reward_predictor=True)

    with pytest.raises(ValueError):
        RTBEnv(use_reward_predictor=True, reward_predictor=LogisticRegression)

    # use_reward_predictor -- success case
    env = RTBEnv(use_reward_predictor=True, reward_predictor=LogisticRegression())
    assert env.simulator.use_reward_predictor

    # step_per_episode -- failure case
    with pytest.raises(ValueError):
        RTBEnv(step_per_episode=0)

    with pytest.raises(ValueError):
        RTBEnv(step_per_episode=-1)

    with pytest.raises(ValueError):
        RTBEnv(step_per_episode=0.5)

    with pytest.raises(ValueError):
        RTBEnv(step_per_episode="1")

    # initial_budget -- failure case
    with pytest.raises(ValueError):
        RTBEnv(initial_budget=0)

    with pytest.raises(ValueError):
        RTBEnv(initial_budget=-1)

    with pytest.raises(ValueError):
        RTBEnv(initial_budget=100.5)

    with pytest.raises(ValueError):
        RTBEnv(initial_budget="1000")

    # n_ads, n_users -- failure case
    with pytest.raises(ValueError):
        RTBEnv(n_ads=0)

    with pytest.raises(ValueError):
        RTBEnv(n_ads=-1)

    with pytest.raises(ValueError):
        RTBEnv(n_ads=0.5)

    with pytest.raises(ValueError):
        RTBEnv(n_ads="1")

    with pytest.raises(ValueError):
        RTBEnv(n_users=0)

    with pytest.raises(ValueError):
        RTBEnv(n_users=-1)

    with pytest.raises(ValueError):
        RTBEnv(n_users=0.5)

    with pytest.raises(ValueError):
        RTBEnv(n_users="1")

    # ad_feature_dim, user_feature_dim -- failure case
    with pytest.raises(ValueError):
        RTBEnv(ad_feature_dim=0)

    with pytest.raises(ValueError):
        RTBEnv(ad_feature_dim=-1)

    with pytest.raises(ValueError):
        RTBEnv(ad_feature_dim=0.5)

    with pytest.raises(ValueError):
        RTBEnv(ad_feature_dim="1")

    with pytest.raises(ValueError):
        RTBEnv(user_feature_dim=0)

    with pytest.raises(ValueError):
        RTBEnv(user_feature_dim=-1)

    with pytest.raises(ValueError):
        RTBEnv(user_feature_dim=0.5)

    with pytest.raises(ValueError):
        RTBEnv(user_feature_dim="1")

    # standard_bid_price_distribution -- failure case
    with pytest.raises(ValueError):
        RTBEnv(standard_bid_price_distribution=10)

    with pytest.raises(ValueError):
        RTBEnv(standard_bid_price_distribution=NormalDistribution(mean=0, std=0))

    with pytest.raises(ValueError):
        RTBEnv(standard_bid_price_distribution=NormalDistribution(mean=-1, std=0))

    with pytest.raises(ValueError):
        RTBEnv(standard_bid_price_distribution=NormalDistribution(mean="1", std=0))

    with pytest.raises(ValueError):
        RTBEnv(standard_bid_price_distribution=NormalDistribution(mean=1, std=-1))

    with pytest.raises(ValueError):
        RTBEnv(standard_bid_price_distribution=NormalDistribution(mean=1, std="1"))

    # standard_bid_price_distribution -- success case
    RTBEnv(standard_bid_price_distribution=NormalDistribution(mean=1, std=0))

    RTBEnv(standard_bid_price_distribution=NormalDistribution(mean=1.5, std=0))

    RTBEnv(standard_bid_price_distribution=NormalDistribution(mean=1, std=1.5))

    # minimum_standard_bid_price -- failure case
    with pytest.raises(ValueError):
        RTBEnv(minimum_standard_bid_price=-1)

    with pytest.raises(ValueError):
        RTBEnv(minimum_standard_bid_price=101)

    with pytest.raises(ValueError):
        RTBEnv(minimum_standard_bid_price="1")

    # minimum_standard_bid_price -- success case
    RTBEnv(minimum_standard_bid_price=0)

    RTBEnv(minimum_standard_bid_price=100)

    RTBEnv(minimum_standard_bid_price=1.5)

    # trend_interval -- failure case
    with pytest.raises(ValueError):
        RTBEnv(trend_interval=0)

    with pytest.raises(ValueError):
        RTBEnv(trend_interval=-1)

    with pytest.raises(ValueError):
        RTBEnv(trend_interval=0.5)

    with pytest.raises(ValueError):
        RTBEnv(trend_interval="1")

    # ad_sampling_rate, user_sampling_rate -- failure case
    with pytest.raises(ValueError):
        RTBEnv(n_ads=1, ad_sampling_rate=1)

    with pytest.raises(ValueError):
        RTBEnv(n_ads=2, ad_sampling_rate=np.array([1]))

    with pytest.raises(ValueError):
        RTBEnv(n_ads=2, ad_sampling_rate=np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        RTBEnv(n_ads=2, ad_sampling_rate=np.array([1, -1]))

    with pytest.raises(ValueError):
        RTBEnv(n_ads=2, ad_sampling_rate=np.array([[1], [1]]))

    with pytest.raises(ValueError):
        RTBEnv(n_ads=2, ad_sampling_rate=np.array([0, 0]))

    with pytest.raises(ValueError):
        RTBEnv(n_users=1, user_sampling_rate=1)

    with pytest.raises(ValueError):
        RTBEnv(n_users=2, user_sampling_rate=np.array([1]))

    with pytest.raises(ValueError):
        RTBEnv(n_users=2, user_sampling_rate=np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        RTBEnv(n_users=2, user_sampling_rate=np.array([1, -1]))

    with pytest.raises(ValueError):
        RTBEnv(n_users=2, user_sampling_rate=np.array([[1], [1]]))

    with pytest.raises(ValueError):
        RTBEnv(n_users=2, user_sampling_rate=np.array([0, 0]))

    # ad_sampling_rate, user_sampling_rate -- success case
    RTBEnv(n_ads=2, ad_sampling_rate=np.array([1, 1]))

    RTBEnv(n_ads=2, ad_sampling_rate=np.array([0.5, 0.4]))

    RTBEnv(n_ads=2, ad_sampling_rate=np.array([0, 1]))

    RTBEnv(n_users=2, user_sampling_rate=np.array([1, 1]))

    RTBEnv(n_users=2, user_sampling_rate=np.array([0.5, 0.4]))

    RTBEnv(n_users=2, user_sampling_rate=np.array([0, 1]))

    # search_volume_distribution -- failure case
    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=10,
            search_volume_distribution=NormalDistribution(
                mean=np.array([1, 2]), std=np.array([1, 2])
            ),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(
                mean=np.array([1, 0]), std=np.array([1, 2])
            ),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(
                mean=np.array([1, -1]), std=np.array([1, 2])
            ),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(
                mean=np.array([1, 1]), std=np.array([-1, 2])
            ),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(
                mean=np.array([1]), std=np.array([2])
            ),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(mean=0, std=2),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(mean=-1, std=2),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(mean="1", std=2),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(mean=1, std=-1),
        )

    with pytest.raises(ValueError):
        RTBEnv(
            step_per_episode=2,
            search_volume_distribution=NormalDistribution(mean=1, std="1"),
        )

    # search_volume_distribution -- success case
    RTBEnv(
        step_per_episode=2,
        search_volume_distribution=NormalDistribution(
            mean=np.array([1, 2]), std=np.array([1, 2])
        ),
    )

    RTBEnv(
        step_per_episode=2,
        search_volume_distribution=NormalDistribution(
            mean=np.array([1.5, 2]), std=np.array([1, 2])
        ),
    )

    RTBEnv(
        step_per_episode=2,
        search_volume_distribution=NormalDistribution(
            mean=np.array([1, 2]), std=np.array([1.5, 2])
        ),
    )

    RTBEnv(
        step_per_episode=2,
        search_volume_distribution=NormalDistribution(
            mean=np.array([1, 2]), std=np.array([0, 2])
        ),
    )

    RTBEnv(
        step_per_episode=2, search_volume_distribution=NormalDistribution(mean=1, std=1)
    )

    RTBEnv(
        step_per_episode=2,
        search_volume_distribution=NormalDistribution(mean=0.5, std=1),
    )

    RTBEnv(
        step_per_episode=2,
        search_volume_distribution=NormalDistribution(mean=1, std=0.5),
    )

    RTBEnv(
        step_per_episode=2, search_volume_distribution=NormalDistribution(mean=1, std=0)
    )

    # minimum_search_volume -- failure_case
    with pytest.raises(ValueError):
        RTBEnv(minimum_search_volume=0)

    with pytest.raises(ValueError):
        RTBEnv(minimum_search_volume=1.5)

    with pytest.raises(ValueError):
        RTBEnv(minimum_search_volume=-1)

    with pytest.raises(ValueError):
        RTBEnv(minimum_search_volume="1")

    # random_state -- failure case
    with pytest.raises(ValueError):
        RTBEnv(random_state=-1)

    with pytest.raises(ValueError):
        RTBEnv(random_state=0.5)

    with pytest.raises(ValueError):
        RTBEnv(random_state="1")


@pytest.mark.parametrize("action", [(-1), (0), ("1"), (0.01), (10.01)])
def test_step_continuous_failure_case(action):
    env = RTBEnv(action_type="continuous")
    env.reset()

    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.parametrize("action", [(0.1), (1), (10.0)])
def test_step_continuous_success_case(action):
    env = RTBEnv(
        objective="click",
        action_type="continuous",
    )
    obs_0 = env.reset()

    for step in range(env.step_per_episode):
        obs, reward, done, info = env.step(action)
        impression, click, conversion = (
            info["impression"],
            info["click"],
            info["conversion"],
        )

        assert obs_0.shape == obs.shape == (7,)
        assert isinstance(reward, int) and reward >= 0
        assert done == (step == env.step_per_episode - 1)
        assert isinstance(impression, int)
        assert isinstance(click, int)
        assert isinstance(conversion, int)
        assert impression >= click >= conversion >= 0
        assert reward == click

    env = RTBEnv(
        objective="conversion",
        action_type="continuous",
    )
    env.reset()

    for step in range(env.step_per_episode):
        obs, reward, done, info = env.step(action)
        conversion = info["conversion"]

        assert reward == conversion


def test_step_discrete():
    env = RTBEnv(
        action_type="discrete",
        action_dim=5,
    )
    valid_action = np.arange(5)
    invalid_action = np.logspace(-1, 1, 5)

    # failure case
    for action in invalid_action:
        env.reset()

        with pytest.raises(ValueError):
            env.step(action)

    # success case
    for action_ in valid_action:
        env.reset()
        env.step(action_)


@pytest.mark.parametrize("n_samples", [(-1), (0), (1.5), ("1")])
def test_fit_reward_predictor_failure_case(n_samples):
    env = RTBEnv(use_reward_predictor=True, reward_predictor=LogisticRegression())

    with pytest.raises(ValueError):
        env.fit_reward_predictor(n_samples)


def test_fit_reward_predictor_success_case():
    env = RTBEnv(
        use_reward_predictor=True,
        reward_predictor=LogisticRegression(),
        ad_feature_dim=2,
        user_feature_dim=2,
    )
    env.fit_reward_predictor(n_samples=1000)

    feature_vectors = np.random.rand(5, 5)
    env.simulator.reward_predictor.predict_proba(feature_vectors)


@pytest.mark.parametrize("n_episodes", [(-1), (0), (1.5), ("1")])
def test_calc_ground_truth_policy_value_failure_case(n_episodes):
    env = RTBEnv()
    random_policy = RandomPolicy(env)

    with pytest.raises(ValueError):
        env.calc_ground_truth_policy_value(
            evaluation_policy=random_policy,
            n_episodes=n_episodes,
        )
