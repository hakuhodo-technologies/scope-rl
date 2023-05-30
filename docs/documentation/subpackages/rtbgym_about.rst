RTBGym
===================================

.. raw:: html

    <h3> Python-based configurative simulation environment for Real-Time Bidding (RTB)</h3>

Overview
~~~~~~~~~~
*RTBGym* is an open-source simulation platform for Real-Time Bidding (RTB) of Display Advertising.
The simulator is particularly intended for reinforcement learning algorithms and follows `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.
We design RTBGym as a configurative environment so that researchers and practitioner can customize the environmental modules including WinningPriceDistribution, ClickThroughRate, and ConversionRate.

Note that, RTBGym is publicized as a sub-package of :doc:`SCOPE-RL <index>`, which streamlines the implementation of offline reinforcement learning (offline RL) and off-policy evaluation and selection (OPE/OPS) procedures.

Basic Setting
~~~~~~~~~~
In RTB, the objective of the RL agent is to maximize some Key Performance Indicators (KPIs; number of clicks or conversions) within an episode under given budget constraints.
We often aim to achieve this goal by adjusting bidding price function parameter :math:`\alpha`. Specifically, we adjust bid price using :math:`\alpha` as follows.

.. math::

    bid_{t,i} = \alpha \cdot r^{\ast}

where :math:`r^{\ast}` denotes a predicted or expected reward (KPIs).

We often formulate this RTB problem as the following Constrained Markov Decision Process (CMDP) as :math:`\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, P_r, \gamma, C \rangle`.

* `timestep` (:math:`1 \leq t \leq T`): One episode (a day or a week) consists of several timesteps (24 hours or seven days, for instance).
* `state` (:math:`s \in \mathcal{S}`): We observe some feedback from the environment at each timestep, which includes the following.

    * timestep
    * remaining budget
    * impression level features (budget consumption rate, cost per mille of impressions, auction winning rate, reward) at the previous timestep
    * adjust rate (RL agent's decision making) at the previous timestep
    
* `action` (:math:`a \in \mathcal{A}`): Agent chooses adjust rate parameter :math:`\alpha` to maximize KPIs.
* `reward` (:math:`r \in \mathbb{R}`): Total number of clicks or conversions obtained during the timestep.
* `constraints` (:math:`C`): The pre-determined episodic budget should not be exceeded.

Note that :math:`\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{P}(\mathcal{S})` is the state transition probability where :math:`\mathcal{T}(s'\mid s,a)` is the probability of observing state :math:`s'` after taking action :math:`a` given state :math:`s`.
:math:`P_r: \mathcal{S} \times \mathcal{A} \times \mathbb{R} \rightarrow [0,1]` is the probability distribution of the immediate reward.
Given :math:`P_r`, :math:`R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}` is the expected reward function where :math:`R(s,a) := \mathbb{E}_{r \sim P_r (r \mid s, a)}[r]` is the expected reward when taking action :math:`a` for state :math:`s`.
We also let :math:`\gamma \in (0,1]` be a discount factor and :math:`C \ge 0` be a budget constraint.
Finally, :math:`\pi: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})` denotes a policy (i.e., agent) where :math:`\pi(a | s)` is the probability of taking action :math:`a` at a given state :math:`s`.

Based on this formulation, our goal is to maximize the expected trajectory-wise reward while satisfying the safety constraints as follows.

.. math::

    \max_{\pi \in \Pi} \, \, \mathbb{E} \left [ \sum_{t=0}^{T} \gamma^t r_t \mid \pi \right ]

.. math::

    \text{s.t.} \, \, \mathbb{E} \left [ \sum_{t=0}^{T} c_t \mid \pi \right ] \leq C

Supported Implementation
~~~~~~~~~~

Standard Environment
----------
    * :class:`RTBEnv-discrete-v0`: Standard RTB environment with discrete action space.
    * :class:`RTBEnv-continuous-v0`: Standard RTB environment with continuous action space.

Custom Environment
----------
    * :class:`RTBEnv`: The basic configurative environment with continuous action space.
    * :class:`CustomizedRTBEnv`: The customized environment given action space and reward predictor.

Configurative Modules
----------
    * :class:`WinningPriceDistribution`: Class to define the winning price distribution of the auction bidding.
    * :class:`ClickThroughRate`: Class to define the click through rate of users.
    * :class:`ConversionRate`: Class to define the conversion rate of users.

Note that, users can customize the above modules by following the abstract class.
We also define the bidding function in the Bidder class and the auction simulation in the Simulator class, respectively.

Quickstart and Configurations
~~~~~~~~~~

We provide an example usage of the standard and customized environment. 
The online/offlline RL and OPE/OPS examples are provides in :doc:`SCOPE-RL's quickstart <quickstart>`.

Standard RTBEnv
----------

Our standard RTBEnv is available from :class:`gym.make()`, 
following the `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.

.. code-block:: python

    # import rtbgym and gym
    import rtbgym
    import gym

    # (1) standard environment for discrete action space
    env = gym.make('RTBEnv-discrete-v0')

    # (2) standard environment for continuous action space
    env_ = gym.make('RTBEnv-continuous-v0')

The basic interaction is performed using only four lines of code as follows.

.. code-block:: python

    obs, info = env.reset(), False
    while not done:
       action = agent.act(obs)
       obs, reward, done, truncated, info = env.step(action)

Let's interact uniform random policy with a continuous action RTB environment. The discrete action case also works in a similar manner.

.. code-block:: python

    # import from other libraries
    from scope_rl.policy import OnlineHead
    from d3rlpy.algos import RandomPolicy as ContinuousRandomPolicy
    from d3rlpy.preprocessing import MinMaxActionScaler
    import matplotlib.pyplot as plt

    # define a random agent (for continuous action)
    agent = OnlineHead(
        ContinuousRandomPolicy(
            action_scaler=MinMaxActionScaler(
                minimum=0.1,  # minimum value that policy can take
                maximum=10,  # maximum value that policy can take
            )
        )
    )

    # (3) basic interaction for continuous action case
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.predict_online(obs)
        obs, reward, done, truncated, info = env.step(action)

Note that, while we use :doc:`SCOPE-RL <index>` and `d3rlpy <https://github.com/takuseno/d3rlpy>`_ here,
RTBGym is compatible with any other libraries that is compatible to the `OpenAI Gym <https://github.com/openai/gym>`_ 
and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.

Customized RTBEnv
----------

Next, we describe how to customize the environment by instantiating the environment.

The list of arguments are given as follows.

* :class:`objective`: Objective KPIs of RTB, which is either "click" or "conversion".
* :class:`cost_indicator`: Timing of arising costs, which is any of "impression", "click", and "conversion".
* :class:`step_per_episode`: Number of timesteps in an episode.
* :class:`initial_budget`: Initial budget (i.e., constraint) for an episode.
* :class:`n_ads`: Number of ads used for auction bidding.
* :class:`n_users`: Number of users used for auction bidding.
* :class:`ad_feature_dim`: Dimensions of the ad feature vectors.
* :class:`user_feature_dim`: Dimensions of the user feature vectors.
* :class:`ad_feature_vector`: Feature vectors that characterizes each ad.
* :class:`user_feature_vector`: Feature vectors that characterizes each user.
* :class:`ad_sampling_rate`: Sampling probabilities to determine which ad (id) is used in each auction.
* :class:`user_sampling_rate`: Sampling probabilities to determine which user (id) is used in each auction.
* :class:`WinningPriceDistribution`: Winning price distribution of auctions.
* :class:`ClickTroughRate`: Click through rate (i.e., click / impression).
* :class:`ConversionRate`: Conversion rate (i.e., conversion / click).
* :class:`standard_bid_price_distribution`: Distribution of the bid price whose average impression probability is expected to be 0.5.
* :class:`minimum_standard_bid_price`: Minimum value for standard bid price.
* :class:`search_volume_distribution`: Search volume distribution for each timestep.
* :class:`minimum_search_volume`: Minimum search volume at each timestep.
* :class:`random_state`: Random state.

Example:

.. code-block:: python

    from rtbgym import RTBEnv
    env = RTBEnv(
        objective="click",  # maximize the number of total impressions
        cost_indicator="click",  # cost arises every time click occurs
        step_per_episode=14,  # 14 days as an episode
        initial_budget=5000,  # budget available for 14 dayas is 5000
        random_state=12345,
    )

Specifically, users can define their own :class:`WinningPriceDistribution`, :class:`ClickThroughRate`, and :class:`ConversionRate` as follows.

Example of Custom Winning Price Distribution:

.. code-block:: python

    # import RTBGym modules
    from rtbgym import BaseWinningPriceDistribution
    from rtbgym.utils import NormalDistribution
    # import other necessary stuffs
    from dataclasses import dataclass
    from typing import Optional, Union, Tuple
    import numpy as np

    @dataclass
    class CustomizedWinningPriceDistribution(BaseWinningPriceDistribution):
        """Initialization."""
        n_ads: int
        n_users: int
        ad_feature_dim: int
        user_feature_dim: int
        step_per_episode: int
        standard_bid_price_distribution: NormalDistribution = NormalDistribution(
            mean=50,
            std=5,
            random_state=12345,
        )
        minimum_standard_bid_price: Optional[Union[int, float]] = None
        random_state: Optional[int] = None

        def __post_init__(self):
            self.random_ = check_random_state(self.random_state)

        def sample_outcome(
            self,
            bid_prices: np.ndarray,
            **kwargs,
        ) -> Tuple[np.ndarray]:
            """Stochastically determine impression and second price for each auction."""
            # sample winning price from simple normal distribution
            winning_prices = self.random_.normal(
                loc=self.standard_bid_price,
                scale=self.standard_bid_price / 5,
                size=bid_prices.shape,
            )
            impressions = winning_prices < bid_prices
            return impressions.astype(int), winning_prices.astype(int)

        @property
        def standard_bid_price(self):
            return self.standard_bid_price_distribution.mean

Example of Custom ClickThroughRate (and Conversion Rate):

.. code-block:: python

    from rtbgym import BaseClickAndConversionRate
    from rtbgym.utils import sigmoid

    @dataclass
    class CustomizedClickThroughRate(BaseClickAndConversionRate):
        """Initialization."""
        n_ads: int
        n_users: int
        ad_feature_dim: int
        user_feature_dim: int
        step_per_episode: int
        random_state: Optional[int] = None

        def __post_init__(self):
            self.random_ = check_random_state(self.random_state)
            self.ad_coef = self.random_.normal(
                loc=0.0,
                scale=0.5,
                size=(self.ad_feature_dim, 10),
            )
            self.user_coef = self.random_.normal(
                loc=0.0,
                scale=0.5,
                size=(self.user_feature_dim, 10),
            )

        def calc_prob(
            self,
            ad_ids: np.ndarray,
            user_ids: np.ndarray,
            ad_feature_vector: np.ndarray,
            user_feature_vector: np.ndarray,
            timestep: Union[int, np.ndarray],
        ) -> np.ndarray:
            """Calculate CTR (i.e., click per impression)."""
            ad_latent = ad_feature_vector @ self.ad_coef
            user_latent = user_feature_vector @ self.user_coef
            ctrs = sigmoid((ad_latent * user_latent).mean(axis=1))
            return ctrs

        def sample_outcome(
            self,
            ad_ids: np.ndarray,
            user_ids: np.ndarray,
            ad_feature_vector: np.ndarray,
            user_feature_vector: np.ndarray,
            timestep: Union[int, np.ndarray],
        ) -> np.ndarray:
            """Stochastically determine whether click occurs in impression=True case."""
            ctrs = self.calc_prob(
                timestep=timestep,
                ad_ids=ad_ids,
                user_ids=user_ids,
                ad_feature_vector=ad_feature_vector,
                user_feature_vector=user_feature_vector,
            )
            clicks = self.random_.rand(len(ad_ids)) < ctrs
            return clicks.astype(int)

Note that, custom conversion rate can be defined in a similar manner.

Wrapper class for custom bidding setup
----------

To customize the bidding setup, we also provide :class:`CustomizedRTBEnv`, which enables discretization or re-definition of the action space.
In addition, users can set their own :class:`reward_predictor`.

The list of arguments are given as follows.

* :class:`original_env`: Original RTB Environment.
* :class:`reward_predictor`: A machine learning model to predict the reward to determine the bidding price.
* :class:`scaler`: Scaling factor (constant value) used for bid price determination. (None for the auto-fitting)
* :class:`action_min`: Minimum value of adjust rate.
* :class:`action_max`: Maximum value of adjust rate.
* :class:`action_type`: Action type of the RL agent, which is either "discrete" or "continuous".
* :class:`n_actions`: Number of "discrete" actions.
* :class:`action_meaning`: Mapping function of agent action index to the actual "discrete" action to take.

Example:

.. code-block:: python

    from rtbgym import CustomizedRTBEnv
    custom_env = CustomizedRTBEnv(
        original_env=env,
        reward_predictor=None,  # use ground-truth (expected) reward as a reward predictor (oracle)
        action_type="discrete",
    )

More examples are available at :doc:`RTBGym Tutorials <_autogallery/rtbgym/index>`.

Citation
~~~~~~~~~~
If you use our pipeline in your work, please cite our paper below.

| **Accelerating Offline Reinforcement Learning Application in Real-Time Bidding and Recommendation: Potential Use of Simulation** [`arXiv <https://arxiv.org/abs/2109.08331>`_]
| Haruka Kiyohara, Kosuke Kawakami, Yuta Saito.

.. code-block::

    @article{kiyohara2021accelerating,
        title={Accelerating Offline Reinforcement Learning Application in Real-Time Bidding and Recommendation: Potential Use of Simulation},
        author={Kiyohara, Haruka and Kawakami, Kosuke and Saito, Yuta},
        journal={arXiv preprint arXiv:2109.08331},
        year={2021}
    }

Contact
~~~~~~~~~~
For any question about the paper and pipeline, feel free to contact: hk844@cornell.edu

Contribution
~~~~~~~~~~
Any contributions to RTBGym are more than welcome!
Please refer to `CONTRIBUTING.md <https://github.com/hakuhodo-technologies/scope-rl/CONTRIBUTING.md>`_ for general guidelines how to contribute to the project.

.. grid::
    :margin: 0

    .. grid-item::
        :columns: 3
        :margin: 0
        :padding: 0

        .. grid::
            :margin: 0

            .. grid-item-card::
                :link: /documentation/subpackages/index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Sub_packages (Back to Top)**

            .. grid-item-card::
                :link: /documentation/subpackages/index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Documentation (Back to Top)**

    .. grid-item::
        :columns: 6
        :margin: 0
        :padding: 0

    .. grid-item::
        :columns: 3
        :margin: 0
        :padding: 0

        .. grid::
            :margin: 0

            .. grid-item-card::
                :link: /documentation/subpackages/rtbgym_api
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Package Reference**
