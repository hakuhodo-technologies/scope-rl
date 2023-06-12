RECGym
===================================

.. raw:: html

    <h3>A Python-based configurative simulation environment for recommender systems</h3>

Overview
~~~~~~~~~~
*RECGym* is an open-source simulation platform for recommender system(REC)
The simulator is particularly intended for reinforcement learning algorithms and follows `OpenAI Gym <https://github.com/openai/gym>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.
We design RECGym as a configurative environment so that researchers and practitioner can customize the environmental modules including UserModel.

Note that RECGym is publicized as a sub-package of :doc:`SCOPE-RL <index>`, which streamlines the implementation of offline reinforcement learning (offline RL) and off-policy evaluation and selection (OPE/OPS) procedures.

Basic Setting
~~~~~~~~~~
In recommendation, the objective of the RL agent is to maximize reward.
We often formulate this recommendation problem as the following (Partially Observable) Markov Decision Process ((PO)MDP) as :math:`\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, P_r \rangle`.

* `state` (:math:`s \in \mathcal{S}`):
    * A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
    * When the true state is unobservable, you can gain observation instead of state.
* `action`(:math:`a \in \mathcal{A}`):  Index of an item to present to the user.
* `reward`(:math:`r \in \mathbb{R}`): User engagement signal as a reward. Either binary or continuous.

Note that :math:`\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{P}(\mathcal{S})` is the state transition probability where :math:`\mathcal{T}(s'\mid s,a)` is the probability of observing state :math:`s'` after taking action :math:`a` given state :math:`s`.
:math:`P_r: \mathcal{S} \times \mathcal{A} \times \mathbb{R} \rightarrow [0,1]` is the probability distribution of the immediate reward.
Given :math:`P_r`, :math:`R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}` is the expected reward function where :math:`R(s,a) := \mathbb{E}_{r \sim P_r (r \mid s, a)}[r]` is the expected reward when taking action :math:`a` for state :math:`s`.
Finally, :math:`\pi: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})` denotes a policy (i.e., agent) where :math:`\pi(a | s)` is the probability of taking action :math:`a` at a given state :math:`s`.

Supported Implementation
~~~~~~~~~~

Standard Environment
----------
    * :class:`RECEnv-v0`: Standard recommender environment with discrete action space.

Custom Environment
----------
    * :class:`RECEnv`: The configurative environment with discrete action space.

Configurative Modules
----------
    * :class:`UserModel`: Class to define the user model of the recommender system.

Note that users can customize the above modules by following the abstract class.

Quickstart and Configurations
~~~~~~~~~~

We provide an example usage of the standard and customized environment.
The online/offline RL and OPE/OPS examples are provides in :doc:`SCOPE-RL's quickstart <quickstart>`.

Standard RECEnv
----------

Our RECEnv is available from :class:`gym.make()`,
following the `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.

.. code-block:: python

    # import recgym and gym
    import recgym
    import gym

    # (1) standard environment for discrete action space
    env = gym.make('RECEnv-v0')

The basic interaction is performed using only four lines of code as follows.

.. code-block:: python

    obs, info = env.reset(), False
    while not done:
       action = agent.act(obs)
       obs, reward, done, truncated, info = env.step(action)

Let's interact uniform random policy with a discrete action REC environment.

.. code-block:: python

    # import from other libraries
    from offlinegym.policy import DiscreteEpsilonGreedyHead
    from d3rlpy.algos import RandomPolicy as DiscreteRandomPolicy
    import matplotlib.pyplot as plt

    # define a random agent
    agent = DiscreteEpsilonGreedyHead(
        base_policy=DiscreteRandomPolicy(),
        n_actions=env.n_items,
        epsilon=1.0,
        name='random',
        random_state = random_state,
    )

    # (2) basic interaction
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.predict_online(obs)
        obs, reward, done, truncated, info = env.step(action)

Note that while we use :doc:`SCOPE-RL <index>` and `d3rlpy <https://github.com/takuseno/d3rlpy>`_ here,
RECGym is compatible with any other libraries that is compatible to the `OpenAI Gym <https://gym.openai.com>`_
and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.

Customized RECEnv
----------

Next, we describe how to customize the environment by instantiating the environment.

The list of arguments are given as follows.

* :class:`UserModel`: User model which defines ``user_prefecture_dynamics`` (e.g., :cite:`dean2022preference`) and ``reward_function``.
* :class:`n_items`: Number of items used for recommendation.
* :class:`n_users`: Number of users used for recommendation.
* :class:`item_feature_dim`: Dimensions of the item feature vectors.
* :class:`user_feature_dim`: Dimensions of the user feature vectors.
* :class:`item_feature_vector`: Feature vectors that characterizes each item.
* :class:`user_feature_vector`: Feature vectors that characterizes each user.
* :class:`reward_type`: Reward type (i.e., continuous / binary).
* :class:`reward_std`: Standard deviation of the reward distribution. Applicable only when reward_type is "continuous".
* :class:`obs_std`: Standard deviation of the observation distribution.
* :class:`step_per_episode`: Number of timesteps in an episode.
* :class:`random_state` : Random state

Example:

.. code-block:: python

    from recgym import RECEnv
    env = RECEnv(
        UserModel = UserModel,
        n_items = 100,  # we use 100 items
        n_users = 100,  # 100 users exists
        item_feature_dim = 5,  #each item has 5 dimensional features
        user_feature_dim = 5,  #each user has 5 dimensional features
        item_feature_vector = None,  #determine item_feature_vector from n_items and item_feature_dim in RECEnv
        user_feature_vector = None,  #determine user_feature_vector from n_users and user_feature_dim in RECEnv
        reward_type = "continuous", #we use continuous reward
        reward_std = 0.0,
        obs_std = 0.0, #not add noise to the observation
        step_per_episode = 10,
        random_state = 12345,
    )

Specifically, users can define their own :class:`UserModel` as follows.

Example of Custom UserModel:

.. code-block:: python

    # import recgym modules
    from recgym import BaseUserModel
    from recgym.types import Action
    # import other necessary stuffs
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np

    @dataclass
    class UserModel(BaseUserModel):
        """Initialization."""
        reward_type: str = "continuous"  # "binary"
        reward_std: float = 0.0
        item_feature_vector: Optional[np.ndarray] = None,
        random_state: Optional[int] = None

        def __post_init__(self):
            self.random_ = check_random_state(self.random_state)

        def user_preference_dynamics(
            self,
            state: np.ndarray,
            action: Action,
            alpha: float = 1.0,
        )-> np.ndarray:
            """Function that determines the user state transition (i.e., user preference) based on the recommended item. user_feature is amplified by the recommended item_feature
            """
            state = (state + alpha * state @ self.item_feature_vector[action] * self.item_feature_vector[action])
            state = state / np.linalg.norm(state, ord=2)
            return state

        def reward_function(
            self,
            state: np.ndarray,
            action: Action,
        )-> float:
            """Reward function. inner product of state and recommended item_feature
            """
            reward = state @ self.item_feature_vector[action]
            if self.reward_type is "continuous":
                reward = reward + self.random_.normal(loc=0.0, scale=self.reward_std)
            return reward


Citation
~~~~~~~~~~
If you use our pipeline in your work, please cite our paper below.

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
Any contributions to RECGym are more than welcome!
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
                :link: /documentation/subpackages/recgym_api
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Package Reference**
