SyntheticGym; a Python-based configurative simulation environment for synthetic simulation (Synthetic)
===================================

Overview
~~~~~~~~~~
*SyntheticGym* is an open-source simulation platform for synthetic simulation(Synthetic)
The simulator is particularly intended for reinforcement learning algorithms and follows `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.
We design SyntheticGym as a configurative environment so that researchers and practitioner can customize the environmental modules including UserModel.

Note that, SyntheticGym is publicized as a sub-package of :doc:`OFRL <index>`, which streamlines the implementation of offline reinforcement learning (offline RL) and off-policy evaluation and selection (OPE/OPS) procedures.

Basic Setting
~~~~~~~~~~
In synthetic simulation, the objective of the RL agent is to maximize reward.
We often formulate this synthetic simulation problem as the following (Partially Observable) Markov Decision Process ((PO)MDP) as :math:`\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, P_r \rangle`.

* `state` (:math:`s \in \mathcal{S}`): 
    * When the true state is unobservable, you can gain observation instead of state.
* `action`(:math:`a \in \mathcal{A}`):  
* `reward`(:math:`r \in \mathbb{R}`): 

Note that :math:`\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{P}(\mathcal{S})` is the state transition probability where :math:`\mathcal{T}(s'\mid s,a)` is the probability of observing state :math:`s'` after taking action :math:`a` given state :math:`s`.
:math:`P_r: \mathcal{S} \times \mathcal{A} \times \mathbb{R} \rightarrow [0,1]` is the probability distribution of the immediate reward.
Given :math:`P_r`, :math:`R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}` is the expected reward function where :math:`R(s,a) := \mathbb{E}_{r \sim P_r (r \mid s, a)}[r]` is the expected reward when taking action :math:`a` for state :math:`s`.
Finally, :math:`\pi: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})` denotes a policy (i.e., agent) where :math:`\pi(a | s)` is the probability of taking action :math:`a` at a given state :math:`s`.

Supported Implementation
~~~~~~~~~~

Standard Environment
----------
    * :class:`SyntheticEnv-v0`: Standard synthetic environment with discrete action space.

Custom Environment
----------
    * :class:`SyntheticEnv`: The configurative environment with discrete action space.

Configurative Modules
----------
    * :class:`StateTransition`: Class to define the state transition of the synthetic simulation.

    * :class:`StateTransition`: Class to define the state transition of the synthetic simulation.
Note that, users can customize the above modules by following the abstract class.

Quickstart and Configurations
~~~~~~~~~~

We provide an example usage of the standard and customized environment. 
The online/offlline RL and OPE/OPS examples are provides in :doc:`OFRL's quickstart <quickstart>`.

Standard SyntheticEnv
----------

Our SyntheticEnv is available from :class:`gym.make()`, 
following the `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.

.. code-block:: python

    # import syntheticgym and gym
    import syntheticgym
    import gym

    # (1) standard environment for discrete action space
    env = gym.make('SyntheticEnv-v0')

The basic interaction is performed using only four lines of code as follows.

.. code-block:: python

    obs, info = env.reset(), False
    while not done:
       action = agent.act(obs)
       obs, reward, done, truncated, info = env.step(action)

Let's interact uniform random policy with a discrete action Synthetic environment. 

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

Note that, while we use :doc:`OFRL <index>` and `d3rlpy <https://github.com/takuseno/d3rlpy>`_ here,
SyntheticGym is compatible with any other libraries that is compatible to the `OpenAI Gym <https://gym.openai.com>`_ 
and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.

Customized SyntheticEnv
----------

Next, we describe how to customize the environment by instantiating the environment.

The list of arguments are given as follows.

* :class:`StateTransition`: .
* :class:`RewardFunction`: .
* :class:`state_dim`: Dimensions of state.
* :class:`action_type`: action type (i.e., continuous / discrete).
* :class:`n_actions`: Number of actions.
* :class:`action_context_dim`: Dimensions of the action context.
* :class:`action_context`: Feature vectors that characterizes each action.
* :class:`reward_type`: Reward type (i.e., continuous / binary).
* :class:`reward_std`: Standard deviation of the reward distribution. Applicable only when reward_type is "continuous".
* :class:`obs_std`: Standard deviation of the observation distribution.
* :class:`step_per_episode`: Number of timesteps in an episode.
* :class:`random_state` : Random state

Example:

.. code-block:: python

    from syntheticgym import SyntheticEnv
    env = SyntheticEnv(
        StateTransition = StateTransition
        RewardFunction = RewardFunction
        state_dim = 10, #each state has 5 dimensional features
        action_type = "continuous", #we use continuous action
        n_actions = 100,  
        action_context_dim = 10,  #each action has 10 dimensional features
        action_context = None,  #determine action_context from n_actions and action_context_dim in SyntheticEnv
        reward_type = "continuous", #we use continuous reward
        reward_std = 0.0,
        obs_std = 0.0, #not add noise to the observation
        step_per_episode = 10,
        random_state = 12345,
    )

Specifically, users can define their own :class:`StateTransition` as follows.

Example of Custom StateTransition:

.. code-block:: python

    # import syntheticgym modules
    from syntheticgym import BaseStateTransition
    from sytheticgym.types import Action
    # import other necessary stuffs
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np

    @dataclass
    class StateTransition(BaseStateTransition):
        state_dim: int = 10
        action_type: str = "continuous",  # "binary"
        action_context_dim: int = 10
        action_context: Optional[np.ndarray] = (None,)
        random_state: Optional[int] = None

        def __post_init__(self):
            self.random_ = check_random_state(self.random_state)

            self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.state_dim))
            self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_context_dim))
            self.state_action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(1, self.action_context_dim))


        def step(
            self,
            state: np.ndarray,
            action: Action,
        ) -> np.ndarray:

            if self.action_type == "continuous":
                state = self.state_coef @ state +  self.action_coef @ action+  state @ self.state_action_coef @ action
            
            elif self.action_type == "discrete":
                state = self.state_coef @ state + self.action_coef @ self.action_context[action] +  state @ self.state_action_coef.T @ self.action_context[action]
                
            state = state / np.linalg.norm(state, ord=2)

            return state


Specifically, users can define their own :class:`RewardFunction` as follows.

Example of Custom RewardFunction:

.. code-block:: python

        # import syntheticgym modules
    from syntheticgym import BaseRewardFunction
    from sytheticgym.types import Action
    # import other necessary stuffs
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np

    @dataclass
    class RewardFunction(BaseRewardFunction):
        reward_type: str = "continuous"  # "binary"
        reward_std: float = 0.0
        state_dim: int = 10
        action_type: str = "continuous",  # "discrete"
        action_context_dim: int = 10
        action_context: Optional[np.ndarray] = (None,)
        random_state: Optional[int] = None

        def __post_init__(self):
            check_scalar(
                self.reward_std,
                name="reward_std",
                target_type=float,
            )

            if self.reward_type not in ["continuous", "binary"]:
                raise ValueError(
                    f'reward_type must be either "continuous" or "binary", but {self.reward_type} is given'
                )

            self.random_ = check_random_state(self.random_state)

            self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(1, self.state_dim))
            self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(1, self.action_context_dim))
            self.state_action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_context_dim))

        def sample(
            self,
            state: np.ndarray,
            action: Action,
        ) -> float:
            if self.action_type == "continuous":
                reward = self.state_coef @ state + self.action_coef @ action +  (state.T @ self.state_action_coef) @ action
            
            elif self.action_type == "discrete":
                reward = self.state_coef @ state + self.action_coef @ self.action_context[action] +  (state.T @ self.state_action_coef ) @ self.action_context[action]

            if self.reward_type == "continuous":
                reward = reward + self.random_.normal(loc=0.0, scale=self.reward_std)

            reward = reward[0][0]

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
For any question about the paper and pipeline, feel free to contact: kiyohara.h.aa@m.titech.ac.jp

Contribution
~~~~~~~~~~
Any contributions to SyntheticGym are more than welcome!
Please refer to `CONTRIBUTING.md <>`_ for general guidelines how to contribute to the project.

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
                :link: /documentation/subpackages/syntheticgym_api
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Package Reference**
