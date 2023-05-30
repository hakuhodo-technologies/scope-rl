BasicGym
===================================

.. raw:: html

    <h3>A Python-based configurative basic simulation environment for RL</h3>

Overview
~~~~~~~~~~
*BasicGym* is a basic simulation platform for RL.
The simulator is particularly intended for reinforcement learning algorithms and follows `OpenAI Gym <https://github.com/openai/gym>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.
We design BasicGym as a configurative environment so that researchers and practitioner can customize the environmental modules including UserModel.

Note that, BasicGym is publicized as a sub-package of :doc:`SCOPE-RL <index>`, which streamlines the implementation of offline reinforcement learning (offline RL) and off-policy evaluation and selection (OPE/OPS) procedures.

Basic Setting
~~~~~~~~~~
We formulate (Partially Observable) Markov Decision Process ((PO)MDP) as :math:`\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, P_r \rangle` containing the following components.

* `state` (:math:`s \in \mathcal{S}`)
* `action` (:math:`a \in \mathcal{A}`)  
* `reward` (:math:`r \in \mathbb{R}`)

Note that :math:`\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{P}(\mathcal{S})` is the state transition probability where :math:`\mathcal{T}(s'\mid s,a)` is the probability of observing state :math:`s'` after taking action :math:`a` given state :math:`s`.
:math:`P_r: \mathcal{S} \times \mathcal{A} \times \mathbb{R} \rightarrow [0,1]` is the probability distribution of the immediate reward.
Given :math:`P_r`, :math:`R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}` is the expected reward function where :math:`R(s,a) := \mathbb{E}_{r \sim P_r (r \mid s, a)}[r]` is the expected reward when taking action :math:`a` for state :math:`s`.
Finally, :math:`\pi: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})` denotes a policy (i.e., agent) where :math:`\pi(a | s)` is the probability of taking action :math:`a` at a given state :math:`s`.

Supported Implementation
~~~~~~~~~~

Standard Environment
----------
    * :class:`BasicEnv-discrete-v0`: Standard synthetic environment with discrete action space.
    * :class:`BasicEnv-continuous-v0`: Standard synthetic environment with continuous action space.

Custom Environment
----------
    * :class:`BasicEnv`: The configurative environment with discrete action space.

Configurative Modules
----------
    * :class:`StateTransitionFunction`: Class to define the state transition of the synthetic simulation.
    * :class:`RewardFunction`: Class to define the reward function of the synthetic simulation.

Note that, users can customize the above modules by following the abstract class.

Quickstart and Configurations
~~~~~~~~~~

We provide an example usage of the standard and customized environment. 
The online/offlline RL and OPE/OPS examples are provides in :doc:`SCOPE-RL's quickstart <quickstart>`.

Standard BasicEnv
----------

Our BasicEnv is available from :class:`gym.make()`, 
following the `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.

.. code-block:: python

    # import basicgym and gym
    import basicgym
    import gym

    # (1) standard environment for continuous action space
    env = gym.make('BasicEnv-continuous-v0')

The basic interaction is performed using only four lines of code as follows.

.. code-block:: python

    obs, info = env.reset(), False
    while not done:
       action = agent.act(obs)
       obs, reward, done, truncated, info = env.step(action)

Let's interact with a uniform random policy. 

.. code-block:: python

    from scope_rl.policy import OnlineHead
    from d3rlpy.algos import RandomPolicy as ContinuousRandomPolicy

    # (1) define a random agent
    agent = OnlineHead(
        ContinuousRandomPolicy(
            action_scaler=MinMaxActionScaler(
                minimum=0.1,  # minimum value that policy can take
                maximum=10,  # maximum value that policy can take
            )
        ),
        name="random",
    )
    agent.build_with_env(env)

    # (2) basic interaction 
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.predict_online(obs)
        obs, reward, done, truncated, info = env.step(action)

Note that, while we use :doc:`SCOPE-RL <index>` and `d3rlpy <https://github.com/takuseno/d3rlpy>`_ here,
BasicGym is compatible with any other libraries that is compatible to the `OpenAI Gym <https://gym.openai.com>`_ 
and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ interface.

Customized BasicEnv
----------

Next, we describe how to customize the environment by instantiating the environment.

The list of arguments are given as follows.

* :class:`step_per_episode`: Number of timesteps in an episode.
* :class:`state_dim`: Dimension of the state.
* :class:`action_type`: Action type of the RL agent.
* :class:`n_actions`: Number of actions in the discrete action case.
* :class:`action_dim`: Dimension of the action (context).
* :class:`action_context`: Feature vectors that characterizes each action. Applicable only when action_type is "discrete".
* :class:`reward_type`: Reward type.
* :class:`reward_std`: Noise level of the reward. Applicable only when reward_type is "continuous".
* :class:`obs_std`: Noise level of the state observation.
* :class:`StateTransitionFunction`: State transition function.
* :class:`RewardFunction`: Mean reward function.
* :class:`random_state` : Random state.

Example:

.. code-block:: python

    from basicgym import BasicEnv
    env = BasicEnv(
        state_dim=10,
        action_type="continuous",  # "discrete"
        action_dim=5,
        reward_type="continuous",  # "ninary"
        reward_std=0.3,
        obs_std=0.3,
        step_per_episode=10,
        random_state=12345,
    )

Specifically, users can define their own :class:`StateTransitionFunction` and :class:`RewardFunction` as follows.

Example of Custom State Transition Function:

.. code-block:: python

    # import basicgym modules
    from basicgym import BaseStateTransitionFunction
    # import other necessary stuffs
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np

    @dataclass
    class CustomizedStateTransitionFunction(BaseStateTransitionFunction):
        state_dim: int
        action_dim: int
        random_state: Optional[int] = None

        def __post_init__(self):
            self.random_ = check_random_state(self.random_state)
            self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.state_dim))
            self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_dim))

        def step(
            self,
            state: np.ndarray,
            action: np.ndarray,
        ) -> np.ndarray:
            state = self.state_coef @ state / self.state_dim +  self.action_coef @ action / self.action_dim
            state = state / np.linalg.norm(state, ord=2)
            return state


Example of Custom Reward Function:

.. code-block:: python

    # import basicgym modules
    from basicgym import BaseRewardFunction
    # import other necessary stuffs
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np

    @dataclass
    class CustomizedRewardFunction(BaseRewardFunction):
        state_dim: int
        action_dim: int
        reward_type: str = "continuous"  # "binary"
        reward_std: float = 0.0
        random_state: Optional[int] = None

        def __post_init__(self):
            self.random_ = check_random_state(self.random_state)
            self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, ))
            self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.action_dim, ))

        def mean_reward_function(
            self,
            state: np.ndarray,
            action: np.ndarray,
        ) -> float:
            reward = self.state_coef.T @ state / self.state_dim + self.action_coef.T @ action / self.action_dim
            return reward

Citation
~~~~~~~~~~
If you use our pipeline in your work, please cite our paper below.

| Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.
| **Towards Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning**
| (a preprint coming soon..)

.. code-block::

   @article{kiyohara2023towards,
      author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
      title = {Towards Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning},
      journal = {A github repository},
      pages = {xxx--xxx},
      year = {2023},
   }

Contact
~~~~~~~~~~
For any question about the paper and pipeline, feel free to contact: hk844@cornell.edu

Contribution
~~~~~~~~~~
Any contributions to BasicGym are more than welcome!
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
                :link: /documentation/subpackages/basicgym_api
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Package Reference**
