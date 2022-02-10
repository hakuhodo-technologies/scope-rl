==========
Quickstart
==========

We show an example workflow of synthetic dataset collection, offline Reinforcement Learning (RL) to Off-Policy Evaluation (OPE).
The workflow mainly consists of following three steps:

* **Synthetic Dataset Generation and Data Preprocessing**: The initial step is to collect logged data using a behavior policy. 
In synthetic setup, we first train the behavior policy through online interaction and then generate dataset with the behavior policy.
In practical situation, we can also utilize the preprocessed logged data from real-world applications.

* **Offline Reinforcement Learning**: Next, we now learn a new (and better) policy from only offline logged data, without any online interactions.

* **Off-Policy Evaluation and Selection**: After learning several candidate policies in an offline manner, we validate their performance in an offline manner and choose the best policy.

In this example, we use `[RTBGym] <https://github.com/negocia-inc/offlinegym/blob/main/rtbgym>`_ (a sub-package of OfflineGym) and `[d3rlpy] <https://github.com/takuseno/d3rlpy>`_. Please satisfy the requirements in advance.


Synthetic Dataset Generation and Data Preprocessing
----------

We start by collecting the logged data useful for offline RL with a behavior policy.

.. code-block:: python

    # implement data collection procedure on the RTBGym environment

    # import offlinegym modules
    >>> from offlinegym.dataset import SyntheticDataset
    >>> from offlinegym.policy import DiscreteEpsilonGreedyHead
    # import d3rlpy algorithms
    >>> from d3rlpy.algos import DoubleDQN
    >>> from d3rlpy.online.buffers import ReplayBuffer
    >>> from d3rlpy.online.explorers import ConstantEpsilonGreedy
    # import rtbgym
    >>> from rtbgym import RTBEnv, CustomizedRTBEnv

    # (0) Setup environment
    >>> env = CustomizedRTBEnv(
            original_env=RTBEnv(random_state=random_state),
            action_type="discrete",
        )

    # (1) Learn a baseline online policy (using d3rlpy)
    # initialize algorithm
    >>> ddqn = DoubleDQN()
    # train an online policy
    >>> ddqn.fit_online(
            env,
            buffer=ReplayBuffer(maxlen=10000, env=env),
            explorer=ConstantEpsilonGreedy(epsilon=0.3),
            n_steps=100000,
            n_steps_per_epoch=1000,
        )

    # (2) Generate logged dataset
    # convert ddqn policy into a stochastic behavior policy
    >>> behavior_policy = DiscreteEpsilonGreedyHead(
            ddqn, 
            n_actions=env.action_space.n,
            epsilon=0.3,
            name="ddqn_epsilon_0.3",
            random_state=random_state,
        )
    # initialize dataset class
    >>> dataset = SyntheticDataset(
            env=env,
            behavior_policy=behavior_policy,
            random_state=random_state,
        )
    # collect logged data using behavior policy
    >>> logged_dataset = dataset.obtain_trajectories(n_episodes=10000)

    >>> print(logged_dataset.keys())

Users can collect logged data from any environment with `[OpenAI Gym] <https://gym.openai.com>'_-like interface using a variety of behavior policies.
Moreover, by preprocessing the logged data, one can also handle their own logged data from real-world applications.


Offline Reinforcement Learning
----------

Now we are ready to learn a new policy only from logged data.
Note that, we use `[d3rlpy] <https://github.com/takuseno/d3rlpy>`_ for offline RL.

.. code-block:: python

    # implement offline RL procedure using OfflineGym and d3rlpy

    # import d3rlpy algorithms
    >>> from d3rlpy.dataset import MDPDataset
    >>> from d3rlpy.algos import DiscreteCQL

    # (3) Learning a new policy from offline logged data (using d3rlpy)
    # convert dataset into d3rlpy's dataset
    >>> offlinerl_dataset = MDPDataset(
            observations=logged_dataset["state"],
            actions=logged_dataset["action"],
            rewards=logged_dataset["reward"],
            terminals=logged_dataset["done"],
            episode_terminals=logged_dataset["done"],
            discrete_action=True,
        )
    # initialize algorithm
    >>> cql = DiscreteCQL()
    # train an offline policy
    >>> cql.fit(
            offlinerl_dataset,
            n_steps=10000,
            scorers={},
        )

For the details of algorithm implementation, please refer to `[d3rlpy's documentation] <https://d3rlpy.readthedocs.io/en/v0.91/>`_.


Off-Policy Evaluation and Selection
----------

Finally, we evaluate the performance of the learned policy using offline logged data. 
We also compare the estimation results from various OPE estimators, Direct Method (DM), Trajectory-wise Importance Sampling (TIS), Step-wise Importance Sampling (SIS), and Doubly Robust (DR).

.. code-block:: python

    # implement OPE procedure using OfflineGym

    # import offlinegym modules
    >>> from offlinegym.ope import CreateOPEInput
    >>> from offlinegym.ope import OffPolicyEvaluation as OPE
    >>> from offlinegym.ope import DiscreteDirectMethod as DM
    >>> from offlinegym.ope import DiscreteTrajectoryWiseImportanceSampling as TIS
    >>> from offlinegym.ope import DiscreteStepWiseImportanceSampling as SIS
    >>> from offlinegym.ope import DiscreteDoublyRobust as DR

    # (4) Evaluate the learned policy in an offline manner
    # we compare ddqn, cql, and random policy
    >>> cql_ = DiscreteEpsilonGreedyHead(
            base_policy=cql, 
            n_actions=env.action_space.n, 
            name="cql", 
            epsilon=0.0, 
            random_state=random_state,
        )
    >>> ddqn_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn, 
            n_actions=env.action_space.n, 
            name="ddqn", 
            epsilon=0.0, 
            random_state=random_state,
        )
    >>> random_ = DiscreteEpsilonGreedyHead(
            base_policy=ddqn, 
            n_actions=env.action_space.n, 
            name="random", 
            epsilon=1.0, 
            random_state=random_state,
        )
    >>> evaluation_policies = [cql_, ddqn_, random_]
    # create input for OPE class
    >>> prep = CreateOPEInput(
            logged_dataset=logged_dataset,
            use_base_model=True,  # use model-based prediction
        )
    >>> input_dict = prep.obtain_whole_inputs(
            evaluation_policies=evaluation_policies,
            env=env,
            n_episodes_on_policy_evaluation=100,
            random_state=random_state,
        )
    # initialize OPE class
    >>> ope = OPE(
            logged_dataset=logged_dataset,
            ope_estimators=[DM(), TIS(), SIS(), DR()],
        )
    # conduct OPE and visualize the result
    >>> ope.visualize_off_policy_estimates(
            input_dict, 
            random_state=random_state, 
            sharey=True,
        )

Users can implement their own OPE estimators by following the interface of :class:`obp.ope.BaseOffPolicyEstimator` class.
:class:`obp.ope.OffPolicyEvaluation` class summarizes and compares the estimation results of various OPE estimators.
It also provides some functions for OPS from several candidate policies.

A formal quickstart example with RTBGym is available `[here] <https://github.com/negocia-inc/offlinegym/blob/main/examples/quickstart>`_.


