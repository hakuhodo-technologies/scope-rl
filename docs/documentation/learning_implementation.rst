==========
Supported Implementation
==========

Our implementation aims to streamline the data collection, (offline) policy learning, and off-policy evaluation/selection (OPE/OPS) procedure.
We rely on `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s implementation of the learning algorithms and provide some useful tools to streamline the above offline RL procedure.

.. _implementation_dataset:

Synthetic Dataset Generation
~~~~~~~~~~
:class:`SyntheticDataset` is an easy-to-use data collection module which is compatible to any `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://gymnasium.farama.org/>`_-like RL environment.

It takes an RL environment as input to instantiate the class.

.. code-block:: python

    # initialize the dataset class
    from scope_rl.dataset import SyntheticDataset
    dataset = SyntheticDataset(
        env=env,
        max_episode_steps=env.step_per_episode,
    )

Then, it collects logged data by a behavior policy (i.e., data collection policy) as follows.

.. code-block:: python

    # collect logged data by a behavior policy
    logged_dataset = dataset.obtain_episodes(
        behavior_policies=behavior_policy,  # BaseHead
        n_trajectories=10000,
        random_state=random_state,
    )

.. _tips_synthetic_dataset:

.. tip::

    .. dropdown:: How to obtain a behavior policy?

        Our :class:`SyntheticDataset` class accepts an instance of :class:`BaseHead` as a behavior policy.

        A policy head converts a `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s deterministic behavior policy to 
        either a deterministic or stochastic policy with functions to calculate propensity scores (i.e., action choice probabilities).

        For example, :class:`DiscreteEpsilonGreedyHead` converts a discrete-action policy to a epsilon-greedy policy as follows.

        .. code-block:: python

            from scope_rl.policy import DiscreteEpsilonGreedyHead
            behavior_policy = DiscreteEpsilonGreedyHead(
                base_policy,  # AlgoBase of d3rlpy
                n_actions=env.action_space.n,
                epsilon=0.3,
                name="eps_03",
                random_state=random_state,
            )


        :class:`ContinuousGaussianHead` converts a continuous-action policy to a stochastic policy as follows.

        .. code-block:: python

            from scope_rl.policy import ContinuousGaussianHead
            behavior_policy = ContinuousGaussianHead(
                base_policy,  # AlgoBase of d3rlpy
                sigma=1.0,
                name="sigma_10",
                random_state=random_state,
            )

        .. seealso::

            For the detail descriptions and additional supported implementations, please refer to the :ref:`Policy Wrappers <implementation_policy_head>` section later in this page.

    .. dropdown:: How to customize the dataset class?

        To customize the dataset class, use :class:`BaseDataset`. The obtained ``logged_dataset`` should contain the following keys for API consistency.

        .. code-block:: python

            key: [
                size,
                n_trajectories,
                step_per_trajectory,
                action_type,
                n_actions,
                action_dim,
                action_keys,
                action_meaning,
                state_dim,
                state_keys,
                state,
                action,
                reward,
                done,
                terminal,
                info,
                pscore,
                behavior_policy,
                dataset_id,
            ]

        .. note::
            
            ``logged_dataset`` can be used for OPE even if ``action_keys``, ``action_meaning``, ``state_keys``, and ``info`` are not provided.
            For API consistency, just leave ``None`` when these keys are unnecessary. 
            
            Moreover, offline RL algorithms, FQE (model-based OPE), and marginal OPE estimators 
            can also work without ``pscore``. 

        .. seealso::

            :doc:`API reference of BaseDataset<_autosummary/dataset/scope_rl.dataset.base>` explains the meaning of each keys in detail.


    .. dropdown:: How to handle multiple logged datasets at once?

        :class:`MultipleLoggedDataset` enables us to smoothly handle multiple logged datasets. 

        Specifically, :class:`MultipleLoggedDataset` saves the paths to each logged dataset and make each dataset accessible through the following command.
        
        .. code-block:: python

            logged_dataset_ = multiple_logged_dataset.get(behavior_policy_name=behavior_policy.name, dataset_id=0)
        
        There are two ways to obtain :class:`MultipleLoggedDataset`.

        The first way is to directly get :class:`MultipleLoggedDataset` as the output of :class:`SyntheticDataset` as follows.

        .. code-block:: python

            synthetic_dataset = SyntheticDataset(
                env=env,
                max_episode_steps=env.step_per_episode,
                ...,
            )
            multiple_logged_dataset_1 = synthetic_dataset.obtain_episodes(
                behavior_policies=[behavior_policy_1, behavior_policy_2],  # when using multiple logged datasets, MultipleLoggedDataset is returned
                n_datasets=1,          
                n_trajectories=10000,
                ...,
            )
            multiple_logged_dataset_2 = synthetic_dataset.obtain_episodes(
                behavior_policies=behavior_policy,
                n_datasets=5,                       # when n_datasets > 1, MultipleLoggedDataset is returned
                n_trajectories=10000,
                ...,
            )

        The second way to define :class:`MultipleLoggedDataset` manually as follows.

        .. code-block:: python

            from scope_rl.utils import MultipleLoggedDataset

            multiple_logged_dataset = MultipleLoggedDataset(
                action_type="discrete",
                path="logged_dataset/",  # either absolute or relative path
            )

            for behavior_policy in behavior_policies:
                single_logged_dataset = dataset.obtain_episodes(
                    behavior_policies=behavior_policy,
                    n_trajectories=10000,
                    ...,
                )

                # add a single_logged_dataset to multiple_logged_dataset
                multiple_logged_dataset.add(
                    single_logged_dataset, 
                    behavior_policy_name=behavior_policy.name,
                )

        .. seealso::

            * :doc:`API reference of MultipleLoggedDataset <_autosummary/scope_rl.utils.MultipleLoggedDataset>`
            .. * :ref:`Tutorial with MultipleLoggedDataset <scope_rl_multiple_tutorial>`

    .. dropdown:: How to collect data in a non-episodic setting?

        When the goal is to evaluate the policy under a stationary distribution (:math:`d^{\pi}(s)`) rather than in an episodic setting 
        (i.e., cartpole or taxi used in :cite:`liu2018breaking` :cite:`uehara2020minimax`), we need to collect data from stationary distribution.

        For this, please consider using :class:`obtain_step` instead of :class:`obtain_episodes` as follows.

        .. code-block:: python

            logged_dataset = dataset.obtain_steps(
                behavior_policies=behavior_policy,
                n_trajectories=10000,
                ...,
            )

.. seealso::

    * :doc:`quickstart` 
    .. * and :ref:`related tutorials <scope_rl_others_tutorial>`

.. _implementation_opl:

Off-Policy Learning
~~~~~~~~~~

Once we obtain the logged dataset, it's time to learn a new policy in an offline manner. 
For this, `d3rlpy <https://github.com/takuseno/d3rlpy>`_ provides various offline RL algorithms that work as follows.

.. code-block:: python

    # import modules
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.algos import DiscreteCQL as CQL
    from d3rlpy.models.encoders import VectorEncoderFactory
    from d3rlpy.models.q_functions import MeanQFunctionFactory
    
    # convert a (single) logged dataset to d3rlpy dataset
    offlinerl_dataset = MDPDataset(
        observations=logged_dataset["state"],
        actions=logged_dataset["action"],
        rewards=logged_dataset["reward"],
        terminals=logged_dataset["done"],
        episode_terminals=logged_dataset["done"],
        discrete_action=True,
    )
    train_episodes, test_episodes = train_test_split(
        offlinerl_dataset, 
        test_size=0.2, 
        random_state=random_state,
    )

    # define an offline RL algorithm
    cql = CQL(
        encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
        q_func_factory=MeanQFunctionFactory(),
    )

    # fit algorithm in an offline manner
    cql.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=10000,
    )

While the above procedure is alreaady simple and easy-to-use, 
we also provide :class:`OffPolicyLearning` as a meta class to further smoothen the OPL procedure with various algorithms.

.. code-block:: python

    # prepare offline RL algorithms
    cql_b1 = CQL(
        encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
        q_func_factory=MeanQFunctionFactory(),
    )
    cql_b2 = CQL(
        encoder_factory=VectorEncoderFactory(hidden_units=[100]),
        q_func_factory=MeanQFunctionFactory(),
    )
    cql_b3 = CQL(
        encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
        q_func_factory=MeanQFunctionFactory(),
    )

    # off-policy learning
    from scope_rl.policy import OffPolicyLearning
    opl = OffPolicyLearning(
        fitting_args={"n_steps": 10000},
    )
    base_policies = opl.learn_base_policy(
        logged_dataset=logged_dataset,
        algorithms=[cql_b1, cql_b2, cql_b3],
        random_state=random_state,
    )

Using :class:`OffPolicyLearning`, we can also convert the deterministic base policies to stochastic (evaluation) policies as follows.

.. code-block:: python

    # policy wrapper
    from scope_rl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead

    policy_wrappers = {
        "eps_00": (
            EpsilonGreedyHead, {
                "epsilon": 0.0,
                "n_actions": env.action_space.n,
            }
        ),
        "eps_03": (
            EpsilonGreedyHead, {
                "epsilon": 0.3,
                "n_actions": env.action_space.n,
            }
        ),
        "eps_07": (
            EpsilonGreedyHead, {
                "epsilon": 0.7,
                "n_actions": env.action_space.n,
            }
        ),
        "softmax": (
            SoftmaxHead, {
                "tau": 1.0,
                "n_actions": env.action_space.n,
            }
        )
    }

    # apply policy wrappers and convert deterministic base policies into stochastic evaluation policies
    eval_policies = opl.apply_head(
        base_policies=base_policies,
        base_policies_name=["cql_b1", "cql_b2", "cql_b3"],
        policy_wrappers=policy_wrappers,
        random_state=random_state,
    )

where we describe the policy wrappers in detail :ref:`in the next section <implementation_policy_head>`.

Also, it is possible to learn the base policy and apply policy wrappers at the same time as follows.

.. code-block:: python

    eval_policies = opl.obtain_evaluation_policy(
        logged_dataset=logged_dataset,
        algorithms=[cql_b1, cql_b2, cql_b3],
        algorithms_name=["cql_b1", "cql_b2", "cql_b3"],
        policy_wrappers=policy_wrappers,
        random_state=random_state,
    )

The obtained evaluation policies are the following (both algorithms and policy wrappers are enumerated).

.. code-block:: python

    >>> [eval_policy.name for eval_policy in eval_policies[0]]

    ['cql_b1_eps_00', 'cql_b1_eps_03', 'cql_b1_eps_07', 'cql_b1_softmax',
     'cql_b2_eps_00', 'cql_b2_eps_03', 'cql_b2_eps_07', 'cql_b2_softmax',
     'cql_b3_eps_00', 'cql_b3_eps_03', 'cql_b3_eps_07', 'cql_b3_softmax']

.. _tip_opl:

.. tip::

    .. dropdown:: How to handle OPL with multiple logged datasets?

        :class:`OffPolicyLearning` is particularly useful when fitting offline RL algorithms on multiple logged dataset.

        We can apply the same algorithms and policies wrappers across multiple datasets by the following command.

        .. code-block:: python

            eval_policies = opl.obtain_evaluation_policy(
                logged_dataset=logged_dataset,                   # MultipleLoggedDataset
                algorithms=[cql_b1, cql_b2, cql_b3],             # single list
                algorithms_name=["cql_b1", "cql_b2", "cql_b3"],  # single list
                policy_wrappers=policy_wrappers,                 # single dict
                random_state=random_state,
            )

        The evaluation policies are returned in a nested list.
        
        The other functions (i.e., :class:`learn_base_policy` and :class:`apply_head`) also work in a manner similar to the above examples.

        .. seealso::

            * :ref:`How to obtain MultipleLoggedDataset? <tips_synthetic_dataset>`
            .. * :ref:`Tutorial with MultipleLoggedDataset <scope_rl_multiple_tutorial>`

.. seealso::

    * :doc:`quickstart` 
    .. * and :ref:`related tutorials <scope_rl_others_tutorial>`

.. _implementation_policy_head:

Policy Wrapper
~~~~~~~~~~

Here, we describe some useful wrapper tools to convert a `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s policy to the behavior/evaluation policies.


======================================================   =============================================
    :ref:`Discrete <implementation_discrete_head>`       EpsilonGreedyHead, SoftmaxHead 
    :ref:`Continuous <implementation_continuous_head>`   GaussianHead, TruncatedGaussianHead, EvalHead
    :ref:`Both (Online) <implementation_online_head>`    OnlineHead
======================================================   =============================================

.. tip::

    .. dropdown:: How to customize the policy head?

        To customize the policy head, use :class:`BaseHead`. Basically, the policy head has two roles.

        1. Enabling online interactions.
        2. Converting a deterministic policy to a stochastic policy. 

        For the first purpose, we already provide the following four functions in the base class:

        * :class:`predict_online`
        * :class:`predict_value_online`
        * :class:`sample_action_online`
        * :class:`stochastic_action_with_pscore_online`

        Please just override these functions for online interactions. :class:`OnlineHead` is also useful for this purpose.

        Next, for the second purpose, you can customize how to convert a deterministic policy to a stochastic policy using following functions.

        * :class:`stochastic_action_with_pscore_online`
        * :class:`calc_action_choice_probability`
        * :class:`calc_pscore_given_action`

        .. seealso::

            * :doc:`Package Reference of BaseHead and implemented policy heads <_autosummary/scope_rl.policy.head>`


.. .. seealso::

..     * :ref:`Related tutorials <scope_rl_others_tutorial>`


.. _implementation_discrete_head:

DiscreteHead
----------
This module transforms a deterministic policy to a stochastic one in discrete action cases.
Specifically, we have two stochastic policies.

    * :class:`DiscreteEpsilonGreedyHead`: :math:`\pi(a | s) := (1 - \epsilon) * \pi_{\mathrm{det}}(a | s) + \epsilon / |\mathcal{A}|`.
    * :class:`DiscreteSoftmaxHead`: :math:`\pi(a | s) := \displaystyle \frac{\exp(Q^{(\pi_{\mathrm{det}})}(s, a) / \tau)}{\sum_{a' \in \mathcal{A}} \exp(Q^{(\pi_{\mathrm{det}})}(s, a') / \tau)}`.

Note that :math:`\epsilon \in [0, 1]` is the degree of exploration :math:`\tau` is the temperature hyperparameter.
DiscreteEpsilonGreedyHead is also used to construct a deterministic evaluation policy in OPE/OPS by setting :math:`\epsilon=0.0`.

.. _implementation_continuous_head:

ContinuousHead
----------
This module transforms a deterministic policy to a stochastic one in continuous action cases.
Specifically, we have two stochastic policies.

    * :class:`ContinuousGaussianHead`: :math:`\pi(a | s) := \mathrm{Normal}(\pi_{\mathrm{det}}(s), \sigma)`.
    * :class:`ContinuousTruncatedGaussianHead`: :math:`\pi(a | s) := \mathrm{TruncatedNormal}(\pi_{\mathrm{det}}(s), \sigma)`.

We also provide the wrapper class of deterministic policy to be used in OPE.

    * :class:`ContinuousEvalHead`: :math:`\pi(s) = \pi_{\mathrm{det}}(s)`.

.. _implementation_online_head:

OnlineHead
----------
This module enables online interaction of the policy (note: `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s policy is particularly designed for batch interactions).

    * :class:`OnlineHead`

Online Evaluation
~~~~~~~~~~
Finally, we provide the series of functions to be used for online performance evaluation in :doc:`scope_rl/ope/online.py <_autosummary/scope_rl.ope.online>`.

.. .. seealso::

..     * :ref:`Related tutorials <scope_rl_others_tutorial>`

(Rollout)

* :class:`rollout_policy_online`

(Statistics)

* :class:`calc_on_policy_policy_value`
* :class:`calc_on_policy_policy_value_interval`
* :class:`calc_on_policy_variance`
* :class:`calc_on_policy_conditional_value_at_risk`
* :class:`calc_on_policy_policy_interquartile_range`
* :class:`calc_on_policy_cumulative_distribution_function`

(Visualization)

* :class:`visualize_on_policy_policy_value`
* :class:`visualize_on_policy_cumulative_distribution_function`
* :class:`visualize_on_policy_conditional_value_at_risk`
* :class:`visualize_on_policy_interquartile_range`

.. raw:: html

    <div class="white-space-20px"></div>

.. grid::
    :margin: 0

    .. grid-item::
        :columns: 3
        :margin: 0
        :padding: 0

        .. grid::
            :margin: 0

            .. grid-item-card::
                :link: online_offline_rl
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Problem Formulation**

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
                :link: evaluation_implementation
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Off_policy Evaluation**

            .. grid-item-card::
                :link: scope_rl_api
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Package Reference**
