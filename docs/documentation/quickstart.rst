Quickstart
==========

We show an example workflow of synthetic dataset collection, offline Reinforcement Learning (RL) to Off-Policy Evaluation (OPE).
The workflow mainly consists of following three steps:

* **Synthetic Dataset Generation and Data Preprocessing**: 
    The initial step is to collect logged data using a behavior policy. In synthetic setup, we first train the behavior policy through online interaction and then generate dataset with the behavior policy. In practical situation, we can also utilize the preprocessed logged data from real-world applications.

* **Offline Reinforcement Learning**: 
    Next, we now learn a new (and better) policy from only offline logged data, without any online interactions.

* **Off-Policy Evaluation and Selection**: 
    After learning several candidate policies in an offline manner, we validate their performance in an offline manner and choose the best policy.

.. card:: workflow
    :img-top: ../_static/images/workflow.png
    :text-align: center

In this example, we use :doc:`RTBGym <rtbgym>` (a sub-package of OFRL) and `d3rlpy <https://github.com/takuseno/d3rlpy>`_. Please satisfy the `requirements <>`_ in advance.

.. seealso::

    * :doc:`distinctive_features` describes the distinctive features of OFRL in detail.
    * :doc:`Overview (online/offline RL) <online_offline_rl>` and :doc:`Overview (OPE/OPS) <ope_ops>` describe the problem settings.

.. _quickstart_dataset:

Synthetic Dataset Generation and Data Preprocessing
~~~~~~~~~~

We start by collecting the logged data using DDQN :cite:`van2016deep` as a behavior policy.

.. code-block:: python

    # implement data collection procedure on the RTBGym environment

    # import ofrl modules
    from ofrl.dataset import SyntheticDataset
    from ofrl.policy import DiscreteEpsilonGreedyHead
    # import d3rlpy algorithms
    from d3rlpy.algos import DoubleDQN
    from d3rlpy.online.buffers import ReplayBuffer
    from d3rlpy.online.explorers import ConstantEpsilonGreedy
    # import rtbgym and gym
    import rtbgym
    import gym
    # random state
    random_state = 12345

    # (0) Setup environment
    env = gym.make("RTBEnv-discrete-v0")

    # (1) Learn a baseline online policy (using d3rlpy)
    # initialize the algorithm
    ddqn = DoubleDQN()
    # train an online policy
    ddqn.fit_online(
        env,
        buffer=ReplayBuffer(maxlen=10000, env=env),
        explorer=ConstantEpsilonGreedy(epsilon=0.3),
        n_steps=100000,
        n_steps_per_epoch=1000,
        update_start_step=1000,
    )

    # (2) Generate logged dataset
    # convert ddqn policy into a stochastic behavior policy
    behavior_policy = DiscreteEpsilonGreedyHead(
        ddqn,
        n_actions=env.action_space.n,
        epsilon=0.3,
        name="ddqn_epsilon_0.3",
        random_state=random_state,
    )
    # initialize the dataset class
    dataset = SyntheticDataset(
        env=env,
        max_episode_steps=env.step_per_episode,
    )
    # collect logged data by a behavior policy
    train_logged_dataset = dataset.obtain_episodes(
        behavior_policies=behavior_policy,
        n_trajectories=10000,
        random_state=random_state,
    )
    test_logged_dataset = dataset.obtain_episodes(
        behavior_policies=behavior_policy,
        n_trajectories=10000,
        random_state= + 1,
    )

Users can collect logged data from any environment with `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_-like interface using a variety of behavior policies.
Moreover, by preprocessing the logged data, one can also handle their own logged data from real-world applications.

.. seealso::

    * :doc:`Related tutorials <_autogallery/ofrl_others/index>`
    * API references of :ref:`dataset modules <ofrl_api_dataset>` and :ref:`policy wrapper (Head) <ofrl_api_policy>`

.. _quickstart_offlinerl:

Offline Reinforcement Learning
~~~~~~~~~~

Now we are ready to learn a new policy only from logged data. Specifically, we learn CQL :cite:`kumar2020conservative` policy here. (Please also refer to :ref:`overview_offline_rl` about the problem setting and the algorithms.)
Note that, we use `d3rlpy <https://github.com/takuseno/d3rlpy>`_ for offline RL.

.. code-block:: python

    # implement offline RL procedure using ofrl and d3rlpy

    # import d3rlpy algorithms
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.algos import DiscreteCQL

    # (3) Learning a new policy from offline logged data (using d3rlpy)
    # convert dataset into d3rlpy's dataset
    offlinerl_dataset = MDPDataset(
        observations=train_logged_dataset["state"],
        actions=train_logged_dataset["action"],
        rewards=train_logged_dataset["reward"],
        terminals=train_logged_dataset["done"],
        episode_terminals=train_logged_dataset["done"],
        discrete_action=True,
    )
    # initialize the algorithm
    cql = DiscreteCQL()
    # train an offline policy
    cql.fit(
        offlinerl_dataset,
        n_steps=10000,
        scorers={},
    )

.. seealso::

    * :doc:`Related tutorials <_autogallery/ofrl_others/index>`
    * :ref:`Problem setting <overview_offline_rl>`
    * :doc:`Supported implementations and useful tools <learning_implementation>` 
    * (external) `d3rlpy's documentation <https://d3rlpy.readthedocs.io/en/latest/>`_

.. _quickstart_ope_ops:

Off-Policy Evaluation (OPE) and Selection (OPS)
~~~~~~~~~~
Finally, we evaluate the performance of the learned policy using offline logged data.

.. _quickstart_basic_ope:

Basic OPE
----------
The goal of (basic) OPE is to accurately estimate the expected performance (i.e., trajectory-wise reward) of a given evaluation policy:

.. math::

    J(\pi) := \mathbb{E}_{\tau} \left [ \sum_{t=0}^{T-1} \gamma^t r_{t} \mid \pi \right ],

where :math:`\pi` is the evaluation policy and :math:`\sum_{t=0}^{T-1} \gamma^t r_{t}` is the trajectory-wise reward. 
(See :doc:`problem setting <ope_ops>` for the detailed notations).

We compare the estimation results from various OPE estimators, Direct Method (DM) :cite:`beygelzimer2009offset` :cite:`le2019batch`, 
Trajectory-wise Importance Sampling (TIS) :cite:`precup2000eligibility`, Step-wise Importance Sampling (SIS) :cite:`precup2000eligibility`, 
and Doubly Robust (DR) :cite:`jiang2016doubly` :cite:`thomas2016data`.

.. code-block:: python

    # implement OPE procedure using OFRL

    # import OFRL modules
    from ofrl.ope import CreateOPEInput
    from ofrl.ope import DiscreteOffPolicyEvaluation as OPE
    from ofrl.ope import DiscreteDirectMethod as DM
    from ofrl.ope import DiscreteTrajectoryWiseImportanceSampling as TIS
    from ofrl.ope import DiscretePerDecisionImportanceSampling as PDIS
    from ofrl.ope import DiscreteDoublyRobust as DR

    # (4) Evaluate the learned policy in an offline manner
    # we compare ddqn, cql, and random policy
    cql_ = DiscreteEpsilonGreedyHead(
        base_policy=cql,
        n_actions=env.action_space.n,
        name="cql",
        epsilon=0.0,
        random_state=random_state,
    )
    ddqn_ = DiscreteEpsilonGreedyHead(
        base_policy=ddqn,
        n_actions=env.action_space.n,
        name="ddqn",
        epsilon=0.0,
        random_state=random_state,
    )
    random_ = DiscreteEpsilonGreedyHead(
        base_policy=ddqn,
        n_actions=env.action_space.n,
        name="random",
        epsilon=1.0,
        random_state=random_state,
    )
    evaluation_policies = [cql_, ddqn_, random_]
    # create input for OPE class
    prep = CreateOPEInput(
        env=env,
        logged_dataset=test_logged_dataset,
        use_base_model=True,  # use model-based prediction
    )
    input_dict = prep.obtain_whole_inputs(
        evaluation_policies=evaluation_policies,
        n_trajectories_on_policy_evaluation=100,
        random_state=random_state,
    )
    # initialize the OPE class
    ope = OPE(
        logged_dataset=test_logged_dataset,
        ope_estimators=[DM(), TIS(), PDIS(), DR()],
    )
    # conduct OPE and visualize the result
    ope.visualize_off_policy_estimates(
        input_dict,
        random_state=random_state,
        sharey=True,
    )

.. card:: 
    :img-top: ../_static/images/ope_policy_value_basic.png
    :text-align: center
    
    Policy Value Estimated by OPE Estimators

Users can implement their own OPE estimators by following the interface of :class:`obp.ope.BaseOffPolicyEstimator`.
In addition, :class:`obp.ope.OffPolicyEvaluation` summarizes and compares the estimation results of various OPE estimators.

.. seealso::

    * :doc:`Related tutorials <_autogallery/basic_ope/index>`
    * :doc:`Problem setting <ope_ops>`
    * :doc:`Supported OPE estimators <evaluation_implementation>` and :doc:`their API reference <_autosummary/ofrl.ope.basic_estimators_discrete>` 
    * (advanced) :ref:`Marginal OPE estimators <implementation_marginal_ope>`, and their :doc:`API reference <_autosummary/ofrl.ope.marginal_ope_discrete>`

.. _quickstart_cumulative_distribution_ope:

Cumulative Distribution OPE
----------
while the basic OPE is beneficial for estimating the average policy performance, we are often also interested in the performance distribution of the evaluation policy
and risk-sensitive performance metrics including conditional value at risk (CVaR).
Cumulative distribution OPE enables to estimate the following cumulative distribution function and risk functions derived by CDF.

.. math::

    F(m, \pi) := \mathbb{E} \left[ \mathbb{I} \left \{ \sum_{t=0}^{T-1} \gamma^t r_t \leq m \right \} \mid \pi \right]

The following shows the example of estimating cumulative distribution function of the trajectory-wise rewards and its statistics 
using Cumulative Distribution OPE estimators :cite:`huang2021off` :cite:`huang2022off` :cite:`chandak2021universal`.

.. code-block:: python

    # import OFRL modules
    from ofrl.ope import DiscreteCumulativeDistributionOffPolicyEvaluation as CumulativeDistributionOPE
    from ofrl.ope import DiscreteCumulativeDistributionDirectMethod as CD_DM
    from ofrl.ope import DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling as CD_IS
    from ofrl.ope import DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust as CD_DR
    from ofrl.ope import DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling as CD_SNIS
    from ofrl.ope import DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust as CD_SNDR

    # (4) Evaluate the learned policy using cumulative distribution function (in an offline manner)
    # we compare ddqn, cql, and random policy defined in the previous section (i.e., (3) of basic OPE procedure)
    # initialize the OPE class
    cd_ope = CumulativeDistributionOPE(
        logged_dataset=test_logged_dataset,
        ope_estimators=[
        CD_DM(estimator_name="cdf_dm"),
        CD_IS(estimator_name="cdf_is"),
        CD_DR(estimator_name="cdf_dr"),
        CD_SNIS(estimator_name="cdf_snis"),
        CD_SNDR(estimator_name="cdf_sndr"),
        ],
    )
    # estimate variance
    variance_dict = cd_ope.estimate_variance(input_dict)
    # estimate CVaR
    cvar_dict = cd_ope.estimate_conditional_value_at_risk(input_dict, alphas=0.3)
    # estimate and visualize cumulative distribution function
    cd_ope.visualize_cumulative_distribution_function(input_dict, n_cols=4)

.. card:: 
    :img-top: ../_static/images/ope_cumulative_distribution_function.png
    :text-align: center
    
    Cumulative Distribution Function Estimated by OPE Estimators

Users can implement their own OPE estimators by following the interface of :class:`obp.ope.BaseCumulativeDistributionOffPolicyEstimator`.
In addition, :class:`obp.ope.DiscreteCumulativeDistributionOffPolicyEvaluation` summarizes and compares the estimation results of various OPE estimators.

.. seealso::

    * :doc:`Related tutorials <_autogallery/cumulative_distribution_ope/index>`
    * :ref:`Problem setting <overview_cumulative_distribution_ope>`
    * :ref:`Supported cumulative distribution OPE estimators <implementation_cumulative_distribution_ope>` 
    and :doc:`their API reference <_autosummary/ofrl.ope.cumulative_distribution_ope_discrete>` 

.. _quickstart_ops:

Off-Policy Selection and Evaluation of OPE/OPS
----------
Finally, we provide the code to conduct OPS, which selects the "best" performing policies among several candidates.

.. code-block:: python

    # import OFRL modules
    from ofrl.ope import OffPolicySelection

    # (5) Conduct Off-Policy Selection
    # Initialize the OPS class
    ops = OffPolicySelection(
        ope=ope,
        cumulative_distribution_ope=cd_ope,
    )
    # rank candidate policy by policy value estimated by (basic) OPE
    ranking_dict = ops.select_by_policy_value(input_dict)
    # rank candidate policy by policy value estimated by cumulative distribution OPE
    ranking_dict_ = ops.select_by_policy_value_via_cumulative_distribution_ope(input_dict)

    # (6) Evaluate OPS/OPE results
    # rank candidate policy by estimated lower quartile and evaluate the selection results
    ranking_df, metric_df = ops.select_by_lower_quartile(
        input_dict,
        alpha=0.3,
        return_metrics=True,
        return_by_dataframe=True,
    )
    # visualize the top k deployment result
    # compared estimators are also easily specified
    ops.visualize_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        compared_estimators=["cdf_dm", "cdf_is", "cdf_dr", "cdf_snis", "cdf_sndr"],
        safety_criteria=1.0,
    )
    # visualize the OPS results with the ground-truth metrics
    ops.visualize_variance_for_validation(
        input_dict,
        share_axes=True,
    )

.. card:: 
    :img-top: ../_static/images/ops_topk_lower_quartile.png
    :text-align: center
    
    Comparison of the Top-k Statistics of 10% Lower Quartile of Policy Value

.. card:: 
    :img-top: ../_static/images/ops_variance_validation.png
    :text-align: center
    
    Validation of Estimated and Ground-truth Variance of Policy Value

.. seealso::

    * :doc:`Related tutorials <_autogallery/ops/index>`
    * :ref:`Problem setting <overview_ops>`
    * :ref:`OPS evaluation protocols <implementation_eval_ope_ops>` and :doc:`their API reference <_autosummary/ofrl.ope.ops>` 

~~~~~

More tutorials with a variety of environments and OPE estimators are available in the next page!

.. raw:: html

    <div class="white-space-5px"></div>

.. grid::

    .. grid-item::
        :columns: 2
        :margin: 0
        :padding: 0

        .. grid::
            :margin: 0

            .. grid-item-card::
                :link: installation
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Quickstart**

    .. grid-item::
        :columns: 8
        :margin: 0
        :padding: 0

    .. grid-item::
        :columns: 2
        :margin: 0
        :padding: 0

        .. grid::
            :margin: 0

            .. grid-item-card::
                :link: _autogallery/index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Tutorial**

            .. grid-item-card::
                :link: index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Documentation**

