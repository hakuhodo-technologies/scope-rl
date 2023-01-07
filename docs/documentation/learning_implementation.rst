==========
Supported Implementation
==========

Our implementation aims to streamline the data collection, (offline) policy learning, and off-policy evaluation/selection (OPE/OPS) procedure.
We rely on `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s implementation of the learning algorithms and provide some useful tools to streamline the above offline RL procedure.

.. _implementation_dataset:

Synthetic Dataset Generation
~~~~~~~~~~
:class:`SyntheticDataset` is an easy-to-use data collection module which is compatible to any `OpenAI Gym <https://gym.openai.com>`_ and `Gymnasium <https://gymnasium.farama.org/>`_-like RL environment.

It takes an RL environment and the behavior policy (i.e., data collection policy) as input to instantiate the class.

.. code-block:: python

    # initialize the dataset class
    from ofrl.dataset import SyntheticDataset
    dataset = SyntheticDataset(
        env=env,
        behavior_policy=behavior_policy,
        max_episode_steps=env.step_per_episode,
        random_state=random_state,
    )

Then, it collects logged data as follows.

.. code-block:: python

    # collect logged data by a behavior policy
    logged_dataset = dataset.obtain_episodes(n_trajectories=10000)

.. tip::

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
            ]

        .. note::
            
            ``logged_dataset`` can be used for OPE even if ``action_keys``, ``action_meaning``, ``state_keys``, ``info`` is not provided.
            For API consistency, just leave ``None`` when these keys are unnecessary. 
            
            Moreover, offline RL algorithms, FQE (model-based OPE), and marginal OPE estimators 
            can also work without ``pscore``. 

        .. seealso::

            :doc:`Package Reference <_autosummary/dataset/ofrl.dataset.base>` explains the meaning of each keys in detail.


    .. dropdown:: How to handle multiple logged datasets at once?

        :class:`MultipleLoggedDataset` enables us to smoothly handle multiple logged datasets. 

        Specifically, :class:`MultipleLoggedDataset` saves the paths to each logged dataset and make each dataset accessible through the following command.
        
        .. code-block:: python

            logged_dataset_0 = multiple_logged_dataset.get(0)
        
        There are two ways to obtain :class:`MultipleLoggedDataset`.

        The first way is to directly get :class:`MultipleLoggedDataset` as the output of :class:`SyntheticDataset` as follows.

        .. code-block:: python

            synthetic_dataset = SyntheticDataset(
                env=env,
                behavior_policy=behavior_policy,
                max_episode_steps=env.step_per_episode,
                random_state=random_state,
            )
            multiple_logged_dataset = synthetic_dataset.obtain_episodes(
                n_datasets=5,          # when n_datasets > 1, MultipleLoggedDataset is returned
                n_trajectories=10000,
            )

        The second way to define :class:`MultipleLoggedDataset` manually as follows.

        .. code-block:: python

            multiple_logged_dataset = MultipleLoggedDataset(
                action_type="discrete",
                path="logged_dataset/",  # either absolute or relative path
            )

            for behavior_policy in behavior_policies:
                synthetic_dataset = SyntheticDataset(
                    env=env,
                    behavior_policy=behavior_policy,
                    max_episode_steps=env.step_per_episode,
                    random_state=random_state,
                )
                single_logged_dataset = synthetic_dataset.obtain_episodes(
                    n_trajectories=10000,
                )

                # add a single logged dataset to multiple_logged_dataset
                multiple_logged_dataset.add(
                    single_logged_dataset, 
                    name=behavior_policy.name,
                )

        .. seealso::

            * :doc:`Package Reference of MultipleLoggedDataset <_autosummary/ofrl.utils.MultipleLoggedDataset>`
            * :doc:`TODO tutorial with MultipleLoggedDataset`

.. seealso::

    * :doc:`quickstart` and :doc:`related tutorials <_autogallery/ofrl_others/index>`

The behavior policy can either be deterministic or stochastic when conducting offline policy learning.
For OPE/OPS, the behavior policy should be a stochastic one.

To convert the d3rlpy's deterministic policy to a stochastic one, we provide several wrapper classes to ease implementation as follows.

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

        To customize the policy head, use :class:`BaseHead`.

.. seealso::

    * :doc:`Related tutorials <_autogallery/ofrl_others/index>`


.. _implementation_discrete_head:

DiscreteHead
----------
This module transforms a deterministic policy to a stochastic one in discrete action case.
Specifically, we have two stochastic policies.

    * :class:`DiscreteEpsilonGreedyHead`: :math:`\pi(a | s) := (1 - \epsilon) * \pi_{\mathrm{det}}(a | s) + \epsilon / |\mathcal{A}|`.
    * :class:`DiscreteSoftmaxHead`: :math:`\pi(a | s) := \displaystyle \frac{\exp(Q^{(\pi_{\mathrm{det}})}(s, a) / \tau)}{\sum_{a' \in \mathcal{A}} \exp(Q^{(\pi_{\mathrm{det}})}(s, a') / \tau)}`.

Note that :math:`\mathbb{I}(\cdot)` is the indicator function and :math:`\tau` is the temperature parameter.
DiscreteEpsilonGreedyHead is also used to construct a deterministic evaluation policy in OPE/OPS.

.. _implementation_continuous_head:

ContinuousHead
----------
This module transforms a deterministic policy to a stochastic one in discrete action case.
Specifically, we have two stochastic policies.

    * :class:`ContinuousGaussianHead`: :math:`\pi(a | s) := \mathrm{Normal}(\pi_{\mathrm{det}}(s), \sigma)`.
    * :class:`ContinuousTruncatedGaussianHead`: :math:`\pi(a | s) := \mathrm{TruncatedNormal}(\pi_{\mathrm{det}}(s), \sigma)`.

We also provide the wrapper class of deterministic policy to be used in OPE.

    * :class:`ContinuousEvalHead`: :math:`\pi(s) = \pi_{\mathrm{det}}(s)`.

.. _implementation_online_head:

OnlineHead
----------
This module enables step-wise interaction of the policy.

    * :class:`OnlineHead`

Online Evaluation
~~~~~~~~~~
Finally, we provide the series of functions to be used for online performance evaluation in :doc:`ofrl/ope/online.py <_autosummary/ofrl.ope.online>`.

.. seealso::

    * :doc:`Related tutorials <_autogallery/ofrl_others/index>`

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

.. seealso::

    * :doc:`Related tutorials <_autogallery/ofrl_others/index>`

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
                :link: ofrl_api
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Package Reference**
