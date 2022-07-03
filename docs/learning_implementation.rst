==========
Supported Implementation
==========

Our implementation aims to streamline the data collection, (offline) policy learning, and off-policy evaluation and selection (OPE/OPS) procedure.
We rely on `d3rlpy's implementation <>`_ of the learning algorithms and provide some useful tools to streamline the above offline RL procedure.

Please also refer to `quickstart <>`_ for the example workflows.

Synthetic Dataset Generation
~~~~~~~~~~
Synthetic Dataset Generation Module is an easy-to-use data collection module which is compatible to any `OpenAI Gym <https://gym.openai.com>`_-like RL environment.

It takes an RL environment and the behavior policy (i.e., data collection policy) as input to instantiate the class.

.. code-block:: python

    # initialize the dataset class
    >>> from offlinegym.dataset import SyntheticDataset
    >>> dataset = SyntheticDataset(
            env=env,
            behavior_policy=behavior_policy,
            random_state=random_state,
        )

Then, it collects logged data as follows.

.. code-block:: python

    # collect logged data using behavior policy
    >>> logged_dataset = dataset.obtain_trajectories(n_episodes=10000)


The behavior policy can either be deterministic or stochastic when conducting offline policy learning. 
For OPE/OPS, the behavior policy should be a stochastic one.

To convert the d3rlpy's deterministic policy to a stochastic one, we provide several wrapper classes to ease implementation as follows.

Policy Wrapper
~~~~~~~~~~

Here, we describe some useful wrapper tools to convert the d3rlpy's policy.

DiscreteHead
----------
This module transforms a deterministic policy to a stochastic one in discrete action case.
Specifically, we have two stochastic policies.

* :class:`DiscreteEpsilonGreedyHead`: :math:`\pi(a | s) := (1 - \epsilon) * \pi_{\mathrm{det}}(a | s) + \epsilon / |\mathcal{A}|`.
* :class:`DiscreteSoftmaxHead`: :math:`\pi(a | s) := \displaystyle \frac{\exp(Q^{(\pi_{\mathrm{det}})}(s, a) / \tau)}{\sum_{a' \in \mathcal{A}} \exp(Q^{(\pi_{\mathrm{det}})}(s, a') / \tau)}`.

Note that :math:`\mathbb{I}(\cdot)` is the indicator function and :math:`\tau` is the temperature parameter.
DiscreteEpsilonGreedyHead is also used to construct a deterministic evaluation policy in OPE/OPS.

ContinuousHead
----------
This module transforms a deterministic policy to a stochastic one in discrete action case.
Specifically, we have two stochastic policies.

* :class:`ContinuousGaussianHead`: :math:`\pi(a | s) := \mathrm{Normal}(\pi_{\mathrm{det}}(s), \sigma)`.
* :class:`ContinuousTruncatedGaussianHead`: :math:`\pi(a | s) := \mathrm{TruncatedNormal}(\pi_{\mathrm{det}}(s), \sigma)`.

We also provide the wrapper class of deterministic policy to be used in OPE.

* :class:`ContinuousEvalHead`: :math:`\pi(s) = \pi_{\mathrm{det}}(s)`.

OnlineHead
----------
This module enables step-wise interaction of the policy.

* :class:`OnlineHead`

Online Evaluation
~~~~~~~~~~
Finally, we provide the series of functions to be used for online performance evaluation in `ope/online.py <>`_.

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

For further descriptions, please also refer to `package reference <>`_.