Example Codes for Assessing OPE Estimators
==========

Here, we show example codes for assessing OPE/OPS results.

.. seealso::

    For preparation, please also refer to the following pages:

    * :doc:`What are Off-Policy Evaluation and Selection? </documentation/ope_ops>`
    * :doc:`Supported Evaluation Protococols for OPE/OPS </documentation/evaluation_implementation>`
    * :doc:`/documentation/sharpe_ratio`
    * :doc:`Supported Implementations for data collection and Offline RL </documentation/learning_implementation>`
    * :doc:`Example codes for basic OPE </documentation/examples/basic_ope>`
    * :doc:`Example codes for cumulative distribution OPE </documentation/examples/cumulative_dist_ope>`
    * :doc:`Example codes for OPS </documentation/examples/ops>`

Prerequisite
~~~~~~~~~~

Here, we assume that an RL environment, a behavior policy, and evaluation policies are given as follows.

* ``behavior_policy``: an instance of :class:`BaseHead`
* ``evaluation_policies``: a list of instance(s) of :class:`BaseHead`
* ``env``: a gym environment (unecessary when using real-world datasets)

Additionally, we assume that the logged datasets, inputs, and either ope or cd_ope instances are ready to use.
For initializing the ope and cd_ope instances, please refer to :doc:`this page </documentation/examples/basic_ope>` 
and :doc:`this page </documentation/examples/cumulative_dist_ope>` as references, respectively.

* ``logged_dataset``: a dictionary containing the logged dataset
* ``input_dict``: a dictionaty containing inputs for OPE
* ``ope``: an instance of :class:`OffPolicyEvaluation`
* ``cd_ope``: an instance of :class:`CumulativeDistributionOPE`

Note that, to run the following example codes, ``input_dict`` should contain on-policy policies of each candidate policy. 
This requirement is automatically satisfied when collecting logged dataset by handing ``env`` over the :class:`CreateInput` class.

In the following examples, we also use a single logged dataset for simplicity.
For the case of using multiple behavior policies or multiple logged datasets, refer to :doc:`/documentation/ecamples/multiple`.

Assessing OPE/OPS results
~~~~~~~~~~
The assessments uses the OPS class.

.. code-block:: python

    from scope_rl.ope import OffPolicySelection

    ops = OffPolicySelection(
        ope=ope,  # either ope or cd_ope must be given
        cumulative_distribution_ope=cd_ope,
    )

Assessments with conventional metrics
----------
The convensional metrics including MSE, RankCorr, Regret, and Type I and II Errors are available in the ops function.

.. code-block:: python

    ranking_dict, metric_dict = ops.select_by_policy_value(
        input_dict=input_dict,
        return_metrics=True,  # enable this option
        return_by_dataframe=True,
    )

To compare Regret@k, specify the following commands.

.. code-block:: python

    ranking_dict, metric_dict = ops.select_by_policy_value(
        input_dict=input_dict,
        top_k_in_eval_metrics=1,  # specify the value of k
        return_metrics=True,
        return_by_dataframe=True,
    )

We can also specify the reward threshold for Type I and II errors as follows.

.. code-block:: python

    ranking_dict, metric_dict = ops.select_by_policy_value(
        input_dict=input_dict,
        safety_threshold=10.0,  # specify the safety threshold
        return_metrics=True,
        return_by_dataframe=True,
    )

To use the value relative to behavior policy as a threshold, use the following option.

.. code-block:: python

    ranking_dict, metric_dict = ops.select_by_policy_value(
        input_dict=input_dict,
        relative_safety_criteria=0.90,  # specify the relative safety threshold
        return_metrics=True,
        return_by_dataframe=True,
    )

Similar evaluations are available in the following functions.

* :class:`select_by_policy_value`
* :class:`select_by_policy_value_lower_bound`
* :class:`select_by_policy_value_via_cumulative_distribution_ope`
* :class:`select_by_conditional_value_at_risk`
* :class:`select_by_lower_quartile`

Assessments with top-:math:`k` deployment results
----------

SCOPE-RL enables to obtain and compare the statistics of policy portfolio formed by each estiamtor as follows.

.. code-block:: python

    topk_metric_df = ops.obtain_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        return_by_dataframe=True,
    )

In the ``topk_metric_df``, you will find the ``k-th``, ``best``, ``worst``, and ``mean`` policy values and ``std`` of policy values amond top- :math:`k`
policy portfolio. We also report the proposed SharpRatio@k metric as ``sharpe_ratio``.

Note that, to additionally report the safety violation rate, specify the following options.

.. code-block:: python

    topk_metric_df = ops.obtain_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        return_safety_violation_rate=True,  # enable this option
        safety_threshold=10.0,  # specify the safety threshold
        return_by_dataframe=True,
    )

To use the value relative to the behavior policy as the safety requirement, use the following option.

.. code-block:: python

    topk_metric_df = ops.obtain_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        return_safety_violation_rate=True,  # enable this option
        relative_safety_criteria=0.90,  # specify the relative safety threshold
        return_by_dataframe=True,
    )

Similar evaluations are available in the following functions.

* :class:`obtain_topk_policy_value_selected_by_standard_ope`
* :class:`obtain_topk_policy_value_selected_by_lower_bound`
* :class:`obtain_topk_policy_value_selected_by_cumulative_distribution_ope`
* :class:`obtain_topk_conditional_value_at_risk_selected_by_cumulative_distirbution_ope`
* :class:`obtain_topk_lower_quartile_selected_by_cumulative_distribution_ope`

We can also evaluate CVaR of top-:math:`k` policies selected based on estimated policy value as follows.

.. code-block:: python

    topk_metric_df = ops.obtain_topk_conditional_value_at_risk_selected_by_standard_ope(
        input_dict=input_dict,
        return_by_dataframe=True,
        ope_alpha=0.3,
    )

We can also evaluate lower quartile of top-:math:`k` policies selected based on estimated policy value as follows.

.. code-block:: python

    topk_metric_df = ops.obtain_topk_lower_quartile_selected_by_standard_ope(
        input_dict=input_dict,
        return_by_dataframe=True,
        ope_alpha=0.3,
    )

Visualizing top-:math:`k` deployment results
----------
SCOPE-RL also provides functions to visualize the above top-:math:`k` policy performances.

.. code-block:: python

    ops.visualize_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        metrics=["k-th", "best", "worst", "mean", "std", "safety_violation_rate", "sharpe_ratio"],  # (default)
        compared_estimators=["dm", "tis", "pdis", "dr"],  # (optional)
        relative_safety_criteria=1.0,  # (optional)
        clip_sharpe_ratio=True,  # (optional)
        ymax_sharpe_ratio=5.0,  # (optional)
        legend=True,  # (optional)
    )

.. card:: 
   :img-top: ../../_static/images/ops_topk_policy_value.png
   :text-align: center

Similar evaluations are available in the following functions.

* :class:`visualize_topk_policy_value_selected_by_standard_ope`
* :class:`visualize_topk_policy_value_selected_by_lower_bound`
* :class:`visualize_topk_policy_value_selected_by_cumulative_distribution_ope`
* :class:`visualize_topk_conditional_value_at_risk_selected_by_cumulative_distirbution_ope`
* :class:`visualize_topk_lower_quartile_selected_by_cumulative_distribution_ope`

Again, the visualization functions are also able to show CVaR and lower quartile of top-:math:`k` policies selected based on estimated policy value as follows.

.. code-block:: python

    # visualize CVaR
    ops.visualize_topk_conditional_value_at_risk_selected_by_standard_ope(
        input_dict=input_dict,
        metrics=["best", "worst", "mean", "std"],  # (optional)
        compared_estimators=["dm", "tis", "pdis", "dr"],  # (optional)
        ope_alpha=0.3,
    )
    # visualize lower quartile
    ops.visualize_topk_lower_quartile_selected_by_standard_ope(
        input_dict=input_dict,
        metrics=["best", "worst", "mean", "std"],  # (optional)
        compared_estimators=["dm", "tis", "pdis", "dr"],  # (optional)
        ope_alpha=0.3,
    )

Visualizing the true and estimated policy performances
----------
Finally, SCOPE-RL also implements functions to compare the true and estimated policy performances via scatter plots.

.. code-block:: python

    ops.visualize_policy_value_for_validation(
        input_dict=input_dict,
        n_cols=4,  # (optional)
    )

.. card:: 
   :img-top: ../../_static/images/ops_validation_policy_value.png
   :text-align: center

Note that, the same y-axes are used with ``sharey`` option.

.. code-block:: python

    ops.visualize_policy_value_for_validation(
        input_dict=input_dict,
        n_cols=4,
        sharey=True,  # enable this option
    )

Similar evaluations are available in the following functions.

* :class:`visualize_policy_value_for_validation`
* :class:`visualize_policy_value_of_cumulative_distribution_ope_for_validation`
* :class:`visualize_variance_for_validation`
* :class:`visualize_lower_quartile_for_validation`
* :class:`visualize_conditional_value_at_risk_for_validation`

.. raw:: html

    <div class="white-space-20px"></div>

.. grid::
    :margin: 0

    .. grid-item::
        :columns: 2
        :margin: 0
        :padding: 0

        .. grid::
            :margin: 0

            .. grid-item-card::
                :link: /documentation/examples/index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Usage**

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
                :link: /documentation/subpackages/multiple
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Multiple Datasets**

