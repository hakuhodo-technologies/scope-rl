Example Codes for Off-Policy Selection
==========

Here, we show example codes for conducting policy selection via OPE (i.e., Off-Policy Selection; OPS).

.. seealso::

    For preparation, please also refer to the following pages:

    * :doc:`What are Off-Policy Evaluation and Selection? </documentation/ope_ops>`
    * :doc:`Supported OPE estimators and OPS methods </documentation/evaluation_implementation>`
    * :doc:`Supported implementations for data collection and Offline RL </documentation/learning_implementation>`
    * :doc:`Example codes for basic OPE </documentation/examples/basic_ope>`
    * :doc:`Example codes for cumulative distribution OPE </documentation/examples/cumulative_dist_ope>`

Prerequisite
~~~~~~~~~~

Here, we assume that an RL environment, a behavior policy, and evaluation policies are given as follows.

* ``behavior_policy``: an instance of :class:`BaseHead`
* ``evaluation_policies``: a list of instance(s) of :class:`BaseHead`
* ``env``: a gym environment (unnecessary when using real-world datasets)

Additionally, we assume that the logged datasets, inputs, and either ope or cd_ope instances are ready to use.
For initializing the ope and cd_ope instances, please refer to :doc:`this page </documentation/examples/basic_ope>` 
and :doc:`this page </documentation/examples/cumulative_dist_ope>` as references, respectively.

* ``logged_dataset``: a dictionary containing the logged dataset
* ``input_dict``: a dictionary containing inputs for OPE
* ``ope``: an instance of :class:`OffPolicyEvaluation`
* ``cd_ope``: an instance of :class:`CumulativeDistributionOPE`

Note that, in the following example, we use a single logged dataset for simplicity.
For the case of using multiple behavior policies or multiple logged datasets, refer to :doc:`/documentation/examples/multiple`.

Off-Policy Selection
~~~~~~~~~~
OPS class calls the ``ope`` and ``cd_ope`` instances for OPE.

.. code-block:: python

    from scope_rl.ope import OffPolicySelection

    ops = OffPolicySelection(
        ope=ope,  # either ope or cd_ope must be given
        cumulative_distribution_ope=cd_ope,
    )

OPS via basic OPE
----------
By default, the following function returns the estimated ranking of evaluation policies with their (estimated) policy value as follows.

.. code-block:: python

    ranking_dict = ops.select_by_policy_value(
        input_dict=input_dict,
    )

To return the results in a dataframe format, enable the following option.

.. code-block:: python

    ranking_df = ops.select_by_policy_value(
        input_dict=input_dict,
        return_by_dataframe=True,  #
    )

With the following option, we can also verify the true (on-policy) policy value. 
Note that, this function is only applicable when the on-policy policy value of evaluation policies are recorded in ``input_dict``.

.. code-block:: python

    ranking_df = ops.select_by_policy_value(
        input_dict=input_dict,
        return_true_values=True,
        return_by_dataframe=True,  #
    )

SCOPE-RL also handles OPS with high-confidence OPE as follows.

.. code-block:: python

    ranking_df = ops.select_by_policy_value_lower_bound(
        input_dict=input_dict,
        cis=["bootstrap", "bernstein", "hoeffding", "ttest"],  # the choices of inequality
        return_by_dataframe=True,
        random_state=12345,
    )

OPS via cumulative distribution OPE
----------
We can also conduct OPS via CD-OPE in a manner similar to basic OPE.

First, the following conduct OPS via policy value estimated by CD-OPE.

.. code-block:: python

    ranking_df = ops.select_by_policy_value_via_cumulative_distribution_ope(
        input_dict=input_dict,
        return_by_dataframe=True,
    )

OPS is also conducted by CVaR and lower quartile as follows.

.. code-block:: python

    # CVaR
    ranking_df = ops.select_by_conditional_value_at_risk(
        input_dict=input_dict,
        return_by_dataframe=True,
        alpha=0.3,  # specify the proportion of the sided region
    )
    # lower quartile
    ranking_df = ops.select_by_lower_quartile(
        input_dict=input_dict,
        return_by_dataframe=True,
        alpha=0.3,  # specify the proportion of the sided region
    )

Obtaining oracle selection results
----------
By default, the following function returns the ranking of evaluation policies with their (ground-truth) policy value as follows.
Note that, this function is only applicable when the on-policy policy value of evaluation policies are recorded in ``input_dict``.

.. code-block:: python

    oracle_selection_dict = ops.obtain_true_selection_result(
        input_dict=input_dict,
    )

To return the results in a dataframe format, enable the following option.

.. code-block:: python

    oracle_selection_df = ops.obtain_true_selection_result(
        input_dict=input_dict,
        return_by_dataframe=True,  #
    )

To return variance, enable the following option.

.. code-block:: python

    oracle_selection_df = ops.obtain_true_selection_result(
        input_dict=input_dict,
        return_variance=True,  #
        return_by_dataframe=True,
    )

To return CVaR and the ranking of candidate policies based on CVaR, enable the following option.

.. code-block:: python

    oracle_selection_df = ops.obtain_true_selection_result(
        input_dict=input_dict,
        return_conditional_value_at_risk=True,  #
        cvar_alpha=0.3,  # specify the proportion of the sided region
        return_by_dataframe=True,
    )

To return the lower quartile and the ranking of candidate policies based on the lower quartile, enable the following option.

.. code-block:: python

    oracle_selection_df = ops.obtain_true_selection_result(
        input_dict=input_dict,
        return_lower_quartile=True,  #
        quartile_alpha=0.3,  # specify the proportion of the sided region
        return_by_dataframe=True,
    )

Calling visualization functions from ope / cd_ope instances
----------
Finally, we should also note that the functions of ope and cd_ope instances are available via ops instance as follows.

.. code-block:: python

    # ope.visualize_off_policy_estimates(...)
    ops.visualize_policy_value_for_selection(...) 

    # cd_ope.visualize_cumulative_distribution_function(...)
    ops.visualize_cumulative_distribution_function_for_selection(...)

    # cd_ope.visualize_policy_value(...)
    ops.visualize_policy_value_of_cumulative_distribution_ope_for_selection(...)

    # cd_ope.visualize_conditional_value_at_risk(...)
    ops.visualize_conditional_value_at_risk_for_selection(...)

    # cd_ope.visualize_interquartile_range(...)
    ops.visualize_interquartile_range_for_selection(...)

.. seealso::

    For the evaluation of OPS results, please also refer to :doc:`/documentation/examples/assessments`.

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
                :link: /documentation/examples/assessments
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Assessments**
