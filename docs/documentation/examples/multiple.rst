Example Codes with Multiple Logged Dataset and Behavior Policies
==========

Here, we show example codes for conducting OPE and OPS with multiple logged datasets.

.. seealso::

    For preparation, please also refer to the following pages about the case with a single logged dataset:

    * :doc:`Data collection and Offline RL </documentation/learning_implementation>`
    * :doc:`Example codes for basic OPE </documentation/examples/basic_ope>`
    * :doc:`Example codes for cumulative distribution OPE </documentation/examples/cumulative_dist_ope>`
    * :doc:`Example codes for OPS </documentation/examples/ops>`
    * :doc:`Example codes for assessing OPE and OPS </documentation/examples/assessments>`

Logged Dataset
~~~~~~~~~~
Here, we assume that an RL environment, behavior policies, and evaluation policies are given as follows.

* ``behavior_policy``: an instance of :class:`BaseHead` or a list of instance(s) of :class:`BaseHead` 
* ``evaluation_policies``: a list of instance(s) of :class:`BaseHead`
* ``env``: a gym environment (unecessary when using real-world datasets)

Then, we can collect multiple logged datasets with a single behavior policy as follows.

.. code-block:: python

    from scope_rl.dataset import SyntheticDataset
    
    # initialize dataset
    dataset = SyntheticDataset(
        env=env,
        max_episode_steps=env.step_per_episode,
    )
    # obtain logged dataset
    multiple_logged_datasets = dataset.obtain_episodes(
        behavior_policies=behavior_policies[0],  # a single behavior policy
        n_datasets=5,  # specify the number of dataset (i.e., number of different random seeeds)
        n_trajectories=10000, 
        random_state=random_state,
    )

Similarly, SCOPE-RL also collects multiple logged datasets with multiple behavior policies as follows.

.. code-block:: python

    multiple_logged_datasets = dataset.obtain_episodes(
        behavior_policies=behavior_policies,  # multiple behavior policies
        n_datasets=5,  # specify the number of dataset (i.e., number of different random seeeds) for each behavior policy
        n_trajectories=10000, 
        random_state=random_state,
    )

The multiple logged datasets are returned as an instance of :class:`MultipleLoggedDataset`. 
Note that, we can also manually create multiple logged datasets as follows.

.. code-block:: python

    from scope_rl.utils import MultipleLoggedDataset

    multiple_logged_dataset = MultipleLoggedDataset(
        action_type="discrete",
        path="logged_dataset/",  # specify the path to the dataset
    )

    for behavior_policy in behavior_policies:
        single_logged_dataset = dataset.obtain_episodes(
            behavior_policies=behavior_policy,  # a single behavior policy
            n_trajectories=10000,
            random_state=random_state,
        )

        # add a single_logged_dataset to multiple_logged_dataset
        multiple_logged_dataset.add(
            single_logged_dataset,
            behavior_policy_name=behavior_policy.name,
            dataset_id=0,
        )

Once you create the multiple logged datasets, each dataset is accessible via the following code.

.. code-block:: python

    single_logged_dataset = multiple_logged_dataset.get(
        behavior_policy_name=behavior_policies[0].name, dataset_id=0,
    )

:class:`MultipleLoggedDataset` also has the following properties.

.. code-block:: python

    # a list of the name of behavior policies
    multiple_logged_dataset.behavior_policy_names

    # a dictionary of the number of datasets for each behavior policy
    multiple_logged_dataset.n_datasets

Inputs
~~~~~~~~~~
The next step is to create the inputs for OPE estimators. 
Here, we show the case of creating inputs for importance-sampling based estimators for brevity.
For the case of creating inputs for model-based and marginal importance-sampling based estimators, please also refer to :doc:`Example Codes for Basic OPE </documentation/examples/basic_ope>`.

We first show the case of creating whole logged datasets stored in ``multiple_logged_datasets`` (, which is essentially the same with the case of using ``single_logged_dataset``).

.. code-block:: python

    from scope_rl.ope import CreateOPEInput

    # initialize class to create inputs
    prep = CreateOPEInput(
        env=env,  # unecessary when using real-world dataset
    )
    # create inputs (e.g., calculating )
    multiple_input_dict = prep.obtain_whole_inputs(
        logged_dataset=multiple_logged_dataset,
        evaluation_policies=evaluation_policies,
        n_trajectories_on_policy_evaluation=100,  # when evaluating OPE (optional)
        random_state=random_state,
    )

The above code returns ``multiple_input_dict`` as an instance of :class:`MultipleInputDict`. 
Each input dictionary is accessble via the following code.

.. code-block:: python

    single_input_dict = multiple_input_dict.get(
        behavior_policy_name=behavior_policies[0].name, dataset_id=0,
    )

:class:`MultipleInputDict` has the following properties.

.. code-block:: python

    # a list of the name of behavior policies
    multiple_input_dict.behavior_policy_names

    # a dictionary of the number of datasets for each behavior policy
    multiple_input_dict.n_datasets

    # a dictionary of the number of evaluation policies of each input dict
    multiple_input_dict.n_eval_policies

    # check if the contained logged datasets use the same evaluation policies
    multiple_input_dict.use_same_eval_policy_across_dataset

Note that, it is also possible to create a single input dict using the :class:`CreateOPEInput` class
by specifying the behavior policy and the dataset id as follows.

.. code-block:: python

    single_input_dict = prep.obtain_whole_inputs(
        logged_dataset=multiple_logged_dataset,
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy
        dataset_id=0,                                    # specify the dataset id
        evaluation_policies=evaluation_policies,
        random_state=random_state,
    )

Off-Policy Evaluation
~~~~~~~~~~
SCOPE-RL enables OPE with multiple logged datasets and multiple input dicts without additional efforts.
Specifically, we can estimate the policy value via basic OPE as follows.

.. code-block:: python

    from scope_rl.ope import OffPolicyEvaluation as OPE
    
    # initialize the OPE class
    ope = OPE(
        logged_dataset=multiple_logged_dataset,  # 
        ope_estimators=estimators,  # a list of OPE estimators
    )
    # estimate policy value and its confidence intervals
    policy_value_df_dict, policy_value_interval_df_dict = ope.summarize_off_policy_estimates(
        input_dict=multiple_input_dict,  #
        random_state=random_state,
    )

The result for each logged dataset is accessible by the following keys.

.. code-block:: python

    policy_value_df_dict[behavior_policies[0].name][dataset_id]

We can also specify the behavior policy and dataset id when calling the function as follows.

.. code-block:: python

    policy_value_df_dict, policy_value_interval_df_dict = ope.summarize_off_policy_estimates(
        input_dict=input_dict,  # either multiple or single input dict
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        dataset_id=0,  # specify the dataset id
        random_state=random_state, 
    )

Next, to compare the OPE result for some specific logged dataset, use the following function.

.. code-block:: python

    ope.visualize_off_policy_estimates(
        input_dict,  # either multiple or single input dict
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        dataset_id=0,  # specify the dataset id
        random_state=random_state, 
    )

.. card:: 
   :img-top: ../../_static/images/multiple_ope_single_policy_value.png
   :text-align: center

We can also compare results with multiple datasets as follows.

.. code-block:: python

    ope.visualize_policy_value_with_multiple_estimates(
        input_dict=multiple_input_dict,
        plot_type="ci",  # 
        hue="policy",
    )

.. card:: 
   :img-top: ../../_static/images/multiple_ope_hist_policy_value.png
   :text-align: center

.. code-block:: python

    ope.visualize_policy_value_with_multiple_estimates(
        input_dict=multiple_input_dict,
        plot_type="violin",  # 
        hue="policy",
    )

.. card:: 
   :img-top: ../../_static/images/multiple_ope_violin_policy_value.png
   :text-align: center

.. code-block:: python

    ope.visualize_policy_value_with_multiple_estimates(
        input_dict=multiple_input_dict,
        plot_type="scatter",  # 
        hue="policy",
    )

.. card:: 
   :img-top: ../../_static/images/multiple_ope_scatter_policy_value.png
   :text-align: center

Cumulative Distribution Off-Policy Evaluation
~~~~~~~~~~
CD-OPE also employs similar implementations with those of basic OPE.

.. code-block:: python

    from scope_rl.ope import CumulativeDistributionOPE
    
    # initialize the OPE class
    cd_ope = CumulativeDistributionOPE(
        logged_dataset=multiple_logged_dataset,  # 
        ope_estimators=estimators,  # a list of OPE estimators
    )
    # estimate policy value and its confidence intervals
    cdf_dict = cd_ope.estimate_cumulative_distribution_function(
        input_dict=multiple_input_dict,  #
    )

The result for each logged dataset is accessible by the following keys.

.. code-block:: python

    cdf_dict[behavior_policies[0].name][dataset_id]

We can also specify the behavior policy and dataset id when calling the function as follows.

.. code-block:: python

    cdf_dict = cd_ope.estimate_cumulative_distribution_function(
        input_dict=multiple_input_dict,  #
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        dataset_id=0,  # specify the dataset id
    )

Similar codes also work for the following functions.

* :class:`estimate_cumulative_distribution_function`
* :class:`estimate_mean`
* :class:`estimate_variance`
* :class:`estimate_conditional_value_at_risk`
* :class:`estimate_interquartile_range`

The following code compares the OPE result for some specific logged dataset.

.. code-block:: python

    cd_ope.visualize_cumulative_distribution_function(
        input_dict,  # either multiple or single input dict
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        dataset_id=0,  # specify the dataset id
    )

.. card:: 
   :img-top: ../../_static/images/multiple_ope_single_cdf.png
   :text-align: center

Similar codes also work for the following functions.

* :class:`visualize_cumulative_distribution_function`
* :class:`visualize_policy_value`
* :class:`visualize_conditional_value_at_risk`
* :class:`visualize_interquartile_range`

Next, SCOPE-RL also visualizes CDF estimated on multiple logged dataset as follows.

The first example shows the case of using a single behavior policy and multiple logged dataset.

.. code-block:: python

    cd_ope.visualize_cumulative_distribution_function_with_multiple_estimates(
        multiple_input_dict, 
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        plot_type="ci_hue",  #
        scale_min=0.0,  # set the reward scale (i.e., x-axis or bins of CDF)
        scale_max=10.0, 
        n_partition=20, 
        n_cols=4,
    )

.. card:: 
   :img-top: ../../_static/images/multiple_ope_single_behavior_cdf.png
   :text-align: center

The next examples compare the results across multiple behavior policies.

.. code-block:: python

    cd_ope.visualize_cumulative_distribution_function_with_multiple_estimates(
        multiple_input_dict, 
        plot_type="ci_behavior_policy",  #
        hue="policy",  #
        scale_min=0.0,
        scale_max=10.0, 
        n_partition=20, 
    )

.. card:: 
   :img-top: ../../_static/images/multiple_ope_all_cdf.png
   :text-align: center

The final example shows CDF for each logged dataset of a single behavior policy.

.. code-block:: python

    cd_ope.visualize_cumulative_distribution_function_with_multiple_estimates(
        multiple_input_dict, 
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        plot_type="enumerate",  #
        hue="policy",  #
        scale_min=0.0, 
        scale_max=10.0, 
        n_partition=20, 
    )

.. card:: 
   :img-top: ../../_static/images/multiple_ope_enum_cdf.png
   :text-align: center

To compare the point-wise estimation result across multiple logged datasets, the following code works.

.. code-block:: python

    ope.visualize_policy_value_with_multiple_estimates(
        multiple_input_dict,
        plot_type="ci",  # "violin", "scatter"
        hue="policy",
    )

.. card:: 
   :img-top: ../../_static/images/multiple_cdope_hist_policy_value.png
   :text-align: center

Similar codes also work for the following functions.

* :class:`visualize_policy_value_with_multiple_estimates`
* :class:`visualize_variance_with_multiple_estimates`
* :class:`visualize_conditional_value_at_risk_with_multiple_estimates`
* :class:`visualize_interquartile_range_with_multiple_estimates`

Off-Policy Selection
~~~~~~~~~~
SCOPE-RL also enables OPS with multiple logged datasets without any additional efforts.

.. code-block:: python

    from scope_rl.ope import OffPolicySelection

    # initialize the OPS class
    ops = OffPolicySelection(
        ope=ope,  # either ope or cd_ope must be given
        cumulative_distribution_ope=cd_ope,
    )
    # OPS based on estimated policy value
    ranking_df_dict, metric_df_dict = ops.select_by_policy_value(
        multiple_input_dict,
        return_metrics=True,
        return_by_dataframe=True,
    )

The result for each logged dataset is accessible by the following keys.

.. code-block:: python

    ranking_df_dict[behavior_policies[0].name][dataset_id]

The following code compares the OPE result for some specific logged dataset.

.. code-block:: python

    ranking_df, metric_df = ops.select_by_policy_value(
        input_dict=input_dict,
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        dataset_id=0,  # specify the dataset id
        return_metrics=True,
        return_by_dataframe=True,
    )

Similar codes also work for the following functions.

* :class:`select_by_policy_value`
* :class:`select_by_policy_value_lower_bound`
* :class:`select_by_policy_value_via_cumulative_distribution_ope`
* :class:`select_by_conditional_value_at_risk`
* :class:`select_by_lower_quartile`
* :class:`obtain_true_selection_result`

Assessments of OPE via top-:math:`k` Policy Selection
~~~~~~~~~~

Next, we show how to assess the top-:math:`k` policy selection with multiple logged datasets.

.. code-block:: python

    topk_metric_df_dict = ops.obtain_topk_policy_value_selected_by_standard_ope(
        input_dict=multiple_input_dict,
        return_by_dataframe=True,
    )

The result for each logged dataset is accessible by the following keys.

.. code-block:: python

    topk_metric_df_dict[behavior_policies[0].name][dataset_id]

The following code compares the top-:math:`k` policies selected by each OPE estimator for some specific logged dataset.

.. code-block:: python

    topk_metric_df = ope.obtain_topk_policy_value_selected_by_standard_ope(
        input_dict,  # either multiple or single input dict
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        dataset_id=0,  # specify the dataset id
        random_state=random_state, 
    )

Similar codes also work for the following functions.

* :class:`obtain_topk_policy_value_selected_by_standard_ope`
* :class:`obtain_topk_policy_value_selected_by_lower_bound`
* :class:`obtain_topk_policy_value_selected_by_cumulative_distribution_ope`
* :class:`obtain_topk_conditional_value_at_risk_selected_by_standard_ope`
* :class:`obtain_topk_conditional_value_at_risk_selected_by_cumulative_distirbution_ope`
* :class:`obtain_topk_lower_quartile_selected_by_standard_ope`
* :class:`obtain_topk_lower_quartile_selected_by_cumulative_distribution_ope`


Visualization functions also works in a similar manner.

.. code-block:: python

    ops.visualize_topk_policy_value_selected_by_standard_ope(
        multiple_input_dict,
        compared_estimators=["dm", "tis", "pdis", "dr"],
        visualize_ci=True,
        safety_threshold=6.0,  # please specify this option instead of `relative_safety_criteria`
        legend=True,
        random_state=random_state,
    )

.. card:: 
   :img-top: ../../_static/images/multiple_topk_policy_value.png
   :text-align: center

When using a single behavior policy, ``relative_safety_criteria`` option becomes available.

.. code-block:: python

    ops.visualize_topk_policy_value_selected_by_standard_ope(
        multiple_input_dict,
        behavior_policy_name=behavior_policies[0].name,
        compared_estimators=["dm", "tis", "pdis", "dr"],
        visualize_ci=True,
        safety_threshold=6.0,  # please specify this option instead of `relative_safety_criteria`
        legend=True,
        random_state=random_state,
    )

When using a single logged dataset, specify both behavior policy name and dataset id.

.. code-block:: python

    ops.visualize_topk_policy_value_selected_by_standard_ope(
        input_dict,  # either multiple or single input dict
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        dataset_id=0,  # specify the dataset id
        compared_estimators=["dm", "tis", "pdis", "dr"],
        visualize_ci=True,
        safety_threshold=6.0,  # please specify this option instead of `relative_safety_criteria`
        legend=True,
        random_state=random_state,
    )

Similar codes also work for the following functions.

* :class:`visualize_topk_policy_value_selected_by_standard_ope`
* :class:`visualize_topk_policy_value_selected_by_lower_bound`
* :class:`visualize_topk_policy_value_selected_by_cumulative_distribution_ope`
* :class:`visualize_topk_conditional_value_at_risk_selected_by_standard_ope`
* :class:`visualize_topk_conditional_value_at_risk_selected_by_cumulative_distirbution_ope`
* :class:`visualize_topk_lower_quartile_selected_by_standard_ope`
* :class:`visualize_topk_lower_quartile_selected_by_cumulative_distribution_ope`

Validating True and Estimated Policy Performance
~~~~~~~~~~
Finally, we also provide funnctions to compare the true and estimated policy performance.

.. code-block:: python

    ops.visualize_policy_value_for_validation(
        multiple_input_dict,
        n_cols=4,
        share_axes=True,
    )

.. card:: 
   :img-top: ../../_static/images/multiple_validation_policy_value.png
   :text-align: center

When using a single behavior policy, specify behavipr policy name.

.. code-block:: python

    ops.visualize_policy_value_for_validation(
        input_dict,  # either multiple or single input dict
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        n_cols=4,
        share_axes=True,
    )

When using a single logged dataset, specify both behavior policy name and dataset id.

.. code-block:: python

    ops.visualize_policy_value_for_validation(
        input_dict,  # either multiple or single input dict
        behavior_policy_name=behavior_policies[0].name,  # specify the behavior policy name
        dataset_id=0,  # specify the dataset id
        n_cols=4,
        share_axes=True,
    )

Similar codes also work for the following functions.

* :class:`visualize_policy_value_for_validation`
* :class:`visualize_policy_value_lower_bound_for_validation`
* :class:`visualize_variance_for_validation`
* :class:`visualize_conditional_value_at_risk_for_validation`
* :class:`visualize_lower_bound_for_validation`

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
                :link: /documentation/examples/real_world
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Real_World Datasets**

