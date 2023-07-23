Example Codes for Cumulative Distribution OPE
==========

Here, we show example codes for conducting cumulative distribution OPE (CD-OPE).

.. seealso::

    For preparation, please also refer to the following pages:

    * :doc:`What is Cumulative Distribution OPE? </documentation/ope_ops>`
    * :ref:`Supported CD-OPE estimators <implementation_cumulative_distribution_ope>`
    * :doc:`Supported implementations for data collection and Offline RL </documentation/learning_implementation>`
    * :doc:`Example codes for basic OPE </documentation/examples/basic_ope>`

Logged Dataset
~~~~~~~~~~

Here, we assume that an RL environment, a behavior policy, and evaluation policies are given as follows.

* ``behavior_policy``: an instance of :class:`BaseHead`
* ``evaluation_policies``: a list of instance(s) of :class:`BaseHead`
* ``env``: a gym environment (unnecessary when using real-world datasets)

Then, we use the behavior policy to collect logged dataset as follows.

.. code-block:: python

    from scope_rl.dataset import SyntheticDataset
    
    # initialize dataset
    dataset = SyntheticDataset(
        env=env,
        max_episode_steps=env.step_per_episode,
    )
    # obtain logged dataset
    logged_dataset = dataset.obtain_episodes(
        behavior_policies=behavior_policy,
        n_trajectories=10000, 
        obtain_info=False,  # whether to record `info` returned by environment (optional)
        random_state=random_state,
    )

Note that, in the following example, we use a single logged dataset for simplicity.
For the case of using multiple behavior policies or multiple logged datasets, refer to :doc:`/documentation/examples/multiple`.

Inputs
~~~~~~~~~~
The next step is to create the inputs for OPE estimators. This procedure is also very similar to that of basic OPE.

OPE with importance sampling-based estimators
----------
When using the importance sampling-based estimators including TIS and SNTIS, 
and hybrid estimators including DR and SNDR, make sure that "pscore" is recorded in the logged dataset.

Then, when using only importance sampling-based estimators, the minimal sufficient codes are the following:

.. code-block:: python

    from scope_rl.ope import CreateOPEInput

    # initialize class to create inputs
    prep = CreateOPEInput(
        env=env,  # unnecessary when using real-world dataset
    )
    # create inputs (e.g., calculating )
    input_dict = prep.obtain_whole_inputs(
        logged_dataset=logged_dataset,
        evaluation_policies=evaluation_policies,
        n_trajectories_on_policy_evaluation=100,  # when evaluating OPE (optional)
        random_state=random_state,
    )

OPE with model-based estimators
----------
When using the model-based estimator (DM) or hybrid methods, we need to additionally obtain value estimation in the input dict.

.. code-block:: python

    # initialize class to create inputs
    prep = CreateOPEInput(
        env=env,
        model_args={  # you can specify the model here (optional)
            "fqe": {
                "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
                "q_func_factory": MeanQFunctionFactory(),
                "learning_rate": 1e-4,
                "use_gpu": torch.cuda.is_available(),
            },
        },
    )
    # create inputs (e.g., calculating )
    input_dict = prep.obtain_whole_inputs(
        logged_dataset=logged_dataset,
        evaluation_policies=evaluation_policies,
        require_value_prediction=True,  # enable this option
        q_function_method="fqe",  # you can specify algorithms here (optional)
        v_function_method="fqe",
        n_trajectories_on_policy_evaluation=100,
        random_state=random_state,
    )

Note that, we can also apply scaling to either state observation or (continuous) action as follows.

.. code-block:: python

    from scope_rl.utils import MinMaxScaler

    prep = CreateOPEInput(
        env=env,
        state_scaler=MinMaxScaler(  #
            minimum=logged_dataset["state"].min(axis=0),
            maximum=logged_dataset["state"].max(axis=0),
        ),
        action_scaler=MinMaxActionScaler(  #
            minimum=env.action_space.low,
            maximum=env.action_space.high,
        ),
        sigma=0.1,  # additional bandwidth hyperparameter (for dice method)
    )

Off-Policy Evaluation
~~~~~~~~~~
After preparing the inputs, SCOPE-RL is capable of handling CD-OPE, again in a manner similar to that of basic OPE.

Here, we use the following OPE estimators. 

.. code-block:: python

    from scope_rl.ope.discrete import CumulativeDistributionDM as CD_DM
    from scope_rl.ope.discrete import CumulativeDistributionTIS as CD_TIS
    from scope_rl.ope.discrete import CumulativeDistributionTDR as CD_TDR
    from scope_rl.ope.discrete import CumulativeDistributionSNTIS as CD_SNTIS
    from scope_rl.ope.discrete import CumulativeDistributionSNTDR as CD_SNTDR

    estimators = [CD_DM(), CD_TIS(), CD_TDR(), CD_SNTIS(), CD_SNTDR()]

Estimating Cumulative Distribution Function (CDF)
----------

The CDF curve is easily estimated as follows.

.. code-block:: python

    from scope_rl.ope import CumulativeDistributionOPE

    # initialize the CD-OPE class
    cd_ope = CumulativeDistributionOPE(
        logged_dataset=logged_dataset,
        ope_estimators=estimators,
    )
    # estimate CDF
    cdf_dict = cd_ope.estimate_cumulative_distribution_function(
        input_dict=input_dict,
    )

The following code visualizes the results to compare OPE estimators.

.. code-block:: python

    cd_ope.visualize_cumulative_distribution_function(
        input_dict=input_dict,
        hue="estimator",  # (default)
        n_cols=4,  # specify the number of columns (optional)
    )

.. card:: 
   :img-top: ../../_static/images/cd_ope_cdf_hue_estimator.png
   :text-align: center

The following code visualizes the results to compare policies.

.. code-block:: python

    cd_ope.visualize_cumulative_distribution_function(
        input_dict=input_dict,
        hue="policy",  # (optional)
        legend=False,
        n_cols=4,
    )

.. card:: 
   :img-top: ../../_static/images/cd_ope_cdf_hue_policy.png
   :text-align: center

Users can also specify the compared OPE estimators as follows.

.. code-block:: python

    cd_ope.visualize_cumulative_distribution_function(
        input_dict=input_dict,
        compared_estimators=["cd_dm", "cd_tis", "cd_tdr"],  # names are assessible by `evaluation_policy.name`
    )

Note that, the x-axis (bins) of CDF is by default set to the reward observed by the behavior policy.
To use the custom bins, specify the reward scale when initializing the class.

.. code-block:: python

    cd_ope = CumulativeDistributionOPE(
        logged_dataset=logged_dataset,
        ope_estimators=estimators,
        use_custom_reward_scale=True,  # setting bins for cdf
        scale_min=0.0,
        scale_max=10.0,
        n_partition=20,
    )

Estimating Mean (i.e., policy value)
----------
Similarly, we can estimate the policy value via CD-OPE as follows.

.. code-block:: python

    policy_value_dict = cd_ope.estimate_mean(
        input_dict=input_dict,
        compared_estimators=["cd_dm", "cd_tis", "cd_tdr"],  # (optional)
    )

The visualization function also has simular arguments.

.. code-block:: python

    cd_ope.visualize_policy_value(
        input_dict=input_dict,
        hue="estimator",  # (default)
    )

.. card:: 
   :img-top: ../../_static/images/cd_ope_mean_hue_estimator.png
   :text-align: center

For the policy value estimate, we additionally provide ``is_relative`` option to visualize the policy value that is relative to that of behavior policy.

.. code-block:: python

    cd_ope.visualize_policy_value(
        input_dict=input_dict,
        hue="policy",  # (optional)
        is_relative=True,  # enable this option
    )

.. card:: 
   :img-top: ../../_static/images/cd_ope_mean_hue_policy.png
   :text-align: center

Note that, the visualization function of policy value accompanies with the visualization of the variance, which we discuss in the following.

Estimating Variance
----------
CD-OPE is able to esitmate the variance of the trajectory-wise reward as follows.

.. code-block:: python

    variance_dict = cd_ope.estimate_variance(
        input_dict=input_dict,
    )

SCOPE-RL shares the visualization function for variance with that of policy value. 
Specifically, the confidence intervals of the trajectory-wise reward is estimated via the variance estimate, assuming that the trajectory-wise reward follows normal distribution.

.. code-block:: python

    cd_ope.visualize_policy_value(
        input_dict=input_dict,
    )

Estimating Conditional Value at Risk (CVaR)
----------
Next, SCOPE-RL also estimates CVaR in a similar manner.

.. code-block:: python

    cvar_dict = cd_ope.estimate_conditional_value_at_risk(
        input_dict=input_dict,
        alpha=0.3,  # specify the proportion of the sided region
    )

We can also get the value of CVaR for multiple values of alpha as follows.

.. code-block:: python

    cvar_dict = cd_ope.estimate_conditional_value_at_risk(
        input_dict=input_dict,
        alpha=np.array([0.1, 0.3]),  # specify the proportions of the sided region
    )

The visualization function depicts CVaR across range of alphas as follows.

.. code-block:: python

    cd_ope.visualize_conditional_value_at_risk(
        input_dict=input_dict,
        alphas=np.linspace(0, 1, 21),  # (default)
        n_cols=4,  # (optional)
    )

.. card:: 
   :img-top: ../../_static/images/cd_ope_cvar.png
   :text-align: center

Estimating Interquartile Range
----------
Finally, SCOPE-RL estimates and visualizes the Interquartile range as follows.

.. code-block:: python

    # estimate the interquartile range
    interquartile_range_dict = cd_ope.estimate_interquartile_range(
        input_dict=input_dict,
        alpha=0.3,  # specify the proportion of the sided region
    )
    # visualize the interquartile range
    cd_ope.visualize_interquartile_range(
        input_dict=input_dict,
        alpha=0.3,  # specify the proportion of the sided region
    )

.. card:: 
   :img-top: ../../_static/images/cd_ope_interquartile_range.png
   :text-align: center

.. seealso::

    For the evaluation of CD-OPE estimators, please also refer to :doc:`/documentation/examples/assessments`.

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
                :link: /documentation/examples/ops
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Off_policy Selection**

            .. grid-item-card::
                :link: /documentation/examples/assessments
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Assessments**
