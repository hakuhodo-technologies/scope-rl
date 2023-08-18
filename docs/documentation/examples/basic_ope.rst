Example Codes for Basic Off-Policy Evaluation
==========

Here, we show example codes for conducting basic off-policy evaluation (OPE).

.. seealso::

    For preparation, please also refer to the following pages:

    * :doc:`What is Off-Policy Evaluation? </documentation/ope_ops>`
    * :ref:`Supported OPE estimators <implementation_basic_ope>`
    * :doc:`Supported implementations for data collection and Offline RL </documentation/learning_implementation>`

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
    # obtain a logged dataset
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
The next step is to create the inputs for OPE estimators. This procedure slightly differs depending on which OPE estimators to use.

OPE with importance sampling-based estimators
----------
When using the importance sampling-based estimators including TIS, PDIS, SNTIS, and SNPDIS, 
and hybrid estimators including DR and SNDR, make sure that "pscore" (i.e., action choice probability of the behavior policy) is recorded in the logged dataset.

Then, when using only importance sampling-based estimators, the minimal sufficient codes are the following:

.. code-block:: python

    from scope_rl.ope import CreateOPEInput

    # initialize class to create inputs
    prep = CreateOPEInput(
        env=env,  # unnecessary when using real-world dataset
    )
    # create inputs
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
            },
        },
    )
    # create inputs
    input_dict = prep.obtain_whole_inputs(
        logged_dataset=logged_dataset,
        evaluation_policies=evaluation_policies,
        require_value_prediction=True,  # enable this option
        q_function_method="fqe",  # you can specify algorithms here (optional)
        v_function_method="fqe",
        n_trajectories_on_policy_evaluation=100,
        random_state=random_state,
    )


OPE with marginal importance sampling-based estimators
----------
Marginal importance sampling-based estimators (e.g., SAMIS, SAMDR, ..) require the estimation of marginal importance weights.

.. code-block:: python

    # initialize class to create inputs
    prep = CreateOPEInput(
        env=env,
        model_args={  # you can specify the model here (optional)
            "dice": {
                "method": "best_dice",
                "q_lr": 1e-4,
                "w_lr": 1e-4,
            },
        },
    )
    # create inputs
    input_dict = prep.obtain_whole_inputs(
        logged_dataset=logged_dataset,
        evaluation_policies=evaluation_policies,
        require_weight_prediction=True,  # enable this option
        w_function_method="dice",  # you can specify algorithms here (optional)
        n_trajectories_on_policy_evaluation=100,
        random_state=random_state,
    )

OPE with Double Reinforcement Learning
----------
Double Reinforcement Learning learns weight and value functions through the cross-fitting procedure :cite:`kallus2020double`. 
This is done by setting the ``k_hold`` parameter as follows.

.. code-block:: python

    input_dict = prep.obtain_whole_inputs(
        logged_dataset=logged_dataset,
        evaluation_policies=evaluation_policies,
        require_value_prediction=True,
        require_weight_prediction=True,
        k_fold=3,  # k > 1 corresponds to cross-fitting
        n_trajectories_on_policy_evaluation=100,
        random_state=random_state,
    )

Scalers for value and weight learning
----------
We can also apply scaling to either state observation or (continuous) action as follows.

.. code-block:: python

    from d3rlpy.preprocessing import MinMaxObservationScaler, MinMaxActionScaler

    prep = CreateOPEInput(
        env=env,
        state_scaler=MinMaxObservationScaler(  #
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
After preparing the inputs, it is time to conduct OPE. 

Here, we use the following OPE estimators. 

.. code-block:: python

    from scope_rl.ope.discrete import DirectMethod as DM
    from scope_rl.ope.discrete import SelfNormalizedPDIS as SNPDIS
    from scope_rl.ope.discrete import SelfNormalizedDR as SNDR

    from scope_rl.ope.discrete import StateMarginalSNIS as SMSNIS
    from scope_rl.ope.discrete import StateMarginalSNDR as SMSNDR

    from scope_rl.ope.discrete import StateActionMarginalSNIS as SAMSNIS
    from scope_rl.ope.discrete import StateActionMarginalSNDR as SAMSNDR

    from scope_rl.ope.discrete import DoubleReinforcementLearning as DRL

    estimators = [DM(), SNPDIS(), SNDR(), SMSNIS(), SMSNDR(), DRL()]


Note that, the following provides the complete list of estimators that are currently implemented in SCOPE-RL.

.. dropdown:: Supported OPE estimators

    (Standard choices)

    * :class:`DirectMethod` (DM)
    * :class:`TrajectoryWiseImportanceSampling` (TIS)
    * :class:`PerDecisionImportanceSampling` (PDIS)
    * :class:`DoublyRobust` (DR)

    * :class:`SelfNormalizedTIS` (SNTIS)
    * :class:`SelfNormalizedPDIS` (SNPDIS)
    * :class:`SelfNormalizedDR` (SNDR)

    (Marginal estimators)

    * :class:`StateMarginalIS` (SMIS)
    * :class:`StateMarginalDR` (SMDR)
    * :class:`StateMarginalSNIS` (SMSNIS)
    * :class:`StateMarginalDR` (SMDR)
    
    * :class:`StateActionMarginalIS` (SAMIS)
    * :class:`StateActionMarginalDR` (SAMDR)
    * :class:`StateActionMarginalSNIS` (SAMSNIS)
    * :class:`StateActionMarginalSNDR` (SAMSNDR)

    * :class:`DoubleReinforcementLearning` (DRL)

    .. seealso::

        * :doc:`Supported OPE estimators </documentation/evaluation_implementation>` summarizes the key properties of each estimator.


We can easily conduct OPE and obtain the results as follows.

.. code-block:: python

    from scope_rl.ope import OffPolicyEvaluation as OPE
    
    # initialize the OPE class
    ope = OPE(
        logged_dataset=logged_dataset,
        ope_estimators=estimators,
    )
    # estimate policy value and its confidence intervals
    policy_value_df_dict, policy_value_interval_df_dict = ope.summarize_off_policy_estimates(
        input_dict=input_dict, 
        random_state=random_state,
    )

SCOPE-RL also offers an easy-to-use visualization function. The following code visualizes the results to compare OPE estimators.

.. code-block:: python

    ope.visualize_off_policy_estimates(
        input_dict, 
        hue="estimator",  # (default)
        random_state=random_state, 
    )

.. card:: 
   :img-top: ../../_static/images/basic_ope_hue_estimator.png
   :text-align: center

The following code visualizes the results to compare candidate (evaluation) policies.

.. code-block:: python

    ope.visualize_off_policy_estimates(
        input_dict, 
        hue="policy",  #
        random_state=random_state, 
    )

.. card:: 
   :img-top: ../../_static/images/basic_ope_hue_policy.png
   :text-align: center

It is also possible to visualize the policy value that is relative to the behavior policy.

.. code-block:: python

    ope.visualize_off_policy_estimates(
        input_dict, 
        hue="policy",
        is_relative=True,  # enable this option
        random_state=random_state, 
    )

.. card:: 
   :img-top: ../../_static/images/basic_ope_is_relative.png
   :text-align: center

Users can also specify the compared OPE estimators as follows.

.. code-block:: python

    ope.visualize_off_policy_estimates(
        input_dict, 
        compared_estimators=["dm", "snpdis", "sndr"],  # names are accessible by `evaluation_policy.name`
        random_state=random_state, 
    )

When ``legend`` is unnecessary, just disable this option.

.. code-block:: python

    ope.visualize_off_policy_estimates(
        input_dict, 
        legend=False,  #
        random_state=random_state, 
    )

To save the figure, specify the directory to save it.

.. code-block:: python

    ope.visualize_off_policy_estimates(
        input_dict, 
        fig_dir="figs/",  # specify the directory
        fig_name="estimated_policy_value.png",  # (default)
        random_state=random_state, 
    )

Choosing the "Spectrum" of OPE for marginal estimators
----------
The implemented OPE estimators can interpolate among naive importance sampling and
marginal importance sampling by specifying the steps to use per-decision importance weight 
(See :ref:`Supported OPE estimators (SOPE) <implementation_sope>` for the details). 
This is done by specifying ``n_step_pdis`` when initializing the class.

.. code-block:: python

    ope = OPE(
        logged_dataset=logged_dataset,
        ope_estimators=estimators,
        n_step_pdis=5,
    )


Choosing a kernel for continuous-action OPE
----------
In continuous-action OPE, the choices of the kernel and the bandwidth hyperparameter can affect the bias-variance tradeoff and the estimation accuracy.
To control the hyperparameter, please use the following arguments.

.. code-block:: python

    policy_value_df_dict, policy_value_interval_df_dict = ope.summarize_off_policy_estimates(
        input_dict=input_dict, 
        action_scaler=MinMaxActionScaler(  # apply scaling of action at each dimension
            minimum=env.action_space.low,
            maximum=env.action_space.high,
        ),
        sigma=0.1,  # bandwidth hyperparameter of the kernel
        random_state=random_state,
    )

Choosing a probability bound for high confidence OPE
----------
Similarly, SCOPE-RL allows to choose the significant level and the inequality to derive a probability bound as follows.

.. code-block:: python

    policy_value_df_dict, policy_value_interval_df_dict = ope.summarize_off_policy_estimates(
        input_dict=input_dict, 
        ci="bootstrap",  # specify inequality (optional)
        alpha=0.05,  # significant level (optional)
        random_state=random_state,
    )

Evaluating the "accuracy" of OPE
----------
Finally, OPE class also provides a function to calculate the estimation accuracy of OPE.

.. code-block:: python

    eval_metric_ope_df = ope.evaluate_performance_of_ope_estimators(
        input_dict, 
        metric="se",  # or "relative-ee"
    )

.. seealso::

    For other metrics to assess OPE results, please also refer to :doc:`/documentation/examples/assessments`.

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
                :link: /documentation/examples/cumulative_dist_ope
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **CD_OPE**

            .. grid-item-card::
                :link: /documentation/examples/assessments
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Assessments**
