==========
Supported Implementation
==========

.. _implementation_create_ope_input:

Create OPE Input
~~~~~~~~~~
Before proceeding to OPE/OPS, we first create :class:`input_dict` to enable a smooth implementation.

.. code-block:: python

            # create input for OPE class
            from scope_rl.ope import CreateOPEInput
            prep = CreateOPEInput(
                env=env,
            )
            input_dict = prep.obtain_whole_inputs(
                logged_dataset=logged_dataset,
                evaluation_policies=evaluation_policies,
                require_value_prediction=True,  # use model-based prediction
                n_trajectories_on_policy_evaluation=100,
                random_state=random_state,
            )

.. _tip_create_input_dict:

.. tip:: 
    
    .. dropdown:: How to create input_dict for multiple logged datasets?

        When obtaining :class:`input_dict` from the same evaluation policies across multiple datasets, try the following command.

        .. code-block:: python

            multiple_input_dict = prep.obtain_whole_inputs(
                logged_dataset=logged_dataset,            # MultipleLoggedDataset
                evaluation_policies=evaluation_policies,  # single list
                ...,
            )

        When obtaining :class:`input_dict` from different evaluation policies for each logged dataset, try the following command.

        .. code-block:: python

            multiple_input_dict = prep.obtain_whole_inputs(
                logged_dataset=logged_dataset,                                 # MultipleLoggedDataset (two logged dataset in this case)
                evaluation_policies=evaluation_policies,                       # nested list or dict that have the same keys with logged_datasets
                ...,
            )

        In both cases, :class:`MultipleInputDict` will be returned.

        :class:`MultipleInputDict` saves the paths to each input_dict and make it accessible through the following command.
            
        .. code-block:: python

            input_dict_ = multiple_input_dict.get(behavior_policy_name=behavior_policy.name, dataset_id=0)

        .. seealso::

            * :ref:`How to obtain MultipleLoggedDataset? <tips_synthetic_dataset>`
            * :ref:`How to handle OPL with MultipleLoggedDataset? <tip_opl>`
            * :doc:`API reference of MultipleInputDict <_autosummary/scope_rl.utils.MultipleInputDict>`
            .. * :ref:`Tutorial with MultipleLoggedDataset <scope_rl_multiple_tutorial>`

    .. dropdown:: How to select models for value/weight learning methods?

        To enable value prediction (for model-based estimators) and weight prediction (for marginal estimators), set ``True`` for the following arguments.

        .. code-block:: python

            input_dict = prep.obtain_whole_inputs(
                ...,
                require_value_prediction=True, 
                require_weight_prediction=True, 
                ...,
            )

        Then, we can customize the choice of weight and value functions using the following arguments.

        .. code-block:: python

            input_dict = prep.obtain_whole_inputs(
                ...,
                q_function_method="fqe",   # one of {"fqe", "dice", "mql"}, default="fqe"
                v_function_method="fqe",   # one of {"fqe", "dice_q", "dice_v", "mql", "mvl"}, default="fqe"
                w_function_method="dice",  # one of {"dice", "mwl"}, default="dice"
                ...,
            )

        To further customize the models, please specify ``model_args`` when initializing :class:`CreateOPEInput` as follows.

        .. code-block:: python

            from d3rlpy.models.encoders import VectorEncoderFactory
            from d3rlpy.models.q_functions import MeanQFunctionFactory

            prep = CreateOPEInput(
                env=env,
                model_args={
                    "fqe": {
                        "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
                        "q_func_factory": MeanQFunctionFactory(),
                        "learning_rate": 1e-4,
                    },
                    "state_action_dual" : {  # "dice"
                        "method": "dual_dice",
                    },
                    "state_action_value": {  # "mql"
                        "batch_size": 64,
                        "lr": 1e-4,
                    },
                }
            )

        where the keys of ``model_args`` are the following.

        .. code-block:: python

            key: [
                "fqe",                  # fqe
                "state_action_dual",    # dice_q
                "state_action_value",   # mql
                "state_action_weight",  # mwl
                "state_dual",           # dice_v
                "state_value",          # mvl
                "state_weight",         # mwl
                "hidden_dim",           # hidden dim of value/weight function, except FQE
            ]

        .. seealso::

            * :doc:`API reference of CreateInputDict <_autosummary/scope_rl.ope.input>`
            * :ref:`API reference of value/weight learning methods <scope_rl_api_ope_weight_and_value_learning>`
            * :ref:`Logics behind value and weight learning methods (How to obtain state(-action) marginal importance weight?) <tip_mariginal_iw>`

    .. dropdown:: How to collect input_dict in a non-episodic setting?

        When the goal is to evaluate the policy under a stationary distribution (:math:`d^{\pi}(s)`) rather than in a episodic setting 
        (i.e., cartpole or taxi used in :cite:`liu2018breaking` :cite:`uehara2020minimax`), we need to (re-)collect initial states from evaluation policies stationary distribution.

        In this case, please turn the following options.

        .. code-block:: python

            input_dict = prep.obtain_whole_inputs(
                ...,
                resample_initial_state=True,
                use_stationary_distribution_on_policy_evaluation=True,  # when env is provided
                ...,
            )

.. seealso::
        
    :doc:`Supported Implementation (learning) <learning_implementation>` describes how to obtain :class:`logged_dataset` using a behavior policy in detail.

.. _implementation_basic_ope:

Basic Off-Policy Evaluation (OPE)
~~~~~~~~~~
The goal of (basic) OPE is to evaluate the following expected trajectory-wise reward of a policy (referred to as policy value).

.. math::

    J(\pi) := \mathbb{E}_{\tau} \left [ \sum_{t=0}^{T-1} \gamma^t r_{t} \mid \pi \right ],

where :math:`\pi` is the (evaluation) policy, :math:`\tau` is the trajectory observed by the evaluation policy, and :math:`r_t` is the immediate reward at each timestep. 
(Please refer to the :doc:`problem setup <ope_ops>` for additional notations.)


Here, we describe the class for conducting OPE and the implemented OPE estimators for estimating the policy value. 
We begin with the :class:`OffPolicyEvaluation` class to streamline the OPE procedure.

.. code-block:: python

    # initialize the OPE class
    from scope_rl.ope import OffPolicyEvaluation as OPE
    ope = OPE(
        logged_dataset=logged_dataset,
        ope_estimators=[DM(), TIS(), PDIS(), DR()],
    )

Using the OPE class, we can obtain the OPE results of various estimators at once as follows.

.. code-block:: python

    ope_dict = ope.estimate_policy_value(input_dict)

.. _tip_ope:

.. tip::

    .. dropdown:: How to conduct OPE with multiple logged datasets?

        Conducting OPE with multiple logged datasets requires no additional efforts.

        First, the same command with the single logged dataset case also works with multiple logged datasets.

        .. code-block:: python

            ope = OPE(
                logged_dataset=logged_dataset,  # MultipleLoggedDataset
                ope_estimators=[DM(), TIS(), PDIS(), DR()],
            )
            multiple_ope_dict = ope.estimate_policy_value(
                input_dict,  # MultipleInputDict
            )

        The returned value is dictionary containing the ope result.

        In addition, we can specify which logged dataset and input_dict to use by setting ``behavior_policy_name`` and ``dataset_id``.

        .. code-block:: python

            multiple_ope_dict = ope.estimate_policy_value(
                input_dict,
                behavior_policy_name=behavior_policy.name,  #
                dataset_id=0,  # specify which logged dataset and input_dict to use
            )

        The basic visualization function also work by specifying the dataset id.

        .. code-block:: python

            ope.visualize_off_policy_estimates(
                input_dict,
                behavior_policy_name=behavior_policy.name,
                dataset_id=0,  #
                ...,
            )

        .. card:: 
            :img-top: ../_static/images/ope_policy_value_basic.png
            :text-align: center
            
            policy value estimated with the specified dataset

        Moreover, we provide additional visualization function for the multiple logged dataset case.

        .. code-block:: python

            ope.visualize_policy_value_with_multiple_estimates(
                input_dict,      # MultipleInputDict
                behavior_policy_name=None,                   # compare estimators with multiple behavior policies
                # behavior_policy_name=behavior_policy.name  # compare estimators with a single behavior policy
                plot_type="ci",  # one of {"ci", "violin", "scatter"}, default="ci"
                ...,
            )

        When the ``plot_type`` is "ci", the plot is somewhat similar to the basic visualization. 
        (The star indicates the ground-truth policy value and the confidence intervals are derived by multiple estimates across datasets.)

        .. card:: 
            :img-top: ../_static/images/ope_policy_value_basic_multiple.png
            :text-align: center
            
            policy value estimated with the multiple datasets

        When the ``plot_type`` is "violin", the plot visualizes the distribution of multiple estimates.
        This is particularly useful to see how the estimation result can vary depending on different datasets or random seeds. 

        .. card:: 
            :img-top: ../_static/images/ope_policy_value_basic_multiple_violin.png
            :text-align: center
            
            policy value estimated with the multiple datasets (violin)

        Finally, when the ``plot_type`` is "scatter", the plot visualizes each estimation with its color specifying the dataset id.
        This function is particularly useful to see how the choice of behavior policy (e.g., their stochasticity) affects the estimation result.

        .. card:: 
            :img-top: ../_static/images/ope_policy_value_basic_multiple_scatter.png
            :text-align: center
            
            policy value estimated with the multiple datasets (scatter)

        .. seealso::

            * :ref:`How to obtain MultipleLoggedDataset? <tips_synthetic_dataset>`
            * :ref:`How to handle OPL with MultipleLoggedDataset? <tip_opl>`
            * :ref:`How to create input_dict for MultipleLoggedDataset? <tip_create_input_dict>`
            .. * :ref:`Tutorial with MultipleLoggedDataset <scope_rl_multiple_tutorial>`
        

.. seealso::

    * :doc:`quickstart` 
    .. * and :ref:`related tutorials <basic_ope_tutorial>`


The OPE class implements the following functions.

(OPE)

* :class:`estimate_policy_value`
* :class:`estimate_intervals`
* :class:`summarize_off_policy_estimates`

(Evaluation of OPE estimators)

* :class:`evaluate_performance_of_ope_estimators`

(Visualization)

* :class:`visualize_off_policy_estimates`

(Visualization with multiple estimates on multiple logged datasets)

* :class:`visualize_policy_value_with_multiple_estimates`

Below, we describe the implemented OPE estimators.

==================================================================================  ================  ================
Standard OPE estimators                                                                    
==================================================================================  ================  ================
:ref:`Direct Method (DM) <implementation_dm>`                                                                    
:ref:`Trajectory-wise Importance Sampling (TIS) <implementation_tis>`             
:ref:`Per-Decision Importance Sampling (PDIS) <implementation_pdis>`              
:ref:`Doubly Robust (DR) <implementation_dr>`                                    
:ref:`Self-Normalized estimators <implementation_sn>`    
==================================================================================  ================  ================


==================================================================================  ================  ================
Marginal OPE estimators                                                                    
==================================================================================  ================  ================
:ref:`State Marginal estimators <implementation_marginal_ope>`                    
:ref:`State-Action Marginal estimators <implementation_marginal_ope>`             
:ref:`Double Reinforcement Learning <implementation_drl>`                         
:ref:`Spectrum of Off-Policy Evaluation <implementation_sope>`     
==================================================================================  ================  ================


==================================================================================  ================  ================
Extensions         
==================================================================================  ================  ================
:ref:`High Confidence Off-Policy Evaluation <implementation_high_confidence_ope>` 
:ref:`Extension to the continuous action space <implementation_continuous_ope>`   
==================================================================================  ================  ================

.. tip::

    .. dropdown:: How to define my own OPE estimator?

        To define your own OPE estimator, use :class:`BaseOffPolicyEstimator`.

        Basically, the common inputs for each functions are the following keys from ``logged_dataset`` and ``input_dict``.

        (logged_dataset)

        .. code-block:: python

            key: [
                size,
                step_per_trajectory,
                action,
                reward,
                pscore,
            ]

        (input_dict)

        .. code-block:: python

            key: [
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                state_action_marginal_importance_weight,
                state_marginal_importance_weight,
                on_policy_policy_value,
                gamma,
            ]
        
        ``n_step_pdis`` is also applicable to marginal estimators and ``action_scaler`` and ``sigma`` are added in the continuous-action case.

        If you want to add other arguments, please add them in the initialization arguments for API consistency.

        Finally, contribution to SCOPE-RL with a new OPE estimator is more than welcome! Please read `the guidelines for contribution (CONTRIBUTING.md) <https://github.com/hakuhodo-technologies/scope-rl/blob/main/CONTRIBUTING.md>`_.

        .. seealso::

            :doc:`API reference of BaseOffPolicyEstimator <_autosummary/scope_rl.ope.estimators_base>` explains the abstract methods.

.. _implementation_dm:

Direct Method (DM)
----------
DM :cite:`beygelzimer2009offset` is a model-based approach which uses the initial state value (estimated by e.g., Fitted Q Evaluation (FQE) :cite:`le2019batch`).
It first learns the Q-function and then leverages the learned Q-function as follows.

.. math::

    \hat{J}_{\mathrm{DM}} (\pi; \mathcal{D}) := \mathbb{E}_n [ \mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} [\hat{Q}(s_0, a_0)] ] = \mathbb{E}_n [\hat{V}(s_0)],

where :math:`\mathcal{D}=\{\{(s_t, a_t, r_t)\}_{t=0}^T\}_{i=1}^n` is the logged dataset with :math:`n` trajectories of data.
:math:`T` indicates step per episode. :math:`\hat{Q}(s_t, a_t)` is the estimated state-action value and :math:`\hat{V}(s_t)` is the estimated state value.

DM has low variance, but can incur bias due to approximation errors.

    * :class:`DiscreteDirectMethod`
    * :class:`ContinuousDirectMethod`

.. note::

    We use the implementation of FQE provided by `d3rlpy <https://github.com/takuseno/d3rlpy>`_.

.. _implementation_tis:

Trajectory-wise Importance Sampling (TIS)
----------

TIS :cite:`precup2000eligibility` uses importance sampling technique to correct the distribution shift between :math:`\pi` and :math:`\pi_0` as follows.

.. math::

    \hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D}) := \mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t w_{1:T-1} r_t \right],

where :math:`w_{0:T-1} := \prod_{t=0}^{T-1} (\pi(a_t | s_t) / \pi_0(a_t | s_t))` is the trajectory-wise importance weight.

TIS enables an unbiased estimation of the policy value. However, when the trajectory length :math:`T` is large, TIS suffers from high variance
due to the product of importance weights.

    * :class:`DiscreteTrajectoryWiseImportanceSampling`
    * :class:`ContinuousTrajectoryWiseImportanceSampling`

.. _implementation_pdis:

Per-Decision Importance Sampling (PDIS)
----------
PDIS :cite:`precup2000eligibility` leverages the sequential nature of the MDP to reduce the variance of TIS.
Specifically, since :math:`s_t` only depends on :math:`s_0, \ldots, s_{t-1}` and :math:`a_0, \ldots, a_{t-1}` and is independent of :math:`s_{t+1}, \ldots, s_{T}` and :math:`a_{t+1}, \ldots, a_{T}`,
PDIS only considers the importance weight of the past interactions when estimating :math:`r_t` as follows.

.. math::

    \hat{J}_{\mathrm{PDIS}} (\pi; \mathcal{D}) := \mathbb{E}_{n} \left[ \sum_{t=0}^{T-1} \gamma^t w_{0:t} r_t \right],

where :math:`w_{0:t} := \prod_{t'=0}^t (\pi_e(a_{t'} | s_{t'}) / \pi_b(a_{t'} | s_{t'}))` is the importance weight of past interactions.

PDIS remains unbiased while reducing the variance of TIS. However, when :math:`t` is large, PDIS still suffers from high variance.

    * :class:`DiscretePerDecisionImportanceSampling`
    * :class:`ContinuousPerDecisionWiseImportanceSampling`

.. _implementation_dr:

Doubly Robust (DR)
----------
DR :cite:`jiang2016doubly` :cite:`thomas2016data` is a hybrid of model-based estimation and importance sampling.
It introduces :math:`\hat{Q}` as a baseline estimation in the recursive form of PDIS and applies importance weighting only on its residual.

.. math::

    \hat{J}_{\mathrm{DR}} (\pi; \mathcal{D})
    := \mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t (w_{0:t} (r_t - \hat{Q}(s_t, a_t)) + w_{0:t-1} \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_t, a)])\right],

DR is unbiased and reduces the variance of PDIS when :math:`\hat{Q}(\cdot)` is reasonably accurate to satisfy :math:`0 < \hat{Q}(\cdot) < 2 Q(\cdot)`. 
However, when the importance weight is quite large, it may still suffer from a high variance.

    * :class:`DiscreteDoublyRobust`
    * :class:`ContinuousDoublyRobust`

.. _implementation_sn:

Self-Normalized estimators
----------
Self-normalized estimators :cite:`kallus2019intrinsically` aim to reduce the scale of importance weight for the variance reduction purpose.
Specifically, it substitute importance weight :math:`w_{\ast}` as follows.

.. math::

    \tilde{w}_{\ast} := w_{\ast} / \mathbb{E}_{n}[w_{\ast}]

where :math:`\tilde{w}_{\ast}` is the self-normalized importance weight.

Self-normalized estimators are no longer unbiased, but has variance bounded by :math:`r_{max}^2` while also being consistent.

(Discrete)

    * :class:`DiscreteSelfNormalizedTrajectoryWiseImportanceSampling`
    * :class:`DiscreteSelfNormalizedPerDecisionImportanceSampling`
    * :class:`DiscreteSelfNormalizedDoublyRobust`

(Continuous)

    * :class:`ContinuousSelfNormalizedTrajectoryWiseImportanceSampling`
    * :class:`ContinuousSelfNormalizedPerDecisionImportanceSampling`
    * :class:`ContinuousSelfNormalizedDoublyRobust`

.. _implementation_marginal_ope:

Marginalized Importance Sampling Estimators
----------
When the length of trajectory (:math:`T`) is large, even per-decision importance weights can exponentially large in the latter part of the trajectory.
To alleviate this, state marginal or state-action marginal importance weights can be used instead of the per-decision importance weight as follows :cite:`liu2018breaking` :cite:`uehara2020minimax`.

.. math::

    w_{s, a}(s, a) &:= d^{\pi}(s, a) / d^{\pi_0}(s, a) \\
    w_s(s) &:= d^{\pi}(s) / d^{\pi_0}(s)

Then, the importance weight is replaced as follows.

.. math::

    w(s_t, a_t) &= w_{s, a}(s_t, a_t) \\
    w(s_t, a_t) &= w_{s}(s_t) w_{t}(s_t, a_t)
    
    
where :math:`w_t(s_t, a_t) = \pi(a_t | s_t) / \pi_0(a_t | s_t)` is the immediate importance weight.

This estimator is particularly useful when policy visits the same or similar states among different trajectories or different timestep.
(e.g., when the state transition is something like :math:`\cdots \rightarrow s_1 \rightarrow s_2 \rightarrow s_1 \rightarrow s_2 \rightarrow \cdots` or when the trajectories always visits some particular state as :math:`\cdots \rightarrow s_{*} \rightarrow s_{1} \rightarrow s_{*} \rightarrow \cdots`)

.. _tip_mariginal_iw:

.. tip::

    .. dropdown:: How to obtain state(-action) marginal importance weight?

        To use marginalized importance sampling estimators, we need to first estimate the state marginal or state-action marginal importance weight.
        A dominant way to do this is to leverage the following relationship between the importance weights and the state-action value function under the assumption that the state visitation probability is consistent across various timesteps :cite:`uehara2020minimax`.

        .. math::

            &\mathbb{E}_{(s, a, r, s') \sim \mathcal{D_{\pi_0}}}[w(s, a) r] \\
            &= \mathbb{E}_{(s, a, r, s') \sim \mathcal{D_{\pi_0}}}[w(s, a)(Q_{\pi}(s, a) - \gamma \mathbb{E}_{a' \sim \pi(a' | s')}[Q(s', a')])] \\
            &= (1 - \gamma) \mathbb{E}_{s_0 \sim d^{\pi}(s_0), a_0 \sim \pi(a_0 | s_0)}[Q_{\pi}(s_0, a_0)]

        The objective of weight learning is to minimize the difference between the middle term and the last term of the above equation when Q-function adversarially maximizes the difference.
        In particular, we provide the following algorithms to estimate state marginal and state-action marginal importance weights (and corresponding state-action value function) via minimax learning.

        * Augmented Lagrangian Method (ALM/DICE) :cite:`yang2020off`: 
            This method simultaneously optimize both :math:`w(s, a)` and :math:`Q(s, a)`. By setting different hyperparameters, 
            ALM can be identical to BestDICE :cite:`yang2020off`, DualDICE :cite:`nachum2019dualdice`, GenDICE :cite:`zhang2020gendice`, 
            AlgaeDICE :cite:`nachum2019algaedice`, and MQL/MWL :cite:`uehara2020minimax`. 

        * Minimax Q-Learning and Weight Learning (MQL/MWL) :cite:`uehara2020minimax`: 
            This method assumes that one of the value function or weight function is expressed by a function class in a reproducing kernel Hilbert space (RKHS) 
            and optimizes only either value function or weight function. 

        .. seealso::

            * :ref:`How to select models for value/weight learning methods? <tip_create_input_dict>` describes how to enable weight learning and select weight learning methods.
            * :ref:`API reference of value/weight learning methods <scope_rl_api_ope_weight_and_value_learning>`
            * :doc:`API reference of CreateInputDict <_autosummary/scope_rl.ope.input>`

We implement state marginal and state-action marginal OPE estimators in the following classes (both for :class:`Discrete-` and :class:`Continuous-` action spaces).

(State Marginal Estimators)

    * :class:`StateMarginalDirectMethod`
    * :class:`StateMarginalImportanceSampling`
    * :class:`StateMarginalDoublyRobust`
    * :class:`StateMarginalSelfNormalizedImportanceSampling`
    * :class:`StateMarginalSelfNormalizedDoublyRobust`

(State-Action Marginal Estimators)

    * :class:`StateActionMarginalImportanceSampling`
    * :class:`StateActionMarginalDoublyRobust`
    * :class:`StateActionMarginalSelfNormalizedImportanceSampling`
    * :class:`StateActionMarginalSelfNormalizedDoublyRobust`

.. _implementation_drl:

Double Reinforcement Learning (DRL)
----------
Comparing DR in the standard and marginal OPE, we notice that their formulation is slightly different as follows.

(DR in standard OPE)

.. math::

    \hat{J}_{\mathrm{DR}} (\pi; \mathcal{D})
    := \mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t (w_{0:t} (r_t - \hat{Q}(s_t, a_t)) + w_{0:t-1} \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_t, a)]) \right],

(DR in marginal OPE)

.. math::

    \hat{J}_{\mathrm{SAM-DR}} (\pi; \mathcal{D})
    &:= \mathbb{E}_{n} [\mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} \hat{Q}(s_0, a_0)] \\
    & \quad \quad + \mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t w_{s, a}(s_t, a_t) (r_t + \gamma \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_{t+1}, a)] - \hat{Q}(s_t, a_t)) \right],

Then, a natural question arises, would it be possible to use marginal importance weight in DR in the standard formulation?

DRL :cite:`kallus2020double` leverages the marginal importance sampling in the standard OPE formulation as follows.

.. math::

    \hat{J}_{\mathrm{DRL}} (\pi; \mathcal{D})
    & := \frac{1}{n} \sum_{k=1}^K \sum_{i=1}^{n_k} \sum_{t=0}^{T-1} (w_s^j(s_{i,t}, a_{i, t}) (r_{i, t} - Q^j(s_{i, t}, a_{i, t})) \\
    & \quad \quad + w_s^j(s_{i, t-1}, a_{i, t-1}) \mathbb{E}_{a \sim \pi(a | s_t)}[Q^j(s_{i, t}, a)] )

DRL achieves the semiparametric efficiency with a consistent value predictor :math:`Q`. 
Therefore, to alleviate the potential bias introduced in :math:`Q`, DRL uses the "cross-fitting" technique to estimate the value function.
Specifically, let :math:`K` is the number of folds and :math:`\mathcal{D}_j` is the :math:`j`-th split of logged data consisting of :math:`n_k` samples.
Cross-fitting trains :math:`w^j` and :math:`Q^j` on the subset of data used for OPE, i.e., :math:`\mathcal{D} \setminus \mathcal{D}_j`.

    * :class:`DiscreteDoubleReinforcementLearning`
    * :class:`ContinuousDoubleReinforcementLearning`

.. tip::

    .. dropdown:: How to obtain Q-hat via cross-fitting?

        To obtain :math:`\hat{Q}` via cross-fitting, please specify ``k_fold`` of :class:`obtain_whole_inputs` of :class:`CreateOPEInput`.

        .. code-block:: python

            prep = CreateOPEInput(
                env=env,
            )
            input_dict = prep.obtain_whole_inputs(
                logged_dataset=logged_dataset,
                evaluation_policies=evaluation_policies,
                require_value_prediction=True,  # use model-based prediction
                k_fold=3,                       # use 3-fold cross-fitting
                random_state=random_state,
            )

        The default :class:`k_fold=1` trains :math:`\hat{Q}` and :math:`\hat{w}` without cross-fitting.

.. _implementation_sope:

Spectrum of Off-Policy Estimators (SOPE)
----------
While state marginal or state-action marginal importance weight effectively alleviates the variance of per-decision importance weight, the estimation error of marginal importance weights
may introduce some bias in estimation. To alleviate this and control the bias-variance tradeoff more flexibly, SOPE uses the following interpolated importance weights :cite:`yuan2021sope`.

.. math::

    w(s_t, a_t) &= 
    \begin{cases}
        \prod_{t'=0}^{k-1} w_t(s_{t'}, a_{t'}) & \mathrm{if} \, t < k \\
        w_{s, a}(s_{t-k}, a_{t-k}) \prod_{t'=t-k+1}^{t} w_t(s_{t'}, a_{t'}) & \mathrm{otherwise}
    \end{cases} \\
    w(s_t, a_t) &= 
    \begin{cases}
        \prod_{t'=0}^{k-1} w_t(s_{t'}, a_{t'}) & \mathrm{if} \, t < k \\
        w_{s}(s_{t-k}) \prod_{t'=t-k}^{t} w_t(s_{t'}, a_{t'}) & \mathrm{otherwise}
    \end{cases}
    
where SOPE uses per-decision importance weight :math:`w_t(s_t, a_t) := \pi(a_t | s_t) / \pi_0(a_t | s_t)` for the :math:`k` most recent timesteps.

.. tip::

    .. dropdown:: How to change the spectrum of (marginal) OPE?

        SOPE is available by specifying :class:`n_step_pdis` in the state marginal and state-action marginal estimators.

        .. code-block:: python

            ope = OPE(
                logged_dataset=logged_dataset,
                ope_estimators=[SMIS(), SMDR(), SAMIS(), SAMDR()],  # any marginal estimators
            )
            estimation_dict = ope.estimate_policy_value(
                input_dict, 
                n_step_pdis=5,  # number of recent timesteps using per-decision importance sampling
            )

        :class:`n_step_pdis=0` is equivalent to the original marginal OPE estimators.

.. _implementation_high_confidence_ope:

High Confidence Off-Policy Evaluation (HCOPE)
----------
To alleviate the risk of optimistic estimation, we are often interested in the confidence intervals and the lower bound of the estimated policy value.
We implement four methods to estimate the confidence intervals :cite:`thomas2015evaluation` :cite:`thomas2015improvement`.

* Hoeffding :cite:`thomas2015evaluation`: 

.. math::

    |\hat{J}(\pi; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{J}(\pi; \mathcal{D})]| \leq \hat{J}_{\max} \displaystyle \sqrt{\frac{\log(1 / \alpha)}{2 n}}.

* Empirical Bernstein :cite:`thomas2015evaluation` :cite:`thomas2015improvement`: 

.. math::

    |\hat{J}(\pi; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{J}(\pi; \mathcal{D})]| \leq \displaystyle \frac{7 \hat{J}_{\max} \log(2 / \alpha)}{3 (n - 1)} + \displaystyle \sqrt{\frac{2 \hat{\mathbb{V}}_{\mathcal{D}}(\hat{J}) \log(2 / \alpha)}{(n - 1)}}.

* Student T-test :cite:`thomas2015improvement`: 

.. math::

    |\hat{J}(\pi; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{J}(\pi; \mathcal{D})]| \leq \displaystyle \frac{T_{\mathrm{test}}(1 - \alpha, n-1)}{\sqrt{n} / \hat{\sigma}}.

* Bootstrapping :cite:`thomas2015improvement` :cite:`hanna2017bootstrapping`: 

.. math::

    |\hat{J}(\pi; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{J}(\pi; \mathcal{D})]| \leq \mathrm{Bootstrap}(1 - \alpha).

Note that, all the above bound holds with probability :math:`1 - \alpha`.
For notations, we denote :math:`\hat{\mathbb{V}}_{\mathcal{D}}(\cdot)` to be the sample variance,
:math:`T_{\mathrm{test}}(\cdot,\cdot)` to be T value,
and :math:`\sigma` to be the standard deviation.

Among the above high confidence interval estimation, hoeffding and empirical bernstein derives lower bound without any distribution assumption of :math:`p(\hat{J})`, which sometimes leads to quite conservative estimation.
On the other hand, T-test is based on the assumption that each sample of :math:`p(\hat{J})` follows the normal distribution.


.. tip::

    .. dropdown:: How to use High-confidence OPE?

        The implementation is available by calling :class:`estimate_intervals` of each OPE estimator as follows.

        .. code-block:: python

            ope = OPE(
                logged_dataset=logged_dataset,
                ope_estimators=[DM(), TIS(), PDIS(), DR()],  # any standard or marginal estimators
            )
            estimation_dict = ope.estimate_intervals(
                input_dict, 
                ci="hoeffding",  # one of {"hoeffding", "bernstein", "ttest", "bootstrap"}
                alpha=0.05,      # confidence level
            )


.. _implementation_continuous_ope:

Extension to the Continuous Action Space
----------
When the action space is continuous, the naive importance weight :math:`w_t = \pi(a_t|s_t) / \pi_0(a_t|s_t) = (\pi(a |s_t) / \pi_0(a_t|s_t)) \cdot \mathbb{I} \{a = a_t \}` rejects almost every actions,
as the indicator function :math:`\mathbb{I}\{a = a_t\}` filters only the action observed in the logged data.

To address this issue, continuous-action OPE estimators apply kernel density estimation technique to smooth the importance weight :cite:`kallus2018policy` :cite:`lee2022local`.

.. math::

    \overline{w}_t = \int_{a \in \mathcal{A}} \frac{\pi(a | s_t)}{\pi_0(a_t | s_t)} \cdot \frac{1}{h} K \left( \frac{a - a_t}{h} \right) da,

where :math:`K(\cdot)` denotes a kernel function and :math:`h` is the bandwidth hyperparameter.
We can use any function as :math:`K(\cdot)` that meets the following qualities:

* 1) :math:`\int xK(x) dx = 0`,
* 2) :math:`\int K(x) dx = 1`,
* 3) :math:`\lim _{x \rightarrow-\infty} K(x)=\lim _{x \rightarrow+\infty} K(x)=0`,
* 4) :math:`K(x) \geq 0, \forall x`.

In our implementation, we use the (distance-based) Gaussian kernel :math:`K(x)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{x^{2}}{2}}`.

.. tip::

    .. dropdown:: How to control the bias-variance tradeoff with a kernel?

        The bandwidth parameter :math:`h` controls the bias-variance tradeoff. 
        Specifically, a large value of :math:`h` leads to a low-variance but high-bias estimation,
        while a small value of :math:`h` leads to a high-variance but low-bias estimation.

        The bandwidth parameter corresponds to ``sigma`` in the :class:`OffPolicyEvaluation` class.

        .. code-block:: python

            ope = OPE(
                logged_dataset=logged_dataset,
                ope_estimators=[DM(), TIS(), PDIS(), DR()],
                sigma=1.0,  # bandwidth hyperparameter
            )

        For multi-dimension actions, we define the kernel with dot product among actions as :math:`K(a, a') := K(a^T a')`.
        To control the scale of each dimension, ``action_scaler``, which is speficied in :class:`OffPolicyEvaluation`, is also useful. 

        .. code-block:: python

            from d3rlpy.preprocessing import MinMaxActionScaler
            ope = OPE(
                logged_dataset=logged_dataset,
                ope_estimators=[DM(), TIS(), PDIS(), DR()],
                sigma=1.0,  # bandwidth hyperparameter
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,
                    maximum=env.action_space.high,
                ),
            )

        .. seealso::

            `(external) d3rlpy's documentation about action_scaler <https://d3rlpy.readthedocs.io/en/latest/references/generated/d3rlpy.preprocessing.MinMaxActionScaler.html#d3rlpy.preprocessing.MinMaxActionScaler>`_

.. _implementation_cumulative_distribution_ope:

Cumulative Distribution Off-Policy Evaluation (CD-OPE)
~~~~~~~~~~

While the basic OPE aims to estimate the average policy performance, we are often also interested in the performance distribution of the evaluation policy.
Cumulative distribution OPE enables flexible estimation of various risk functions such as variance and conditional value at risk (CVaR) using the cumulative distribution function (CDF) :cite:`huang2021off` :cite:`huang2022off` :cite:`chandak2021universal`.

(Cumulative Distribution Function)

.. math::

    F(m, \pi) := \mathbb{E} \left[ \mathbb{I} \left \{ \sum_{t=0}^{T-1} \gamma^t r_t \leq m \right \} \mid \pi \right]
    
(Risk Functions derived by CDF)

* Mean: :math:`\mu(F) := \int_{G} G \, \mathrm{d}F(G)`
* Variance: :math:`\sigma^2(F) := \int_{G} (G - \mu(F))^2 \, \mathrm{d}F(G)`
* :math:`\alpha`-quartile: :math:`Q^{\alpha}(F) := \min \{ G \mid F(G) \leq \alpha \}`
* Conditional Value at Risk (CVaR): :math:`\int_{G} G \, \mathbb{I}\{ G \leq Q^{\alpha}(F) \} \, \mathrm{d}F(G)`

where we let :math:`G := \sum_{t=0}^{T-1} \gamma^t r_t` to represent the random variable of trajectory wise reward
and :math:`dF(G) := \mathrm{lim}_{\Delta \rightarrow 0} F(G) - F(G- \Delta)`.

To estimate both CDF and various risk functions, we provide the following :class:`CumulativeDistributionOffPolicyEvaluation` class.

.. code-block:: python

    # initialize the OPE class
    from scope_rl.ope import CumulativeDistributionOffPolicyEvaluation as CumulativeDistributionOPE
    cd_ope = CumulativeDistributionOPE(
        logged_dataset=logged_dataset,
        ope_estimators=[CD_DM(), CD_IS(), CD_DR()],
    )

It estimates the cumulative distribution of the trajectory wise reward and various risk functions as follows.

.. code-block:: python

    cdf_dict = cd_ope.estimate_cumulative_distribution_function(input_dict)
    variance_dict = cd_ope.estimate_variance(input_dict)

.. _tip_cumulative_distribution_ope:

.. tip::

    .. dropdown:: How to conduct Cumulative Distribution OPE with multiple logged datasets?

        Conducting Cumulative Distribution OPE with multiple logged datasets requires no additional efforts.

        First, the same command with the single logged dataset case also works with multiple logged datasets.

        .. code-block:: python

            ope = CumulativeDistributionOPE(
                logged_dataset=logged_dataset,  # MultipleLoggedDataset
                ope_estimators=[CD_DM(), CD_IS(), CD_DR()],
            )
            multiple_cdf_dict = ope.estimate_cumulative_distribution_function(
                input_dict,  # MultipleInputDict
            )

        The returned value is the dictionary containing the ope result.

        In addition, we can specify which logged dataset and input_dict to use by setting ``behavior_policy_name`` and ``dataset_id``.

        .. code-block:: python

            multiple_ope_dict = ope.estimate_cumulative_distribution_function(
                input_dict,
                behavior_policy_name=behavior_policy.name,  #
                dataset_id=0,  # specify which logged dataset and input_dict to use
            )

        The basic visualization function also work by specifying the dataset id.

        .. code-block:: python

            ope.visualize_cumulative_distribution_function(
                input_dict,
                behavior_policy_name=behavior_policy.name,  #
                dataset_id=0,  #
                random_state=random_state,
            )

        .. card:: 
            :img-top: ../_static/images/ope_cumulative_distribution_function.png
            :text-align: center
            
            cumulative distribution function estimated with the specified dataset

        Moreover, we provide additional visualization function for the multiple logged dataset case.

        The following visualizes confidence intervals of cumulative distribution function.

        .. code-block:: python

            ope.visualize_cumulative_distribution_function_with_multiple_estimates(
                input_dict,      # MultipleInputDict
                behavior_policy_name=None,                   # compare estimators with multiple behavior policies
                # behavior_policy_name=behavior_policy.name  # compare estimators with a single behavior policy
                random_state=random_state,
            )

        .. card:: 
            :img-top: ../_static/images/ope_cumulative_distribution_function_multiple.png
            :text-align: center
            
            cumulative distribution function estimated with the multiple datasets

        On contrary, the following visualizes the distribution of multiple estimates of ponit-wise policy performance 
        (e.g., policy value, variance, conditional value at risk, lower quartile). 

        .. code-block:: python

            ope.visualize_policy_value_with_multiple_estimates(
                input_dict,      # MultipleInputDict
                plot_type="ci",  # one of {"ci", "violin", "scatter"}, default="ci"
                random_state=random_state,
            )

        When the ``plot_type`` is "ci", the plot is somewhat similar to the basic visualization. 
        (The star indicates the ground-truth policy value and the confidence intervals are derived by multiple estimates across datasets.)

        .. card:: 
            :img-top: ../_static/images/ope_cumulative_policy_value_basic_multiple.png
            :text-align: center
            
            policy value estimated with the multiple datasets

        When the ``plot_type`` is "violin", the plot visualizes the distribution of multiple estimates.
        This is particularly useful to see how the estimation result can vary depending on different datasets or random seeds. 

        .. card:: 
            :img-top: ../_static/images/ope_cumulative_policy_value_basic_multiple_violin.png
            :text-align: center
            
            policy value estimated with the multiple datasets (violin)

        Finally, when the ``plot_type`` is "scatter", the plot visualizes each estimation with its color specifying the dataset id.
        This function is particularly useful to see how the choice of behavior policy (e.g., their stochasticity) affects the estimation result.

        .. card:: 
            :img-top: ../_static/images/ope_cumulative_policy_value_basic_multiple_scatter.png
            :text-align: center
            
            policy value estimated with the multiple datasets (scatter)

        .. seealso::

            * :ref:`How to obtain MultipleLoggedDataset? <tips_synthetic_dataset>`
            * :ref:`How to handle OPL with MultipleLoggedDataset? <tip_opl>`
            * :ref:`How to create input_dict for MultipleLoggedDataset? <tip_create_input_dict>`
            .. * :ref:`Tutorial with MultipleLoggedDataset <scope_rl_multiple_tutorial>`

.. seealso::

    * :doc:`quickstart` 
    .. * and :ref:`related tutorials <cumulative_distribution_ope_tutorial>`

:class:`CumulativeDistributionOffPolicyEvaluation` implements the following functions.

(Cumulative Distribution Function)

* :class:`estimate_cumulative_distribution_function`

(Risk Functions and Statistics)

* :class:`estimate_mean`
* :class:`estimate_variance`
* :class:`estimate_conditional_value_at_risk`
* :class:`estimate_interquartile_range`

(Visualization)

* :class:`visualize_policy_value`
* :class:`visualize_conditional_value_at_risk`
* :class:`visualize_interquartile_range`
* :class:`visualize_cumulative_distribution_function`

(Visualization with multiple estimates on multiple logged datasets)

* :class:`visualize_policy_value_with_multiple_estimates`
* :class:`visualize_variance_with_multiple_estimates`
* :class:`visualize_cumulative_distribution_function_with_multiple_estimates`
* :class:`visualize_lower_quartile_with_multiple_estimates`
* :class:`visualize_cumulative_distribution_function_with_multiple_estimates`


(Others)

* :class:`obtain_reward_scale`

Below, we describe the implemented cumulative distribution OPE estimators.

==================================================================================  ================  ================
:ref:`Direct Method (DM) <implementation_cd_dm>`                                                                    
:ref:`Trajectory-wise Importance Sampling (TIS) <implementation_cd_tis>`             
:ref:`Trajectory-wise Doubly Robust (DR) <implementation_cd_tdr>`                                    
Self-Normalized estimators
Extension to the continuous action space   
==================================================================================  ================  ================

.. tip::

    .. dropdown:: How to define my own cumulative distribution OPE estimator?

        To define your own OPE estimator, use :class:`BaseCumulativeDistributionOffPolicyEstimator`. 

        Basically, the common inputs for each functions are ``reward_scale`` (np.ndarray indicating x-axis of cumulative distribution function) 
        and the following keys from ``logged_dataset`` and ``input_dict``.

        (logged_dataset)

        .. code-block:: python

            key: [
                size,
                step_per_trajectory,
                action,
                reward,
                pscore,
            ]

        (input_dict)

        .. code-block:: python

            key: [
                evaluation_policy_action,
                evaluation_policy_action_dist,
                state_action_value_prediction,
                initial_state_value_prediction,
                state_action_marginal_importance_weight,
                state_marginal_importance_weight,
                on_policy_policy_value,
                gamma,
            ]
        
        ``action_scaler`` and ``sigma`` are also added in the continuous-action case.

        If you want to add other arguments, please add them in the initialization arguments for API consistency.

        Finally, contribution to SCOPE-RL with a new OPE estimator is more than welcome! Please read `the guidelines for contribution (CONTRIBUTING.md) <https://github.com/hakuhodo-technologies/scope-rl/blob/main/CONTRIBUTING.md>`_.

        .. seealso::

            :doc:`API reference of BaseOffPolicyEstimator <_autosummary/scope_rl.ope.estimators_base>` explains the abstract methods.

.. _implementation_cd_dm:

Direct Method (DM)
----------

DM adopts model-based approach to estimate the cumulative distribution function.

.. math::

        \hat{F}_{\mathrm{DM}}(m, \pi; \mathcal{D}) := \mathbb{E}_{n} [\mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} \hat{G}(m; s_0, a_0)]

where :math:`\hat{F}(\cdot)` is the estimated cumulative distribution function and :math:`\hat{G}(\cdot)` is the estimated conditional distribution.

DM is vulnerable to the approximation error, but has low variance.

    * :class:`DiscreteCumulativeDistributionDirectMethod`
    * :class:`ContinuousCumulativeDistributionDirectMethod`

.. _implementation_cd_tis:

Trajectory-wise Importance Sampling (TIS)
----------

TIS corrects the distribution shift by applying importance sampling technique on the cumulative distribution estimation.

.. math::

        \hat{F}_{\mathrm{TIS}}(m, \pi; \mathcal{D}) := \mathbb{E}_{n} \left[ w_{0:T-1} \mathbb{I} \left \{\sum_{t=0}^{T-1} \gamma^t r_t \leq m \right \} \right]

TIS is unbiased but can suffer from high variance.
In particular, :math:`\hat{F}_{\mathrm{TIS}}(\cdot)` sometimes becomes more than one when the variance is high.
Therefore, we correct CDF as :math:`\hat{F}^{\ast}_{\mathrm{TIS}}(m, \pi; \mathcal{D}) := \min(\max_{m' \leq m} \hat{F}_{\mathrm{TIS}}(m', \pi; \mathcal{D}), 1)` :cite:`huang2021off`.

    * :class:`DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling`
    * :class:`ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling`

.. _implementation_cd_tdr:

Trajectory-wise Doubly Robust (TDR)
----------

TDR combines TIS and DM to reduce the variance while being unbiased.

.. math::

    \hat{F}_{\mathrm{TDR}}(m, \pi; \mathcal{D})
    := \mathbb{E}_{n} \left[ w_{0:T-1} \left( \mathbb{I} \left \{\sum_{t=0}^{T-1} \gamma^t r_t \leq m \right \} - \hat{G}(m; s_0, a_0) \right) \right]
    + \hat{F}_{\mathrm{DM}}(m, \pi; \mathcal{D})

TDR reduces the variance of TIS while being unbiased, leveraging the model-based estimate (i.e., DM) as a control variate.
Since :math:`\hat{F}_{\mathrm{TDR}}(\cdot)` may be less than zero or more than one, we should apply the following transformation to bound :math:`\hat{F}_{\mathrm{TDR}}(\cdot) \in [0, 1]` :cite:`huang2021off`.

.. math::

    \hat{F}^{\ast}_{\mathrm{TIS}}(m, \pi; \mathcal{D}) := \mathrm{clip}(\max_{m' \leq m} \hat{F}_{\mathrm{TIS}}(m', \pi; \mathcal{D}), 0, 1).

Note that, this estimator is not equivalent to the (recursive) DR estimator defined by :cite:`huang2022off`. We are planning to implement the recursive version in a future update of the software.

    * :class:`DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust`
    * :class:`ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust`

Finally, we also provide the self-normalized estimators for TIS and TDR.
They use the self-normalized importance weight :math:`\tilde{w}_{\ast} := w_{\ast} / \mathbb{E}_{n}[w_{\ast}]` for the variance reduction purpose.

    * :class:`DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling`
    * :class:`DiscreteCumulativeDistributionSelfNormalizedDoublyRobust`
    * :class:`ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling`
    * :class:`ContinuousCumulativeDistributionSelfNormalizedDoublyRobust`

.. _implementation_eval_ope_ops:

Evaluation Metrics of OPE/OPS
~~~~~~~~~~
Finally, we describe the metrics to evaluate the quality of OPE estimators and its OPS result.

* Mean Squared Error (MSE) :cite:`paine2020hyperparameter` :cite:`voloshin2021empirical` :cite:`fu2021benchmarks`: 
    This metrics measures the estimation accuracy as :math:`\sum_{\pi \in \Pi} (\hat{J}(\pi; \mathcal{D}) - J(\pi))^2 / |\Pi|`.

* Regret@k :cite:`paine2020hyperparameter` :cite:`fu2021benchmarks`: 
    This metrics measures how well the selected policy(ies) performs. In particular, Regret@1 indicates the expected performance difference between the (oracle) best policy and the selected policy as :math:`J(\pi^{\ast}) - J(\hat{\pi}^{\ast})`, where :math:`\pi^{\ast} := {\arg\max}_{\pi \in \Pi} J(\pi)` and :math:`\hat{\pi}^{\ast} := {\arg\max}_{\pi \in \Pi} \hat{J}(\pi; \mathcal{D})`.

* Spearman's Rank Correlation Coefficient :cite:`paine2020hyperparameter` :cite:`fu2021benchmarks`: 
    This metrics measures how well the raking of the candidate estimators are preserved in the OPE result.

* Type I and Type II Error Rate: 
    This metrics measures how well an OPE estimator validates whether the policy performance surpasses the given safety threshold or not.

To ease the comparison of candidate (evaluation) policies and the OPE estimators, we provide the :class:`OffPolicySelection` class.

.. code-block:: python

    # Initialize the OPS class
    from scope_rl.ope import OffPolicySelection
    ops = OffPolicySelection(
        ope=ope,
        cumulative_distribution_ope=cd_ope,
    )

The :class:`OffPolicySelection` class returns both the OPE results and the OPS metrics as follows.

.. code-block:: python

    ranking_df, metric_df = ops.select_by_policy_value(
        input_dict,
        return_metrics=True,
        return_by_dataframe=True,
    )

Moreover, the OPS class enables us to validate the best/worst/mean performance of top k deployment and how well the safety requirement is satisfied.

.. code-block:: python

    ops.visualize_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        safety_criteria=1.0,
    )

Finally, the OPS class also implements the modules to compare the OPE result and the true policy metric as follows.

.. code-block:: python

    ops.visualize_policy_value_for_validation(
        input_dict=input_dict,
        n_cols=4,
        share_axes=True,
    )

.. tip::

    .. dropdown:: How to conduct OPS with multiple logged datasets?

        Conducting OPS with multiple logged datasets requires no additional efforts.

        First, the same command with the single logged dataset case also works with multiple logged datasets.

        .. code-block:: python

            ops = OffPolicySelection(
                ope=ope,                             # initialized with MultipleLoggedDataset
                cumulative_distribution_ope=cd_ope,  # initialized with MultipleLoggedDataset
            )
            ranking_df, metric_df = ops.select_by_policy_value(
                input_dict,  # MultipleInputDict
                return_metrics=True,
                return_by_dataframe=True,
            )

        The returned value is dictionary containing the ops result.

        Next, visualization functions for OPS demonstrate the aggregated ops result by default.
        For example, the average topk performance and its confidence intervals is shown for topk visualization.

        .. code-block:: python

            ops.visualize_topk_policy_value_selected_by_standard_ope(
                input_dict=input_dict,
                safety_criteria=1.0,
            )

        .. card:: 
            :img-top: ../_static/images/ops_topk_policy_value_multiple.png
            :text-align: center
            
            top-k deployment result with multiple logged datasets

        In the validation visualization, colors indicate the dataset ids. 
        This function is particularly useful to see how the choice of behavior policy (e.g., their stochasticity) affects the estimation result.

        .. code-block:: python

            ops.visualize_policy_value_for_validation(
                input_dict=input_dict,
                n_cols=4,
                share_axes=True,
            )

        .. card:: 
            :img-top: ../_static/images/ops_validation_policy_value_multiple.png
            :text-align: center
            
            validation results on multiple logged datasets

        Note that, when the ``behavior_policy_name`` and ``dataset_id`` is specified, the methods show the result on the specified dataset.

        .. seealso::

            * :ref:`How to obtain MultipleLoggedDataset? <tips_synthetic_dataset>`
            * :ref:`How to handle OPL with MultipleLoggedDataset? <tip_opl>`
            * :ref:`How to create input_dict for MultipleLoggedDataset? <tip_create_input_dict>`
            * :ref:`How to conduct OPE with MultipleLoggedDataset? <tip_ope>`
            * :ref:`How to conduct Cumulative Distribution OPE with MultipleLoggedDataset? <tip_cumulative_distribution_ope>`
            .. * :ref:`Tutorial with MultipleLoggedDataset <scope_rl_multiple_tutorial>`

.. seealso::

    * :doc:`quickstart` 
    .. * and :ref:`related tutorials <ops_tutorial>`

The OPS class implements the following functions.

(OPS)

* :class:`obtain_oracle_selection_result`
* :class:`select_by_policy_value`
* :class:`select_by_policy_value_via_cumulative_distribution_ope`
* :class:`select_by_policy_value_lower_bound`
* :class:`select_by_lower_quartile`
* :class:`select_by_conditional_value_at_risk`

(Visualization)

* :class:`visualize_policy_value_for_selection`
* :class:`visualize_cumulative_distribution_function_for_selection`
* :class:`visualize_policy_value_for_selection`
* :class:`visualize_policy_value_of_cumulative_distribution_ope_for_selection`
* :class:`visualize_conditional_value_at_risk_for_selection`
* :class:`visualize_interquartile_range_for_selection`

(Visualization with multiple estimates on multiple logged datasets)

* :class:`visualize_policy_value_with_multiple_estimates_standard_ope`
* :class:`visualize_policy_value_with_multiple_estimates_cumulative_distribution_ope`
* :class:`visualize_variance_with_multiple_estimates`
* :class:`visualize_cumulative_distribution_function_with_multiple_estimates`
* :class:`visualize_lower_quartile_with_multiple_estimates`
* :class:`visualize_cumulative_distribution_function_with_multiple_estimates`

(Visualization of top k performance)

* :class:`visualize_topk_policy_value_selected_by_standard_ope`
* :class:`visualize_topk_policy_value_selected_by_cumulative_distribution_ope`
* :class:`visualize_topk_policy_value_selected_by_lower_bound`
* :class:`visualize_topk_conditional_value_at_risk_selected_by_standard_ope`
* :class:`visualize_topk_conditional_value_at_risk_selected_by_cumulative_distribution_ope`
* :class:`visualize_topk_lower_quartile_selected_by_standard_ope`
* :class:`visualize_topk_lower_quartile_selected_by_cumulative_distribution_ope`

(Visualization for validation)

* :class:`visualize_policy_value_for_validation`
* :class:`visualize_policy_value_of_cumulative_distribution_ope_for_validation`
* :class:`visualize_policy_value_lower_bound_for_validation`
* :class:`visualize_variance_for_validation`
* :class:`visualize_lower_quartile_for_validation`
* :class:`visualize_conditional_value_at_risk_for_validation`

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
                :link: ope
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Problem Formulation**

            .. grid-item-card::
                :link: learning_implementation
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Offline RL**

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
                :link: visualization
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Visualization tools**

            .. grid-item-card::
                :link: scope_rl_api
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Package Reference**
