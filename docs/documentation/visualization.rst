:html_theme.sidebar_secondary.remove:

==========
Visualization Tools
==========

SCOPE-RL also provides user-friendly tools to visually compare and understand the performance of OPE methods.
Specifically, the following figures are all available by calling only one function from either :class:`OffPolicyEvaluation` or :class:`OffPolicySelection` as follows.

.. card:: 
    :width: 75%
    :margin: auto

    .. code-block:: python

        # initialize the OPE class
        ope = OPE(
            logged_dataset=logged_dataset,
            ope_estimators=[DM(), TIS(), PDIS(), DR()],
        )
        # conduct OPE and visualize the result
        # by calling only one function!
        ope.visualize_off_policy_estimates(
            input_dict,
            random_state=random_state,
            sharey=True,
        )

.. raw:: html

    <div class="white-space-20px"></div>


Then, the above code produces the following visualization result.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_policy_value_basic.png
    :text-align: center
    
    Policy value estimated by (standard) OPE Estimators

.. raw:: html

    <div class="white-space-20px"></div>

Similarly, the visualization tools are also available for cumulative distribution OPE (CD-OPE).

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_cumulative_distribution_function.png
    :text-align: center
    
    Cumulative distribution function estimated by CD-OPE Estimators

.. raw:: html

    <div class="white-space-5px"></div>

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_conditional_value_at_risk.png
    :text-align: center
    
    Conditional value at risk (CVaR) estimated by CD-OPE Estimators

.. raw:: html

    <div class="white-space-5px"></div>

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_policy_value_by_cd_ope.png
    :text-align: center
    
    Policy value and its confidence interval derived by variance estimated by CD-OPE Estimators

.. raw:: html

    <div class="white-space-5px"></div>

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_interquartile_range.png
    :text-align: center
    
    Interquartile range estimated by CD-OPE Estimators

.. raw:: html

    <div class="white-space-20px"></div>

Moreover, the evaluation of OPE/OPS can also be done by visualizing the top-:math:`k` Risk-Return Tradeoff (RRT) metrics.
Note that, the following figures are applicable to all the point-wise performance estimate including expected policy value, variance, CVaR, and lower quartile.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ops_topk_policy_value_multiple.png
    :text-align: center

    Example of evaluating OPE/OPS methods with top-:math:`k` RRT metrics

.. raw:: html

    <div class="white-space-20px"></div>

Furthermore, when conducting OPE on multiple logged datasets collected by various behavior policies, 
SCOPE-RL also enables to discuss how the quality of dataset may affect the performance of OPE.

First, the following three figures are applicable to the point-wise estimate of expected policy value, variance, CVaR, and lower quartile.
In the following example, we can learn that OPE results can be particularly unstable when using "ddqn_epsilon_0.1" as the behavior policy, which is more deterministic than other behavior policies.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_policy_value_basic_multiple.png
    :text-align: center
    
    Policy value estimated on the multiple datasets collected by various behavior policies (box)

.. raw:: html

    <div class="white-space-5px"></div>

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_policy_value_basic_multiple_violin.png
    :text-align: center
    
    Policy value estimated on the multiple datasets collected by various behavior policies (violin)

.. raw:: html

    <div class="white-space-5px"></div>

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_policy_value_basic_multiple_scatter.png
    :text-align: center
    
    Policy value estimated on the multiple datasets collected by various behavior policies (scatter)

.. raw:: html

    <div class="white-space-20px"></div>

Next, we demonstrate the example of comparing cumulative distribution function estimated on multiple logged datasets collected by various behavior policies.
In the figure, we observe that the cumulative distribution OPE results do not change greatly across various behavior policies.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_cumulative_distribution_function_multiple.png
    :text-align: center
    
    Cumulative distribution function estimated on the multiple datasets collected by various behavior policies

.. raw:: html

    <div class="white-space-20px"></div>

Finally, we compare the true policy value (x-axis) and estimated policy value (y-axis) in the following figure.
For TIS, PDIS, and DR, the result suggests that the variance of OPE estimation becomes particularly large when using near-deterministic behavior policy named "sac_sigma_0.5".
On the other hand, for SNTIS and SNPDIS, we found that the choice of behavior policy can heavily affects the estimation result of OPE -- OPE results are almost the same across various evaluation policies in the bottom left figures.
This kind of visualization is again available for all point-wise estimates including expected policy value, variance, CVaR, and lower quartile.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ops_validation_policy_value_multiple.png
    :text-align: center
    
    Validation results of the policy value estimation on multiple logged datasets collected by various behavior policies 

.. raw:: html

    <div class="white-space-5px"></div>

.. seealso:: 

    * :doc:`quickstart` and :doc:`related example codes </documentation/examples/multiple>`

.. raw:: html

    <div class="white-space-5px"></div>

.. grid::
    :margin: 0

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

                <<< Prev
                **Supported Implementation**

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
                :link: sharpe_ratio
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **SharpeRatio**

            .. grid-item-card::
                :link: scope_rl_api
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Package Reference**

