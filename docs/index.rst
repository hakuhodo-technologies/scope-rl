:html_theme.sidebar_secondary.remove:

.. card:: SCOPE-RL
    :class-title: top-page-title-title
    :class-body: top-page-title-desc
    :text-align: center
    :shadow: none

    Towards Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning

    .. button-ref:: documentation/index
      :ref-type: doc
      :color: primary
      :shadow:

      Getting Started

    .. button-ref:: documentation/installation
      :ref-type: doc

      Installation

.. raw:: html

    <h2>Why SCOPE-RL?</h2>

.. grid-item::
    :class: top-page-list

    * End-to-end implementation of Offline RL and OPE

    * Variety of OPE estimators and standardized evaluation protocol of OPE 
    
    * Provide Cumulative Distribution OPE for risk function estimation

    * Validate potential risks of OPS in deploying poor policies

.. raw:: html

    <div class="top-page-link">
        <a href="documentation/distinctive_features.html">>>> Explore!</a>
    </div>

.. raw:: html

    <h2>Try SCOPE-RL in two lines of code!</h2>

    <div class="top-page-list">
        <ul>
            <li>Compare policy performance via OPE</li>
        </ul>
    </div>

.. code-block:: python

    # initialize the OPE class
    ope = OPE(
        logged_dataset=logged_dataset,
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


.. raw:: html

    <div class="top-page-link">
        <a href="documentation/quickstart.html">>>> See more</a>
    </div>
    <div class="white-space-5px"></div>

.. raw:: html

    <div class="top-page-list">
        <ul>
            <li>Compare cumulative distribution function (CDF) via OPE</li>
        </ul>
    </div>

.. code-block:: python

    # initialize the OPE class
    cd_ope = CumulativeDistributionOPE(
        logged_dataset=logged_dataset,
        ope_estimators=[
        CD_DM(estimator_name="cdf_dm"),
        CD_IS(estimator_name="cdf_is"),
        CD_DR(estimator_name="cdf_dr"),
        CD_SNIS(estimator_name="cdf_snis"),
        CD_SNDR(estimator_name="cdf_sndr"),
        ],
    )
    # estimate and visualize cumulative distribution function
    cd_ope.visualize_cumulative_distribution_function(input_dict, n_cols=4)

.. card:: 
    :img-top: ../_static/images/ope_cumulative_distribution_function.png
    :text-align: center
    
    Cumulative Distribution Function Estimated by OPE Estimators

.. raw:: html

    <div class="top-page-link">
        <a href="documentation/quickstart.html">>>> See more</a>
    </div>
    <div class="white-space-5px"></div>

.. raw:: html

    <div class="top-page-list">
        <ul>
            <li>Validate top-k performance and risks of OPS</li>
        </ul>
    </div>

.. code-block:: python

    # Initialize the OPS class
    ops = OffPolicySelection(
        ope=ope,
        cumulative_distribution_ope=cd_ope,
    )
    # visualize the top k deployment result
    ops.visualize_topk_lower_quartile_selected_by_cumulative_distribution_ope(
        input_dict=input_dict,
        ope_alpha=0.10,
        safety_threshold=9.0,
    )

.. card:: 
    :img-top: ../_static/images/ops_topk_lower_quartile.png
    :text-align: center
    
    Comparison of the Top-k Statistics of 10% Lower Quartile of Policy Value

.. raw:: html

    <div class="top-page-link">
        <a href="documentation/quickstart.html">>>> See more</a>
    </div>
    <div class="white-space-5px"></div>

.. raw:: html

    <div class="top-page-list">
        <ul>
            <li>Understand the trend of estimation errors</li>
        </ul>
    </div>

.. code-block:: python

    # Initialize the OPS class
    ops = OffPolicySelection(
        ope=ope,
        cumulative_distribution_ope=cd_ope,
    )
    # visualize the OPS results with the ground-truth metrics
    ops.visualize_variance_for_validation(
        input_dict,
        share_axes=True,
    )

.. card:: 
    :img-top: ../_static/images/ops_variance_validation.png
    :text-align: center
    
    Validation of Estimated and Ground-truth Variance of Policy Value

.. raw:: html

    <div class="top-page-link">
        <a href="documentation/quickstart.html">>>> See more</a>
    </div>
    <div class="white-space-5px"></div>

.. .. raw:: html

..     <h2>Explore more with SCOPE-RL</h2>

..     <div class="top-page-gallery-link">
..         <a href="documentation/_autogallery/index.html">Tutorials</a>
..     </div>

.. .. card-carousel:: 4

..     .. card:: Basic Off-Policy Evaluation
..         :img-top: .png

..     .. card:: Marginal Off-Policy Evaluation
..         :img-top: .png

..     .. card:: Cumulative Distribution Off-Policy Evaluation
..         :img-top: .png

..     .. card:: Off-Policy Selection
..         :img-top: .png

..     .. card:: Evaluation of OPE/OPS
..         :img-top: .png

..     .. card:: Ablation with various value functions
..         :img-top: .png

..     .. card:: Ablation with xxx
..         :img-top: .png

..     .. card:: Handling multiple datasets
..         :img-top: .png

..     .. card:: Evaluating with various behavior policies
..         :img-top: .png

..     .. card:: Evaluating on non-episodic setting
..         :img-top: .png

.. .. raw:: html

..     <div class="top-page-gallery-link">
..         <a href="documentation/subpackages/index.html">Applications</a>
..     </div>


.. .. card-carousel:: 4

..     .. card:: Example on Real-Time Bidding
..         :img-top: .png

..     .. card:: Example on Recommendation
..         :img-top: .png

..     .. card:: Example on xxx
..         :img-top: .png

..     .. card:: Example on xxx
..         :img-top: .png


.. raw:: html

    <h2>Citation</h2>


If you use our pipeline or the SharpRatio@k metric in your work, please cite our paper below.

.. card::

    | Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.
    | **SCOPE-RL: Towards Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning** [`arXiv <>`_] [`Proceedings <>`_]
    | (a preprint coming soon..)

    .. code-block::

        @article{kiyohara2023scope,
            author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
            title = {SCOPE-RL: Towards Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning},
            journal = {A github repository},
            pages = {xxx--xxx},
            year = {2023},
        }

.. raw:: html

    <div class="white-space-5px"></div>
    <h2>Join us!</h2>

Any contributions to SCOPE-RL are more than welcome!

* `Guidelines for contribution (CONTRIBUTING.md) <>`_
* `Google Group (scope-rl@googlegroups.com) <https://groups.google.com/g/scope-rl>`_!

If you have any questions, feel free to contact: hk844 [at] cornell.edu

.. raw:: html

   <div style="visibility: hidden;">

Welcome!
========

.. raw:: html

   </div>

.. toctree::
    :maxdepth: 1
    :hidden:

    Installation <documentation/installation>
    Quickstart <documentation/quickstart>
    .. Tutorial <documentation/_autogallery/index>
    Documentation <documentation/index>
    FAQs <documentation/frequently_asked_questions>
    News <documentation/news>
    Sub-packages <documentation/subpackages/index>
    Release Notes <https://github.com/negocia-inc/scope-rl/releases>
    Proceedings <https://github.com/negocia-inc/scope-rl/404>

.. grid::

    .. grid-item::
        :columns: 2
        :margin: 0
        :padding: 0

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
                :link: documentation/installation
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Installation**

