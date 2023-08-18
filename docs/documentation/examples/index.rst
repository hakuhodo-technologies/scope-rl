.. _example:

Example Codes
==========

SCOPE-RL
----------

.. _basic_ope_example:

Basic and High-Confidence Off-Policy Evaluation (OPE):
~~~~~~~~~~

.. card::
    :link: /documentation/examples/basic_ope
    :link-type: doc

    .. grid::
        :gutter: 1

        .. grid-item::
            :columns: 4

            .. grid:: 1
                :gutter: 1

                **Basic OPE**

            .. grid:: 1
                :gutter: 1

                .. grid-item-card:: 
                    :img-background: /_static/images/ope_policy_value_basic.png
                    :shadow: none

        .. grid-item::
            :columns: 8

            .. grid:: 1
                :gutter: 1
                :padding: 1

                .. grid-item:: 

                    * Logged datasets and inputs
                    * Basic Off-Policy Evaluation (DM, PDIS, DR, ..)
                    * Marginal Off-Policy Evaluation (SMIS, SMDR, SAMIS, SAMDR, ..)
                    * High-Confidence Off-Policy Evaluation (Hoeffding, Bernstein, ..)
                    * Extention to continuous action space

.. _cumulative_distribution_ope_example:

Cumulative Distribution OPE (CD-OPE):
~~~~~~~~~~

.. card::
    :link: /documentation/examples/cumulative_dist_ope
    :link-type: doc

    .. grid::
        :gutter: 1

        .. grid-item::
            :columns: 4

            .. grid:: 1
                :gutter: 1

                **Cumulative Distribution OPE**

            .. grid:: 1
                :gutter: 1

                .. grid-item-card:: 
                    :img-background: /_static/images/cd_ope_interquartile_range.png
                    :shadow: none

        .. grid-item::
            :columns: 8

            .. grid:: 1
                :gutter: 1
                :padding: 1

                .. grid-item:: 

                    * Logged datasets and inputs
                    * Estimating Cumulative Distribution Function
                    * Estimating risk-functions (mean, variance, CVaR, ..)

.. _off_policy_selection_example:

Off-Policy Selection
~~~~~~~~~~

.. card::
    :link: /documentation/examples/ops
    :link-type: doc

    .. grid::
        :gutter: 1

        .. grid-item::
            :columns: 4

            .. grid:: 1
                :gutter: 1

                **Off-Policy Selection (OPS)**

            .. grid:: 1
                :gutter: 1

                .. grid-item-card:: 
                    :img-background: /_static/images/ops_topk_thumbnail.png
                    :shadow: none

        .. grid-item::
            :columns: 8

            .. grid:: 1
                :gutter: 1
                :padding: 1

                .. grid-item:: 

                    * OPS via Basic OPE
                    * OPS via Cumulative Distribution OPE
                    * Obtaining oracle selection results

.. _assessment_example:

Assessing OPE Estimators
~~~~~~~~~~

.. card::
    :link: /documentation/examples/assessments
    :link-type: doc

    .. grid::
        :gutter: 1

        .. grid-item::
            :columns: 4

            .. grid:: 1
                :gutter: 1

                **Off-Policy Selection (OPS)**

            .. grid:: 1
                :gutter: 1

                .. grid-item-card:: 
                    :img-background: /_static/images/ops_validation_thumbnail.png
                    :shadow: none

        .. grid-item::
            :columns: 8

            .. grid:: 1
                :gutter: 1
                :padding: 1

                .. grid-item:: 

                    * Conventional "accuracy" metrics
                    * Top-:math:`k` risk-return tradeoff metrics
                    * Validation visualization

.. _custom_estimator_example

Implementing Custom OPE Estimators:
~~~~~~~~~~

.. card::
    :link: /documentation/examples/custom_estimators
    :link-type: doc

    .. grid::
        :gutter: 1

        .. grid-item::
            :columns: 4

            .. grid:: 1
                :gutter: 1

                **Basic OPE (Continuous)**

            .. grid:: 1
                :gutter: 1

                .. grid-item-card:: 
                    :img-background: /_static/images/ope_policy_value_basic.png
                    :shadow: none

        .. grid-item::
            :columns: 8

            .. grid:: 1
                :gutter: 1
                :padding: 1

                .. grid-item:: 

                    * Custom Basic OPE estimators
                    * Custom Cumulative Distribution OPE estimators

.. _multiple_dataset_example:

Handling Multiple Datasets:
~~~~~~~~~~

.. card::
    :link: /documentation/examples/multiple
    :link-type: doc

    .. grid::
        :gutter: 1

        .. grid-item::
            :columns: 4

            .. grid:: 1
                :gutter: 1

                **Basic OPE (Continuous)**

            .. grid:: 1
                :gutter: 1

                .. grid-item-card:: 
                    :img-background: /_static/images/multiple_topk_thumbnail.png
                    :shadow: none

        .. grid-item::
            :columns: 8

            .. grid:: 1
                :gutter: 1
                :padding: 1

                .. grid-item:: 

                    * Logged datasets and inputs
                    * (Basic) Off-Policy Evaluation
                    * Cumulative Distribution Off-Policy Evaluation
                    * Off-Policy Selection
                    * Assessments of OPE and OPS

Handling Real-World Datasets:
~~~~~~~~~~

.. card::
    :link: /documentation/examples/real_world
    :link-type: doc

    .. grid::
        :gutter: 1

        .. grid-item::
            :columns: 4

            .. grid:: 1
                :gutter: 1

                **Basic OPE (Continuous)**

            .. grid:: 1
                :gutter: 1

                .. grid-item-card:: 
                    :img-background: /_static/images/ope_policy_value_basic.png
                    :shadow: none

        .. grid-item::
            :columns: 8

            .. grid:: 1
                :gutter: 1
                :padding: 1

                .. grid-item:: 

                    * Logged dataset
                    * Input dict

.. raw:: html

    <div class="white-space-5px"></div>

.. seealso::

    For the data collection and integration with d3rlpy in policy learning, please also refer to :doc:`this page </documentation/learning_implementation>`.

.. seealso::

    The comprehensive quickstart examples with the provided sub-packages are available in the GitHub repository:

    * `Quickstart with RTBGym <https://github.com/hakuhodo-technologies/scope-rl/tree/main/examples/quickstart/rtb>`_
    * `Quickstart with RECGym <https://github.com/hakuhodo-technologies/scope-rl/tree/main/examples/quickstart/rec>`_
    * `Quickstart with BasicGym <https://github.com/hakuhodo-technologies/scope-rl/tree/main/examples/quickstart/basic>`_

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
                :link: /documentation/quickstart
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Quickstart**

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
                :link: /documentation/examples/basic_ope
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Basic OPE**
