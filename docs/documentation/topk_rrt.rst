:html_theme.sidebar_secondary.remove:

==========
Top-:math:`k` Risk Return Tradeoff Metircs
==========

We describe the **top-** :math:`k` **Risk-Return Tradeoff (RRT)** for evaluating the off-policy selection (OPS) result.
Note that, for the basic problem formulation of Off-Policy Evaluation and Selection (OPE/OPS), please also refer to :doc:`Overview (OPE/OPS) <ope_ops>`.

.. seealso::

    The top-:math:`k` RRT metrics are the main contribution of our paper **"SCOPE-RL: Towards Risk-Return Assessments of Off-Policy Evaluation in Offline RL."** 
    A preprint is available at `arXiv <>`_.

Background
~~~~~~~~~~
While OPE is useful for estimating the policy performance of a new policy using offline logged data, 
OPE sometimes produces erroneous estimation due to *counterfactual estimation* and *distribution shift* between the behavior and evaluation policies.
Therefore, in practical situations, we cannot solely rely on OPE results to choose the production policy, but instead, combine OPE results and online A/B tests for policy evaluation and selection :cite:`kurenkov2022showing`.
Specifically, the practical workflow often begins by filtering out poor-performing policies based on OPE results, then conducting A/B tests on the remaining top-:math:`k`
policies to identify the best policy based on reliable online evaluation, as illustrated in the following figure.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ops_workflow.png
    :text-align: center

    Practical workflow of policy evaluation and selection

.. raw:: html

    <div class="white-space-20px"></div>

In the following sections, we consider this practical workflow and aim to evaluate the key properties of OPE estimators in selecting90
the top-k candidate policies for deployment in subsequent online A/B tests.


Issues of existing evaluation protocols of OPE/OPS
~~~~~~~~~~
To evaluate and compare the performance of OPE estimators, the following three metrics are often used:

* Mean Squared Error (MSE) :cite:`voloshin2021empirical`: This metric measures the estimation accuracy of OPE esimator :math:`\hat{J}`.
* Rank Correlation (RankCorr) :cite:`paine2020hyperparameter, fu2021benchmarks`: This metric measures how well the ranking of the candidate estimators is preserved in the OPE.
* Regret @ :math:`k` :cite:`doroudi2017importance`: This metric measures how well the best policy among the top-:math:`k` policies selected by an estimator performs. In particular, Regret@1 measures performance difference between the true best policy and the best policy estimated by the OPE estimator.

In the above metrics, MSE measures the accuracy of OPE estimation, while the latter two assess the accuracy of downstream policy selection tasks. 
By combining these metrics, especially the latter two, we can quantify how likely an OPE estimator can choose a near-optimal policy in policy selection when solely relying on the OPE result. 
However, a critical shortcoming of the current evaluation protocol is that these metrics do not assess potential risks experienced during online A/B tests in more practical two-stage selection combined with online A/B tests. 
For instance, let us now consider the following toy situation as an illustrative example. 

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/toy_example_1.png
    :text-align: center

    Toy example 1: overestimation vs. underestimation

.. raw:: html

    <div class="white-space-20px"></div>

In the above example, all three metrics report the same metric values for both estimators X and Y. 
However, since estimator X underestimates near-optimal policies and estimator Y overestimates poor-performing policies, there exists a non-negligible difference in their risk-return tradeoff. 
Unfortunately, all the existing metrics fails to detect by the difference between estimator X and Y as follows. 

============== ============ ============
(estimator)    X            Y
**MSE**        11.3         11.3
**RankCorr**   0.413        0.413
**Regret@3**   0.0          0.0
============== ============ ============

.. raw:: html

    <div class="white-space-5px"></div>

This motivates the development of a new evaluation protocol that can quantify the risk-return tradeoff of OPE estimators.


Evaluating the top-:math:`k` risk-return tradeoff in policy deployment
~~~~~~~~~~
Motivated by the lack of comprehensive risk-return assessments in OPE in existing literature, we propose a new evaluation protocol called top-:math:`k` Risk-Return Tradeoff (RRT). 
Our key idea is to view the set of top-:math:`k` candidate policies selected by an OPE estimator as its *policy portfolio*, drawing inspiration from risk-return assessments in finance :cite:`connor2010portfolio`. 
Specifically, we measure the risk, return, and efficiency of an estimator using the following metrics:

* **best @** :math:`k` (*return, the larger the better*): This metric reports the best policy performance among the selected top-:math:`k` policies. Similar to regret@ :math:`k`, it measures how well an OPE estimator identifies a high-performing policy.
* **worst @** :math:`k`, **mean@**:math:`k` (*risk, the larger the better*): These metrics report the worst and mean performance among the top-:math:`k` policies selected by an estimator. These metrics quantify how likely an OPE estimator mistakenly chooses poorly-performing policies as promising.
* **safety violation rate @** :math:`k` (*risk, the smaller the better*): This metric reports the probability of deployed policies violating a pre-defined safety requirement (such as the performance of the behavior policy).
* **Sharpe ratio @** :math:`k` (*efficiency, the larger the better*): Analogous to the original Sharpe ratio used in the field of finance :cite:`sharpe1998sharpe`, we define this metric as follows.

.. math::

        \text{sharpe_ratio@}k := \frac{\text{best@}k - J(\pi_b)}{\text{std@}k},
    
here std@ :math:`k` is the standard deviation of policy value among the top-:math:`k` policies and :math:`J(\pi_b)` is the policy value of the behavior policy. 
This metric values the return (best@ :math:`k`) over the risk-free baseline (:math:`J(\pi_b)`) while taking risk into consideration by measuring the standard deviation.

Reporting these risk, return, and efficiency metrics under varying numbers of selected policies (online evaluation budgets) :math:`k` is crucial to evaluate and understand the risk-return tradeoff of OPE estimators. 
Below, we showcase how our top-:math:`k` RRT provides valuable insights for comparing OPE estimators in two practical scenarios.

.. raw:: html

    <div class="white-space-5px"></div>

**Toy example 1: Overestimation vs. Underestimation.**
The first case is the previously mentioned example of evaluating estimator X (which underestimates the near-best policy) and estimator Y (which overestimates the poor-performing policies) in the above figure.
While the conventional metrics fail to distinguish the two estimators, our top-:math:`k` RRT metrics reports the following results: 

.. card:: 
    :img-top: ../_static/images/topk_toy1.png
    :text-align: center

    Top-:math:`k` RRT metrics for the toy example 1

.. raw:: html

    <div class="white-space-20px"></div>

In this case, the plots of the "worst" and "mean" policy values in show that Y is riskier than X, particularly when the online evaluation budget :math:`k` is small. 
In contrast, the plot of the "best" policy value demonstrates that the return of the top-:math:`k` deployment is the same for both X and Y, suggesting that in terms of risk-return tradeoff, X is preferable to Y in this specific example. 
The Sharpe ratio also follows the above analysis by scoring X higher than Y, especially for small values of :math:`k`.

.. raw:: html

    <div class="white-space-5px"></div>

**Toy example 2: Conservative vs. High-Stakes.**
Another example involves evaluating a conservative OPE (estimator W, which always underestimates) and a uniform random OPE (estimator Z) as shown in the following figure. 

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/toy_example_2.png
    :text-align: center

    Toy example 2: conservative vs. high-stakes

.. raw:: html

    <div class="white-space-20px"></div>

In this case, the typical metrics again give W and Z almost the same values, making it difficult to decide which estimator to use in practical situations.

============== ============ ============
(estimator)    W            Z
**MSE**        60.1         58.6
**RankCorr**   0.079        0.023
**Regret@1**   9.0          9.0
============== ============ ============

.. raw:: html

    <div class="white-space-5px"></div>

In contrast, our top-:math:`k` RRT metrics report the following results, which clearly distinguishes the two estimators:

.. card:: 
    :img-top: ../_static/images/topk_toy2.png
    :text-align: center

    Top-:math:`k` RRT metrics (risk, return) for the toy example 2

.. raw:: html

    <div class="white-space-5px"></div>

.. card:: 
    :img-top: ../_static/images/sharpe_ratio_2.png
    :text-align: center

    Top-:math:`k` RRT metrics (efficiency) for the toy example 2

.. raw:: html

    <div class="white-space-20px"></div>

In the plots of the "best" and "worst" policy values, we observe that Z quickly deploys both near-best and near-worst policies with small :math:`k`, while W is slower in deploying them. 
It is thus clear that W is a low-risk, low-return estimator, while Z is high-risk, high-return. 
It is also reasonable that the efficiency (Sharpe ratio) is competitive between W and Z, and the superiority between the two can change with the behavior policy and its value :math:`J(\pi_b)`. 
Top-:math:`k` RRT is thus much more informative with respect to the risk-return assessments of OPE.

OPE benchmarks with top-:math:`k` RRT
~~~~~~~~~~

.. grid::
    :margin: 0

    .. grid-item::
        :columns: 3
        :margin: 0
        :padding: 0

        .. grid::
            :margin: 0

            .. grid-item-card::
                :link: ope_ops
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Problem Formulation**

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
                :link: quickstart
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Quickstart**
