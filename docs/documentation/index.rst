:html_theme.sidebar_secondary.remove:

SCOPE-RL
===================================

.. raw:: html

    <h3>A Python library for offline reinforcement learning, off-policy evaluation, and selection</h3>


.. .. card:: logo
..     :img-top: ../_static/images/logo.png
..     :text-align: center
    
Overview
~~~~~~~~~~
*SCOPE-RL* is an open-source Python library for offline Reinforcement Learning (RL) and Off-Policy Evaluation and Selection (OPE/OPS).
This library aims to facilitate an easy, flexible and reliable experiment in offline RL research, as well as to provide a streamlined implementation for practitioners.
SCOPE-RL includes a series of modules to implement synthetic dataset generation and dataset preprocessing and methods for conducting and evaluating OPE/OPS.

SCOPE-RL is applicable to any RL environment with `OpenAI Gym <https://github.com/openai/gym>`_ or `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_-like interface.
The library is also compatible with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, which provides the algorithm implementation of both online and offline RL methods.

Our software facilitates implementation, evaluation and algorithm comparison related to the following research topics:

.. card:: 
   :width: 75%
   :margin: auto
   :img-top: ../_static/images/offline_rl_workflow.png
   :text-align: center

   workflow of offline RL, OPE, and online A/B testing

.. raw:: html

    <div class="white-space-20px"></div>

* **Offline Reinforcement Learning**: 
   Offline RL aims to learn a new policy from only offline logged data collected by a behavior policy. SCOPE-RL enables a flexible experiment using customized dataset on diverse environments collected by various behavior policies.

* **Off-Policy Evaluation**: 
   OPE aims to evaluate the policies of a counterfactual policy using only offline logged data. SCOPE-RL supports the basic implementations of OPE estimators and streamline the experimental procedure to evaluate OPE estimators.

* **Off-Policy Selection**: 
   OPS aims to select the top-:math:`k` policies from several candidate policies using offline logged data. Typically, the final production policy is chosen based on the online A/B tests results of the selected top-:math:`k` policies.
   SCOPE-RL supports the basic implementations of OPS methods and provide some metrics to evaluate OPS result.

.. note::

   This documentation aims to provide a gentle introduction to offline RL and OPE/OPS in the following steps.

   .. 2. Provide a variety of examples of conducting offline RL and OPE/OPS in practical problem settings in :doc:`Quickstart <quickstart>` and :doc:`Tutorial <tutorial>`.
   1. Explain the basic concepts in :doc:`Overview (online/offline RL) <online_offline_rl>` and :doc:`Overview (OPE/OPS) <ope_ops>`.
   2. Provide a variety of examples of conducting offline RL and OPE/OPS in practical problem settings in :doc:`Quickstart <quickstart>`.
   3. Describe the algorithms and implementations in detail in :doc:`Supported Implementation <evaluation_implementation>` and :doc:`Package Reference <scope_rl_api>`.

   **You can also find the distinctive features of SCOPE-RL here:** :doc:`distinctive_features`

Implementation
~~~~~~~~~~

Data Collection Policy and Offline RL
----------
SCOPE-RL override `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s implementation for the base algorithms.
We provide a class to handle synthetic dataset deneration, off-policy learning with multiple algorithms, and
wrapper classes for transforming the policy into a stochastic policy as follows.

Meta class
^^^^^^
* SyntheticDataset
* OffPolicyLearning

Discrete
^^^^^^
* Epsilon Greedy
* Softmax

Continuous
^^^^^^
* Gaussian
* Truncated Gaussian

Basic OPE
----------

.. card:: 
    :img-top: ../_static/images/ope_policy_value_basic.png
    :text-align: center
    
    Policy Value Estimated by OPE Estimators

SCOPE-RL provides a variety of OPE estimators both in discrete and continuous action spaces.
Moreover, SCOPE-RL also implements meta class to handle OPE with multiple estimators at once and provide generic classes of OPE estimators to facilitate research development.

Basic estimators
^^^^^^
* Direct Method (DM) :cite:`beygelzimer2009offset` :cite:`le2019batch`
* Trajectory-wise Importance Sampling (TIS) :cite:`precup2000eligibility`
* Per-Decision Importance Sampling (PDIS) :cite:`precup2000eligibility`
* Doubly Robust (DR) :cite:`jiang2016doubly` :cite:`thomas2016data`
* Self-Normalized Trajectory-wise Importance Sampling (SNTIS) :cite:`precup2000eligibility` :cite:`kallus2019intrinsically`
* Self-Normalized Per-Decision Importance Sampling (SNPDIS) :cite:`precup2000eligibility` :cite:`kallus2019intrinsically`
* Self-Normalized Doubly Robust (SNDR) :cite:`jiang2016doubly` :cite:`thomas2016data` :cite:`kallus2019intrinsically`

State Marginal Estimators
^^^^^^
* State Marginal Direct Method (SM-DM) :cite:`uehara2020minimax`
* State Marginal Importance Sampling (SM-IS) :cite:`liu2018breaking` :cite:`uehara2020minimax` :cite:`yuan2021sope`
* State Marginal Doubly Robust (SM-DR) :cite:`liu2018breaking` :cite:`uehara2020minimax` :cite:`yuan2021sope`
* State Marginal Self-Normalized Importance Sampling (SM-SNIS) :cite:`liu2018breaking` :cite:`uehara2020minimax` :cite:`yuan2021sope`
* State Marginal Self-Normalized Doubly Robust (SM-SNDR) :cite:`liu2018breaking` :cite:`uehara2020minimax` :cite:`yuan2021sope`

State-Action Marginal Estimators
^^^^^^
* State-Action Marginal Importance Sampling (SAM-IS) :cite:`uehara2020minimax` :cite:`yuan2021sope`
* State-Action Marginal Doubly Robust (SAM-DR) :cite:`uehara2020minimax` :cite:`yuan2021sope`
* State-Action Marginal Self-Normalized Importance Sampling (SAM-SNIS) :cite:`uehara2020minimax` :cite:`yuan2021sope`
* State-Action Marginal Self-Normalized Doubly Robust (SAM-SNDR) :cite:`uehara2020minimax` :cite:`yuan2021sope`

Double Reinforcement Learning
^^^^^^
* Double Reinforcement Learning :cite:`kallus2020double`

Weight and Value Learning Methods
^^^^^^
* Augmented Lagrangian Method (ALM/DICE) :cite:`yang2020off`
   * BestDICE :cite:`yang2020off`
   * GradientDICE :cite:`zhang2020gradientdice`
   * GenDICE :cite:`zhang2020gendice`
   * AlgaeDICE :cite:`nachum2019algaedice`
   * DualDICE :cite:`nachum2019dualdice`
   * MQL/MWL :cite:`uehara2020minimax`
* Minimax Q-Learning and Weight Learning (MQL/MWL) :cite:`uehara2020minimax`

High Confidence OPE
^^^^^^
* Bootstrap :cite:`thomas2015improvement` :cite:`hanna2017bootstrapping`
* Hoeffding :cite:`thomas2015evaluation`
* (Empirical) Bernstein :cite:`thomas2015evaluation` :cite:`thomas2015improvement`
* Student T-test :cite:`thomas2015improvement`

Cumulative Distribution OPE
----------

.. card:: 
    :img-top: ../_static/images/ope_cumulative_distribution_function.png
    :text-align: center

    Cumulative Distribution Function Estimated by OPE Estimators

SCOPE-RL also provides cumulative distribution OPE estimators, which enables practitioners to evaluate various risk metrics (e.g., conditional value at risk) for safety assessment.
Meta class and generic abstract class are available also for cumulative distribution OPE.

Estimators
^^^^^^
* Direct Method (DM) :cite:`huang2021off`
* Trajectory-wise Importance Sampling (TIS) :cite:`huang2021off` :cite:`chandak2021universal`
* Trajectory-wise Doubly Robust (TDR) :cite:`huang2021off`
* Self-Normalized Trajectory-wise Importance Sampling (SNTIS) :cite:`huang2021off` :cite:`chandak2021universal`
* Self-Normalized Trajectory-wise Doubly Robust (SNDR) :cite:`huang2021off`

Metrics of Interest
^^^^^^
* Cumulative Distribution Function (CDF)
* Mean (i.e., policy value)
* Variance
* Conditional Value at Risk (CVaR)
* Interquartile Range

Off-Policy Selection Metrics
----------

.. card:: 
    :img-top: ../_static/images/ops_topk_lower_quartile.png
    :text-align: center
    
    Comparison of the Top-k Statistics of 10% Lower Quartile of Policy Value

Finally, SCOPE-RL also standardizes the evaluation protocol of OPE in two axes, first by measuring the accuracy of OPE over the whole candidate policies, 
and second by evaluating the gains and costs in top-k deployment (e.g., the best and worst performance in top-k deployment).
The streamlined implementations and visualization of OPS class provide informative insights on offline RL and OPE performance.

OPE metrics
^^^^^^
* Mean Squared Error :cite:`paine2020hyperparameter` :cite:`voloshin2021empirical` :cite:`fu2021benchmarks`
* Spearman's Rank Correlation Coefficient :cite:`paine2020hyperparameter` :cite:`fu2021benchmarks`
* Regret :cite:`paine2020hyperparameter` :cite:`fu2021benchmarks`
* Type I and Type II Error Rates

OPS metrics (performance of top :math:`k` deployment policies)
^^^^^^
* {Best/Worst/Mean/Std} of {policy value/conditional value at risk/lower quartile}
* Safety violation rate
* Sharpe ratio (our proposal)

.. seealso::

   We also provide configurative RL environments as sub-packages of this library.

   * :doc:`RTBGym <subpackages/rtbgym_about>`: Real-Time Bidding (RTB) of online advertisement
   * :doc:`RECGym <subpackages/recgym_about>`: Recommendation in e-commerce
   * :doc:`BasicGym <subpackages/basicgym_about>`: Basic environment


Citation
~~~~~~~~~~
If you use our pipeline in your work, please cite our paper below.

| Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.
| **Towards Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning**
| (a preprint coming soon..)

.. code-block::

   @article{kiyohara2023towards,
      author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
      title = {Towards Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning},
      journal = {A github repository},
      pages = {xxx--xxx},
      year = {2023},
   }

Google Group
~~~~~~~~~~
Feel free to follow our updates from our google group: `scope-rl@googlegroups.com <https://groups.google.com/g/scope-rl>`_.

Contact
~~~~~~~~~~
For any question about the paper and pipeline, feel free to contact: hk844@cornell.edu

Contribution
~~~~~~~~~~
Any contributions to SCOPE-RL are more than welcome!
Please refer to `CONTRIBUTING.md <https://github.com/hakuhodo-technologies/scope-rl/CONTRIBUTING.md>`_ for general guidelines how to contribute to the project.

Table of Contents
~~~~~~~~~~

.. toctree::
   :maxdepth: 3
   :caption: Getting Started:

   installation
   quickstart
   .. _autogallery/index
   distinctive_features

.. toctree::
   :maxdepth: 3
   :caption: Online & Offline RL:

   online_offline_rl
   learning_implementation

.. toctree::
   :maxdepth: 3
   :caption: Off-Policy Evaluation & Selection:

   ope_ops
   evaluation_implementation
   visualization

.. toctree::
   :maxdepth: 3
   :caption: Our Proposal:

   sharpe_ratio

.. toctree::
   :maxdepth: 1
   :caption: Sub-packages:

   Gallery of Sub-packages <subpackages/index>

.. toctree::
   :maxdepth: 2
   :caption: Package References:

   scope_rl_api
   subpackages/rtbgym_api
   subpackages/recgym_api
   subpackages/basicgym_api

.. toctree::
   :maxdepth: 1
   :caption: See also:

   Github <https://github.com/hakuhodo-technologies/scope-rl>
   LICENSE <https://github.com/hakuhodo-technologies/scope-rl/blob/main/LICENSE>
   frequently_asked_questions
   News <news>
   Release Notes <https://github.com/hakuhodo-technologies/scope-rl/releases>
   Proceedings <https://github.com/hakuhodo-technologies/scope-rl/404>
   references

.. grid::
    :margin: 0

    .. grid-item::
        :columns: 3
        :margin: 0
        :padding: 0

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
                :link: distinctive_features
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Why SCOPE-RL?**

            .. grid-item-card::
                :link: /documentation/quickstart
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Quickstart**
