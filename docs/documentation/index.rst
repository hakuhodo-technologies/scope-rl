OFRL; a Python library for offline reinforcement learning, off-policy evaluation, and selection
===================================

.. card:: logo
    :img-top: ../_static/images/logo.png
    :text-align: center
    
Overview
~~~~~~~~~~
*OFRL* is an open-source Python library for offline Reinforcement Learning (RL) and Off-Policy Evaluation and Selection (OPE/OPS).
This library aims to facilitate an easy, flexible and reliable experiment in offline RL research, as well as to provide a streamlined implementation for practitioners.
OFRL includes a series of modules to implement synthetic dataset generation and dataset preprocessing and methods for conducting and evaluating OPE/OPS.

OFRL is applicable to any RL environment with `OpenAI Gym <https://gym.openai.com>`_ or `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_-like interface.
The library is also compatible with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, which provides the algorithm implementation of both online and offline RL methods.

Our software facilitates implementation, evaluation and algorithm comparison related to the following research topics:

* **Offline Reinforcement Learning**: 
   Offline RL aims to learn a new policy from only offline logged data collected by a behavior policy. OFRL enables a flexible experiment using customized dataset on diverse environments collected by various behavior policies.

* **Off-Policy Evaluation**: 
   OPE aims to evaluate the policies of a counterfactual policy using only offline logged data. OFRL supports the basic implementations of OPE estimators and streamline the experimental procedure to evaluate OPE estimators.

* **Off-Policy Selection**: 
   OPS aims to select the best policy from several candidate policies using offline logged data. OFRL supports the basic implementations of OPS methods and provide some metrics to evaluate OPS result.

.. card:: workflow
    :img-top: ../_static/images/workflow.png
    :text-align: center

.. note::

   This documentation aims to provide a gentle introduction to offline RL and OPE/OPS in the following steps.

   1. Explain the basic concepts in :doc:`Overview (online/offline RL) <online_offline_rl>` and :doc:`Overview (OPE/OPS) <ope_ops>`.
   2. Provide a variety of examples of conducting offline RL and OPE/OPS in practical problem settings in :doc:`Quickstart <quickstart>` and :doc:`Tutorial <tutorial>`.
   3. Describe the algorithms and implementations in detail in :doc:`Supported Implementation <evaluation_implementation>` and :doc:`Package Reference <ofrl_api>`.

   You can also find the distinctive features of OFRL in :doc:`distinctive_features`

Implementation
~~~~~~~~~~

Data Collection Policy and Offline RL
----------
OFRL override `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s implementation for the base algorithm.
We provide a wrapper class for transforming the policy into a stochastic policy as follows.

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

OPRL provides a variety of OPE estimators both in discrete and continuous action spaces.
Moreover, OFRL also implements meta class to handle OPE with multiple estimators at once and provide generic classes of OPE estimators to facilitate research development.

Basic estimators
^^^^^^
* Direct Method (DM) :cite:`beygelzimer2009offset` :cite:`le2019batch`
* Trajectory-wise Importance Sampling (TIS) :cite:`precup2000eligibility`
* Per-Decision Importance Sampling (PDIS) :cite:`precup2000eligibility`
* Doubly Robust (DR) :cite:`jiang2016doubly` :cite:`thomas2016data`
* Self-Normalized Trajectory-wise Importance Sampling (SNTIS) :cite:`precup2000eligibility` :cite:`kallus2020optimal`
* Self-Normalized Per-Decision Importance Sampling (SNPDIS) :cite:`precup2000eligibility` :cite:`kallus2020optimal`
* Self-Normalized Doubly Robust (SNDR) :cite:`jiang2016doubly` :cite:`thomas2016data` :cite:`kallus2020optimal`

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
* Augmented Lagrangian Method (ALM) :cite:`yang2020off`
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

OFRL also provides cumulative distribution OPE estimators, which enables practitioners to evaluate various risk metrics (e.g., conditional value at risk) for safety assessment.
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

Finally, OFRL also standardizes the evaluation protocol of OPE in two axes, first by measuring the accuracy of OPE over the whole candidate policies, 
and second by evaluating the gains and costs in top-k deployment (e.g., the best and worst performance in top-k deployment).
The streamlined implementations and visualization of OPS class provide informative insights on offline RL and OPE performance.

OPE metrics
^^^^^^
* Mean Squared Error :cite:`paine2020hyperparameter` :cite:`voloshin2021empirical` :cite:`fu2021benchmarks`
* Spearman's Rank Correlation Coefficient :cite:`paine2020hyperparameter` :cite:`fu2021benchmarks`
* Regret :cite:`paine2020hyperparameter` :cite:`fu2021benchmarks`
* Type I and Type II Error Rates

OPS metrics (performance of top k deployment policies)
^^^^^^
* {Best/Worst/Mean} of {policy value/conditional value at risk/lower quartile}
* Safety violation rate

.. seealso::

   We also provide configurative RL environments as sub-packages of this library.

   * :doc:`RTBGym <rtbgym_about>`: Real-Time Bidding (RTB) of online advertisement


Citation
~~~~~~~~~~
If you use our pipeline in your work, please cite our paper below.

| **Title** [`arXiv <>`_] [`Proceedings <>`_]
| Authors.

.. code-block::

   @article{kiyohara2023xxx
      title={},
      author={},
      journal={},
      year={},
   }

Contact
~~~~~~~~~~
For any question about the paper and pipeline, feel free to contact: kiyohara.h.aa@m.titech.ac.jp

Contribution
~~~~~~~~~~
Any contributions to OFRL are more than welcome!
Please refer to `CONTRIBUTING.md <>`_ for general guidelines how to contribute to the project.

Table of Contents
~~~~~~~~~~

.. toctree::
   :maxdepth: 3
   :caption: Getting Started:

   installation
   quickstart
   _autogallery/index
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

.. toctree::
   :maxdepth: 1
   :caption: Sub-packages:

   Gallery of Sub-packages <subpackages/index>
   subpackages/rtbgym_about
   subpackages/recgym_about

.. toctree::
   :maxdepth: 2
   :caption: Package References:

   ofrl_api
   subpackages/rtbgym_api
   subpackages/recgym_api

.. toctree::
   :maxdepth: 1
   :caption: See also:

   Github <https://github.com/negocia-inc/ofrl>
   LICENSE <https://github.com/negocia-inc/ofrl/blob/main/LICENSE>
   frequently_asked_questions
   News <news>
   Release Notes <https://github.com/negocia-inc/ofrl/releases>
   Proceedings <https://github.com/negocia-inc/ofrl/404>
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
                **Why_OFRL?**

            .. grid-item-card::
                :link: /documentation/index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Quickstart**
