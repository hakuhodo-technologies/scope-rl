.. OFRL documentation master file, created by
   sphinx-quickstart on Thu Jan 20 15:25:38 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OFRL; a Python library for offline reinforcement learning, off-policy evaluation, and selection
===================================

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

This website contains pages with example implementations that demonstrates the usage of this library.
The package reference page consists of the full reference documentation for the currently implemented modules.

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

Estimation Methods of Marginal Importance Weights
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
----------
* Bootstrap :cite:`thomas2015improvement` :cite:`hanna2017bootstrapping`
* Hoeffding :cite:`thomas2015evaluation`
* (Empirical) Bernstein :cite:`thomas2015evaluation` :cite:`thomas2015improvement`
* Student T-test :cite:`thomas2015improvement`

Cumulative Distribution OPE
----------
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

In addition to the offline RL/OPE related resources, we provide a configurative RL environment for Real-Time Bidding (RTB) as a sub-package of this library.
Please refer to `RTBGym's documentation <>`_ for the details.

Citation
~~~~~~~~~~
If you use our pipeline in your work, please cite our paper below.

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
   tutorial
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

   rtbgym_about

.. toctree::
   :maxdepth: 2
   :caption: Package References:

   ofrl
   rtbgym

.. toctree::
   :maxdepth: 1
   :caption: See also:

   Github <https://github.com/negocia-inc/ofrl>
   LICENSE <https://github.com/negocia-inc/ofrl/blob/main/LICENSE>
   frequently_asked_questions
   release_notes
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
