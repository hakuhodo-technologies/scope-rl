:html_theme.sidebar_secondary.remove:

==========
Why SCOPE-RL?
==========

Motivation
~~~~~~~~~~

Sequential decision making is ubiquitous in many real-world applications including recommender, search, and advertising systems. 
While a *logging* or *behavior* policy interacts with users to optimize such sequential decision making, it also produces logged data valuable for learning and evaluating future policies.
For example, a search engine often records a user's search query (state), the document presented by the behavior policy (action), the user response such as a click observed for the presented document (reward), and the next user behavior including a more specific search query (next state). 
Making most of these logged data to evaluate a counterfactual policy is particularly beneficial in practice, as it can be a safe and cost-effective substitute for online A/B tests. 

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/real_world_interaction.png
    :text-align: center

    An example of sequential decision makings in the real-world

.. raw:: html

    <div class="white-space-20px"></div>

**Off-Policy Evaluation (OPE)**, which studies how to accurately estimate the performance of an *evaluation* policy using only offline logged data, is thus gaining growing interest.
Especially in the reinforcement learning (RL) setting, a variety of theoretically grounded OPE estimators have been proposed to accurately estimate the expected reward :cite:`precup2000eligibility, jiang2016doubly, thomas2016data, farajtabar2018more, le2019batch, kallus2019intrinsically,liu2018breaking, uehara2020minimax`. 
Moreover, several recent work on cumulative distribution OPE :cite:`huang2021off, huang2022off, chandak2021universal` also aim at estimating the cumulative distribution function (CDF) and risk functions (e.g., variance, conditional value at risk (CVaR), and interquartile range) of an evaluation policy. 
These risk functions provide informative insights on policy performance especially from safety perspectives, which are thus crucial for practical decision making.

Unfortunately, despite these recent advances in OPE of RL policies, only a few existing platforms :cite:`fu2021benchmarks, voloshin2021empirical` are available for extensive OPE studies and benchmarks for plactical applications. 
Moreover, those existing platform lacks the following important properties:

Most *offline RL* plaforms:

* ... provide only a basic OPE estimators.

Existing *OPE* platforms:

* ... have a limited flexibly of environment and the choice of offline RL methods (i.e., **limited compatibility with gym/gymnasium and offline RL libraries**).
* ... support only the standard OPE framework and lack the implementation of **cumulative distribution OPE**.
* ... only focus on the accuracy of OPE/OPS and do not take the **risk-return tradeoff** of the policy selection into account.
* ... do not support user-friendly **visualization tools** to interpret the OPE results.
* ... do not provide well-described **documentations**.

It is critical to fill the above gaps to further facilitate the OPE research and its practical applications.
This is why we build **SCOPE-RL, the first end-to-end platform for offline RL and OPE, which puts an emphasis on the OPE modules**.


Key contributions
~~~~~~~~~~

The distinctive features of our SCOPE-RL platform are summarized as follows.

* :ref:`feature_end_to_end`

* :ref:`feature_variety_ope`

* :ref:`feature_cd_ope`

* :ref:`feature_sharpe_ratio`

Below, we describe each advantage one by one. 
Note that, for a quick comparison with the exising platforms, please refer to :ref:`the following section <feature_comparison>`.

.. _feature_end_to_end:

End-to-end implementation of Offline RL and OPE
----------

While existing platforms support flexible implementations on either offline RL or OPE, we aim to bridge the offline RL and OPE processes and streamline an end-to-end procedure for the first time.
Specifically, as shown in the bottom figure, our module mainly consists of the following four modules:

.. card:: 
   :width: 75%
   :margin: auto
   :img-top: ../_static/images/scope_workflow.png
   :text-align: center

   Workflow of offline RL and OPE streamlined by SCOPE-RL

.. raw:: html

    <div class="white-space-20px"></div>

* Dataset module
* Off-Policy Learning (OPL) module
* Off-Policy Evaluation (OPE) module
* Off-Policy Selection (OPS) module

First, the *Dataset* module handles the data collection from RL environments.
Since our Dataset module is compatible with `OpenAI Gym <https://github.com/openai/gym>`_ or `Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_-like environments, SCOPE-RL is applicable to a variety of environmental settings.
Moreover, SCOPE-RL supports compatibility with `d3rlpy <https://github.com/takuseno/d3rlpy>`_, which provides implementations of various online and offline RL algorithms. 
This also allows us test the performance of offline RL and OPE with various behavior policies or other experimental settings.

Next, the *OPL* module provides an easy-to-handle wrapper for learning new policies with various offline RL algorithms.
While `d3rlpy <https://github.com/takuseno/d3rlpy>`_ has already supported user-friedly API, their implementation is basically intended to use offline RL algorithms one by one.
Therefore, to further make the end-to-end offline RL and OPE process smoothly connected, our OPL wrapper enables to handle multiple datasets and multiple algorithms in a single class.

.. Please refer to :ref:`this page <>` for the details. 

Finally, the *OPE* and *OPS* modules are particularly our focus. 
As we will review in the following sub-sections, we implement a variety of OPE estimators from the basic choices :cite:`le2019batch, precup2000eligibility, jiang2016doubly, thomas2016data`, 
advanced ones :cite:`kallus2020double, uehara2020minimax, liu2018breaking, yang2020off, yuan2021sope`, and estimators for the cutting-edge cumulative distribution OPE :cite:`huang20210ff, huang2022off, chandak2021universal`.
Moreover, we provide the meta-class to handle OPE/OPS experiments and the abstract base implementation of OPE estimators. 
This allows researchers to quickly test their own algorithms with this platform and also help practitioners empirically learn the property of various OPE methods.

.. _feature_variety_ope:

Variety of OPE estimators and evaluation protocol of OPE
----------

SCOPE-RL provides the implementation of various OPE estimators in both discrete and continuous action settings.
In the standard OPE, which aim to estimate the expected performance of the given evaluation policy, we implement the OPE estimators listed below. 
These implementations are as comprehensive as the existing platforms for OPE including :cite:`fu2021benchmarks, voloshin2021empirical`.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_policy_value_variety.png
    :text-align: center

    Example of estimating policy value using various OPE estimators


.. raw:: html

    <div class="white-space-20px"></div>

.. seealso::

    The detailed descriptions of each estimator and evaluation metrics are in :doc:`Supported Implemetation (OPE/OPS) <evaluation_implementation>`.


.. raw:: html

    <div class="white-space-5px"></div>


**Basic estimators**

* (abstract base)
* Direct Method (DM) :cite:`beygelzimer2009offset` :cite:`le2019batch`
* Trajectory-wise Importance Sampling (TIS) :cite:`precup2000eligibility`
* Per-Decision Importance Sampling (PDIS) :cite:`precup2000eligibility`
* Doubly Robust (DR) :cite:`jiang2016doubly` :cite:`thomas2016data`
* Self-Normalized Trajectory-wise Importance Sampling (SNTIS) :cite:`precup2000eligibility` :cite:`kallus2020optimal`
* Self-Normalized Per-Decision Importance Sampling (SNPDIS) :cite:`precup2000eligibility` :cite:`kallus2020optimal`
* Self-Normalized Doubly Robust (SNDR) :cite:`jiang2016doubly` :cite:`thomas2016data` :cite:`kallus2020optimal`

.. raw:: html

    <div class="white-space-5px"></div>

**State Marginal Estimators**

* (abstract base)
* State Marginal Direct Method (SM-DM) :cite:`uehara2020minimax`
* State Marginal Importance Sampling (SM-IS) :cite:`liu2018breaking` :cite:`uehara2020minimax`
* State Marginal Doubly Robust (SM-DR) :cite:`liu2018breaking` :cite:`uehara2020minimax`
* State Marginal Self-Normalized Importance Sampling (SM-SNIS) :cite:`liu2018breaking` :cite:`uehara2020minimax`
* State Marginal Self-Normalized Doubly Robust (SM-SNDR) :cite:`liu2018breaking` :cite:`uehara2020minimax`
* Spectrum of Off-Policy Evaluation (SOPE) :cite:`yuan2021sope`

.. raw:: html

    <div class="white-space-5px"></div>

**State-Action Marginal Estimators**

* (abstract base)
* State-Action Marginal Importance Sampling (SAM-IS) :cite:`uehara2020minimax`
* State-Action Marginal Doubly Robust (SAM-DR) :cite:`uehara2020minimax`
* State-Action Marginal Self-Normalized Importance Sampling (SAM-SNIS) :cite:`uehara2020minimax`
* State-Action Marginal Self-Normalized Doubly Robust (SAM-SNDR) :cite:`uehara2020minimax`
* Spectrum of Off-Policy Evaluation (SOPE) :cite:`yuan2021sope`

.. raw:: html

    <div class="white-space-5px"></div>

**Double Reinforcement Learning**

* Double Reinforcement Learning :cite:`kallus2020double`

.. raw:: html

    <div class="white-space-5px"></div>

**Weight and Value Learning Methods**

* Augmented Lagrangian Method (ALM/DICE) :cite:`yang2020off`
   BestDICE :cite:`yang2020off` / GradientDICE :cite:`zhang2020gradientdice` / GenDICE :cite:`zhang2020gendice` / AlgaeDICE :cite:`nachum2019algaedice` / DualDICE :cite:`nachum2019dualdice` / MQL/MWL :cite:`uehara2020minimax`
* Minimax Q-Learning and Weight Learning (MQL/MWL) :cite:`uehara2020minimax`

.. raw:: html

    <div class="white-space-5px"></div>

**High Confidence OPE**

* Bootstrap :cite:`thomas2015improvement` :cite:`hanna2017bootstrapping`
* Hoeffding :cite:`thomas2015evaluation`
* (Empirical) Bernstein :cite:`thomas2015evaluation` :cite:`thomas2015improvement`
* Student T-test :cite:`thomas2015improvement`

.. raw:: html

    <div class="white-space-5px"></div>
    <div class="white-space-5px"></div>

Moreover, we streamline the evaluation protocol of OPE/OPS with the following metrics.

**OPE metrics**

* Mean Squared Error :cite:`paine2020hyperparameter` :cite:`voloshin2021empirical` :cite:`fu2021benchmarks`
* Spearman's Rank Correlation Coefficient :cite:`paine2020hyperparameter` :cite:`fu2021benchmarks`
* Regret :cite:`paine2020hyperparameter` :cite:`fu2021benchmarks`
* Type I and Type II Error Rates

.. raw:: html

    <div class="white-space-5px"></div>

**OPS metrics** (performance of top :math:`k` deployment policies)

* {Best/Worst/Mean/Std} of policy performance
* Safety violation rate
* Sharpe ratio (our proposal)

Note that, the above top-:math:`k` metrics are the proposal in our research paper **"
Towards Assessing and Benchmarking Risk-Return Tradeoff of Off-Policy Evaluation in Reinforcement Learning"**.  
:doc:`sharpe_ratio` describe these metrics in details, and we also discuss about these metrics briefly in :ref:`the later sub-section <feature_sharpe_ratio>`.

.. _feature_cd_ope:

Cumulative Distribution OPE for risk function estimation
----------

Besides the standard OPE, SCOPE-RL differentiates itself from other OPE platforms by supporting the cumulative distribution OPE for the first time.
Roughly, cumulative distribution OPE aims to estimate the whole performance distribution of the policy performance, not just the expected performance as the standard OPE does.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ope_cumulative_distribution_function.png
    :text-align: center

    Example of estimating cumulative distribution function (CDF) via OPE

.. raw:: html

    <div class="white-space-20px"></div>

By estimating the cumulative distribution function (CDF), we can derive the following statistics of the policy performance:

* Mean (i.e., policy value)
* Variance
* Conditional Value at Risk (CVaR)
* Interquartile Range

Knowing the whole performance distribution or deriving the risk metrics including CVaR is particularly beneficial in a real-life situation where the safety matters. 
For example, in recommender systems, we are interested in stably providing good-quality products rather than sometimes providing an extremely good one but sometimes hurting user satisfaction seriously with bad items.
Moreover, in the self-diriving cars, the catastrophic accidents should be avoided even if its probability is small (e.g., less than 10%). 
We believe that the release of cumulative distribution OPE implementations will boost the applicability of OPE in practical situations.


.. _feature_sharpe_ratio:

Risk-Return Assessments of OPS
----------

Our SCOPE-RL is also unique in that it provides risk-return assessments of Off-Policy Selection (OPS). 

While OPE is useful for estimating the policy performance of a new policy using offline logged data, 
OPE sometimes produces erroneous estimation due to *counterfactual estimation* and *distribution shift* between the behavior and evaluation policies.
Therefore, in practical situations, we cannot solely rely on OPE results to choose the production policy, but instead, combine OPE results and online A/B tests for policy evaluation and selection :cite:`kurenkov2022showing`.
Specifically, the practical workflow often begins by filtering out poor-performing policies based on OPE results, then conducting A/B tests on the remaining top-:math:`k`
policies to identify the best policy based on reliable online evaluation, as illustrated in the following figure.

.. card:: 
    :width: 50%
    :margin: auto
    :img-top: ../_static/images/ops_workflow.png
    :text-align: center

    Practical workflow of policy evaluation and selection

.. raw:: html

    <div class="white-space-20px"></div>

While the conventional metrics of OPE focus on the "accuracy" of OPE and OPS measured by mean-squared error (MSE) :cite:`uehara2022review, voloshin2021empirical`, rank correlation :cite:`paine2020hyperparameter, fu2021benchmarks`, and regret :cite:`doroudi2017importance, tang2021model`, 
we measure risk, return, and efficiency of the selected top-:math:`k` policy with the following metrics.

.. card:: 
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/ops_topk_policy_value_multiple.png
    :text-align: center

    Example of evaluating OPE/OPS methods with top-:math:`k` RRT metrics

.. raw:: html

    <div class="white-space-20px"></div>

* best @ :math:`k` (*return*)
* worsk @ :math:`k`, mean @ :math:`k` (*risk*)
* safety violation rate @ :math:`k` (*risk*)
* Sharpe ratio @ :math:`k` (*efficiency*, our proposal)

.. seealso::

    Among the top-:math:`k` risk-return tradeoff metrics, SharpRatio is the main propossal of our research paper 
    **"Towards Assessing and Benchmarking Risk-Return Tradeoff of Off-Policy Evaluation in Reinforcement Learning"**. 
    We describe the motivation and contributions of the SharpRatio metric in :doc:`sharpe_ratio`.


.. _feature_comparison:


Comparisons with the existing platforms
~~~~~~~~~~

Finally, we provide a comprehensive comparion with the existing offline RL and OPE platforms.

.. card:: 
    :width: 75%
    :margin: auto
    :img-bottom: ../_static/images/distinctive_features.png
    :text-align: center

    Comparing SCOPE-RL with existing offline RL and OPE platforms

.. raw:: html

    <div class="white-space-20px"></div>

The criteria of each colums is given as follows:

* "data collection": |:white_check_mark:| means that the platform is compatible with Gymnasium environments :cite:`brockman2016openai` and thus is able to handle various settings. 
* "offline RL": |:white_check_mark:| means that the platform implements a variety of offline RL algorithms or the platform is compatible to one of offline RL libraries. In particular, our SCOPE-RL supports compatibility to `d3rlpy <https://github.com/takuseno/d3rlpy>`_ :cite:`seno2021d3rlpy`.
* "OPE": |:white_check_mark:| means that the platform implements various OPE estimators other than the standard choices including Direct Method :cite:`le2019batch`, Importance Sampling :cite:`precup2000eligibility`, and Doubly Robust :cite:`jiang2016doubly`. (limited) means that the platform supports only these standard estimators.
* "CD-OPE": is the abbreviation of Cumulative Distribution OPE, which estimates the cumulative distribution function of the return under evaluation policy :cite:`huang20210ff, huang2022off, chandak2021universal`. 

In summary, **our unique contribution is 
(1) to provide the first end-to-end platform for offline RL, OPE, and OPS,
(2) to support cumulative distribution ope for the first time, and
(3) to implement (the proposed) top-** :math:`k` **risk-return tradeoff metics for the risk assessments of OPS.**
Additionally, we provide a user-friendly :doc:`visualization tools <visualization>`, :doc:`documentation <index>`, and `quickstart examples <https://github.com/hakuhodo-technologies/scope-rl/tree/main/examples/quickstart>`_ to facilitate a quick benckmarking and practical application. 
We hope that SCOPE-RL will serve as a important milestone for the future development of OPE research.

.. We also provide an :doc:`OPE tutorial <_autogallery/index>` with SCOPE-RL experiments for educational purpose. 
.. We hope that SCOPE-RL will serve as a important milestone for the future development of OPE research.

Note that, the compared platforms include the following:

(offline RL platforms)

* d3rlpy :cite:`seno2021d3rlpy`
* CORL :cite:`tarasov2022corl`
* RLlib :cite:`liang2018rllib`
* Horizon :cite:`gauci2018horizon` 

(application-specific testbeds)

* NeoRL :cite:`qin2021neorl`
* RecoGym :cite:`rohde2018recogym`
* RL4RS :cite:`wang2021rl4rs`
* AuctionGym :cite:`jeunen2022learning`

(OPE platforms)

* DOPE :cite:`fu2021benchmarks`
* COBS :cite:`voloshin2021empirical`
* OBP :cite:`saito2021open`

.. raw:: html

    <div class="white-space-5px"></div>

**Remark**

Our implementations are highly inspired by `OpenBanditPipeline (OBP) <https://zr-obp.readthedocs.io/en/latest/>`_ :cite:`saito2021open`, which has demonstrated success in enabling flexible OPE experiments in contextual bandits. 
We hope that SCOPE-RL will also serve as a quick prototyping and benchmarking toolkit for OPE of RL policies, as done by OBP in non-RL settings.

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
                :link: index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Documentation (Back to Top)**

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
                :link: online_offline_rl
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Problem Formulation**

            .. grid-item-card::
                :link: quickstart
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Quickstart**
