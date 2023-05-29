==========
Overview
==========

We describe the problem setup and prevalent approaches of online/offline Reinforcement Learning (RL).

.. _overview_online_rl:

Online Reinforcement Learning
~~~~~~~~~~
We consider a general reinforcement learning setup, which is formalized by Markov Decision Process (MDP) as :math:`\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, P_r, \gamma \rangle`.
:math:`\mathcal{S}` is the state space and :math:`\mathcal{A}` is the action space, which is either discrete or continuous. 
Let :math:`\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{P}(\mathcal{S})` is the state transition probability where :math:`\mathcal{T}(s' | s,a)` is the probability of observing state :math:`s'` after taking action :math:`a` given state :math:`s`. 
:math:`P_r: \mathcal{S} \times \mathcal{A} \times \mathbb{R} \rightarrow [0,1]` is the probability distribution of the immediate reward. 
Given :math:`P_r`, :math:`R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}` is the expected reward function where :math:`R(s,a) := \mathbb{E}_{r \sim P_r (r | s, a)}[r]` is the expected reward when taking action :math:`a` for state :math:`s`. 
We also let :math:`\gamma \in (0,1]` be a discount factor. Finally, :math:`\pi: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})` denotes a *policy* where :math:`\pi(a| s)` is the probability of taking action :math:`a` at a given state :math:`s`. 
Note that we also denote :math:`d_0` as the initial state distribution.

.. card:: Description of Reinforcement Learning
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/onlinerl_setup.png
    :text-align: center

.. raw:: html

    <div class="white-space-20px"></div>

The goal of RL is to maximize the following expected cumulative reward (i.e., policy value) of an episode that consists of total :math:`T` timesteps.

.. math::

    \max_{\pi \in \Pi} \, J(\pi) := \mathbb{E}_{\tau \sim p_{\pi}(\tau)} \left [ \sum_{t=0}^{T-1} \gamma^t r_t | \pi \right ]

where :math:`\gamma` is a discount rate and :math:`\tau := (s_t, a_t, s_{t+1}, r_t)_{t=0}^{T-1}` is the trajectory of the policy which is sampled from 
:math:`p_{\pi}(\tau) := d_0(s_0) \prod_{t=0}^{T-1} \pi(a_t | s_t) \mathcal{T}(s_{t+1} | s_t, a_t) P_r(r_t | s_t, a_t)`.

There are several approaches to maximize the policy value. Below, we review three basic methods, On-Policy Policy Gradient :cite:`kakade2001natural` :cite:`silver2014deterministic`, 
Q-Learning :cite:`watkins1992q` :cite:`mnih2013playing`, and Actor-Critic :cite:`konda1999actor` :cite:`degris2012off`.

On-Policy Policy Gradient
----------
One of the most naive approach to maximize the policy value is to directly learn a policy through gradient ascent as follows :cite:`kakade2001natural` :cite:`silver2014deterministic`.

.. math::

    \theta_{k+1} \leftarrow \theta_{k} + \nabla J(\pi_{\theta_k})

where :math:`\theta` is a set of policy parameter. 

We can estimate the policy gradient :math:`J(\pi)` via on-policy estimation as follows.

.. math::

    \nabla J(\pi) \approx \mathbb{E}_n \left [ \sum_{t=0}^{T-1} \nabla \log \pi(a_t | s_t) \gamma^t r_t \right ]

where :math:`\mathbb{E}_n [\cdot]` takes empirical average over :math:`n` trajectories sampled from online interactions.

The benefit of On-Policy Policy Gradient is that it enables an unbiased estimation of the policy value as :math:`n` grows. 
However, as the algorithm needs :math:`n` trajectories collected by :math:`\pi_{k-1}` every time the policy is updated to :math:`\pi_{k}`, the algorithm is known to suffer from *sample inefficiency* and instability.

Q-Learning
----------
To pursue the sample efficiency, Q-Learning instead takes Off-Policy approach, which leverages a large amount of data in the replay buffer :cite:`mnih2013playing`.
Specifically, it aims to learn the following state value :math:`V(s_t)` and state-action value :math:`Q(s_t, a_t)` using the data collected by previous online interactions :cite:`watkins1992q`.

.. math::

    V(s_t) := \mathbb{E}_{\tau_{t:T-1} \sim p_{\pi}(\tau_{t:T-1} | s_t)} \left[ \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} \right]

.. math::

    Q(s_t, a_t) := \mathbb{E}_{\tau_{t:T-1} \sim p_{\pi}(\tau_{t:T-1} | s_t, a_t)} \left[ \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} \right]

where :math:`\tau_{t:T-1}` is the trajectory from timestep :math:`t` to :math:`T-1`.

Using the recursive connection between :math:`V(\cdot)` and :math:`Q(\cdot)`, we can derive the following Bellman equation.

.. math::

    Q(s_t, a_t) = r_t + \mathbb{E}_{(s_{t+1}, a_{t+1}) \sim \mathcal{T}(s_{t+1} | s_t, a_t) \pi(a_{t+1} | s_{t+1})} [ Q(s_t+1, a_{t+1}) ]

Temporal Difference (TD) learning leverages this recursive formula to learn Q-function (i.e., :math:`Q`). 
In particular, when we use a greedy policy, Q-Function is reduces to as follows.

.. math::

    \hat{Q}_{k+1} \leftarrow {\arg \min}_{Q_{k+1}} \mathbb{E}_n \left[ \left( Q_{k+1}(s_t, a_t) - (r_t + \hat{Q}_k(s_{t+1}, \pi_k(s_{t+1}))) \right)^2 \right]

where :math:`n` state-action pairs are randomly sampled from the replay buffer, which stores the past observations :math:`(s_t, a_t, s_{t+1}, r_t)`.
Based on this Q-function, the greedy policy :math:`\pi_k` chooses actions as follows.

.. math::

    \pi_k(a_t | s_t) := \mathbb{I} \{ {\arg \max}_{a_t \in \mathcal{A}}  \hat{Q}_k(s_t, a_t) \}, 

where :math:`\mathbb{I} \{ \cdot \}` is the indicator function. 

Though this strategy enhances sample efficiency compared to On-Policy Policy Gradient, 
Q-learning is known to suffer from approximation error when the *deadly triad* conditions -- bootstrapping (i.e., TD learning), function approximation, and off-policy -- are simultaneously satisfied :cite:`van2018deep`. 

As a result, :math:`\hat{Q}(\cdot)` can fail to estimate the true state-action value, which may lead to a sub-optimal policy.

To alleviate the estimation error of :math:`\hat{Q}(\cdot)`, we often use epsilon-greedy policy, which chooses actions randomly with probability :math:`\epsilon`.
Such *exploration* helps improve the quality of :math:`\hat{Q}(\cdot)` by collecting additional data to fit Q-function to the state-action pairs that have not seen in the replay buffer. 

Actor-Critic
----------
Actor-critic :cite:`konda1999actor` :cite:`degris2012off` is a hybrid of Policy Gradient and Q-Learning.
It first estimate the Q-function and then calculate the advantage of choosing actions (:math:`A(s, a) := Q(s, a) - V(s)`) to derive an approximated policy gradient as follows.

.. math::

    \hat{Q}_{k+1} \leftarrow {\arg \min}_{Q_{k+1}} \mathbb{E}_n \left[ \left( Q_{k+1}(s_t, a_t) - (r_t + \hat{Q}_k(s_{t+1}, \pi_{\theta_k}(s_{t+1}))) \right)^2 \right]

.. math::

    \theta_{k+1} \leftarrow \theta_{k} + \mathbb{E}_n \left[ \sum_{t=0}^{T-1} \nabla \log \pi_{\theta_k}(a_t | s_t) \gamma^t \hat{A}(s_t, a_t) \right]

where :math:`\hat{A}(s_t, a_t) := \hat{Q}(s_t, a_t) - \mathbb{E}_{a \sim \pi_{\theta_k}(a_t | s_t)} \left[ \hat{Q}(s_t, a) \right]` 
and :math:`\pi_{\theta_k}(s_{t+1})` is an action sampled from :math:`\pi_{\theta_k}(\cdot)`.

Compared to the (vanilla) On-policy Policy Gradient, Actor-Critic stabilizes the policy gradient and enhances sample efficiency by the use of :math:`\hat{Q}`.
Moreover, in continuous action space, Actor-Critic is often more suitable than Q-learning, which requires discretization of the action space to choose actions.

.. _overview_offline_rl:

Offline Reinforcement Learning
~~~~~~~~~~
While online learning is a powerful framework to learn a (near) optimal policy through interaction, however, it also entails risk of taking sub-optimal or even unsafe actions, especially in the initial learning phase :cite:`levine2020offline`.
Moreover, updating a policy in a online manner may also require huge implementation costs (in applications such as recommender systems and robotics) :cite:`matsushima2020deployment`.

.. card:: Description of Offline Reinforcement Learning
    :width: 75%
    :margin: auto
    :img-top: ../_static/images/offlinerl_concept.png
    :text-align: center

.. raw:: html

    <div class="white-space-20px"></div>

To overcome the above issue, offline RL aims to learn a new policy in an `offline` manner, leveraging the logged data collected by a past deployment policy. 
Specifically, let us assume that we are accessible to the logged dataset :math:`\mathcal{D}` consisting of :math:`n` trajectories, each of which is generated by a behavior policy :math:`\pi_0` as follows.

.. math::

    \tau := \{ (s_t, a_t, s_{t+1}, r_t) \}_{t=0}^{T} \sim p(s_0) \prod_{t=0}^{T} \pi_0(a_t | s_t) \mathcal{T}(s_{t+1} | s_t, a_t) P_r (r_t | s_t, a_t)

A key ingredient here is that we can observe feedback only for the actions chosen by the behavior policy. 
Therefore, when learning a new policy in an offline manner, we need to answer the counterfactual question, 

.. card:: 
    :text-align: center

    *"What if a new policy chooses a different action from that of behavior policy?"*

Further, the state and reward observations in the logged dataset are also biased since state transition and data collection heavily depend on the action chosen by the behavior policy. 
Therefore, we need to tackle the `distributional shift` between the behavior policy and a new policy and deal with the out-of-distribution problem. 

The problem of Extrapolation Error
----------
Apparently, Q-learning seems to be compatible with the offline setting, as it uses large amount of data to learn Q-function.
However, Q-function is known to suffer from `extrapolation error` :cite:`fujimoto2019off` 
due to the distribution shift and the deadly triad conditions (i.e., the combination of the bootstrapping, function approximation, and off-policy) :cite:`van2018deep`.

To investigate why the extrapolation error arises, let us recall the following TD loss of the Q-learning.

.. math::

    \hat{\mathcal{L}}_{\mathrm{TD}}(\theta, \mathcal{D}) \propto \mathbb{E}_n \left[ \left( Q_{\theta}(s_t, a_t) - (r_t + \hat{Q}_{\mathrm{target}}(s_{t+1}, \pi(s_{t+1}))) \right)^2 \right]

where :math:`Q_{\theta}` is the currently learning Q-function and :math:`\theta` is its parameters. 
:math:`\hat{Q}_{\mathrm{target}}` is the previous Q-function, which is used as the `target`. :math:`\pi` is the policy derived from :math:`\hat{Q}_{\mathrm{target}}`.

What is problematic here is that we have to calculate the TD loss using :math:`(s_t, a_t, r_t, s_{t+1}, a_{t+1}=\pi(s_{t+1}))`, while we are only accessible to :math:`(s_t, a_t, r_t, s_{t+1})` in the logged data.
Moreover, since :math:`\pi` chooses the action that maximizes :math:`\hat{Q}_{\mathrm{target}}`, :math:`\pi` tends to choose unobserved (or out-of-distribution) action whose :math:`\hat{Q}_{\mathrm{target}}` is coincidentally higher or overestimated than true Q-function.
As a result, :math:`Q_{\theta}(s_t, a_t)` also propagates the overestimation error, which eventually leads to a sub-optimal and often unsafe policy.

Below, we describe several approaches that addresses the aforementioned issue.

Behavior Divergence Regularization and Behavior Cloning
----------
One way to mitigate the extrapolation error is to directly regularize the distribution shift.

For example, BRAC :cite:`wu2019behavior` regularizes the discrepancy between the behavior and learning policies at :math:`s_{t+1}` as follows.

(objective)

.. math::

    \max_{\pi \in \Pi} \, J(\pi) := \mathbb{E}_{\tau \sim p_{\pi}(\tau)} \left [ \sum_{t=0}^{T-1} \gamma^t r_t - \alpha D(\pi, \pi_0) | \pi \right ]

(TD loss)

.. math::

    \hat{\mathcal{L}}_{\mathrm{TD}}(\theta, \mathcal{D}) \propto \mathbb{E}_n \left[ \left( Q_{\theta}(s_t, a_t) - (r_t + \hat{Q}_{\mathrm{target}}(s_{t+1}, \pi(s_{t+1})) - \alpha D(\pi(\cdot | s_{t+1}), \pi_0(\cdot | s_{t+1}))) \right)^2 \right]

where :math:`\alpha` is the weight of the divergence regularization and :math:`D(\cdot, \cdot)` is some divergence metrics such as KL-divergence or Wassertein distance.
This method effectively reduces the :math:`\hat{Q}_{\mathrm{target}}` of out-distribution-action, thereby mitigate the overestimation. 
However, the divergence regularization may also restrict the generalizability because it keeps the learned policy too similar to the behavior policy even when the Q-function is adequately accurate (e.g., when the :math:`\pi_0` is uniform random or follows a multi-modal distribution). 

Another way to regularize the distribution shift is to force :math:`\pi` to imitate :math:`\pi` in the policy optimization phase (not in Q-learning phase) as follows.

For example, TD3+BC :cite:`fujimoto2021minimalist` imposes a strong behavior cloning regularization when the average Q-value is large.

.. math::

    \pi \leftarrow {\arg\max}_{\pi \in \Pi} \, \mathbb{E}_{n} \left[ \lambda \hat{Q}(s_t, \pi(s_t)) - (\pi(s_t) - a_t)^2 \right]

where the first term facilitates value optimization (based on :math:`\hat{Q}`), whilst the second term promotes the behavior cloning. The weight parameter :math:`\lambda` is defined as follows.

.. math::

    \lambda = \frac{\alpha}{\mathbb{E}_n \left[ |Q(s_t, a_t)| \right]}

where :math:`\alpha` is the predefined hyperparameter.
Intuitively, :math:`\lambda` becomes small when the average Q-value is large. Therefore, in such cases, :math:`\pi` imitates :math:`\pi_0` more because :math:`\hat{Q}` is unreliable.
On the other hand, when :math:`\hat{Q}` estimates well and the average Q-value is not that large, :math:`\pi` maximizes :math:`\hat{Q}`. 

Uncertainty Estimation
----------
The second approach to deal with the overestimation bias of :math:`\hat{Q}` is to derive the lower bound of the Q-value based on estimation uncertainty.
This approach is somewhat similar to BRAC, but does not have to penalize the distribution shift as long as the Q-function is accurate.

For example, BEAR :cite:`kumar2019stabilizing` estimates the Q-function as follows.

.. math::

    \hat{\mathcal{L}}_{\mathrm{TD}}(\theta, \mathcal{D}) \propto \mathbb{E}_n \left[ \left( Q_{\theta}(s_t, a_t) - (r_t + \hat{Q}_{\mathrm{pess}}(s_{t+1}, \pi(s_{t+1})) \right)^2 \right]

The pessimistic Q-function is learned through ensembling :math:`m` different Q-functions as follows.

.. math::

    \hat{Q}_{\mathrm{pess}}(s) := \max_{a \in \mathcal{A}} \left( \lambda \min_{j = 1,2, \ldots, m} \hat{Q}_j(s, a) + (1 - \lambda) \max_{j' = 1, 2, \ldots ,m} \hat{Q}_{j'}(s, a) \right)

where :math:`\lambda` is the hyperparameter that determines the degree of optimism/pessimism. A large value of :math:`\lambda` leads to a pessimistic Q-function.

Besides, we can penalize with the standard deviation as follows.

.. math::

    \hat{Q}_{\mathrm{pess}}(s) := \max_{a \in \mathcal{A}} \left( \mathbb{E}_m [\hat{Q}_j(s, a)] - \sqrt{\mathbb{V}_m [\hat{Q}_j(s, a)]} \right)

where :math:`\mathbb{E}_m[\cdot]` and :math:`\mathbb{V}_m[\cdot]` is the mean and variance among :math:`m` different Q-functions.

Conservative Q-Learning
----------
To derive the conservative Q-function without explicitly quantifying the uncertainty, CQL :cite:`kumar2020conservative` minimizes the Q-value of the out-of-distribution state-action pairs while also minimizing the TD loss.

.. math::

    Q \leftarrow \max_{Q} \min_{\mu} \, & \alpha \left( \mathbb{E}_n \left[ Q(s_t, \mu(s_t)) - Q(s_t, \pi_0(s_t)) \right]  \right) \\
    & \quad \quad + \mathbb{E}_n \left[ \left( Q(s_t, a_t) - (r_t + \hat{Q}(s_{t+1}, \pi(s_{t+1}))) \right)^2 \right]

where :math:`\alpha` is the hyperparameter to balance the loss function. 
The first term aims to minimize the maximum Q-value of the policy :math:`\mu` to alleviate the overestimation while maximizing the Q-value of the behavior policy. 
By adding this loss function, CQL effectively learn the Q-function under the state-action pairs supported by :math:`\pi_0`, while being conservative to the out-of-distribution action. 
However, CQL is also known to be too conservative to generalize well. Many advanced algorithms including COMBO :cite:`yu2021combo` (, which exploits model-based data augmentation for OOD observations)
have been developed to improve the generalizability of CQL. 

Implicit Q-Learning
----------

One of the limitations of the above approaches is that they may sacrifice the generalizability due to the explicit regularization on the out-of-distribution state-action pairs.

To tackle this issue, IQL :cite:`kostrikov2021offline` aims to learn a conservative policy without the explicit out-of-distribution regularization.
For this, IQL first estimates the state-value function (V-function) with the asymmetric loss to penalize the optimism as follows.

.. math::

    \hat{\mathcal{L}}_{V}(\psi) = \mathbb{E}_n [ L_2^{\lambda} (\hat{Q}_{\theta}(s_t, a_t) - V_{\psi}(s_t)) ]

where :math:`\hat{Q}_{\theta}` and :math:`V_{\psi}` is learned distinctly, with different parameters :math:`\theta` and :math:`\psi`, respectively. 
:math:`L_2^{\lambda}(z)` is the asymmetric loss function, which is defined as follows.

.. math::

    L_2^{\lambda}(z) := |\tau - \mathbb{I}(z < 0)| z^2

where :math:`\tau` is the parameter to control the asymmetricity. When :math:`\tau > 0.5`, the loss function penalizes the positive value of :math:`z` more.
Therefore, :math:`\hat{V}` learned with :math:`\tau \rightarrow 1` indicates the maximum Q-value among the observed state-action pairs, 
while that learned with :math:`\tau = 0.5` indicates the average Q-value among those pairs.
This prevents the propagation of the overestimation bias, even when the basic TD loss is used to learn the Q-function as follows.

.. math::

    \hat{\mathcal{L}}_{Q}(\theta) = \mathbb{E}_n [ (\hat{Q}_{\theta}(s_t, a_t) - (r_t + \hat{V}_{\psi}(s_{t+1}))) ]

.. seealso::

    * :doc:`Supported implementations and useful tools <learning_implementation>` 
    * :doc:`Quickstart <quickstart>` and :doc:`related tutorials <_autogallery/scope_rl_others/index>`

.. seealso::

    For further taxonomies, algorithms, and descriptions, we refer readers to survey papers :cite:`levine2020offline` :cite:`prudencio2022survey`. 
    `awesome-offline-rl <https://github.com/hanjuku-kaso/awesome-offline-rl>`_ also provides a comprehensive list of literature.

.. seealso::

    After learning a new policy, we are often interested in the performance validation. 
    We describe the problem formulation of Off-Policy Evaluation (OPE) and Selection (OPS) in :doc:`Overview (OPE/OPS) <ope_ops>`.

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
                :link: ope_ops
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Off_Policy Evaluation**

            .. grid-item-card::
                :link: learning_implementation
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Supported Implementation**