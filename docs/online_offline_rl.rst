==========
Overview
==========

We describe the problem setup and prevalent approaches of online/offline Reinforcement Learning (RL).

Online Reinforcement Learning
~~~~~~~~~~
We consider a general reinforcement learning setup, which is formalized by Markov Decision Process (MDP) as :math:`\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, P_r, \gamma \rangle`.
:math:`\mathcal{S}` is the state space and :math:`\mathcal{A}` is the action space, which is either discrete or continuous. 
Let :math:`\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{P}(\mathcal{S})` is the state transition probability where :math:`\mathcal{T}(s' | s,a)` is the probability of observing state :math:`s'` after taking action :math:`a` given state :math:`s`. 
:math:`P_r: \mathcal{S} \times \mathcal{A} \times \mathbb{R} \rightarrow [0,1]` is the probability distribution of the immediate reward. 
Given :math:`P_r`, :math:`R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}` is the expected reward function where :math:`R(s,a) := \mathbb{E}_{r \sim P_r (r | s, a)}[r]` is the expected reward when taking action :math:`a` for state :math:`s`. 
We also let :math:`\gamma \in (0,1]` be a discount factor. Finally, :math:`\pi: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})` denotes a *policy* where :math:`\pi(a| s)` is the probability of taking action :math:`a` at a given state :math:`s`. 
Note that we also denote :math:`d_0` as the initial state distribution.

The goal of RL is to maximize the following expected cumulative reward (i.e., policy value) of an episode that consists of total :math:`T` timesteps.

.. math::

    \max_{\pi \in \Pi} J(\pi) := \mathbb{E}_{\tau \sim p_{\pi}(\tau)} \left [ \displaystyle \sum_{t=0}^{T-1} \gamma^t r_t | \pi \right ]

where :math:`\gamma` is a discount rate and :math:`\tau := (s_t, a_t, s_{t+1}, r_t)_{t=0}^{T-1}` is the trajectory of the policy which is sampled from 
:math:`p_{\pi}(\tau) := d_0(s_0) \prod_{t=0}^{T-1} \pi(a_t | s_t) \mathcal{T}(s_{t+1} | s_t, a_t) P_r(r_t | s_t, a_t)`.

There are several approaches to maximize the policy value. Below, we review three basic methods, On-Policy Policy Gradient, Q-Learning, and Actor-Critic.


On-Policy Policy Gradient
----------
One of the most naive approach to maximize the policy value is to directly learn a policy through gradient ascent as follows.

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
To pursue the sample efficiency, Q-Learning instead takes Off-Policy approach.
Specifically, it aims to learn the following state value :math:`V(s_t)` and state-action value :math:`Q(s_t, a_t)` using the data collected by previous online interactions.

.. math::

    V(s_t) := \mathbb{E}_{\tau_{t:T-1} \sim p_{\pi}(\tau_{t:T-1} | s_t)} \left[ \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} \right]

    Q(s_t, a_t) := \mathbb{E}_{\tau_{t:T-1} \sim p_{\pi}(\tau_{t:T-1} | s_t, a_t)} \left[ \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'} \right]

where :math:`\tau_{t:T-1}` is the trajectory from timestep :math:`t` to :math:`T-1`.

Using the recursive connection between :math:`V(\cdot)` and :math:`Q(\cdot)`, we can derive the following Bellman equation to learn Q-function (i.e., :math:`Q`).

.. math::

    Q(s_t, a_t) = r_t + \mathbb{E}_{(s_{t+1}, a_{t+1}) \sim \mathcal{T}(s_{t+1} | s_t, a_t) \pi(a_{t+1} | s_{t+1})} [ Q(s_t+1, a_{t+1}) ]

For example, when we use a greedy policy, Q-Learning learns Q-Function and update policy alternately as follows.

.. math::

    \hat{Q}_{k+1} \leftarrow {\arg \min}_{Q_{k+1}} \mathbb{E}_n [ \left( Q_{k+1}(s_t, a_t) - (r_t + \hat{Q}_k(s_{t+1}, \pi_k(s_{t+1}))) \right)^2 ]

where :math:`n` state-action pairs are randomly sampled from the replay buffer, which stores the past observations :math:`(s_t, a_t, s_{t+1}, r_t)`.
:math:`\pi_k` chooses actions as :math:`\pi_k(a_t | s_t) := \mathbb{I} \{ {\arg \max}_{a_t \in \mathcal{A}}  \hat{Q}_k(s_t, a_t) \}`, where :math:`I \{\cdot\}` is the indicator function.

Though this strategy enhances sample efficiency compared to On-Policy Policy Gradient, this method can suffer from bias in estimation.
That is, when :math:`\hat{Q}(\cdot)` fails estimate the true state-action value, the action choice easily becomes sub-optimal.

To alleviate the estimation error of :math:`\hat{Q}(\cdot)`, we often use epsilon-greedy policy, which chooses random actions with probability :math:`\epsilon`.
Such *exploration* helps improve the quality of \hat{Q}(\cdot) by alleviating the estimation error on unseen state-action pairs. 


Actor-Critic
----------
Actor-critic is a hybrid of Policy Gradient and Q-Learning.
It first estimate Q-function and then calculate the advantage of choosing actions (:math:`A(s, a) := Q(s, a) - V(s)`) to derive an approximated policy gradient as follows.

.. math::

    \hat{Q}_{k+1} \leftarrow {\arg \min}_{Q_{k+1}} \mathbb{E}_n \left [ \left( Q_{k+1}(s_t, a_t) - (r_t + \hat{Q}_k(s_{t+1}, \pi_k(s_{t+1}))) \right)^2 \right ]

    \theta_{k+1} \leftarrow \theta_{k} + \nabla \mathbb{E}_n \left [ \sum_{t=0}^{T-1} \nabla \log \pi(a_t | s_t) \gamma^t \hat{A}(s_t, a_t) \right ]

where :math:`\hat{A}(s_t, a_t) := \hat{Q}(s_t, a_t) - \mathbb{E}_{a \sim \pi(a_t | s_t)} \left[ \hat{Q}(s_t, a) \right]`.

Compared to the (vanilla) On-policy Policy Gradient, Actor-Critic stabilizes the policy gradient and enhances sample efficiency by the use of :math:`\hat{Q}`.
Note that, compared to Q-learning, Actor-Critic is more suitable in continuous action space because we do not have to discretize the action space to choose actions.


Case Study: Does Online Learning almost always works well?
----------
TODO: Example of running online policies in RTBGym.

Online learning can still be unsafe in the initial learning phase due to sub-optimal actions choice.
Moreover, frequent online policy update is often costly in many practical systems.
Those factors often makes practical application of reinforcement learning challenging particularly in the initial trial, despite its potential benefit after a successful deployment.


Offline Reinforcement Learning
~~~~~~~~~~
So far, we have seen that learning a policy in an online environment is demanding in many practical systems.
Then, the next question should be: *Aren't their any ways to tackle the risks and costs of online learning?* To answer, we now introduce offline RL.