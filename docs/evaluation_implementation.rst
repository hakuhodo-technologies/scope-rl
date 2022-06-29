==========
Supported Implementation
==========

We are currently supporting the Episodic RL setting.

Please also refer to `quickstart <>`_ for the example usages.

Basic Off-Policy Evaluation (OPE)
~~~~~~~~~~
Here, we describe how the implemented estimators estimates the policy value. Please refer to the `problem setup <>`_ for notations.

Direct Method (DM)
----------
DM is a model-based approach which uses initial state value estimated by Fitted Q Evaluation (FQE).
It first learns the Q-function and then leverages the learned Q-function as follows.

.. math::

    \hat{J}_{\mathrm{DM}} (\pi; \mathcal{D}) := \mathbb{E}_n \left[ \mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} [\hat{Q}(s_0, a_0)] \right],

where :math:`\mathcal{D}=\{\{(s_t, a_t, r_t)\}_{t=0}^T\}_{i=1}^n` is logged dataset with :math:`n` trajectories of data.
:math:`T` indicates step per episode. :math:`\hat{Q}(s_t, a_t)` is estimated state-action value.

DM has low variance, but can incur bias caused by approximation errors.

Note that, we use the implementation of FQE provided in `d3rlpy <https://github.com/takuseno/d3rlpy>`_.

Trajectory-wise Importance Sampling (TIS)
----------

TIS uses importance sampling technique to correct the distribution shift between :math:`\pi` and :math:`\pi_0` as follows.

.. math::

    \hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D}) := \mathbb{E}_{n} \left[\sum_{t=0}^T \gamma^t w_{1:T} r_t \right],

where :math:`w_{0:T} := \prod_{t=1}^T (\pi(a_t | s_t) / \pi_0(a_t | s_t))` is the importance weight.

TIS enables an unbiased estimation of the policy value. However, when the trajectory length :math:`T` is large, TIS suffers from high variance
due to the product of importance weights.

Per-Decision Importance Sampling (PDIS)
----------
PDIS leverages the sequential nature of the MDP to reduce the variance of TIS. 
Specifically, since :math:`s_t` only depends on :math:`s_0, \ldots, s_{t-1}` and :math:`a_0, \ldots, a_{t-1}` and is independent of :math:`s_{t+1}, \ldots, s_{T}` and :math:`a_{t+1}, \ldots, a_{T}`,
PDIS only considers the importance weight of the past interactions when estimating :math:`r_t` as follows.

.. math::

    \hat{J}_{\mathrm{PDIS}} (\pi; \mathcal{D}) := \mathbb{E}_{n} \left[ \sum_{t=0}^T \gamma^t w_{0:t} r_t \right],

where :math:`w_{0:t} := \prod_{t'=0}^t (\pi_e(a_{t'} \mid s_{t'}) / \pi_b(a_{t'} \mid s_{t'}))` is the importance weight of past interactions.

PDIS remains unbiased while reducing the variance of TIS. However, when :math:`t` is large, PDIS still suffers from high variance.

Doubly Robust (DR)
----------
DR is a hybrid of model-based estimation and importance sampling.
It introduces :math:`\hat{Q}` as a baseline estimation in the recursive form of PDIS and applies importance weighting only on its residual. 

.. math::

    \hat{J}_{\mathrm{DR}} (\pi; \mathcal{D})
    := \mathbb{E}_{n} \left[\sum_{t=0}^T \gamma^t (w_{0:t} (r_t - \hat{Q}(s_t, a_t)) + w_{0:t-1} \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_t, a)])\right],

DR is unbiased and reduces the variance of IPS when :math:`\hat{Q}(\cdot)` is reasonably accurate to satisfy :math:`0 < \hat{Q}(\cdot) < 2 Q(\cdot)`.

Self-Normalized estimators
----------
Self-normalized estimators aims to reduce the scale of importance weight for the variance reduction purpose.
Specifically, it substitute importance weight :math:`w_{\ast}` as follows.

.. math::

    \tilde{w}_{\ast} := w_{\ast} / \mathbb{E}_{n}[w_{\ast}]

where :math:`\tilde{w}_{\ast}` is the self-normalized importance weight.

Self-normalized estimators has variance bounded by :math:`r_{max}^2` while also being consistent.

Extension to the continuous action space
----------
When the action space is continuous, the naive importance weight :math:`w_t = \pi(a_t|s_t) / \pi_0(a_t|s_t) = (\pi(a |s_t) / \pi_0(a_t|s_t)) \cdot \mathbb{I}(a = a_t)` rejects almost every actions,
as :math:`\mathbb{I}(a = a_t)` filters only the action observed in the logged data.

To address this issue, continuous OPE estimators apply kernel density estimation technique to smooth the importance weight.

.. math::

    \overline{w}_t = \int_{a \in \mathcal{A}} \frac{\pi(a \mid s_t)}{\pi_0(a_t | s_t)} \cdot \frac{1}{h} K \left( \frac{a - a_t}{h} \right) da, 

where :math:`K(\cdot)` denotes a kernel function and :math:`h` is the bandwidth hyperparameter. 
We can use any function as :math:`K(\cdot)` that meets the following qualities: 

* 1) :math:`\int xK(x) dx = 0`, 
* 2) :math:`\int K(x) dx = 1`, 
* 3) :math:`\lim _{x \rightarrow-\infty} K(x)=\lim _{x \rightarrow+\infty} K(x)=0`, 
* 4) :math:`K(x) \geq 0, \forall x`. 

In our implementation, we use the (truncated) Gaussian kernel :math:`K(x)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{x^{2}}{2}}`. 


High Confidence Off-Policy Evaluation (HC-OPE)
----------
To alleviate the risk of optimistic estimation, we are often interested in the confidence intervals and the lower bound of the estimated policy value.
We implement four methods to estimate the confidence intervals.

* Hoeffding: :math:`|\hat{J}(\pi; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{J}(\pi; \mathcal{D})]| \leq \hat{J}_{\max} \displaystyle \sqrt{\frac{\log(1 / \alpha)}{2 n}}`.

* Empirical Bernstein: :math:`|\hat{J}(\pi; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{J}(\pi; \mathcal{D})]| \leq \displaystyle \frac{7 \hat{J}_{\max} \log(2 / \alpha)}{3 (n - 1)} + \displaystyle \sqrt{\frac{2 \hat{\mathbb{V}}_{\mathcal{D}}(\hat{J}) \log(2 / \alpha)}{(n - 1)}}`.

* Student T-test: :math:`|\hat{J}(\pi; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{J}(\pi; \mathcal{D})]| \leq \displaystyle \frac{T_{\mathrm{test}}(1 - \alpha, n-1)}{\sqrt{n} / \hat{\sigma}}`.

* Bootstrapping: :math:`|\hat{J}(\pi; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{J}(\pi; \mathcal{D})]| \leq \mathrm{Bootstrap}(1 - \alpha)`.

Note that, all the above bound holds with probability :math:`1 - \alpha`. 
For notations, we denote :math:`\hat{\mathbb{V}}_{\mathcal{D}}(\cdot)` to be the sample variance, 
:math:`T_{\mathrm{test}}(\cdot,\cdot)` to be T value, 
and :math:`\sigma` to be the standard deviation.

Among the above high confidence interval estimation, hoeffding and empirical bernstein derives lower bound without any distributional assumption of :math:`p(\hat{J})`, which sometimes leads to quite conservative estimation.
On the other hand, T-test is based on the assumption that each sample of :math:`p(\hat{J})` follows the normal distribution.

Cumulative Distributional Off-Policy Evaluation (CD-OPE)
~~~~~~~~~~

Direct Method (DM)
----------

DM adopts model-based approach to estimate the cumulative distribution function.

.. math::

        \hat{F}_{\mathrm{DM}}(m, \pi; \mathcal{D}) := \mathbb{E}_{n} \left[ \mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} \hat{G}(m; s_0, a_0) \right]

where :math:`\hat{F}(\cdot)` is the estimated cumulative distribution function and :math:`\hat{G}(\cdot)` is the estimated conditional distribution.

DM is vulnerable to the approximation error, but has low variance.

Trajectory-wise Importance Sampling (TIS)
----------

TIS corrects the distribution shift by applying importance sampling technique on the cumulative distribution estimation.

.. math::

        \hat{F}_{\mathrm{TIS}}(m, \pi; \mathcal{D}) := \mathbb{E}_{n} \left[ w_{1:T} \mathbb{I} \left \{\sum_{t=0}^T \gamma^t r_t \leq m \right \} \right]

TIS is unbiased but can suffer from high variance.

Trajectory-wise Doubly Robust (TDR)
----------

TDR combines TIS and DM to reduce the variance while being unbiased.

.. math::

    \hat{F}_{\mathrm{TDR}}(m, \pi; \mathcal{D})
    := \mathbb{E}_{n} \left[ w_{1:T} \left( \mathbb{I} \left \{\sum_{t=0}^T \gamma^t r_t \leq m \right \} - \hat{G}(m; s_0, a_0) \right) \right]
    + \hat{F}_{\mathrm{DM}}(m, \pi; \mathcal{D})

TDR reduces the variance of TIS while being unbiased, leveraging the model-based estimate (i.e., DM) as a control variate.

Evaluation Metrics of OPE/OPS
~~~~~~~~~~
Finally, we describe the metrics to evaluate the quality of OPE estimators and its OPS result.

* Mean Squared Error (MSE): This metrics measures the estimation accuracy as :math:`\sum_{\pi \in \Pi} (\hat{J}(\pi; \mathcal{D}) - J(\pi))^2 / |\Pi|`.

* Regret@k: This metrics measures how well the selected policy(ies) performs. In particular, Regret@1 indicates the expected performance difference between the (oracle) best policy and the selected policy as :math:`J(\pi^{\ast}) - J(\hat{\pi}^{\ast})`, where :math:`\pi^{\ast} := {\arg\max}_{\pi \in \Pi} J(\pi)` and :math:`\hat{\pi}^{\ast} := {\arg\max}_{\pi \in \Pi} \hat{J}(\pi; \mathcal{D})`.

* Spearman's Rank Correlation Coefficient: This metrics measures how well the raking of the candidate estimators are preserved in the OPE result.

* Type I and Type II Error Rate: This metrics measures how well an OPE estimator validates whether the policy performance surpasses the given safety threshold or not.

