Example Codes for Custom OPE Estimators
==========

Here, we show example codes for implementing custom OPE estimators.


Estimator for Basic OPE
~~~~~~~~~~

To define a custom OPE etimator, use :class:`BaseOffPolicyEstimator` and implement the following abstract methods.

* :class:`_estimate_trajectory_value`: estimate policy value for each trajectory in the logged dataset
* :class:`estimate_policy_value`: estimate policy value for a logged dataset
* :class:`estimate_interval`: estimate confidence intervals for OPE estimate for a logged dataset

We provide an example of implementing a naive average estimator, which returns the naive average of the reward observed in the logged dataset as follows.

.. code-block:: python

    from scope_rl.ope import BaseOffPolicyEstimator
    from dataclasses import dataclass

    @dataclass
    class NaiveAverageEstimator(BaseOffPolicyEstimator):

        estimator_name = "naive_average"

        def __post__init__(self):
            self.action_type = "discrete"  # "continuous"

        def _estimate_trajectory_value(
            self,
            step_per_trajectory: int,  # length of a trajectory
            reward: np.ndarray,  # step-wise reward
            gamma: float = 1.0,  # discount factor
            **kwargs,
        ) -> np.ndarray:
            """Estimate policy value for each data tuple in the logged dataset."""
            reward = reward.reshape((-1, step_per_trajectory))
            discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
            estimated_trajectory_value = (discount[np.newaxis, :] * reward).sum(
                axis=1
            )
            return estimated_trajectory_value

        def estimate_policy_value(
            self,
            step_per_trajectory: int,
            reward: np.ndarray,
            gamma: float = 1.0,
            **kwargs,
        ) -> float:
            """Estimate policy value for a logged dataset."""
            estimated_policy_value = self._estimate_trajectory_value(
                step_per_trajectory=step_per_trajectory,
                reward=reward,
                gamma=gamma,
            ).mean()
            return estimated_policy_value

        def estimate_interval(
            step_per_trajectory: int,
            reward: np.ndarray,
            gamma: float = 1.0,
            alpha: float = 0.05,
            ci: str = "bootstrap",
            n_bootstrap_samples: int = 10000,
            random_state: Optional[int] = None,
            **kwargs,
        ) -> Dict[str, float]:
            """Estimate confidence intervals for OPE estimate for a logged dataset"""
            estimated_trajectory_value = self._estimate_trajectory_value(
                step_per_trajectory=step_per_trajectory,
                action=action,
                reward=reward,
                pscore=pscore,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
                gamma=gamma,
            )
            return self._estimate_confidence_interval[ci](
                samples=estimated_trajectory_value,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

To add the inequality to derive confidence intervals, override the following property.

.. code-block:: python

    from scope_rl.ope import DiscreteTrajectoryWiseImportanceSampling

    def estimate_condifence_interval_by_ttest(
        samples: np.ndarray,
        alpha: float = 0.05,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence intervals by Student's T-test."""
        n = len(samples)
        t = scipy.stats.t.ppf(1 - alpha, n - 1)
        mean = samples.mean()
        ci = t * samples.std(ddof=1) / np.sqrt(n)
        return {
            "mean": mean,
            f"{100 * (1. - alpha)}% CI (lower)": mean - ci,
            f"{100 * (1. - alpha)}% CI (upper)": mean + ci,
        }

    @dataclass
    class CustomHighConfidenceTIS(DiscreteTrajectoryWiseImportanceSampling):

        @property
        def _estimate_confidence_interval(self) -> Dict[str, Callable]:
            return {
                "custom_ci": estimate_confidence_interval_by_ttest
            }

Note that, an OPE estimator can take the following inputs from ``logged_dataset`` and ``input_dict``.

(logged datasets)

* ``step_per_trajectory``: number of steps in a trajectory
* ``state``: state observation
* ``action``: action chosen by the behavior policy
* ``reward``: reward observation
* ``done``: whether an episode ends or not (due to the consequence of agent action)
* ``terminal``: whether an episode terminates or not (due to fixed episode lengths)
* ``pscore``: probability of the behavior policy choosing the observed action

(input dict)

* ``evaluation_policy_action``: action chosen by the evaluation policy (continuous action case)
* ``evaluation_policy_action_dist``: action distribution of the evaluation policy (discrete action case)
* ``state_action_value_prediction``: predicted value for observed state-action pairs
* ``initial_state_value_prediction``: predicted value for observed initial actions
* ``state_action_marginal_importance_weight``: estimated state-action marginal importance weight
* ``state_marginal_importance_weight``: estimated state-marginal importance weight
* ``on_policy_policy_value``: on-policy policy value of the evaluation policy
* ``gamma``: discount factor


.. dropdown:: Other auxiliary information

    * ``size``: number of data tuples
    * ``n_trajectories``: number of trajectories
    * ``action_type``: type of action, either "discrete" or "continuous"
    * ``n_actions``: number of actions (discrete action case)
    * ``action_dim``: dimension of actions (continuous action case)
    * ``action_keys``: disctionary containing the name of actions (optional)
    * ``action_meaning``: np.ndarray to map action index to actual actions
    * ``state_dim``: dimension of states 
    * ``state_keys``: disctionary containing the name of states (optional)
    * ``info``: info obtained during the interaction of the agent
    * ``behavior_policy``: name of the behavior policy
    * ``evaluation_policy``: name of the evaluation policy
    * ``dataset_id``: dataset id 


Estimator for Cumulative Distribution OPE
~~~~~~~~~~

To define a custom CD-OPE estimator, use :class:`BaseCumulativeDistributionOPEEstimator` and implement the following abstract methods.

* :class:`estimate_cumulative_distribution_function`: estimate cumulative distribution function (CDF)
* :class:`estimate_mean`: estimate policy value from CDF
* :class:`estimate_variance`: estimate variance from CDF
* :class:`estimate_conditional_value_at_risk`: estimate conditional value at risk (CVaR) from CDF
* :class:`estimate_interquartile_range`: estimate interquartile range from CDF

We provide an example of implementing a naive cdf estimator, which does not apply importance sampling as follows.

.. code-block:: python

    from scope_rl.ope import BaseCumulativeDistributionOPEEstimator

    @dataclass
    class NaiveCumulativeDistributionEstimator(BaseCumulativeDistributionOPEEstimator):

        estimator_name = "naive_cdf"

        def __post__init__(self):
            self.action_type = "discrete"  # "continuous"

        def estimate_cumulative_distribution_function(
            self,
            step_per_trajectory: int,
            reward: np.ndarray,
            reward_scale: np.ndarray,  # bins of the CDF
            gamma: float = 1.0,
            **kwargs,
        ) -> Tuple[np.ndarray]:
            """Estimate cumulative distribution function."""
            reward = reward.reshape((-1, step_per_trajectory))
            discount = np.full(step_per_trajectory, gamma).cumprod() / gamma
            trajectory_reward = (discount[np.newaxis, :] * reward).sum(
                axis=1
            )

            sort_idxes = trajectory_wise_reward.argsort()
            sorted_importance_weight = trajectory_wise_importance_weight[sort_idxes]
            cumulative_density = np.clip(sorted_importance_weight.cumsum() / n, 0, 1)

            trajectory_wise_reward = np.clip(
                trajectory_wise_reward, reward_scale.min(), reward_scale.max()
            )
            histogram = np.histogram(
                trajectory_wise_reward, bins=reward_scale, density=False
            )[0]

            idx = histogram.cumsum().astype(int) - 1
            idx = np.where(idx < 0, 0, idx)

            cumulative_density = cumulative_density[idx]
            return np.insert(cumulative_density, 0, 0)

        def estimate_mean(
            self,
            reward: np.ndarray,
            reward_scale: np.ndarray,
            gamma: float = 1.0,
            **kwargs,
        ) -> float:
            """Estimate mean (i.e., policy value) from CDF."""
            cumulative_density = self.estimate_cumulative_distribution_function(
                step_per_trajectory=step_per_trajectory,
                reward=reward,
                reward_scale=reward_scale,
                gamma=gamma,
            )
            return (np.diff(cumulative_density) * reward_scale[1:]).sum()

        def estimate_variance(
            self,
            reward: np.ndarray,
            reward_scale: np.ndarray,
            gamma: float = 1.0,
            **kwargs,
        ) -> float:
            """Estimate variance from CDF."""
            cumulative_density = self.estimate_cumulative_distribution_function(
                step_per_trajectory=step_per_trajectory,
                reward=reward,
                reward_scale=reward_scale,
                gamma=gamma,
            )
            mean = (np.diff(cumulative_density) * reward_scale[1:]).sum()
            return (np.diff(cumulative_density) * (reward_scale[1:] - mean) ** 2).sum()

        def estimate_conditional_value_at_risk(
            self,
            reward: np.ndarray,
            reward_scale: np.ndarray,
            gamma: float = 1.0,
            alphas: Optional[np.ndarray] = None,  # the proportions of the sided region
            **kwargs,
        ):
            """Estimate CVaR from CDF."""
            if alphas is None:
                alphas = np.linspace(0, 1, 21)

            cumulative_density = self.estimate_cumulative_distribution_function(
                step_per_trajectory=step_per_trajectory,
                reward=reward,
                reward_scale=reward_scale,
                gamma=gamma,
            )

            cvar = np.zeros_like(alphas)
            for i, alpha in enumerate(alphas):
                idx_ = np.nonzero(cumulative_density[1:] > alpha)[0]
                if len(idx_) == 0:
                    cvar[i] = (
                        np.diff(cumulative_density) * reward_scale[1:]
                    ).sum() / cumulative_density[-1]
                elif idx_[0] == 0:
                    cvar[i] = reward_scale[1]
                else:
                    lower_idx_ = idx_[0]
                    relative_probability_density = (
                        np.diff(cumulative_density)[: lower_idx_ + 1]
                        / cumulative_density[lower_idx_ + 1]
                    )
                    cvar[i] = (
                        relative_probability_density * reward_scale[1 : lower_idx_ + 2]
                    ).sum()

            return cvar

        def estimate_interquartile_range(
            self,
            reward: np.ndarray,
            reward_scale: np.ndarray,
            gamma: float = 1.0,
            alphas: float = 0.05,  # the proportion of the sided region
            **kwargs,
        ) -> Dict[str, float]:
            """Estimate interquartile range from CDF."""

            cumulative_density = self.estimate_cumulative_distribution_function(
                step_per_trajectory=step_per_trajectory,
                reward=reward,
                reward_scale=reward_scale,
                gamma=gamma,
            )

            lower_idx_ = np.nonzero(cumulative_density > alpha)[0]
            median_idx_ = np.nonzero(cumulative_density > 0.5)[0]
            upper_idx_ = np.nonzero(cumulative_density > 1 - alpha)[0]

            estimated_interquartile_range = {
                "median": self._target_value_given_idx(
                    median_idx_, reward_scale=reward_scale
                ),
                f"{100 * (1. - alpha)}% quartile (lower)": self._target_value_given_idx(
                    lower_idx_,
                    reward_scale=reward_scale,
                ),
                f"{100 * (1. - alpha)}% quartile (upper)": self._target_value_given_idx(
                    upper_idx_,
                    reward_scale=reward_scale,
                ),
            }

            return estimated_interquartile_range

Note that, the available inputs are the same with basic OPE.

.. seealso::

    Finally, contribution to SCOPE-RL with a new OPE estimator is more than welcome! Please read `the guidelines for contribution (CONTRIBUTING.md) <https://github.com/hakuhodo-technologies/scope-rl/blob/main/CONTRIBUTING.md>`_.

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
                :link: /documentation/examples/index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Usage**

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
                :link: /documentation/examples/real_world
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Real-World Datasets**
