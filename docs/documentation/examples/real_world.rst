Guidelines for Preparing Real-World Datasets
==========

Here, we provide the guideline for preparing logged datasets and inputs that are compatible to SCOPE-RL.

Logged Datasets
~~~~~~~~~~

In real-world experiments, ``logged_dataset`` should contain the following keys. 
For the keys that are (optional), please use ``None`` values when the data is unavailable.

.. code-block:: python

    key: [
        size,
        n_trajectories,
        step_per_trajectory,
        action_type,
        n_actions,
        action_dim,
        action_keys,
        action_meaning,
        state_dim,
        state_keys,
        state,
        action,
        reward,
        done,
        terminal,
        info,
        pscore,
        behavior_policy,
        dataset_id,
    ]

* ``step_per_trajectory``: number of steps in a trajectory, *int*
* ``n_trajectories``: number of trajectories, *int (optional)*
* ``size``: number of data tuples, which is given by the multiplication of ``n_trajectories`` and ``step_per_trajectory``, *int (optional)*
* ``state``: state observation, *np.ndarray of shape (size, )*
* ``action``: action chosen by the behavior policy, *np.ndarray*
* ``reward``: reward observation, *np.ndarray of shape (size, )*
* ``done``: whether an episode ends or not (due to the consequence of agent action), *np.ndarray of (size, )*
* ``terminal``: wether an episode terminates or not (due to fixed episode lengths), *np.ndarray of (size, )*
* ``pscore``: probability of the behavior policy choosing the observed action, *np.ndarray of (size, ), (optional)*

* ``action_type``: type of action, *str (either "discrete" or "continuous")*
* ``n_actions``: number of actions (discrete action case), *int (optional)*
* ``action_dim``: dimension of actions (continuous action case), *int (optional)*
* ``action_keys``: disctionary containing the name of actions, *dict (optional)*
* ``action_meaning``: np.ndarray to map action index to actual actions, *dict (optional)*
* ``state_dim``: dimension of states, *int, (optional)*
* ``state_keys``: disctionary containing the name of states, *int (optional)*
* ``info``: info obtained during the interaction of the agent, *dict (optional)*
* ``behavior_policy``: name of the behavior policy, *str*
* ``dataset_id``: dataset id , *int*

Note that, when ``pscore`` is available, the importance sampling-based estimators are applicable to OPE.
The shape of ``action`` is *(size, n_actions)* in discrete action cases, while it is *(size, action_dim)* in continuous action cases.

Input Dict
~~~~~~~~~~
Then, ``input_dict`` should contain the following keys for each evaluation policy (in ``input_dict[evaluation_policy_name]``).
For the keys that are (optional), please use ``None`` values when the data is unavailable.

.. code-block:: python

    key: [evaluation_policy_name][
        evaluation_policy_action,
        evaluation_policy_action_dist,
        state_action_value_prediction,
        initial_state_value_prediction,
        state_action_marginal_importance_weight,
        state_marginal_importance_weight,
        on_policy_policy_value,
        gamma,
        behavior_policy,
        evaluation_policy,
        dataset_id,
    ]

* ``evaluation_policy_action``: action chosen by the evaluation policy (continuous action case), *np.ndarray of shape (size, )*
* ``evaluation_policy_action_dist``: action distribution of the evaluation policy (discrete action case), *np.ndarray of shape (size, n_actions)*
* ``state_action_value_prediction``: predicted value for observed state-action pairs, *np.ndarray*
* ``initial_state_value_prediction``: predicted value for observed initial actions, *np.ndarray pf shape (n_trajectories, ) (optional)*
* ``state_action_marginal_importance_weight``: estimated state-action marginal importance weight, *np.ndarray of (size, ) (optional)*
* ``state_marginal_importance_weight``: estimated state-marginal importance weight, *np.ndarray of (size, ) (optional)*
* ``on_policy_policy_value``: on-policy policy value of the evaluation policy, *float (optional)*
* ``gamma``: discount factor, *float*

* ``behavior_policy``: name of the behavior policy, *str*
* ``evaluation_policy``: name of the evaluation policy, *str*
* ``dataset_id``: dataset id , *int*

Note that, when ``state_action_value_prediction`` and ``initial_state_value_predictions`` are available, 
the model-based and hybrid estimators (e.g., DM and DR) are applicable to OPE.
On the other side, when ``state_action_marginal_importance_weight`` and ``state_marginal_importance_weight`` are available, 
the marginal importance-sampling based estimators are apllicable to OPE.
Finally, the assessments of OPE methods become feasible when ``on-policy policy value`` is available.

The shape of ``state_action_value_prediction`` is *(size, n_actions)* in discrete action cases, while it is *(size, 2)* in continuous action cases.
In continuous action case, index 0 of ``axis=1`` should contain the predicted values for the actions chosen by the behavior policy, whreas index 1 of ``axis=1`` should contain those of evaluation policy. 

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
                :link: /documentation/subpackages/custom_estimators
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                Next >>>
                **Custom OPE Estimators**

