==========
OFRL Package Reference
==========

.. _ofrl_api_dataset:

dataset module
----------
.. autosummary::
    :toctree: _autosummary/dataset
    :recursive:
    :nosignatures:

    ofrl.dataset.base
    ofrl.dataset.synthetic

.. _ofrl_api_policy:

policy module
----------
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:
    :template: module_head

    ofrl.policy.head

.. _ofrl_api_ope:

ope module
----------

.. _ofrl_api_ope_pipeline:

pipeline
^^^^^^
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.ope.input
    ofrl.ope.ope
    ofrl.ope.ops

.. _ofrl_api_ope_estimators:

OPE estimators
^^^^^^
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.ope.estimators_base
    ofrl.ope.basic_estimators_discrete
    ofrl.ope.basic_estimators_continuous
    ofrl.ope.marginal_estimators_discrete
    ofrl.ope.marginal_estimators_continuous
    ofrl.ope.cumulative_distribution_estimators_discrete
    ofrl.ope.cumulative_distribution_estimators_continuous

.. _ofrl_api_ope_weight_and_value_learning:

weight and value learning methods
^^^^^^
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:
    :template: module_weight_value_learning

    ofrl.ope.weight_value_learning.base
    ofrl.ope.weight_value_learning.function
    ofrl.ope.weight_value_learning.augmented_lagrangian_learning_discrete
    ofrl.ope.weight_value_learning.augmented_lagrangian_learning_continuous
    ofrl.ope.weight_value_learning.minimax_weight_learning_discrete
    ofrl.ope.weight_value_learning.minimax_weight_learning_continuous
    ofrl.ope.weight_value_learning.minimax_value_learning_discrete
    ofrl.ope.weight_value_learning.minimax_value_learning_continuous

.. _ofrl_api_ope_utils:

others
^^^^^^
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.ope.online

.. _ofrl_api_utils:

others
----------
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.utils

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
