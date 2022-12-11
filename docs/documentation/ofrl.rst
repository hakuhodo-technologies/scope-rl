==========
OFRL Package Reference
==========

dataset module
----------
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.dataset.base
    ofrl.dataset.synthetic

policy module
----------
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.policy.head

ope module
----------

pipeline
^^^^^^
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.ope.input
    ofrl.ope.ope
    ofrl.ope.ops

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

weight and value learning methods
^^^^^^
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.ope.weight_value_learning.augmented_lagrangian_learning_discrete
    ofrl.ope.weight_value_learning.augmented_lagrangian_learning_continuous
    ofrl.ope.weight_value_learning.minimax_weight_learning_discrete
    ofrl.ope.weight_value_learning.minimax_weight_learning_continuous
    ofrl.ope.weight_value_learning.minimax_value_learning_discrete
    ofrl.ope.weight_value_learning.minimax_value_learning_continuous

others
^^^^^^
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.ope.online

others
----------
.. autosummary::
    :toctree: _autosummary
    :recursive:
    :nosignatures:

    ofrl.utils
