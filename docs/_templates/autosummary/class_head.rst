{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :exclude-members: build_with_dataset,build_with_env,copy_policy_from,copy_policy_optim_from,copy_q_function_from,copy_q_function_optim_from,fitter,update,inner_update,create_impl,inner_create_impl,get_action_type,load_model,save_model,from_json,save,save_policy,set_grad_step,reset_optimizer_states,impl,grad_step,action_size,batch_size,gamma,config,reward_scaler,observation_scaler,action_scaler,fit,fit_online,observation_shape,predict,predict_value,sample_action

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }},

.. autosummary::
    :nosignatures:
{% for item in methods %}
    {%- if not item.startswith('_') %}
    .. automethod:: {{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: {{ _('Functions') }}

.. autosummary::
    :nosignatures:
{% for item in functions %}
    .. automethod:: {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
