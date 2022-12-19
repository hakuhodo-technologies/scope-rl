{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :exclude-members: build_with_dataset,build_with_env,copy_policy_from,copy_q_function_from,fitter,generate_new_data,create_impl,get_action_type,get_params,load_model,save_model,from_json,save_params,save_policy,set_active_logger,set_grad_step,set_params,impl,grad_step,n_frames,action_size,batch_size,gamma,n_steps,reward_scaler,scaler,fit,fit_online,update,collect,action_logger,action_scalar,observation_space,predict,predict_value,fit_batch_online,sample_action

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
