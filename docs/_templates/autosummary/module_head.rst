{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :exclude-members: build_with_dataset,build_with_env,copy_policy_from,copy_policy_optim_from,copy_q_function_from,copy_q_function_optim_from,fitter,update,inner_update,create_impl,inner_create_impl,get_action_type,load_model,save_model,from_json,save,save_policy,set_grad_step,reset_optimizer_states,impl,grad_step,action_size,batch_size,gamma,config,reward_scaler,observation_scaler,action_scaler,fit,fit_online,observation_shape,predict,predict_value,sample_action

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :nosignatures:
      :template: class_head
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. autosummary::
   :toctree:
   :recursive:
   :template: module_head
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}