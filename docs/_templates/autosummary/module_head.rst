{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :exclude-members: build_with_dataset,build_with_env,copy_policy_from,copy_q_function_from,fitter,generate_new_data,create_impl,get_action_type,get_params,load_model,save_model,from_json,save_params,save_policy,set_active_logger,set_grad_step,set_params,impl,grad_step,n_frames,action_size,batch_size,gamma,n_steps,reward_scaler,scaler,fit,fit_online,update,collect,action_logger,action_scalar,observation_space,predict,predict_value,fit_batch_online,sample_action

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