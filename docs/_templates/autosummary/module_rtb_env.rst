{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :exclude-members: standard_bid_price,close,render,np_random,render_mode,unwrapped,spec,reward_range

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
      :template: class_rtb_env
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
   :template: module_rtb_env
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}