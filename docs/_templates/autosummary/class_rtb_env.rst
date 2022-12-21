{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :exclude-members: standard_bid_price,close,render,np_random,render_mode,unwrapped,spec,reward_range

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
