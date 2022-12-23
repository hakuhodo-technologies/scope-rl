{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :inherited-members:

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }}

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
