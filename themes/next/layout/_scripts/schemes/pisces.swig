{%
  set scripts = [
    'src/affix.js',
    'src/schemes/pisces.js'
  ]
%}

{% for script in scripts %}
  {% if theme.instantclick.enabled %}
    <script type="text/javascript" src="{{ url_for(theme.js) }}/{{ script }}?v={{ version }}" data-no-instant></script>
  {% else %}
    <script type="text/javascript" src="{{ url_for(theme.js) }}/{{ script }}?v={{ version }}"></script>
  {% endif %}
{% endfor %}
