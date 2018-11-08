{%
  set js_commons = [
    'src/utils.js'
  ]
%}

{% if theme.motion.enable %}
        <script type="text/javascript" src="{{ url_for(theme.js) }}/src/motion.js?v={{ version }}" data-instant></script>
{% endif %}

{% for common in js_commons %}

  {% if theme.instantclick.enabled %}
    {% set dni = false %}
    {% for i in theme.instantclick.data_no_instant %}
      {% if common.indexOf(i+".js") > 0 %}
         {% set dni = true %}
      {% endif %}
    {% endfor %}
    {% if dni %}
      <script type="text/javascript" src="{{ url_for(theme.js) }}/{{ common }}?v={{ version }}" data-no-instant></script>
    {% else %}
      {% set di = false %}
      {% for i in theme.instantclick.data_instant %}
        {% if common.indexOf(i+".js") > 0 %}
          {% set di = true %}
        {% endif %}
      {% endfor %}
      {% if di %}
        <script type="text/javascript" src="{{ url_for(theme.js) }}/{{ common }}?v={{ version }}" data-instant></script>
      {% else %}
        <script type="text/javascript" src="{{ url_for(theme.js) }}/{{ common }}?v={{ version }}"></script>
      {% endif %}
    {% endif %}
  {% else %}
    <script type="text/javascript" src="{{ url_for(theme.js) }}/{{ common }}?v={{ version }}"></script>
  {% endif %}

{% endfor %}
