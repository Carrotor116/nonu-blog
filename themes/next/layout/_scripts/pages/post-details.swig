{% if theme.instantclick.enabled %}
  <script type="text/javascript" src="{{ url_for(theme.js) }}/src/scrollspy.js?v={{ version }}" data-no-instant></script>
  <script type="text/javascript" src="{{ url_for(theme.js) }}/src/post-details.js?v={{ version }}" data-no-instant></script>
{% else %}
  <script type="text/javascript" src="{{ url_for(theme.js) }}/src/scrollspy.js?v={{ version }}"></script>
  <script type="text/javascript" src="{{ url_for(theme.js) }}/src/post-details.js?v={{ version }}"></script>
{% endif %}

