foundry_private_sources += files([
  'plugin.c',
  'plugin-simple-template-provider.c',
])

foundry_private_sources += [
  gnome.compile_resources(
    'plugin-simple-templates-resources',
    'simple-templates.gresource.xml',
    c_name: '_plugin_simple_templates',
  )
]
