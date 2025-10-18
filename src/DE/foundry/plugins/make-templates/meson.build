foundry_private_sources += files([
  'plugin.c',
])

foundry_private_sources += [
  gnome.compile_resources(
    'plugin-make-templates-resources',
    'make-templates.gresource.xml',
    c_name: '_plugin_make_templates',
  ),
]
