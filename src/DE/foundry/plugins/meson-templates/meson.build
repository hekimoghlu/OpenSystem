foundry_private_sources += files([
  'plugin.c',
])

gtk4_project_conf = configuration_data()
gtk4_project_conf.set('is_adwaita', 'false')
gtk4_project_conf.set('name', 'gtk4-app')
gtk4_project_conf.set('title', 'GTK 4 Application')
gtk4_project = configure_file(
          input: 'gtk4.project.in',
         output: 'gtk4.project',
  configuration: gtk4_project_conf,
)

gnome_project_conf = configuration_data()
gnome_project_conf.set('is_adwaita', 'true')
gnome_project_conf.set('name', 'gnome-app')
gnome_project_conf.set('title', 'GNOME Application')
gnome_project = configure_file(
          input: 'gtk4.project.in',
         output: 'gnome.project',
  configuration: gnome_project_conf,
)

foundry_private_sources += [
  gnome.compile_resources(
    'plugin-meson-templates-resources',
    'meson-templates.gresource.xml',
    c_name: '_plugin_meson_templates',
    dependencies: [gnome_project, gtk4_project],
  )
]
