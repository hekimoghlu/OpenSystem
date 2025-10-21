connections_sources = [
  'actions-popover.vala', 
  'assistant.vala',
  'application.vala',
  'connection.vala',
  'collection-view.vala',
  'collection-view-child.vala',
  'database.vala',
  'display-view.vala',
  'main.vala',
  'notifications.vala',
  'dialogs.vala',
  'onboarding-dialog.vala',
  'onboarding-dialog-page.vala',
  'preferences-widgets.vala',
  'rdp-connection.vala',
  'rdp-preferences-window.vala',
  'topbar.vala',
  'vnc-connection.vala',
  'vnc-preferences-window.vala',
  'window.vala',
]

if spice_dep_found
  add_project_arguments('-D', 'HAS_SPICE', language: 'vala')

  connections_sources += [
    'spice-connection.vala',
    'spice-preferences-window.vala',
  ]
endif

connections_sources += gnome.compile_resources('connections-resources',
  'connections.gresource.xml',
  c_name: 'connections'
)

c_args = [
    '-DGETTEXT_PACKAGE="gnome-connections"',
    '-w',
]

vala_args = [
  '--enable-experimental',
  '--disable-since-check',
  '--target-glib=2.50',
]

executable('gnome-connections', connections_sources,
  include_directories: config_h_dir, 
  vala_args: vala_args,
  c_args: c_args,
  dependencies: connections_deps,
  install: true,
  install_rpath: join_paths (get_option ('prefix'), get_option ('libdir'), meson.project_name ())
)
