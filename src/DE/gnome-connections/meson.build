project('gnome-connections', ['c', 'vala'],
          version: '49.0',
    meson_version: '>= 0.59.0',
  default_options: [ 'warning_level=2',
                   ],
)

cc = meson.get_compiler ('c')
i18n = import('i18n')
valac = meson.get_compiler ('vala')
gnome = import('gnome')

src_dir = join_paths (meson.source_root (), 'src')
locale_dir = join_paths (get_option ('prefix'), get_option ('localedir'))

conf = configuration_data ()
conf.set_quoted ('GETTEXT_PACKAGE', meson.project_name ())
conf.set_quoted ('LOCALEDIR', locale_dir)
conf.set_quoted ('VERSION', '@0@-@VCS_TAG@'.format (meson.project_version()))
conf.set_quoted ('PACKAGE_TARNAME', meson.project_name ())

if get_option ('profile') == 'development'
  profile = '.Devel'
else
  profile = ''
endif

application_id = 'org.gnome.Connections@0@'.format (profile)
conf.set_quoted ('PROFILE', profile)
conf.set_quoted ('APPLICATION_ID', application_id)

connections_deps = [
  valac.find_library ('config', dirs: src_dir),
  cc.find_library('m'),
  dependency('gio-2.0', version: '>= 2.50'),
  dependency('gtk+-3.0', version: '>= 3.22'),
  dependency('gtk-vnc-2.0', version: '> 0.4.4'),
  dependency('gvncpulse-1.0'),
  dependency('libhandy-1', version: '>= 1.6.0'),
  dependency ('libxml-2.0', version: '>= 2.7.8'),
  dependency ('libsecret-1'),
]

gtk_frdp_dep = dependency('gtk-frdp-0.2', required: false)
if gtk_frdp_dep.found()
  connections_deps += gtk_frdp_dep
else
  gtk_frdp_dep = subproject('gtk-frdp', default_options: [
    'package_subdir=' + meson.project_name()
  ])

  connections_deps += gtk_frdp_dep.get_variable('gtk_frdp_vapi')
endif

spice_dep = dependency ('spice-client-gtk-3.0', required: false)
spice_vala_dep = valac.find_library ('spice-client-gtk-3.0', required: false)
spice_dep_found = spice_dep.found() and spice_vala_dep.found()
conf.set10('HAS_SPICE', spice_dep_found)
if spice_dep_found
  connections_deps += [
    spice_dep,
    spice_vala_dep,
  ]
endif

config_h = declare_dependency (
  sources: vcs_tag (
    command: ['git', 'rev-parse', '--short', 'HEAD'],
    fallback: '',
    input: configure_file (
      output: 'config.h.in',
      configuration: conf
    ),
    output: 'config.h'
  )
)
connections_deps += config_h

#configure_file (output: 'config.h', configuration: conf)
config_h_dir = include_directories ('.')

subdir('data')
subdir('help')
subdir('src')
subdir('po')

gnome.post_install(
  glib_compile_schemas: true,
  gtk_update_icon_cache: true,
  update_desktop_database: true,
)
