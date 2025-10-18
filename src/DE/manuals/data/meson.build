desktop_config = configuration_data()
desktop_config.set('APP_ID', app_id)
desktop_file_in = configure_file(
          input: 'app.devsuite.Manuals.desktop.in.in',
         output: 'app.devsuite.Manuals.desktop.in',
  configuration: desktop_config,
)
desktop_file = i18n.merge_file(
        input: desktop_file_in,
       output: '@0@.desktop'.format(app_id),
         type: 'desktop',
       po_dir: '../po',
      install: true,
  install_dir: get_option('datadir') / 'applications'
)

desktop_utils = find_program('desktop-file-validate', required: false)
if desktop_utils.found()
  test('Validate desktop file', desktop_utils, args: [desktop_file])
endif

appstream_config = configuration_data()
appstream_config.set('APP_ID', app_id)
appstream_file_in = configure_file(
          input: 'app.devsuite.Manuals.metainfo.xml.in.in',
         output: 'app.devsuite.Manuals.metainfo.xml.in',
  configuration: appstream_config,
)
appstream_file = i18n.merge_file(
        input: appstream_file_in,
       output: '@0@.metainfo.xml'.format(app_id),
       po_dir: '../po',
      install: true,
  install_dir: join_paths(get_option('datadir'), 'metainfo'),
)


appstreamcli = find_program('appstreamcli', required: false, disabler: true)
test('Validate appstream file', appstreamcli,
     args: ['validate', '--no-net', '--explain', appstream_file])

compile_schemas = find_program('glib-compile-schemas', required: false, disabler: true)
test('Validate schema file',
     compile_schemas,
     args: ['--strict', '--dry-run', meson.current_source_dir()])

dbusconf = configuration_data()
dbusconf.set('app_id', app_id)
dbusconf.set('bindir', join_paths(get_option('prefix'), get_option('bindir')))
configure_file(
          input: 'app.devsuite.Manuals.service.in',
         output: '@0@.service'.format(app_id),
  configuration: dbusconf,
        install: true,
    install_dir: join_paths(get_option('datadir'), 'dbus-1', 'services'),
)

gnome.compile_schemas(build_by_default: true)
install_data('app.devsuite.Manuals.gschema.xml',
  install_dir: join_paths(get_option('datadir'), 'glib-2.0/schemas')
)

subdir('icons')
