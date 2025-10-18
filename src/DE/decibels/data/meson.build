blueprints = custom_target('blueprints',
  input: files(
    'shortcuts-dialog.blp',
    'drag-overlay.blp',
    'empty.blp',
    'error.blp',
    'header.blp',
    'playback-rate-button.blp',
    'player.blp',
    'volume-button.blp',
    'window.blp',
  ),
  output: '.',
  command: [find_program('blueprint-compiler'), 'batch-compile', '@OUTPUT@', '@CURRENT_SOURCE_DIR@', '@INPUT@'],
)

desktop_file = i18n.merge_file(
  input: configure_file(
    input: 'org.gnome.Decibels.desktop.in.in',
  	output: application_id + '.desktop.in',
    configuration: conf
  ),
  output: application_id + '.desktop',
  type: 'desktop',
  po_dir: '../po',
  install: true,
  install_dir: decibels_datadir / 'applications'
)

desktop_utils = find_program('desktop-file-validate', required: false)
if desktop_utils.found()
  test('Validate desktop file', desktop_utils, args: [desktop_file])
endif

appstream_file = i18n.merge_file(
  input: configure_file(
  	input: 'org.gnome.Decibels.metainfo.xml.in.in',
  	output: application_id + '.metainfo.xml.in',
  	configuration: conf
	),
  output: application_id + '.metainfo.xml',
  po_dir: '../po',
  install: true,
  install_dir: decibels_datadir / 'metainfo'
)

# Validate Appdata
appstreamcli = find_program('appstreamcli', required: false)
if (appstreamcli.found())
  test('Validate appdata file',
    appstreamcli,
    args: ['validate', '--no-net', '--explain', appstream_file],
    workdir: meson.current_build_dir()
  )
endif

data_res = gnome.compile_resources(
  application_id + '.data',
  configure_file(
  	input: 'org.gnome.Decibels.data.gresource.xml.in',
  	output: application_id + '.data.gresource.xml',
  	configuration: conf
	),
  gresource_bundle: true,
  install: true,
  install_dir: decibels_pkgdatadir,
  dependencies: [appstream_file, blueprints],
)

configure_file(
  input: 'org.gnome.Decibels.service.in',
  output: application_id + '.service',
  configuration: conf,
  install: true,
  install_dir: decibels_datadir / 'dbus-1/services',
)

subdir('icons')
