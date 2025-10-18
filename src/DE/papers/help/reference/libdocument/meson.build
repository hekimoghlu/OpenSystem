libppsdocument_toml = configure_file(
  input: 'libppsdocument.toml.in',
  output: 'libppsdocument-' + pps_api_version + '.toml',
  configuration: toml_conf,
  install: true,
  install_dir: docs_dir / 'libppsdocument',
)

custom_target('libppsdocument-doc',
  input: [ libppsdocument_toml, libppsdocument_gir[0] ],
  output: 'libppsdocument',
  command: [
    gidocgen,
    'generate',
    '--quiet',
    '--fatal-warnings',
    '--config=@INPUT0@',
    '--output-dir=@OUTPUT@',
    '--no-namespace-dir',
    '--content-dir=@0@'.format(meson.current_source_dir()),
    '@INPUT1@',
  ],
  build_by_default: true,
  install: true,
  install_dir: docs_dir,
)
