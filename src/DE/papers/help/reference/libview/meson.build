expand_content_md_files = [
  'profiling.md',
]

libppsview_toml = configure_file(
  input: 'libppsview.toml.in',
  output: 'libppsview.toml',
  configuration: toml_conf,
  install: true,
  install_dir: docs_dir / 'libppsview',
)

custom_target('libppsview-doc',
  input: [ libppsview_toml, libppsview_gir[0] ],
  output: 'libppsview',
  command: [
    gidocgen,
    'generate',
    '--quiet',
    '--fatal-warnings',
    '--config=@INPUT0@',
    '--add-include-path=@0@'.format(meson.current_build_dir() / '..' / '..' / '..' / 'libdocument'),
    '--output-dir=@OUTPUT@',
    '--no-namespace-dir',
    '--content-dir=@0@'.format(meson.current_source_dir()),
    '@INPUT1@',
  ],
  depends: [ libppsdocument_gir[0] ],
  build_by_default: true,
  depend_files: [ expand_content_md_files ],
  install: true,
  install_dir: docs_dir,
)
