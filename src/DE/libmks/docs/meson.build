toml_conf = configuration_data()
toml_conf.set('version', meson.project_version())

gidocgen = find_program('gi-docgen')

gidocgen_common_args = [
  '--quiet',
  '--no-namespace-dir',
]

if get_option('werror')
  gidocgen_common_args += ['--fatal-warnings']
endif

docs_dir = get_option('datadir') / 'doc'

if get_option('docs')
  source_toml = configure_file(
    input: 'Mks.toml.in',
    output: 'Mks.toml',
    configuration: toml_conf,
    install: true,
    install_dir: docs_dir / 'libmks1',
  )

  custom_target('mks-doc',
    input: [ source_toml, libmks_gir[0] ],
    output: 'libmks1',
    command: [
      gidocgen,
      'generate',
      gidocgen_common_args,
      '--config=@INPUT0@',
      '--output-dir=@OUTPUT@',
      '--content-dir=@0@'.format(meson.current_source_dir()),
      '@INPUT1@',
    ],
    build_by_default: true,
    install: true,
    install_dir: docs_dir,
  )
endif
