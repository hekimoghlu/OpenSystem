expand_content_md_files = [
  'overview.md',
  'integration-client.md',
  'integration-server.md',
  'cloudproviders-dbus.md',
]

codegen = find_program('gdbus-codegen')
dbus_doc_files = custom_target(
  'dbus-doc',
  command: [codegen, '--generate-md', 'dbus', '@INPUT@', '--output-directory', '@OUTDIR@'],
  input: meson.project_source_root() / 'data/cloud-providers-dbus.xml',
  output: [
    'dbus-org.freedesktop.CloudProviders.Account.md',
    'dbus-org.freedesktop.CloudProviders.Provider.md',
  ]
)

toml_data = configuration_data()
toml_data.set('VERSION', meson.project_version())

libcloudproviders_toml = configure_file(
  input: 'libcloudproviders.toml.in',
  output: 'libcloudproviders.toml',
  configuration: toml_data
)

dependency(
  'gi-docgen', version: '>= 2021.1',
  fallback: ['gi-docgen', 'dummy_dep'],
  native: true,
  required: get_option('documentation')
)

gidocgen = find_program('gi-docgen')

docs_dir = datadir / 'doc'

custom_target('libcloudproviders-doc',
  input: [ libcloudproviders_toml, libcloudproviders_gir[0], dbus_doc_files ],
  output: 'libcloudproviders-@0@'.format(libcloudproviders_api_version),
  command: [
    gidocgen,
    'generate',
    '--quiet',
    '--config=@INPUT0@',
    '--output-dir=@OUTPUT@',
    '--no-namespace-dir',
    '--content-dir=@0@'.format(meson.current_source_dir()),
    '--content-dir=@0@'.format(meson.current_build_dir()),
    '@INPUT1@',
  ],
  depend_files: [ expand_content_md_files ],
  build_by_default: true,
  install: true,
  install_dir: docs_dir,
)

test('Validate docs',
  gidocgen,
  args: [
    'check',
    '--config', libcloudproviders_toml,
    '--add-include-path=@0@'.format(meson.current_build_dir() / '../src'),
    libcloudproviders_gir[0],
  ],
  depends: libcloudproviders_gir[0],
  suite: ['doc'],
)

