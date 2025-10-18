api_version = '0.1'

govf_sources = [
  'govf-disk.c',
  'govf-package.c',
]

govf_headers = [
  'govf-disk.h',
  'govf-package.h',
  'govf.h',
]

version_split = meson.project_version().split('.')
MAJOR_VERSION = version_split[0]
MINOR_VERSION = version_split[1]
MICRO_VERSION = version_split[2]

version_conf = configuration_data()
version_conf.set('VERSION', meson.project_version())
version_conf.set('MAJOR_VERSION', MAJOR_VERSION)
version_conf.set('MINOR_VERSION', MINOR_VERSION)
version_conf.set('MICRO_VERSION', MICRO_VERSION)

govf_header_subdir = join_paths(package_subdir, 'govf')
govf_header_dir = join_paths(get_option('includedir'), package_subdir)

cc = meson.get_compiler('c')
vala = meson.get_compiler('vala')

govf_deps = [
  dependency('libarchive'),
  dependency('gio-2.0', version: '>= 2.50'),
  dependency('libxml-2.0'),
  cc.find_library('m'),
]

govf_lib = shared_library('govf-' + api_version,
  govf_sources,
  dependencies: govf_deps,
  install: true,
  install_dir: libdir
)

govf_dep = declare_dependency(
  sources: govf_headers,
  dependencies: govf_deps,
  link_with: govf_lib,
  include_directories: include_directories('.'),
)

install_headers(govf_headers, subdir: govf_header_subdir)

govf_gir = gnome.generate_gir(govf_lib,
  sources: govf_sources + govf_headers,
  nsversion: api_version,
  namespace: 'Govf',
  symbol_prefix: 'govf',
  identifier_prefix: 'Govf',
  link_with: govf_lib,
  includes: ['Gio-2.0'],
  install: true,
  install_dir_gir: girdir,
  install_dir_typelib: typelibdir,
  extra_args: [ '--c-include=govf.h', '--quiet' ],
)

govf_vapi = gnome.generate_vapi('govf-' + api_version,
  sources: govf_gir[0],
  packages: [ 'gio-2.0' ],
  install: true,
  install_dir: vapidir,
  metadata_dirs: [ meson.current_source_dir() ],
)

pkg = import('pkgconfig')

pkg.generate(
  description: 'A shared library for ...',
    libraries: govf_lib,
         name: 'govf',
     filebase: 'govf-' + api_version,
      version: meson.project_version(),
      subdirs: 'govf',
     requires: 'glib-2.0',
  install_dir: join_paths(libdir, 'pkgconfig')
)
