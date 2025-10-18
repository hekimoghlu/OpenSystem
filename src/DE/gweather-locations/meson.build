project('gweather-locations',
  version: '2025.1',
  default_options: [
    'warning_level=2',
  ],
  license: 'GPL-2.0-or-later',
  meson_version: '>=1.0',
)

GETTEXT_PACKAGE = 'gweather-locations'

py = import('python').find_installation('python3', modules: ['gi'])
gen_locations_variant = find_program('build-aux/gen_locations_variant.py')

subdir('data')
subdir('po')

pkgconfig = import('pkgconfig')
pkgconfig.generate(
  name: meson.project_name(),
  description: 'Locations database for libgweather',
  install_dir: get_option('datadir') / 'pkgconfig',
  variables: {
    'locations_dtd': '${datadir}/gweather-locations/locations.dtd',
    'locations_xml': '${datadir}/gweather-locations/Locations.xml',
    'locations_db': '${libdir}/gweather-locations/Locations.bin',
  },
  uninstalled_variables: {
    'locations_dtd': meson.project_source_root() / 'data' / 'locations.dtd',
    'locations_xml': meson.project_source_root() / 'data' / 'Locations.xml',
    'locations_db': meson.project_build_root() / 'data' / 'Locations.bin',
  },
)
