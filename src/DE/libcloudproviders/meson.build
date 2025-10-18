project ('libcloudproviders', 'c',
  version: '0.3.6',
  meson_version: '>=0.63.0',
  license: 'LGPL-3.0-or-later',
  default_options: [
    'c_std=gnu99',
    'warning_level=2',
  ]
)

libcloudproviders_minor_version = meson.project_version().split('.')[1].to_int()
libcloudproviders_micro_version = meson.project_version().split('.')[2].to_int()

libcloudproviders_api_version = '0.3'
libcloudproviders_soversion = '0'
libcloudproviders_library_version = '@0@.@1@.@2@'.format(libcloudproviders_soversion, libcloudproviders_minor_version, libcloudproviders_micro_version)

cc = meson.get_compiler('c')

test_cflags = [
  '-Wno-unused-parameter',
  '-Wno-missing-field-initializers',
  '-Wstrict-prototypes',
  '-Wmissing-prototypes',
  '-Werror=implicit-function-declaration',
  '-Werror=pointer-arith',
  '-Werror=init-self',
  ['-Werror=format-security','-Werror=format=1'],
  '-Werror=date-time',
]
global_cflags = []
foreach cflag : test_cflags
  if cc.has_multi_arguments(cflag)
    global_cflags += cflag
  endif
endforeach
add_project_arguments(global_cflags, language: 'c')

glib_req_version = '2.56'
glib_ver = '>= @0@'.format(glib_req_version)
glib = dependency ('glib-2.0', version: glib_ver)
gio = dependency ('gio-2.0', version: glib_ver)

host_os = host_machine.system().to_lower()
if host_os.contains('mingw') or host_os.contains('cygwin') or host_os.contains('windows')
    gio_platform = dependency ('gio-windows-2.0', version: glib_ver)
else
    gio_platform = dependency ('gio-unix-2.0', version: glib_ver)
endif

add_project_arguments(
  '-DGLIB_VERSION_MIN_REQUIRED=GLIB_VERSION_@0@_@1@'.format(glib_req_version.split('.')[0], glib_req_version.split('.')[1]),
  '-DGLIB_VERSION_MAX_ALLOWED=GLIB_VERSION_@0@_@1@'.format(glib_req_version.split('.')[0], glib_req_version.split('.')[1]),
  language: 'c'
)

prefix = get_option ('prefix')
libdir = get_option ('libdir')
libexecdir = get_option ('libexecdir')
bindir = get_option ('bindir')
datadir = get_option ('datadir')
servicedir = join_paths (datadir, 'dbus-1', 'services')
incdir = join_paths (prefix, get_option('includedir'), 'cloudproviders')
girdir= join_paths(datadir, 'gir-1.0')
typelibdir = join_paths(libdir, 'girepository-1.0')

gnome = import('gnome')

subdir ('src')
subdir ('test')
if get_option ('documentation')
    subdir ('docs')
endif
summary({
  'bindir': bindir,
  'libdir': libdir,
  'libexecdir': libexecdir,
  'datadir': datadir,
  'includedir': get_option('includedir'),
}, section: 'Directories')
summary({
  'Introspection': get_option('introspection'),
  'Vala bindings': get_option('vapigen'),
  'Documentation': get_option('documentation'),
  'Installed tests': get_option('installed-tests'),
}, section: 'Configuration')
