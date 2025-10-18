project('libmks', 'c',
          version: '0.1.6',
    meson_version: '>= 0.62.0',
  default_options: ['warning_level=2', 'werror=false', 'c_std=gnu11'],
)

api_version = '1'

cc = meson.get_compiler('c')
gnome = import('gnome')
pkg = import('pkgconfig')

project_c_args = ['-DMKS_COMPILATION']

# Setup config.h and ensure -I includes path to it
config_h = configuration_data()
config_h.set_quoted('GETTEXT_PACKAGE', 'libmks-@0@'.format(api_version))
config_h.set_quoted('PACKAGE_VERSION', meson.project_version())
if get_option('tracing')
  config_h.set10('MKS_DEBUG', true)
endif
add_project_arguments(['-I' + meson.project_build_root()], language: 'c')

# Check dependencies
glib_req_version = '2.75.0'
gtk_req_version = '4.13'

glib_req = '>= @0@'.format(glib_req_version)
gtk_req = '>= @0@'.format(gtk_req_version)

glib_major = glib_req_version.split('.')[0].to_int()
glib_minor = glib_req_version.split('.')[1].to_int()
gtk_major = gtk_req_version.split('.')[0].to_int()
gtk_minor = gtk_req_version.split('.')[1].to_int()

if glib_minor % 2 == 1
  glib_minor = glib_minor + 1
endif
if gtk_minor % 2 == 1
  gtk_minor = gtk_minor + 1
endif

libgio_dep = dependency('gio-2.0', version: glib_req)
libgiounix_dep = dependency('gio-unix-2.0', version: glib_req)
libgtk_dep = dependency('gtk4', version: gtk_req, fallback: ['gtk'])

# Ensure we restrict ABI usage to what we claim
project_c_args += [
  '-DGLIB_VERSION_MIN_REQUIRED=GLIB_VERSION_@0@_@1@'.format(glib_major, glib_minor),
  '-DGLIB_VERSION_MAX_ALLOWED=GLIB_VERSION_@0@_@1@'.format(glib_major, glib_minor),
  '-DGDK_VERSION_MIN_REQUIRED=GDK_VERSION_@0@_@1@'.format(gtk_major, gtk_minor),
  '-DGDK_VERSION_MAX_ALLOWED=GDK_VERSION_@0@_@1@'.format(gtk_major, gtk_minor),
]

test_c_args = [
  '-Wcast-align',
  '-Wdeclaration-after-statement',
  '-Werror=address',
  '-Werror=array-bounds',
  '-Werror=empty-body',
  '-Werror=implicit',
  '-Werror=implicit-function-declaration',
  '-Werror=incompatible-pointer-types',
  '-Werror=init-self',
  '-Werror=int-conversion',
  '-Werror=int-to-pointer-cast',
  '-Werror=main',
  '-Werror=misleading-indentation',
  '-Werror=missing-braces',
  '-Werror=missing-include-dirs',
  '-Werror=nonnull',
  '-Werror=overflow',
  '-Werror=parenthesis',
  '-Werror=pointer-arith',
  '-Werror=pointer-to-int-cast',
  '-Werror=redundant-decls',
  '-Werror=return-type',
  '-Werror=sequence-point',
  '-Werror=shadow',
  '-Werror=strict-prototypes',
  '-Werror=trigraphs',
  '-Werror=undef',
  '-Werror=write-strings',
  '-Wformat-nonliteral',
  '-Wignored-qualifiers',
  '-Wimplicit-function-declaration',
  '-Wlogical-op',
  '-Wmissing-declarations',
  '-Wmissing-format-attribute',
  '-Wmissing-include-dirs',
  '-Wmissing-noreturn',
  '-Wnested-externs',
  '-Wno-cast-function-type',
  '-Wno-dangling-pointer',
  '-Wno-missing-field-initializers',
  '-Wno-sign-compare',
  '-Wno-unused-parameter',
  '-Wold-style-definition',
  '-Wpointer-arith',
  '-Wredundant-decls',
  '-Wstrict-prototypes',
  '-Wswitch-default',
  '-Wswitch-enum',
  '-Wundef',
  '-Wuninitialized',
  '-Wunused',
  '-fno-strict-aliasing',
  ['-Werror=format-security', '-Werror=format=2'],
]
if get_option('buildtype') != 'plain'
  test_c_args += '-fstack-protector-strong'
endif
foreach arg: test_c_args
  if cc.has_multi_arguments(arg)
    project_c_args += arg
  endif
endforeach
hidden_visibility_args = []
if get_option('default_library') != 'static'
  if host_machine.system() == 'windows'
    config_h.set('DLL_EXPORT', true)
    if cc.get_id() == 'msvc'
      config_h.set('_MKS_EXTERN', '__declspec(dllexport) extern')
    elif cc.has_argument('-fvisibility=hidden')
      config_h.set('_MKS_EXTERN', '__attribute__((visibility("default"))) __declspec(dllexport) extern')
      hidden_visibility_args = ['-fvisibility=hidden']
    endif
  elif cc.has_argument('-fvisibility=hidden')
    config_h.set('_MKS_EXTERN', '__attribute__((visibility("default"))) extern')
    hidden_visibility_args = ['-fvisibility=hidden']
  endif
endif
project_c_args += hidden_visibility_args
add_project_arguments(project_c_args, language: 'c')

release_args = []
project_link_args = []
test_link_args = [
  '-Wl,-z,relro',
  '-Wl,-z,now',
]
if not get_option('buildtype').startswith('debug') and not meson.is_subproject()
  add_project_arguments(['-DG_DISABLE_CAST_CHECKS'], language: 'c')
  release_args += [ '-DG_DISABLE_ASSERT' ]
  test_link_args += [
    '-Wl,-Bsymbolic',
    '-fno-plt',
  ]
endif
foreach link_arg: test_link_args
  if cc.has_link_argument(link_arg)
    project_link_args += link_arg
  endif
endforeach
add_project_link_arguments(project_link_args, language: 'c')

subdir('lib')
subdir('tools')

have_tests = get_option('tests')
if have_tests
 subdir('tests')
endif
summary('Enabled', have_tests, section: 'Tests')

if get_option('docs') and get_option('introspection').enabled()
  subdir('docs')
endif

configure_file(output: 'config.h', configuration: config_h)
