project('libspelling', 'c',
          version: '0.4.10',
    meson_version: '>= 0.62.0',
  default_options: [ 'warning_level=2', 'werror=false', 'c_std=gnu11', ],
)

libgtk_dep = dependency('gtk4', version: '>= 4.15.5')
gtksource_dep = dependency('gtksourceview-5', version: '>= 5.10.0')
libgio_dep = dependency('gio-2.0')

cc = meson.get_compiler('c')

prefix = get_option('prefix')
datadir = join_paths(prefix, get_option('datadir'))

config_h = configuration_data()
config_h.set_quoted('PACKAGE_VERSION', meson.project_version())
config_h.set_quoted('GETTEXT_PACKAGE', 'libspelling')
config_h.set_quoted('PACKAGE_LOCALE_DIR', join_paths(get_option('prefix'), get_option('datadir'), 'locale'))
config_h.set10('HAVE_ENCHANT', get_option('enchant').enabled())

# libsysprof-capture support for profiling
if get_option('sysprof')
  libsysprof_capture_dep = dependency('sysprof-capture-4',
    required: get_option('sysprof'),
    default_options: [
      'examples=false',
      'gtk=false',
      'help=false',
      'install-static=false',
      'libsysprof=false',
      'sysprofd=none',
      'tests=false',
      'tools=false',
    ],
    fallback: ['sysprof', 'libsysprof_capture_dep'],
  )
  config_h.set('HAVE_SYSPROF', libsysprof_capture_dep.found())
  profiler_enabled = true
else
  libsysprof_capture_dep = disabler()
  profiler_enabled = false
endif

# Detect and set symbol visibility
if get_option('default_library') != 'static'
  if host_machine.system() == 'windows'
    config_h.set('DLL_EXPORT', true)
    if cc.get_id() == 'msvc'
      config_h.set('_SPELLING_EXTERN', '__declspec(dllexport) extern')
    elif cc.has_argument('-fvisibility=hidden')
      config_h.set('_SPELLING_EXTERN', '__attribute__((visibility("default"))) __declspec(dllexport) extern')
    endif
  elif cc.has_argument('-fvisibility=hidden')
    config_h.set('_SPELLING_EXTERN', '__attribute__((visibility("default"))) extern')
  endif
endif

project_c_args = []
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
add_project_arguments(project_c_args, language: 'c')

release_args = []
global_link_args = []
test_link_args = [
  '-Wl,-z,relro',
  '-Wl,-z,now',
  '-Wl,-z,noexecstack',
]
if not get_option('buildtype').startswith('debug')
  add_global_arguments('-DG_DISABLE_CAST_CHECKS', language: 'c')
  release_args += [ '-DG_DISABLE_ASSERT' ]
  test_link_args += [
    '-Wl,-Bsymbolic',
    '-fno-plt',
  ]
endif
foreach link_arg: test_link_args
  if cc.has_link_argument(link_arg)
    global_link_args += link_arg
  endif
endforeach
add_project_link_arguments(global_link_args, language: 'c')

configure_file(output: 'config.h', configuration: config_h)

subdir('lib')
subdir('po')
subdir('test')
subdir('testsuite')

if get_option('docs')
  subdir('docs')
endif
