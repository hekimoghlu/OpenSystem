api_version = '1'

foundry_debug_conf = configuration_data()
foundry_debug_conf.set10('ENABLE_TRACING', get_option('development'))
foundry_debug_conf.set('BUGREPORT_URL', 'https://gitlab.gnome.org/chergert/foundry/issues')
foundry_debug_h = configure_file(
          input: 'foundry-debug.h.in',
         output: 'foundry-debug.h',
  configuration: foundry_debug_conf,
        install: true,
    install_dir: get_option('includedir') / 'libfoundry-@0@'.format(api_version)
)

foundry_sources += files([
  'foundry-action-muxer.c',
  'foundry-context.c',
  'foundry-contextual.c',
  'foundry-inhibitor.c',
  'foundry-init.c',
  'foundry-service.c',
  'foundry-util.c',
])

foundry_private_sources += files([
  'foundry-string-object.c',
])

foundry_private_sources += [
  gnome.gdbus_codegen('foundry-ipc',
             sources: 'app.devsuite.foundry.xml',
    interface_prefix: 'app.devsuite.foundry.',
           namespace: 'FoundryIpc',
  ),

  gnome.compile_resources('foundry-resources', 'foundry.gresource.xml',
    c_name: '_foundry',
  ),

  gnome.genmarshal('foundry-marshal',
              internal: true,
                prefix: 'foundry_marshal',
               sources: ['foundry-marshal.defs'],
    valist_marshallers: true,
  ),
]

foundry_headers += files([
  'foundry.h',
  'foundry-action-muxer.h',
  'foundry-context.h',
  'foundry-contextual.h',
  'foundry-inhibitor.h',
  'foundry-init.h',
  'foundry-service.h',
  'foundry-types.h',
  'foundry-util.h',
  'foundry-version-macros.h',
])

version_split = meson.project_version().split('.')
version_conf = configuration_data()
version_conf.set('VERSION', meson.project_version())
version_conf.set('MAJOR_VERSION', version_split[0])
version_conf.set('MINOR_VERSION', version_split[1])
if version_split[2][0] in ['a','b','r']
  version_conf.set('MICRO_VERSION', 0)
else
  version_conf.set('MICRO_VERSION', version_split[2])
endif

configure_file(
          input: 'foundry-version.h.in',
         output: 'foundry-version.h',
  configuration: version_conf,
        install: true,
    install_dir: get_option('includedir') / 'libfoundry-@0@'.format(api_version)
)

libfoundry_config_conf = configuration_data()

foundry_deps = [
  glib_dep,
  gio_unix_dep,
  gom_dep,
  dex_dep,
  json_dep,
  peas_dep,
  sysprof_dep,
  libeggbitset_static_dep,
]

foundry_static_deps = [
  contrib_static_dep,
]

foundry_c_args = [
  '-DFOUNDRY_COMPILATION',
  '-DG_LOG_DOMAIN="Foundry"',
]

foundry_include_directories = [
  include_directories('.'),
  include_directories('..' / 'contrib'),
]

subdir('build')
subdir('cli')
subdir('commands')
subdir('config')
subdir('dependencies')
subdir('diagnostics')
subdir('device')
subdir('files')
subdir('input')
subdir('jsonrpc')
subdir('licenses')
subdir('logging')
subdir('markup')
subdir('operations')
subdir('platform')
subdir('plugins')
subdir('run')
subdir('sdk')
subdir('search')
subdir('settings')
subdir('symbols')
subdir('tests')
subdir('tweaks')

if get_option('feature-debugger')
  subdir('debugger')

  if get_option('feature-dap')
    subdir('dap')
  endif
endif

if get_option('feature-docs')
  subdir('documentation')
endif

if get_option('feature-text')
  subdir('completion')
  subdir('editing')
  subdir('hover')
endif

if get_option('feature-flatpak')
  subdir('flatpak')
endif

if get_option('feature-git')
  subdir('git')
endif

if get_option('feature-llm')
  subdir('llm')
endif

if get_option('feature-lsp')
  subdir('lsp')
endif

if get_option('feature-templates')
  subdir('templates')
endif

if get_option('feature-terminal')
  subdir('terminal')
endif

if get_option('feature-vcs')
  subdir('vcs')
endif

configure_file(
          input: 'libfoundry-config.h.in',
         output: 'libfoundry-config.h',
  configuration: libfoundry_config_conf,
        install: true,
    install_dir: get_option('libdir') / 'libfoundry-@0@'.format(api_version) / 'include',
)

