api_version = '1'
so_version = '2'

libspelling_private_sources = [
  'cjhtextregion.c',
  'spelling-cursor.c',
  'spelling-empty-provider.c',
  'spelling-engine.c',
  'spelling-job.c',
  'spelling-menu.c',
]

libspelling_public_sources = [
  'spelling-init.c',
  'spelling-checker.c',
  'spelling-dictionary.c',
  'spelling-language.c',
  'spelling-provider.c',
  'spelling-text-buffer-adapter.c',
]

libspelling_public_headers = [
  'libspelling.h',
  'spelling-checker.h',
  'spelling-dictionary.h',
  'spelling-init.h',
  'spelling-language.h',
  'spelling-provider.h',
  'spelling-text-buffer-adapter.h',
  'spelling-types.h',
  'spelling-version-macros.h',
]

libspelling_deps = [
  libgio_dep,
  libgtk_dep,
  gtksource_dep,
]

if profiler_enabled
  libspelling_deps += [libsysprof_capture_dep]
endif

if get_option('enchant').enabled()
  subdir('enchant')
endif

libspelling_sources = libspelling_private_sources + libspelling_public_sources

version_split = meson.project_version().split('.')
version_conf = configuration_data()
version_conf.set('VERSION', meson.project_version())
version_conf.set('MAJOR_VERSION', version_split[0])
version_conf.set('MINOR_VERSION', version_split[1])
version_conf.set('MICRO_VERSION', version_split[2])

configure_file(
          input: 'spelling-version.h.in',
         output: 'spelling-version.h',
  configuration: version_conf,
        install: true,
    install_dir: join_paths(get_option('includedir'), 'libspelling-' + api_version)
)

libspelling_static = static_library( 'spelling-' + api_version, libspelling_sources,
           dependencies: libspelling_deps,
  gnu_symbol_visibility: 'hidden',
                 c_args: ['-DLIBSPELLING_COMPILATION'],
    include_directories: [include_directories('..'), include_directories('.')],
                install: get_option('install-static'),
)

libspelling_static_dep = declare_dependency(
           link_whole: libspelling_static,
         dependencies: libspelling_deps,
  include_directories: [include_directories('.')],
)

libspelling = shared_library('spelling-' + api_version,
           dependencies: [libspelling_static_dep],
  gnu_symbol_visibility: 'hidden',
              soversion: so_version,
                version: '@0@.0.0'.format(so_version),
        darwin_versions: '@0@.0'.format(so_version),
                install: true,
                 c_args: ['-DLIBSPELLING_COMPILATION'] + release_args,
)

libspelling_dep = declare_dependency(
            link_with: libspelling,
         dependencies: libspelling_deps,
  include_directories: [include_directories('.')],
)
meson.override_dependency('libspelling-' + api_version, libspelling_dep)

install_headers(libspelling_public_headers, subdir: 'libspelling-' + api_version)

pkg = import('pkgconfig')
pkg.generate(
  description: 'A spellcheck library for GTK',
    libraries: libspelling,
         name: 'libspelling-' + api_version,
     filebase: 'libspelling-' + api_version,
      version: meson.project_version(),
      subdirs: 'libspelling-' + api_version,
     requires: ['gio-2.0', 'gtk4', 'gtksourceview-5'],
  install_dir: join_paths(get_option('libdir'), 'pkgconfig')
)

if get_option('introspection').enabled()
  gnome = import('gnome')
  libspelling_gir = gnome.generate_gir(libspelling,
              nsversion: api_version,
                sources: [libspelling_public_sources, libspelling_public_headers],
              namespace: 'Spelling',
          symbol_prefix: 'spelling',
      identifier_prefix: 'Spelling',
               includes: ['Gio-2.0', 'Gtk-4.0', 'GtkSource-5'],
        export_packages: 'libspelling-' + api_version,
                install: true,
                 header: 'libspelling.h',
             extra_args: '-DLIBSPELLING_COMPILATION',
  )

  if get_option('vapi')
    spelling_vapi = gnome.generate_vapi('libspelling-' + api_version,
            sources: libspelling_gir.get(0),
            install: true,
        install_dir: join_paths(datadir, 'vala', 'vapi'),
           packages: ['gio-2.0', 'gtk4', 'gtksourceview-5'],
    )
  endif
endif
