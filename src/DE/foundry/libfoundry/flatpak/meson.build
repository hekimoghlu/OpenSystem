 libfoundry_config_conf.set10('FOUNDRY_FEATURE_FLATPAK', true)

foundry_sources += files([
  'foundry-flatpak-arch-options.c',
  'foundry-flatpak-extension.c',
  'foundry-flatpak-extensions.c',
  'foundry-flatpak-list.c',
  'foundry-flatpak-manifest.c',
  'foundry-flatpak-manifest-loader.c',
  'foundry-flatpak-module.c',
  'foundry-flatpak-modules.c',
  'foundry-flatpak-options.c',
  'foundry-flatpak-serializable.c',
  'foundry-flatpak-source.c',
  'foundry-flatpak-source-archive.c',
  'foundry-flatpak-source-bzr.c',
  'foundry-flatpak-source-dir.c',
  'foundry-flatpak-source-extra-data.c',
  'foundry-flatpak-source-file.c',
  'foundry-flatpak-source-git.c',
  'foundry-flatpak-source-inline.c',
  'foundry-flatpak-source-patch.c',
  'foundry-flatpak-source-script.c',
  'foundry-flatpak-source-shell.c',
  'foundry-flatpak-source-svn.c',
  'foundry-flatpak-sources.c',
])

foundry_headers += files([
  'foundry-flatpak-arch-options.h',
  'foundry-flatpak-extension.h',
  'foundry-flatpak-extensions.h',
  'foundry-flatpak-list.h',
  'foundry-flatpak-manifest.h',
  'foundry-flatpak-manifest-loader.h',
  'foundry-flatpak-module.h',
  'foundry-flatpak-modules.h',
  'foundry-flatpak-options.h',
  'foundry-flatpak-serializable.h',
  'foundry-flatpak-source.h',
  'foundry-flatpak-source-archive.h',
  'foundry-flatpak-source-bzr.h',
  'foundry-flatpak-source-dir.h',
  'foundry-flatpak-source-extra-data.h',
  'foundry-flatpak-source-file.h',
  'foundry-flatpak-source-git.h',
  'foundry-flatpak-source-inline.h',
  'foundry-flatpak-source-patch.h',
  'foundry-flatpak-source-script.h',
  'foundry-flatpak-source-shell.h',
  'foundry-flatpak-source-svn.h',
  'foundry-flatpak-sources.h',
])

foundry_deps += [
  dependency('json-glib-1.0'),
  dependency('yaml-0.1'),
]

foundry_include_directories += [include_directories('.')]
