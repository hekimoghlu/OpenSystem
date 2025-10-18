foundry_private_sources += files([
  'plugin.c',
  'plugin-flatpak.c',
  'plugin-flatpak-autogen-stage.c',
  'plugin-flatpak-build-addin.c',
  'plugin-flatpak-bundle-stage.c',
  'plugin-flatpak-commit-stage.c',
  'plugin-flatpak-config.c',
  'plugin-flatpak-config-provider.c',
  'plugin-flatpak-dependency.c',
  'plugin-flatpak-dependency-provider.c',
  'plugin-flatpak-dependencies-stage.c',
  'plugin-flatpak-download-stage.c',
  'plugin-flatpak-export-stage.c',
  'plugin-flatpak-prepare-stage.c',
  'plugin-flatpak-sdk.c',
  'plugin-flatpak-sdk-install.c',
  'plugin-flatpak-sdk-provider.c',
  'plugin-flatpak-simple-stage.c',
  'plugin-flatpak-util.c',
  'plugin-flatpak-aux.c',
])

if get_option('feature-docs')
  foundry_private_sources += files([
    'plugin-flatpak-documentation-bundle.c',
    'plugin-flatpak-documentation-provider.c',
  ])
endif

foundry_deps += [
  dependency('flatpak'),
]

schemas += files(['app.devsuite.foundry.flatpak.gschema.xml.in'])
