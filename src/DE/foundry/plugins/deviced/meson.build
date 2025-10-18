if not get_option('feature-flatpak')
  error('plugin-deviced requires feature-flatpak')
endif

libdeviced_dep = dependency('libdeviced', version: '>= 3.27.4')

foundry_deps += [
  libdeviced_dep,
]

foundry_private_sources += files([
  'plugin.c',
  'plugin-deviced-deploy-strategy.c',
  'plugin-deviced-device.c',
  'plugin-deviced-device-info.c',
  'plugin-deviced-device-provider.c',
])

foundry_deviced = executable('foundry-deviced', 'foundry-deviced.c',
       install: true,
   install_dir: get_option('libexecdir'),
  dependencies: [libdeviced_dep],
)
