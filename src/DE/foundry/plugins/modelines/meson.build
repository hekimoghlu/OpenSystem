foundry_private_sources += files([
  'modeline.c',
  'plugin.c',
  'plugin-modelines-settings-provider.c',
])

executable('test-modeline', ['test-modeline.c'],
  dependencies: [dependency('gobject-2.0')],
  include_directories: include_directories('../../contrib'),
)
