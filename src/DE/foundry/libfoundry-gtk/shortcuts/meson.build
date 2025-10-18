foundry_gtk_sources += files([
  'foundry-menu-manager.c',
  'foundry-menu-proxy.c',
  'foundry-shortcut-bundle.c',
  'foundry-shortcut-info.c',
  'foundry-shortcut-manager.c',
  'foundry-shortcut-observer.c',
  'foundry-shortcut-provider.c',
])

foundry_gtk_headers += files([
  'foundry-menu-manager.h',
  'foundry-menu-proxy.h',
  'foundry-shortcut-bundle.h',
  'foundry-shortcut-info.h',
  'foundry-shortcut-manager.h',
  'foundry-shortcut-observer.h',
  'foundry-shortcut-provider.h',
])

foundry_gtk_deps += [
  dependency('template-glib-1.0', version: '>= 3.37.0'),
]

foundry_gtk_include_directories += [include_directories('.')]
