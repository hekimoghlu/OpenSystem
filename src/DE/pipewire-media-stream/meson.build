project(
  'pipewire-media-stream',
  'c',
  version: '0.1.0',
  meson_version: '>= 0.50.0',
)

gnome = import('gnome')

sources = [
  'pw-media-stream.c',
  'window.c',
  'main.c',
]

deps = [
  dependency('epoxy'),
  dependency('libadwaita-1'),
  dependency('libportal-gtk4'),
  dependency('libpipewire-0.3', version: '>= 0.3.40'),
  dependency('gtk4', version: '>= 4.13'),
]

sources += gnome.compile_resources(
  'pw-media-stream-resources',
  'media-stream.gresource.xml',
  c_name: 'pw',
)

executable(
  'pipewire-media-stream',
  sources,
  dependencies: deps,
  install: true,
)
