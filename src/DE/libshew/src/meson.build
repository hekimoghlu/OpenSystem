# SPDX-FileCopyrightText: 2020 Florian Müllner <fmuellner@gnome.org>
# SPDX-License-Identifier: LGPL-2.1-or-later

shew_public_headers = files(
  'shew-external-window.h',
  'shew-window-exporter.h',
)

shew_sources = [
  'shew-external-window-wayland.c',
  'shew-external-window-x11.c',
  'shew-external-window.c',
  'shew-window-exporter.c',
]

libshew = library(full_name,
  sources: shew_sources,
  dependencies: [gtk_dep, x11_dep],
  install_dir: pkglibdir,
  install: true,
)

libshew_dep = declare_dependency(
  link_with: libshew,
  dependencies: [gtk_dep],
  include_directories: include_directories('.'),
)

meson.override_dependency('libshew', libshew_dep)

libshew_gir = gnome.generate_gir(libshew,
  sources: shew_sources + shew_public_headers,
  nsversion: api_version,
  namespace: 'Shew',
  includes: ['Gdk-4.0', 'Gtk-4.0'],
  extra_args: ['--quiet'],
  fatal_warnings: get_option('werror'),
  install_gir: false,
  install_typelib: true,
  install_dir_typelib: typelibdir,
)
