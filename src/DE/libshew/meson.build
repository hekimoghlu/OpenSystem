# SPDX-FileCopyrightText: 2020 Florian Müllner <fmuellner@gnome.org>
# SPDX-License-Identifier: LGPL-2.1-or-later

project('shew', 'c',
  version: '48.1',
  meson_version: '>= 1.1.0',
  license: 'LGPL-2.1-or-later',
)

assert(meson.is_subproject(), 'This project is only intended to be used as a subproject!')

gnome = import('gnome')
pkg = import('pkgconfig')

api_version = '0'
full_name = '@0@-@1@'.format(meson.project_name(), api_version)

package_version = meson.project_version()
package_name = get_option('package_name')
assert(package_name != '', 'package_name must be specified')

pkglibdir = join_paths(get_option('libdir'), package_name)

typelibdir = join_paths(pkglibdir, 'girepository-1.0')

gtk_dep = dependency('gtk4')
x11_dep = dependency('x11', required: false)

subdir('src')

if get_option('tests')
  add_test_setup('plain')
  add_test_setup('headless',
    exe_wrapper: [
      find_program('dbus-run-session'),
      find_program('mutter'),
      '--headless',
      '--',
    ],
  )

  subdir('tests')
endif
