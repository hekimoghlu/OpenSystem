# SPDX-FileCopyrightText: 2025 Florian Müllner <fmuellner@gnome.org>
# SPDX-License-Identifier: LGPL-2.1-or-later

project(
  'test-project',
  'c',
  version: '0',
  license: 'LGPL-2.1-or-later',
)

libshew_dep = dependency('libshew',
  default_options: [
    'package_name=@0@'.format(meson.project_name()),
    'tests=true',
  ],
)
