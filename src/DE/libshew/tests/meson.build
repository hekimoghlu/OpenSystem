# SPDX-FileCopyrightText: 2025 Florian Müllner <fmuellner@gnome.org>
# SPDX-License-Identifier: LGPL-2.1-or-later

exported_client = executable('exported-client',
  sources: 'exported-client.c',
  dependencies: [
    libshew_dep,
  ],
)

test_cases = [
  {
    'name': 'exporter',
    'sources': [
      'test-exporter.c',
    ],
  },
  {
    'name': 'external-window',
    'sources': [
      'test-external-window.c',
    ],
    'depends': [
      exported_client,
    ],
  },
]

foreach test_case: test_cases
  test_executable = executable('shew-' + test_case['name'],
    sources: test_case['sources'],
    dependencies: [
      libshew_dep,
    ],
  )

  test(test_case['name'], test_executable,
    env: {
      'G_TEST_SRCDIR': meson.current_source_dir(),
      'G_TEST_BUILDDIR': meson.current_build_dir(),
      'GSETTINGS_BACKEND': 'memory',
      'GTK_A11Y': 'none',
    },
    depends: test_case.get('depends', []),
  )
endforeach
