# Copyright 2024 Collabora Ltd.
# SPDX-License-Identifier: LGPL-2.1-or-later

test_env = environment()
test_env.set('G_TEST_SRCDIR', meson.current_source_dir())
test_env.set('G_TEST_BUILDDIR', meson.current_build_dir())
test_env.set('LINT_WARNINGS_ARE_ERRORS', '1')

lint_scripts = [
  'black.sh',
  'flake8.sh',
]

if have_bash
  foreach test_name : lint_scripts
    test(
      test_name, files(test_name),
      env : test_env,
      suite : ['lint', 'no-valgrind'],
      protocol : 'tap',
    )
  endforeach
endif