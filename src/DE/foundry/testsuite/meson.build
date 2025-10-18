lib_test_env = [
  'G_TEST_SRCDIR=@0@'.format(meson.current_source_dir()),
  'G_TEST_BUILDDIR=@0@'.format(meson.current_build_dir()),
  'GSETTINGS_BACKEND=memory',
  'GSETTINGS_SCHEMA_DIR=@0@/data/settings'.format(meson.project_build_root()),
  'G_DEBUG=gc-friendly',
  'MALLOC_CHECK_=2',
]

lib_testsuite_c_args = [
  '-DG_ENABLE_DEBUG',
  '-UG_DISABLE_ASSERT',
  '-UG_DISABLE_CAST_CHECKS',
]

lib_testsuite = {
  'test-cli-command' : {},
  'test-file' : {},
  'test-json' : {},
  'test-settings' : {},
  'test-tweaks' : {},
}

lib_testsuite_deps = [
  foundry_static_dep,
]

if get_option('feature-flatpak')
  lib_testsuite += {
    'test-flatpak-builder-manifest' : {},
  }
endif

if get_option('feature-llm') and get_option('plugin-ollama')
  lib_testsuite += {
    'test-ollama' : {'skip': true},
  }
endif

if get_option('feature-text')
  lib_testsuite += {
    'test-simple-text-buffer' : {},
  }
endif

if get_option('feature-git')
  lib_testsuite += {
    'test-git-uri' : {},
  }
endif

foreach test, params: lib_testsuite
  test_exe = executable(test, ['@0@.c'.format(test)],
                 c_args: lib_testsuite_c_args,
           dependencies: lib_testsuite_deps,
    include_directories: [include_directories('..'), include_directories('.')],
  )
  if not params.get('skip', false)
    test(test, test_exe, env: lib_test_env)
  endif
endforeach

subdir('tools')
