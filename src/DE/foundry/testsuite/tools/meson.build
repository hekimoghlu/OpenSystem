readline_dep = dependency('readline', required: false)
if not readline_dep.found()
  readline_dep = disabler()
endif

lib_testsuite_tools = {
  'test-auth-prompt' : {},
  'test-dap' : {'dap': true},
  'list-palettes' : {'gtk': true},
  'list-tweaks' : {'gtk': true},
  'test-view' : {'gtk': true, 'text': true},
  'test-terminal' : {'gtk': true},
  'test-diagnostics' : {'text': true},
  'test-build-manager-gtk' : {'gtk': true},
  'test-chat' : {'llm': true, 'gtk': true},
  'test-git-monitor' : {'git': true},
  'fdb' : {'dap': true, 'sources': ['egg-line.c'], 'dependencies': [readline_dep]},
}

foreach test, params: lib_testsuite_tools
  tool_deps = [lib_testsuite_deps]
  if params.get('gtk', false)
    if not get_option('gtk')
      continue
    endif
    tool_deps += [foundry_gtk_dep, dependency('gtk4')]
  endif
  if params.get('text', false) and not get_option('feature-text')
    continue
  endif
  if params.get('llm', false) and not get_option('feature-llm')
    continue
  endif
  if params.get('git', false) and not get_option('feature-git')
    continue
  endif
  if params.get('dap', false) and not get_option('feature-dap')
    continue
  endif
  sources = params.get('sources', [])
  dependencies = params.get('dependencies', [])
  tool_exe = executable(test, ['@0@.c'.format(test)] + sources,
                 c_args: lib_testsuite_c_args,
           dependencies: tool_deps + dependencies,
    include_directories: [include_directories('..'), include_directories('.')],
  )
endforeach

if get_option('gtk') and get_option('feature-debugger')
  gresource = gnome.compile_resources('test-debugger-gtk-resources',
                                      'test-debugger-gtk.gresource.xml',
                                      source_dir: '.',
                                      c_name: 'test_debugger_gtk')
  tool_exe = executable('test-debugger-gtk', ['test-debugger-gtk.c'] + gresource,
                 c_args: lib_testsuite_c_args,
           dependencies: lib_testsuite_deps + [foundry_gtk_dep, dependency('gtk4')],
    include_directories: [include_directories('..'), include_directories('.')],
  )
endif

if get_option('plugin-ctags') and get_option('feature-text')
  executable('test-ctags', ['test-ctags.c'], dependencies: [foundry_static_dep])
  executable('test-ctags-builder', ['test-ctags-builder.c'], dependencies: [foundry_static_dep])
endif
