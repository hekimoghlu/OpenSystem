tools = {
  'mks-connect': {},
  'mks': {},
}

foreach tool, params: tools
  tool_exe = executable(tool, '@0@.c'.format(tool),
    dependencies: libmks_dep,
         install: get_option('install-tools'),
  )
endforeach
