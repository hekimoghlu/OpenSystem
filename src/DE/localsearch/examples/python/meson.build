python = find_program('python3')

sandbox_python_args = [run_uninstalled, '--store-tmpdir']

examples = [
  'query-async',
  'query-sync',
]

foreach example_name: examples
  file = meson.current_source_dir() / '@0@.py'.format(example_name)
  test(example_name, python,
    args: sandbox_python_args + [file],
    suite: 'examples')
endforeach
