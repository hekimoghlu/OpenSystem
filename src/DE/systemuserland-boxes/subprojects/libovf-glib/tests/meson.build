test_parser = executable(
  'parser', [
    'parser.c',
  ],
  dependencies: govf_dep,
)

configure_file(
  input: 'Fedora_23.ovf',
  output: 'Fedora_23.ovf',
  copy: true,
)

configure_file(
  input: 'Fedora_23.ova',
  output: 'Fedora_23.ova',
  copy: true
)

test ('parser', test_parser)
