backend_sources = files(
  'comics-document.c',
  'pps-archive.c',
)

backend_incs = backends_common_incs

backend_deps = backends_common_deps + [
  libarchive_dep,
]

test_name = 'test-pps-archive'

test_sources = files(
  'pps-archive.c',
  'test-pps-archive.c',
)

executable(
  test_name,
  test_sources,
  include_directories: backend_incs,
  dependencies: backend_deps,
)
