libtracker_extract_tests = [
    'guarantee',
    'utils',
    'xmp',
]

if charset_library_name != 'none'
    libtracker_extract_tests += ['encoding']
endif

if libexif.found()
    libtracker_extract_tests += ['exif']
endif

libtracker_extract_test_deps = [
    tracker_miners_common_dep, tracker_extract_dep
]

foreach base_name: libtracker_extract_tests
    source = 'tracker-@0@-test.c'.format(base_name)
    binary_name = 'tracker-@0@-test'.format(base_name)

    binary = executable(binary_name, source,
      dependencies: libtracker_extract_test_deps,
      c_args: test_c_args)

    test(base_name, binary,
      protocol: test_protocol,
      suite: 'extract')
endforeach

