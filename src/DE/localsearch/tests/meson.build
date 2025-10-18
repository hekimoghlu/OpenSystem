subdir('test-utils')

subdir('common')
subdir('indexer')

if have_tracker_extract
  subdir('extract-utils')
endif

subdir('services')

test_bus_conf_file = configure_file(
  input: 'test-bus.conf.in',
  output: 'test-bus.conf',
  configuration: conf)

if get_option('functional_tests')
  subdir('functional-tests')
endif
