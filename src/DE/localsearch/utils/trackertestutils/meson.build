testutils_dir = tracker_internal_libs_dir / get_option('test_utils_dir')

sources = [
  '__init__.py',
  '__main__.py',
  'dbusdaemon.py',
  'dconf.py',
  'helpers.py',
  'mainloop.py',
  'psutil_mini.py',
  'sandbox.py',
  'storehelper.py',
]

install_data(sources, install_dir: testutils_dir / 'trackertestutils')
