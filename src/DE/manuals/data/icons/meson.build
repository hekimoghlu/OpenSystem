icondir = join_paths(get_option('datadir'), 'icons', 'hicolor')
symbolicdir = join_paths(icondir, 'symbolic', 'apps')
scalabledir = join_paths(icondir, 'scalable', 'apps')

if get_option('development')
  install_data('manuals-devel.svg', rename: '@0@.svg'.format(app_id), install_dir: scalabledir)
else
  install_data('manuals.svg', rename: '@0@.svg'.format(app_id), install_dir: scalabledir)
endif

install_data('manuals-symbolic.svg', rename: '@0@-symbolic.svg'.format(app_id), install_dir: symbolicdir)
