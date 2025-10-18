# Shared common code
subdir('common')

# Filesystem indexer
if have_tracker_miner_fs
  subdir('indexer')
endif

# Public data extract tool & modules
if get_option('extract')
  subdir('extractor')
endif

# Control Proxy
subdir('control')

# Metadata writeback service & modules
if get_option('writeback')
  subdir('writeback')
endif

# CLI subcommands
subdir('cli')

meson.add_install_script('create-miner-symlinks.sh', dbus_services_dir, tracker_miner_services_dir, get_option('domain_prefix'), have_tracker_miner_fs.to_string())
