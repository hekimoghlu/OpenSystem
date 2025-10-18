# This directory contains service files used by dbus-daemon to automatically
# activate the daemons as needed. These files are used when running Tracker
# from the build tree.

tracker_miner_fs_test_dbus_service_file = configure_file(
    input: 'org.freedesktop.LocalSearch3.service.in',
    output: 'org.freedesktop.LocalSearch3.service',
    configuration: conf)

tracker_index_test_dbus_service_file = configure_file(
    input: 'org.freedesktop.LocalSearch3.Control.service.in',
    output: 'org.freedesktop.LocalSearch3.Control.service',
    configuration: conf)

tracker_writeback_test_dbus_service_file = configure_file(
    input: 'org.freedesktop.LocalSearch3.Writeback.service.in',
    output: 'org.freedesktop.LocalSearch3.Writeback.service',
    configuration: conf)
