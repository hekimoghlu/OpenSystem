schemas = {
  'org.freedesktop.Tracker.Extract.gschema.xml': 'org.freedesktop.Tracker@0@.Extract.gschema.xml'.format(tracker_api_major),
  'org.freedesktop.Tracker.FTS.gschema.xml': 'org.freedesktop.Tracker@0@.FTS.gschema.xml'.format(tracker_api_major),
  'org.freedesktop.Tracker.Miner.Files.gschema.xml': 'org.freedesktop.Tracker@0@.Miner.Files.gschema.xml'.format(tracker_api_major),
}

dir_conf = configuration_data()

dirs = {
  'default_index_recursive_dirs': 'INDEX_RECURSIVE_DIRS',
  'default_index_single_dirs': 'INDEX_SINGLE_DIRS'
}
foreach opt_name, variable : dirs
  array = []
  foreach dir : get_option(opt_name)
    # See https://github.com/mesonbuild/meson/issues/4007
    escaped_dir = '&amp;'.join(dir.split('&'))
    array += [ '\'' + escaped_dir + '\'' ]
  endforeach
  # Remove the trailing comma
  str = ', '.join(array)
  dir_conf.set(variable, str)
endforeach

schemas_file_deps = []
foreach source, dest : schemas
    configure_file(
        input: source,
        output: dest,
        install: true,
        configuration: dir_conf,
        install_dir: gsettings_schema_dir)
    schemas_file_deps += source
endforeach

tracker_miners_settings_enums = custom_target('tracker-miners-common-settings-enums',
    input: '../src/common/tracker-enums.h',
    output: 'org.freedesktop.TrackerMiners@0@.enums.xml'.format(tracker_api_major),
    command: [glib_mkenums,
              '--comments', '<!-- @comment@ -->',
              '--fhead', '<schemalist>',
              '--vhead', '<@type@ id="org.freedesktop.TrackerMiners@0@.@EnumName@">'.format(tracker_api_major),
              '--vprod', '    <value nick="@valuenick@" value="@valuenum@"/>',
              '--vtail', '  </@type@>',
              '--ftail', '</schemalist>', '@INPUT@'],
    capture: true,
    install: true,
    install_dir: gsettings_schema_dir)


# Compile schemas locally so that the functional-tests can use them without
# them being installed into /usr.
#
compile_schemas_deps = [tracker_miners_settings_enums]

custom_target('tracker-miners-compile-schemas',
  output: 'gschemas.compiled',
  command: [glib_compile_schemas, meson.current_build_dir()],
  build_by_default: true,
  depends: compile_schemas_deps,
  depend_files: schemas_file_deps)

tracker_miners_uninstalled_gsettings_schema_dir = meson.current_build_dir()

indexer_dbus_interface = files ('org.freedesktop.Tracker3.Miner.xml')

install_data(
    indexer_dbus_interface,
    install_dir: join_paths(get_option('prefix'), get_option('datadir'), 'dbus-1', 'interfaces'))
