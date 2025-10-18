files_extract = files(
    'tracker-miner-files-methods.c',
    'tracker-indexing-tree-methods.c',
)

miner_fs_enums = gnome.mkenums_simple('tracker-miner-enum-types',
    sources: 'tracker-miner-enums.h',
)

miner_fs_resources = gnome.compile_resources(
    'miner-fs-resources',
    'org.freedesktop.Tracker3.Miner.Files.gresource.xml',
    c_name: 'tracker_miner_fs',
)

private_sources = [
    'tracker-file-notifier.c',
    'tracker-files-interface.c',
    'tracker-indexing-tree.c',
    'tracker-lru.c',
    'tracker-miner-fs.c',
    'tracker-monitor.c',
    'tracker-monitor-glib.c',
    'tracker-priority-queue.c',
    'tracker-task-pool.c',
    'tracker-sparql-buffer.c',
    'tracker-utils.c',
    files_extract,
]

if have_fanotify
    private_sources += 'tracker-monitor-fanotify.c'
endif

libtracker_miner_private = static_library(
    'tracker-miner-private',
    miner_fs_resources[0], miner_fs_resources[1],
    miner_fs_enums[0], miner_fs_enums[1],
    private_sources,
    dependencies: [tracker_miners_common_dep, tracker_sparql, gmodule, gudev],
    c_args: tracker_c_args,
)

tracker_miner_dep = declare_dependency(
    sources: miner_fs_enums[1],
    link_with: libtracker_miner_private,
    include_directories: include_directories('.')
)

sources = [
    'tracker-application.c',
    'tracker-config.c',
    'tracker-controller.c',
    'tracker-extract-watchdog.c',
    'tracker-main.c',
    'tracker-miner-files.c',
    'tracker-storage.c',
]

tracker_miner_fs_deps = [
    tracker_miners_common_dep,
    tracker_miner_dep,
    gudev,
]

executable('localsearch-@0@'.format(tracker_api_major),
    sources,
    miner_fs_resources[0], miner_fs_resources[1],
    miner_fs_enums[0], miner_fs_enums[1],
    dependencies: tracker_miner_fs_deps,
    c_args: [
        tracker_c_args,
        '-DBUILDROOT="@0@"'.format(meson.global_build_root()),
        '-DBUILD_EXTRACTDIR="@0@"'.format(meson.build_root() / 'src' / 'extractor'),
        '-DLIBEXECDIR="@0@"'.format(get_option('prefix') / get_option('libexecdir')),
    ],
    install: true,
    install_dir: get_option('libexecdir'),
    install_rpath: tracker_internal_libs_dir,
)

if get_option('domain_prefix') != 'org.freedesktop'
    tmp_conf = configuration_data()
    tmp_conf.merge_from(conf)
    tmp_conf.set ('DBUS_NAME', 'LocalSearch3')
    tmp_conf.set ('DOMAIN_PREFIX', get_option('domain_prefix'))

    domain_ontology_localsearch_service_file = configure_file(
        input: 'org.freedesktop.Tracker3.Miner.Files.service.in',
        output: get_option('domain_prefix') + '.LocalSearch3.service',
        configuration: tmp_conf,
        install: true,
        install_dir: dbus_services_dir,
    )
endif

localsearch_service_file = configure_file(
    input: 'org.freedesktop.Tracker3.Miner.Files.service.in',
    output: 'org.freedesktop.LocalSearch3.service',
    configuration: configuration_data({
        'DOMAIN_PREFIX': 'org.freedesktop',
        'DBUS_NAME': 'LocalSearch3',
        'SYSTEMD_SERVICE': 'SystemdService=localsearch-3.service',
        'libexecdir': conf.get('libexecdir'),
    }),
    install: true,
    install_dir: dbus_services_dir,
)


if get_option('domain_prefix') != 'org.freedesktop'
    legacy_conf = configuration_data()
    legacy_conf.merge_from(conf)
    legacy_conf.set ('DBUS_NAME', 'Tracker3.Miner.Files')
    legacy_conf.set ('DOMAIN_PREFIX', get_option('domain_prefix'))

    legacy_domain_ontology_service_file = configure_file(
        input: 'org.freedesktop.Tracker3.Miner.Files.service.in',
        output: get_option('domain_prefix') + '.Tracker3.Miner.Files.service',
        configuration: legacy_conf,
        install: true,
        install_dir: dbus_services_dir,
    )
endif

legacy_service_file = configure_file(
    input: 'org.freedesktop.Tracker3.Miner.Files.service.in',
    output: 'org.freedesktop.Tracker3.Miner.Files.service',
    configuration: configuration_data({
        'DOMAIN_PREFIX': 'org.freedesktop',
        'DBUS_NAME': 'Tracker3.Miner.Files',
        'SYSTEMD_SERVICE': 'SystemdService=localsearch-3.service',
        'libexecdir': conf.get('libexecdir'),
    }),
    install: true,
    install_dir: dbus_services_dir,
)

# This file allows starting the service automatically using XDG autostart.
# systemd user service activation is preferred if available.
desktop_file = configure_file(
    input: 'tracker-miner-fs.desktop.in',
    output: 'localsearch-@0@.desktop'.format(tracker_api_major),
    configuration: conf,
    install: true,
    install_dir: xdg_autostart_dir)

if get_option('systemd_user_services')
  # This file allows starting the service as a systemd user service.
  configure_file(
      input: 'tracker-miner-fs.service.in',
      output: 'localsearch-@0@.service'.format(tracker_api_major),
      configuration: conf,
      install: true,
      install_dir: systemd_user_services_dir)
endif
