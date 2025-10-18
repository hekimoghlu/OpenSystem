enums = gnome.mkenums('tracker-miners-enum-types',
  sources: 'tracker-enums.h',
  c_template: 'tracker-miners-enum-types.c.template',
  h_template: 'tracker-miners-enum-types.h.template',
)

tracker_miners_common_enum_header = enums[1]

tracker_miners_common_sources = [
  'tracker-dbus.c',
  'tracker-debug.c',
  'tracker-error-report.c',
  'tracker-extract-info.c',
  'tracker-file-utils.c',
  'tracker-ioprio.c',
  'tracker-miner.c',
  'tracker-miner-proxy.c',
  'tracker-module-manager.c',
  'tracker-sched.c',
  'tracker-systemd.c',
  'tracker-term-utils.c',
  'tracker-type-utils.c',
  'tracker-utils.c',
  'tracker-locale.c',
  'tracker-seccomp.c',
  enums[0], enums[1],
]

if have_landlock
  tracker_miners_common_sources += 'tracker-landlock.c'
endif

tracker_miners_common_dependencies = [glib, gio, gio_unix, libmath, tracker_sparql]

if battery_detection_library_name == 'upower'
  tracker_miners_common_sources += 'tracker-power-upower.c'
  tracker_miners_common_dependencies += upower
elif battery_detection_library_name == 'hal'
  tracker_miners_common_sources += 'tracker-power-hal.c'
  tracker_miners_common_dependencies += hal
endif

tracker_miners_common_private_dependencies = [charset_library]
if libseccomp.found()
  tracker_miners_common_private_dependencies += [libseccomp]
endif

if build_machine.system() == 'openbsd'
  libkvm = meson.get_compiler('c').find_library('kvm')
  tracker_miners_common_dependencies += libkvm
endif

libtracker_miners_common = static_library('tracker-miners-common',
  tracker_miners_common_sources,
  dependencies: tracker_miners_common_dependencies + tracker_miners_common_private_dependencies,
  c_args: tracker_c_args + [
    '-DTRACKERSHAREDIR="@0@"'.format(tracker_versioned_name),
    '-DLIBDIR="@0@"'.format(get_option ('libdir')),
    '-DDATADIR="@0@"'.format(get_option ('datadir')),
    '-DPREFIX="@0@"'.format(get_option ('prefix')),
    '-DBUILDROOT="@0@"'.format(meson.build_root()),
    '-DSRCROOT="@0@"'.format(meson.source_root()),
    '-DLIBEXECDIR="@0@"'.format(join_paths(get_option('prefix'), get_option('libexecdir'))),
    '-DTRACKER_EXTRACTOR_RULES_DIR="@0@"'.format(tracker_extract_rules_dir),
    '-DTRACKER_EXTRACTORS_DIR="@0@"'.format(tracker_extract_modules_dir)
  ],
  include_directories: [configinc, srcinc],
)

commoninc = include_directories('.')

tracker_miners_common_dep = declare_dependency(
  sources: [tracker_miners_common_enum_header],
  link_with: libtracker_miners_common,
  dependencies: tracker_miners_common_dependencies,
  include_directories: [configinc, srcinc, commoninc],
)
