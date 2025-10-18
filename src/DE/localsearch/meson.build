project('localsearch', 'c',
        version: '3.10.1',
        meson_version: '>=0.51')

gnome = import('gnome')
i18n = import('i18n')
python = import('python')
cc = meson.get_compiler('c')

# This is the X.Y used in -llibtracker-FOO-X.Y
tracker_api_version = '3.0'
tracker_api_major = tracker_api_version.split('.')[0]
tracker_versioned_name = 'localsearch@0@'.format(tracker_api_major)

libdir = join_paths(get_option('prefix'), get_option('libdir'))
datadir = join_paths(get_option('prefix'), get_option('datadir'))

glib_required = '2.76.0'

tracker_sparql = dependency('tinysparql-3.0', version: '>=3.8')
dbus = dependency('dbus-1', version: '>= 1.3.1')
exempi = dependency('exempi-2.0', version: '>= 2.1.0', required: get_option('xmp'))
gexiv2 = dependency('gexiv2-0.16', required: false)
if not gexiv2.found()
  gexiv2 = dependency('gexiv2', required: get_option('raw'))
endif
gio = dependency('gio-2.0', version: '>=' + glib_required)
gio_unix = dependency('gio-unix-2.0', version: '>=' + glib_required)
glib = dependency('glib-2.0', version: '>=' + glib_required)
gudev = dependency('gudev-1.0')
gmodule = dependency('gmodule-2.0', version: '>=' + glib_required)
gobject = dependency('gobject-2.0', version: '>=' + glib_required)
gobject_introspection = dependency('gobject-introspection-1.0')
gstreamer = dependency('gstreamer-1.0', required: false)
gstreamer_audio = dependency('gstreamer-audio-1.0', required: false)
gstreamer_tag = dependency('gstreamer-tag-1.0', required: false)
gupnp_dlna = dependency('gupnp-dlna-2.0', version: '>= 0.9.4', required: false)
libcue = dependency('libcue', version: '>= 2.0.0', required: get_option('cue'))
libexif = dependency('libexif', version: '>= 0.6', required: get_option('exif'))
libgsf = dependency('libgsf-1', version: '>= 1.14.24', required: get_option('gsf'))
libgxps = dependency('libgxps', required: get_option('xps'))
libiptcdata = dependency('libiptcdata', required: get_option('iptc'))
libjpeg = dependency('libjpeg', required: get_option('jpeg'))
libosinfo = dependency('libosinfo-1.0', version: '>= 0.2.9', required: get_option('iso'))
libpng = dependency('libpng', version: '>= 0.89', required: get_option('png'))
libseccomp = dependency('libseccomp', version: '>= 2.0', required: false)
libtiff = dependency('libtiff-4', required: get_option('tiff'))
libxml2 = dependency('libxml-2.0', version: '>= 2.6', required: get_option('xml'))
poppler = dependency('poppler-glib', version: '>= 0.16.0', required: get_option('pdf'))
totem_plparser = dependency('totem-plparser', required: get_option('playlist'))
webp = dependency('libwebpdemux', required: false)

libgif = cc.find_library('gif', required: get_option('gif'))
libmath = cc.find_library('m', required: false)

dbusmock = python.find_installation(
  'python3', modules: ['dbusmock'], required: get_option('functional_tests'))

have_tracker_extract = get_option('extract')
have_tracker_miner_fs = get_option('miner_fs')
have_tracker_writeback = get_option('writeback')

if get_option('man')
  a2x = find_program('a2x')
endif

sha256sum = find_program('sha256sum')

add_project_arguments('-Wformat', '-Wformat-security', language: 'c')

# There are various gchar vs. unsigned char warnings that occur in extract
# modules, it's not worth adding casts everywhere so we disable the warning.
add_project_arguments('-Wno-pointer-sign', language: 'c')

add_project_arguments('-DTRACKER_COMPILATION', language: 'c')
add_project_arguments('-DG_LOG_DOMAIN="Tracker"', language: 'c')
add_project_arguments('-DG_LOG_STRUCTURED=1', language: 'c')

if get_option('buildtype') == 'plain'
  # Default case is to allow LOCALSEARCH_DEBUG which is very useful for
  # triaging user-reported issues.
  # Use --buildtype=release to disable debug completely.
  debug = true
else
  debug = get_option('debug')
endif
optimization = get_option('optimization')

debug_cflags = []
if debug
  debug_cflags += '-DG_ENABLE_DEBUG'
  if optimization in ['0', 'g']
    debug_cflags += '-DG_ENABLE_CONSISTENCY_CHECKS'
  endif
elif optimization in ['2', '3', 's']
  debug_cflags += '-DG_DISABLE_CAST_CHECKS'
endif

add_project_arguments(debug_cflags, language: 'c')

if gobject_introspection.found()
  typelib_dir = gobject_introspection.get_variable(pkgconfig: 'typelibdir',
                                                   pkgconfig_define: [ 'libdir', libdir ])
endif

##################################################################
# Check for libtracker-common: battery/mains power detection
##################################################################

battery_detection_help = 'You can use the "battery_detection" config option to select a different backend or set `-Dbattery_detection=none` to disable this feature.'
battery_detection_library_name = get_option('battery_detection')
if battery_detection_library_name == 'upower'
  upower = dependency('upower-glib',
    version: '>= 0.9.0',
    not_found_message: battery_detection_help)

  battery_detection_library = upower
elif battery_detection_library_name == 'hal'
  hal = dependency('hal',
    version: '>= 0.5',
    not_found_message: battery_detection_help)

  battery_detection_library = hal
endif

##########################################
# Check for Fanotify support
##########################################

have_fanotify = cc.has_header('sys/fanotify.h', required: get_option('fanotify'))

##########################################
# Check for btrfs ioctls
##########################################

if cc.has_header_symbol('linux/btrfs.h', 'BTRFS_IOC_INO_LOOKUP')
  have_btrfs_ioctl = true
else
  have_btrfs_ioctl = false
endif

##########################################
# Check for landlock
##########################################

have_landlock = cc.has_header('linux/landlock.h', required: get_option('landlock'))

# If landlock feature is explicitly enabled and header is present, do not perform
# a runtime test, as this might be e.g. a distribution build in an isolated build
# environment and/or on an older kernel.  If feature is auto, then run-check the
# kernel as well.  https://gitlab.gnome.org/GNOME/tracker-miners/-/issues/300
if have_landlock and get_option('landlock').auto()
  landlock_check = cc.run('''
      #include <fcntl.h>
      #include <linux/landlock.h>
      #include <sys/syscall.h>
      #include <unistd.h>

      int main (int argc, char *argv[]) {
        /* Get supported landlock ABI */
        int abi = syscall (SYS_landlock_create_ruleset, NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
        return abi < 0 ? 1 : 0;
      }
  ''',
  name: 'landlock is enabled in kernel')

  if not (landlock_check.compiled() and landlock_check.returncode() == 0)
    error('Landlock was auto-enabled in build options, but is disabled in the kernel')
  endif
endif

####################################################################
# This section is for tracker-extract dependencies
####################################################################

##########################################################################
# Check for tracker-extract: icu or enca for detecting Cyrillic in MP3s
##########################################################################

charset_library_help = 'You can use the "charset_detection" config option to select a different backend or set `-Dcharset_detection=none` to disable this feature.'
charset_library_name = get_option('charset_detection')

if charset_library_name == 'icu'
  icu_i18n = dependency('icu-i18n',
    version: '>= 4.8.1.1',
    not_found_message: charset_library_help)
  icu_uc = dependency('icu-uc',
    version: '>= 4.8.1.1',
    not_found_message: charset_library_help)

  charset_library = declare_dependency(
     dependencies: [icu_uc, icu_i18n]
  )
elif charset_library_name == 'enca'
  enca = dependency('enca',
    version: '>= 1.9',
    not_found_message: charset_library_help)

  charset_library = enca
endif

####################################################################
# Check for tracker-extract: libav
####################################################################

avformat = dependency('libavformat', version: '>= 0.8.4', required: get_option('libav'))

##################################################################
# Check for libseccomp
##################################################################

host_system = host_machine.system()
host_cpu = host_machine.cpu()

# You can find a list of architectures that seccomp supports in the seccomp
# project README. See: https://github.com/seccomp/libseccomp/
unsupported_cpus = [
  'alpha',
  'ia64',
  'm68k',
  'parisc',
  'parisc64',
  'riscv64',
  'sh4',
  'sparc',
  'sparc64',
]

system_supports_seccomp = host_system == 'linux'
cpu_supports_seccomp = not unsupported_cpus.contains(host_cpu)
seccomp_required = system_supports_seccomp and cpu_supports_seccomp and get_option('seccomp') and get_option('extract')

if not libseccomp.found() and seccomp_required
  error('Libseccomp is necessary for sandboxed metadata extraction')
endif

####################################################################
# D-Bus service files
####################################################################

dbus_services_dir = get_option('dbus_services_dir')
if dbus_services_dir == ''
  dbus_services_dir = dbus.get_variable(pkgconfig: 'session_bus_services_dir',
                                        pkgconfig_define: [ 'datadir', datadir ])
endif

####################################################################
# systemd user services
####################################################################

if get_option('systemd_user_services')
  systemd_user_services_dir = get_option('systemd_user_services_dir')

  if systemd_user_services_dir == ''
    systemd = dependency('systemd', version: '>= 242', required: false)
    if systemd.found()
      systemd_user_services_dir = systemd.get_variable(pkgconfig: 'systemduserunitdir',
                                                       pkgconfig_define: ['prefix', get_option('prefix')])
    else
      error('systemd user services were enabled, but systemd was not found. ' +
            'Please set the systemd_user_services_dir option if you want to ' +
            'install them, or set systemd_user_services=false to disable.')
    endif
  endif
endif

if get_option('bash_completion')
  bash_completion_dir = get_option('bash_completion_dir')

  if bash_completion_dir == ''
    bash_completion_dir = join_paths(datadir, 'bash-completion', 'completions')
  endif
endif

conf = configuration_data()

have_malloc_trim = meson.get_compiler('c').has_function('malloc_trim')

# Config that goes in config.h
conf.set('GUARANTEE_METADATA', get_option('guarantee_metadata') == true)
conf.set('USING_UNZIPPSFILES', get_option('unzip_ps_gz_files') == true)
conf.set('GIO_SUPPORTS_CREATION_TIME', glib.version().version_compare('>=2.70.0'))

conf.set('HAVE_ENCA', charset_library_name == 'enca')
conf.set('HAVE_EXEMPI', exempi.found())
conf.set('HAVE_GSTREAMER_1_20', gstreamer.version() >= '1.20.0')
conf.set('HAVE_POWER', battery_detection_library_name != 'none')
conf.set('HAVE_LIBCUE', libcue.found())
conf.set('HAVE_LIBICU_CHARSET_DETECTION', charset_library_name == 'icu')
conf.set('HAVE_LIBEXIF', libexif.found())
conf.set('HAVE_LIBIPTCDATA', libiptcdata.found())
conf.set('HAVE_LIBSECCOMP', seccomp_required and libseccomp.found())
conf.set('HAVE_MALLOC_TRIM', have_malloc_trim)
conf.set('HAVE_GUPNP_DLNA', gupnp_dlna.found())
conf.set('HAVE_FANOTIFY', have_fanotify)
conf.set('HAVE_BTRFS_IOCTL', have_btrfs_ioctl)
conf.set_quoted('DOMAIN_PREFIX', get_option('domain_prefix'))

if get_option('domain_prefix') != 'org.freedesktop'
  if get_option('miner_fs_cache_location') == ''
    conf.set_quoted('MINER_FS_CACHE_LOCATION', get_option('domain_prefix') + '/miner/files')
  else
    conf.set_quoted('MINER_FS_CACHE_LOCATION', get_option('miner_fs_cache_location'))
  endif
else
  conf.set_quoted('MINER_FS_CACHE_LOCATION', 'tracker3')
endif

conf.set('HAVE_GETLINE', cc.has_function('getline', prefix : '#include <stdio.h>'))
conf.set('HAVE_POSIX_FADVISE', cc.has_function('posix_fadvise', prefix : '#include <fcntl.h>'))
conf.set('HAVE_STATVFS64', cc.has_header_symbol('sys/statvfs.h', 'statvfs64', args: '-D_LARGEFILE64_SOURCE'))
conf.set('HAVE_STRNLEN', cc.has_function('strnlen', prefix : '#include <string.h>'))
conf.set('HAVE_MEMFD_CREATE', cc.has_function('memfd_create', prefix : '#define _GNU_SOURCE\n#include <sys/mman.h>'))
conf.set('HAVE_LANDLOCK', have_landlock)

conf.set_quoted('LOCALEDIR', get_option('prefix') / get_option('localedir'))
conf.set_quoted('SHAREDIR', get_option('prefix') / get_option('datadir'))
conf.set('TRACKER_MINERS_DIR', '"@0@"'.format(join_paths(get_option('prefix'), get_option('datadir'), tracker_versioned_name, 'miners')))

conf.set('GETTEXT_PACKAGE', '"@0@"'.format(tracker_versioned_name))
conf.set('PACKAGE_VERSION', '"@0@"'.format(meson.project_version()))
conf.set('TRACKER_VERSION', '"@0@"'.format(meson.project_version()))

# Config that goes in some other generated files (.desktop, .pc, etc)
conf.set('abs_top_builddir', meson.current_build_dir())
conf.set('exec_prefix', get_option('prefix'))
conf.set('bindir', join_paths(get_option('prefix'), get_option('bindir')))
conf.set('datadir', datadir)
conf.set('datarootdir', join_paths(get_option('prefix'), get_option('datadir')))
conf.set('includedir', join_paths(get_option('prefix'), get_option('includedir')))
conf.set('libdir', libdir)
conf.set('libexecdir', join_paths(get_option('prefix'), get_option('libexecdir')))
conf.set('prefix', get_option('prefix'))
conf.set('TRACKER_API_VERSION', tracker_api_version)
conf.set('VERSION', meson.project_version())

configure_file(input: 'config-miners.h.meson.in',
               output: 'config-miners.h',
               configuration: conf)

# Global compiler defines. We avoid add_global_arguments() as it breaks
# when we are included by another project as a subproject.
tracker_c_args = [
  # Needed for O_NOATIME, and probably other stuff.
  '-D_GNU_SOURCE',

  # Needed for statvfs64, and probably other stuff
  '-D_LARGEFILE64_SOURCE',
]

configinc = include_directories('./')
srcinc = include_directories('src/')

build_root = meson.current_build_dir()

# We use tracker-miners-3.0 rather than tracker3-miners inside the lib64
# directory, following the existing convention in that directory.
tracker_api_versioned_name = 'localsearch-@0@'.format(tracker_api_version)
tracker_internal_libs_dir = join_paths(get_option('prefix'), get_option('libdir'), tracker_api_versioned_name)

tracker_extract_modules_dir = join_paths(get_option('prefix'), get_option('libdir'), tracker_api_versioned_name, 'extract-modules')
tracker_writeback_modules_dir = join_paths(get_option('prefix'), get_option('libdir'), tracker_api_versioned_name, 'writeback-modules')
tracker_extract_rules_dir = join_paths(get_option('prefix'), get_option('datadir'), tracker_versioned_name, 'extract-rules')
tracker_miner_services_dir = join_paths(get_option('prefix'), get_option('datadir'), tracker_versioned_name, 'miners')

gsettings_schema_dir = join_paths(get_option('prefix'), get_option('datadir'), 'glib-2.0', 'schemas')
po_dir = join_paths(meson.current_source_dir(), 'po')
xdg_autostart_dir = join_paths(get_option('sysconfdir'), 'xdg', 'autostart')

glib_compile_schemas = find_program('glib-compile-schemas')
glib_mkenums = find_program('glib-mkenums')

subdir('utils')
subdir('data')
subdir('src')
subdir('docs')

test_c_args = tracker_c_args + [
  '-DTOP_BUILDDIR="@0@/"'.format(meson.build_root()),
  '-DTOP_SRCDIR="@0@/"'.format(meson.source_root()),
]

if get_option('tests_tap_protocol')
  test_protocol = 'tap'
else
  test_protocol = 'exitcode'
endif

subdir('tests')

run_uninstalled_conf = configuration_data()
run_uninstalled_conf.set('tracker_uninstalled_cli_dir', tracker_uninstalled_cli_dir)
run_uninstalled_conf.set('tracker_uninstalled_domain_rule', meson.current_build_dir() / 'tests' / 'functional-tests' / 'config' / 'test-domain.rule')
run_uninstalled_conf.set('tracker_uninstalled_testutils_dir', tracker_uninstalled_testutils_dir)

if have_tracker_extract
  run_uninstalled_conf.set('tracker_uninstalled_extract_rules_dir', tracker_uninstalled_extract_rules_dir)
  run_uninstalled_conf.set('tracker_uninstalled_extractors_dir', uninstalled_tracker_extract_dir)
endif

if have_tracker_writeback
  run_uninstalled_conf.set('tracker_uninstalled_writeback_modules_dir', tracker_uninstalled_writeback_modules_dir)
endif
run_uninstalled_conf.set('uninstalled_gsettings_schema_dir', tracker_miners_uninstalled_gsettings_schema_dir)
run_uninstalled = configure_file(
  input: 'run-uninstalled.in',
  output: 'run-uninstalled',
  configuration: run_uninstalled_conf)

subdir('examples')

subdir('po')

meson.add_install_script('meson_integration_commands.sh', glib_compile_schemas.path(), gsettings_schema_dir)

summary = [
  '\nBuild Configuration:',
  '    Prefix:                                 ' + get_option('prefix'),
  '    Source code location:                   ' + meson.source_root(),
  '    Compiler:                               ' + cc.get_id(),
  '    Debug:                                  ' + get_option('debug').to_string(),
  '    Optimization:                           ' + get_option('optimization'),
  '    Domain prefix:                          ' + get_option('domain_prefix'),
  '\nFeature Support:',
  '    File monitoring:                        @0@glib'.format(have_fanotify ? 'fanotify ' : ''),
  '    Landlock:                               ' + have_landlock.to_string(),
  '    BTRFS subvolumes:                       ' + have_btrfs_ioctl.to_string(),
  '    Battery/mains power detection:          ' + battery_detection_library_name,
  '    Releasing heap memory with malloc_trim: ' + have_malloc_trim.to_string(),
  '    Store creation time:                    ' + glib.version().version_compare('>=2.70.0').to_string(),
  '    Bash completion support:                ' + get_option('bash_completion').to_string() + ' (' + bash_completion_dir + ')',
  '\nData Miners / Writebacks:',
  '    FS (File System):                       ' + have_tracker_miner_fs.to_string(),
  '   Extract (secondary extraction):     ' + have_tracker_extract.to_string(),
  '   Writeback (writing changes back):   ' + have_tracker_writeback.to_string(),
]

if have_tracker_extract
  summary += [
    '\nMetadata Extractors:',
    '    Support PNG:                            ' + libpng.found().to_string(),
    '    Support PDF:                            ' + poppler.found().to_string(),
    '    Support XPS:                            ' + libgxps.found().to_string(),
    '    Support GIF:                            @0@ (xmp: @1@)'.format(libgif.found().to_string(), exempi.found().to_string()),
    '    Support JPEG:                           @0@ (xmp: @1@, exif: @2@, iptc: @3@)'.format(
        libjpeg.found().to_string(), exempi.found().to_string(), libexif.found().to_string(), libiptcdata.found().to_string()),
    '    Support WEBP:                           ' + webp.found().to_string(),
    '    Support RAW:                            ' + gexiv2.found().to_string(),
    '    Support TIFF:                           @0@ (xmp: @1@, exif: @2@, iptc: @3@)'.format(
        libtiff.found().to_string(), exempi.found().to_string(), libexif.found().to_string(), libiptcdata.found().to_string()),
    '    Support MS & Open Office:               ' + libgsf.found().to_string(),
    '    Support XML / HTML:                     ' + libxml2.found().to_string(),
    '    Support embedded / sidecar XMP:         ' + exempi.found().to_string(),
    '    Support A/V formats:                    ' + avformat.found().to_string(),
    '    Support cue sheet parsing:              ' + libcue.found().to_string(),
    '    Support playlists (w/ Totem):           ' + totem_plparser.found().to_string(),
    '    Support ISO image parsing:              ' + libosinfo.found().to_string(),
    '    Support AbiWord document parsing:       ' + get_option('abiword').to_string(),
    '    Support MP3 parsing:                    ' + get_option('mp3').to_string(),
    '    Support MP3 tag charset detection:      ' + charset_library_name,
    '    Support PS parsing:                     ' + get_option('ps').to_string(),
    '    Support text parsing:                   ' + get_option('text').to_string(),
    '    Support icon parsing:                   ' + get_option('icon').to_string(),
  ]
endif


if have_tracker_writeback
  summary += [
    '\nWriteback Formats:',
    '    Audio files using GStreamer:            ' + (gstreamer.found() and gstreamer_tag.found() and gstreamer_audio.found()).to_string(),
    '    XMP:                                    ' + exempi.found().to_string(),
  ]
endif

message('\n'.join(summary))

unsafe = false
if not get_option('seccomp')
  warning('Seccomp sandboxing is disabled.')
  unsafe = true
endif
if get_option('landlock').disabled() or not have_landlock
  warning('Landlock sandboxing is disabled.')
  unsafe = true
endif

if unsafe
  warning('Run at your own risk. Distribution is discouraged.')
endif
