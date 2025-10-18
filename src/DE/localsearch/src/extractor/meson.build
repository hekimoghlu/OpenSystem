subdir('utils')

tracker_uninstalled_extract_rules_dir = join_paths(meson.current_build_dir(), 'uninstalled-rules')
uninstalled_tracker_extract_dir = meson.current_build_dir()
uninstalled_tracker_extract_path = join_paths(meson.current_build_dir(), 'localsearch-extractor-3')

# Extract modules are tracked here as arrays with the following elements:
#
#   [ name, source-files, rule-files, dependencies ]
#
# This might be nicer if Meson supported dictionary types.

modules = [
  ['extract-bmp', 'tracker-extract-bmp.c', ['10-bmp.rule'], [tracker_miners_common_dep]],
  ['extract-desktop', 'tracker-extract-desktop.c', ['10-desktop.rule'], [tracker_miners_common_dep]],
  ['extract-dummy', [], ['10-comics.rule', '10-ebooks.rule', '10-folder.rule', '10-svg.rule', '15-games.rule', '15-executable.rule'], [tracker_miners_common_dep]]
]

if get_option('abiword')
  modules += [['extract-abw', 'tracker-extract-abw.c', ['10-abw.rule'], [tracker_miners_common_dep]]]
endif

if libgsf.found()
  modules += [['extract-epub', 'tracker-extract-epub.c', ['10-epub.rule'], [libgsf, tracker_miners_common_dep]]]
  modules += [['extract-msoffice', 'tracker-extract-msoffice.c', ['10-msoffice.rule'], [libgsf, tracker_miners_common_dep]]]
  modules += [['extract-msoffice-xml', 'tracker-extract-msoffice-xml.c', ['11-msoffice-xml.rule'], [libgsf, tracker_miners_common_dep]]]
  modules += [['extract-oasis', 'tracker-extract-oasis.c', ['10-oasis.rule'], [libgsf, tracker_miners_common_dep]]]
endif

if gexiv2.found()
  modules += [['extract-raw', 'tracker-extract-raw.c', ['10-raw.rule'], [gexiv2, tracker_miners_common_dep]]]
endif

if libgif.found()
  modules += [['extract-gif', 'tracker-extract-gif.c', ['10-gif.rule'], [libgif, tracker_miners_common_dep]]]
endif

if libxml2.found()
  modules += [['extract-html', 'tracker-extract-html.c', ['10-html.rule'], [libxml2]]]
endif

if libjpeg.found()
  modules += [['extract-jpeg', 'tracker-extract-jpeg.c', ['10-jpeg.rule'], [libjpeg, tracker_miners_common_dep]]]
endif

if avformat.found()
  rules = ['15-libav-guess.rule', '90-libav-audio-generic.rule', '90-libav-video-generic.rule']
  extract_libav_files = ['tracker-extract-libav.c', 'tracker-cue-sheet.c']
  if gupnp_dlna.found()
    extract_libav_files += ['tracker-gupnp.c']
  endif
  modules += [['extract-libav', extract_libav_files, rules, [avformat, libcue, gupnp_dlna, tracker_miners_common_dep]]]
endif

if get_option('icon')
  modules += [['extract-icon', 'tracker-extract-icon.c', ['10-ico.rule'], []]]
endif

if libosinfo.found()
  modules += [['extract-iso', 'tracker-extract-iso.c', ['11-iso.rule'], [libosinfo]]]
endif

if get_option('mp3')
  modules += [['extract-mp3', 'tracker-extract-mp3.c', ['10-mp3.rule'], [tracker_miners_common_dep]]]
endif

if poppler.found()
  modules += [['extract-pdf', 'tracker-extract-pdf.c', ['10-pdf.rule'], [poppler, tracker_miners_common_dep]]]
endif

if totem_plparser.found()
  modules += [['extract-playlist', 'tracker-extract-playlist.c', ['15-playlist.rule'], [totem_plparser]]]
endif

if libpng.found()
  modules += [['extract-png', 'tracker-extract-png.c', ['10-png.rule'], [libpng, tracker_miners_common_dep]]]
endif

if get_option('ps')
  modules += [['extract-ps', 'tracker-extract-ps.c', ['10-ps.rule'], [tracker_miners_common_dep]]]
endif

if get_option('text')
  modules += [['extract-text', 'tracker-extract-text.c', ['15-text.rule'], []]]
endif

if libtiff.found()
  modules += [['extract-tiff', 'tracker-extract-tiff.c', ['10-tiff.rule'], [libtiff, tracker_miners_common_dep]]]
endif

if libgxps.found()
  modules += [['extract-xps', 'tracker-extract-xps.c', ['10-xps.rule'], [libgxps, tracker_miners_common_dep]]]
endif

if libcue.found()
  modules += [['extract-disc-generic', 'tracker-extract-disc-generic.c', ['90-disc-generic.rule'], [libcue, tracker_miners_common_dep]]]
endif

if webp.found()
  modules += [['extract-webp', 'tracker-extract-webp.c', ['10-webp.rule'], [webp, tracker_miners_common_dep]]]
endif

enabled_rule_files = []

foreach module : modules
  name = module[0]
  sources = module[1]
  rules = module[2]
  dependencies = module[3]

  shared_module(name, sources,
    c_args: tracker_c_args,
    dependencies: [tracker_extract_dep, tracker_miners_common_dep] + dependencies,
    include_directories: [configinc, commoninc],
    install: true,
    install_dir: tracker_extract_modules_dir)

  hash = run_command('calculate-hash.sh', [sources] + files_extract, check: true)

  foreach rule : rules
    configure_file(input: rule,
                   output: '@PLAINNAME@',
                   install: true,
                   install_dir: tracker_extract_rules_dir,
                   configuration: { 'hash': hash.stdout() })
  endforeach

  enabled_rule_files += rules
endforeach

extract_resources = gnome.compile_resources(
    'extract-resources',
    'org.freedesktop.Tracker3.Extract.gresource.xml',
    c_name: 'tracker_extract',
)

tracker_extract_sources = [
  'tracker-decorator.c',
  'tracker-extract.c',
  'tracker-extract-controller.c',
  'tracker-extract-persistence.c',
  'tracker-read.c',
  'tracker-main.c',
  extract_resources,
]

tracker_extract_dependencies = [
  tracker_miners_common_dep, tracker_extract_dep
]

if libgsf.found()
  tracker_extract_sources += 'tracker-gsf.c'
  tracker_extract_dependencies += libgsf
endif

executable('localsearch-extractor-@0@'.format(tracker_api_major),
  tracker_extract_sources,
  # Manually add the root dir to work around https://github.com/mesonbuild/meson/issues/1387
  c_args: tracker_c_args + ['-I' + meson.build_root()],
  export_dynamic: true,
  dependencies: tracker_extract_dependencies,
  install: true,
  install_dir: join_paths(get_option('prefix'), get_option('libexecdir')),
  install_rpath: tracker_internal_libs_dir)

# Populate a directory inside the build tree with the extract rules that are
# enabled in this build configuration.
setup_extract_rules = join_paths(meson.current_source_dir(), 'setup-extract-rules.sh')
result = run_command(setup_extract_rules, meson.current_source_dir(), tracker_uninstalled_extract_rules_dir, enabled_rule_files, check: true)
if result.returncode() != 0
    output = result.stdout().strip()
    errortxt = result.stderr().strip()
    error('Failed to run @0@.\n@1@\n@2@'.format(setup_extract_rules, errortxt, output))
endif
