python = find_program('python3')

subdir('mockvolumemonitor')
subdir('test-extractor-modules')

# Configure functional tests to run completely from source tree.
testconf = configuration_data()

config_json_full_path = meson.current_build_dir() / 'config' / 'configuration.json'

testconf.set('MOCK_VOLUME_MONITOR_DIR', meson.current_build_dir() / 'mockvolumemonitor')
testconf.set('TEST_ROOT', meson.current_build_dir())
testconf.set('TEST_CLI_DIR', tracker_uninstalled_cli_dir)
testconf.set('TEST_DBUS_DAEMON_CONFIG_FILE', build_root / 'tests' / 'test-bus.conf')
testconf.set('TEST_DOMAIN_ONTOLOGY_RULE', meson.current_build_dir() / 'config' / 'test-domain.rule')
testconf.set('TEST_GSETTINGS_SCHEMA_DIR', tracker_miners_uninstalled_gsettings_schema_dir)
testconf.set('TEST_TAP_ENABLED', get_option('tests_tap_protocol').to_string())

if have_tracker_extract
  testconf.set('TEST_EXTRACTOR_RULES_DIR', tracker_uninstalled_extract_rules_dir)
  testconf.set('TEST_EXTRACTORS_DIR', uninstalled_tracker_extract_dir)
  testconf.set('TRACKER_EXTRACT_PATH', uninstalled_tracker_extract_path)
endif
 
if have_tracker_writeback
  testconf.set('TEST_WRITEBACK_MODULES_DIR', tracker_uninstalled_writeback_modules_dir)
endif

subdir('config')

extractor_tests = [
  'audio/mp3-id3v2.2-1',
  'audio/mp3-id3v2.3-empty-artist-album',
  'audio/mp3-id3v2.3-vbr-1',
  'audio/mp3-id3v2.4-1',
  'audio/mp3-id3v2.4-2',
  'audio/mp3-id3v2.4-vbr-1',
  'desktop/application',
  'desktop/application-broken-1',
  'desktop/application-broken-2',
  'desktop/application-broken-3',
  'desktop/application-broken-4',
  'desktop/application-hidden',
  'desktop/link-broken-1',
  'desktop/link-wikipedia-tracker',
  'images/bmp-basic-1',
  'images/bmp-corrupt-1',
  'images/bmp-corrupt-2',
  'images/bmp-corrupt-3',
  'images/bmp-corrupt-4',
  'images/ico-basic-1',
  'text/text-non-utf8-1',
]

if get_option('abiword')
  extractor_tests += 'office/abw-1'
  extractor_tests += 'office/abw-2'
  extractor_tests += 'office/abw-3'
endif

# Gstreamer 1.17/1.18 introduced support for some additional
# MusicBrainz tags
if gstreamer.found() and gstreamer.version() >= '1.17.0'
  extractor_tests += 'audio/flac-musicbrainz'
  extractor_tests += 'audio/vorbis-musicbrainz'
else
  extractor_tests += 'audio/flac'
  extractor_tests += 'audio/vorbis'
endif

if libjpeg.found() and exempi.found() and libexif.found()
  extractor_tests += [
    'images/jpeg-basic',
    'images/jpeg-region-of-interest',
    'images/jpeg-gps-location',
    'images/jpeg-xmp-sidecar-1',
  ]

  if libiptcdata.found()
    extractor_tests += [
      'images/jpeg-iptc-tags',
      'images/jpeg-iptcdata',
      'images/jpeg-iptcdata-records',
    ]
  endif
endif

if libgif.found()
  extractor_tests += [
    'images/gif-comment-extension-block',
    'images/gif-corrupted-image',
  ]

  if exempi.found()
    extractor_tests += 'images/gif-xmp'
    extractor_tests += 'images/gif-xmp-sidecar-1'
  endif
endif

if libpng.found()
  extractor_tests += 'images/png-basic'
  extractor_tests += 'images/png-photo-1'
  extractor_tests += 'images/png-screenshot-1'

  if gexiv2.found()
    extractor_tests += 'images/png-exif-old-1'
  endif

  if exempi.found()
    extractor_tests += 'images/png-photo-with-sidecar-1'
    extractor_tests += 'images/png-region-of-interest'
  endif
endif

if gexiv2.found()
  extractor_tests += 'images/raw-cr2'
endif

if webp.found()
  extractor_tests += 'images/webp-basic-1'
endif

if libtiff.found()
  extractor_tests += 'images/tiff-basic'
  extractor_tests += 'images/tiff-photo-1'
  extractor_tests += 'images/tiff-xmp-sidecar-1'
endif

if totem_plparser.found()
  extractor_tests += 'playlists/playlist-test-1'
endif

if libcue.found()
  extractor_tests += 'disc-images/psx-game-image-1'
  extractor_tests += 'disc-images/turbografx-game-image-1'
endif

if libosinfo.found()
  extractor_tests += [
    'disc-images/bootable-1',
    'disc-images/bootable-2',
    'disc-images/non-bootable-1',
  ]
endif

if libgsf.found()
  extractor_tests += [
    'office/epub-doc-1',
    'office/epub-doc-2',
    'office/oasis-doc',
    'office/odg-drawing-1',
    'office/odp-presentation-1',
    'office/ods-spreadsheet-1',
    'office/office-doc',
    'office/office-xml-doc-1',
    'office/powerpoint',
    'office/pdf-doc',
    'office/pdf-encrypted',
    'office/ps-doc',
    'office/ps-doc-2',
    'office/ps-doc-atend',
    'office/pptx-presentation-1',
    'office/xls-spreadsheet-1',
    'office/xlsx-spreadsheet-1',
  ]

  if exempi.found()
    extractor_tests += [
      'office/pdf-doc-with-sidecar-1',
    ]
  endif
endif

if get_option('unzip_ps_gz_files')
  extractor_tests += 'office/psgz-doc'
endif

if libxml2.found()
  extractor_tests += 'office/html-1'
endif

if libgxps.found()
   extractor_tests += 'office/xps-doc-1'
endif

misbehavior_tests = [
  'misbehavior/exit',
  'misbehavior/no-data',
  'misbehavior/no-error',
  'misbehavior/stall',
  'misbehavior/wrong-sparql',
]

if libseccomp.found() and have_landlock and get_option('sandbox_tests')
  misbehavior_tests += [
    'malice/dbus-access',
    'malice/exec',
    'malice/fs-read',
    'malice/fs-read-trunc',
    'malice/fs-write',
    'malice/net-access',
  ]
endif

functional_tests = [
  'test_cli',
  'test_cli_index',
  'test_cli_info',
  'test_cli_inhibit',
  'test_cli_reset',
  'test_cli_search',
  'test_cli_status',
  'test_cli_tag',
  'test_config_changes',
  'test_offline_config_changes',
  'test_extractor_decorator',
  'test_fts_basic',
  'test_fts_file_operations',
  'test_fts_stopwords',
  'test_memory_pressure',
  'test_miner_basic',
  'test_miner_removable_media',
  'test_miner_resource_removal',
  'test_power',
  'test_systemd',
]

if libcue.found()
  functional_tests += 'test_extractor_flac_cuesheet'
endif

if have_tracker_writeback and libjpeg.found() and libgif.found() and libpng.found() and libtiff.found() and exempi.found() and libexif.found()
  functional_tests += [
    'test_writeback_images',
  ]
endif

if have_tracker_writeback and gstreamer.found()
  functional_tests += 'test_writeback_audio'
endif

extractor_tests += [
  'video/mkv-basic',
  'video/mov-basic',
  'video/mp4-basic',
  'video/mp4-video-without-audio',
]

test_env = environment()

test_env.prepend('PYTHONPATH', tracker_uninstalled_testutils_dir)
test_env.set('TRACKER_FUNCTIONAL_TEST_CONFIG', config_json_full_path)
test_env.set('TRACKER_EXTRACT_DEADLINE', '30')
test_env.set('TZ', 'Europe/Amsterdam')
test_env.prepend('MANPATH', uninstalled_manpath)

foreach t: extractor_tests
  data = meson.current_source_dir() / 'data/extractor-content' / t + '.expected.json'
  test_name = t
  test_parts = t.split('/')
  test_suite = []
  if test_parts.length() == 2
    test_name = test_parts[1]
    test_suite = [test_parts[0]]
  endif
  test(test_name, python,
    args: [
      meson.current_source_dir() / 'test_extractor_generic.py',
      data
    ],
    env: test_env,
    protocol: test_protocol,
    suite: ['extractor'] + test_suite)
endforeach

foreach t: functional_tests
  file = meson.current_source_dir() / '@0@.py'.format(t)
  test(t, python,
    args: [file],
    env: test_env,
    protocol: test_protocol,
    suite: ['functional'],
    timeout: 120)
endforeach

foreach t: misbehavior_tests
  rules = meson.current_source_dir() / 'test-extractor-modules'
  data = meson.current_build_dir() / 'test-extractor-modules' / t
  test_parts = t.split('/')
  test_name = test_parts[1]
  test_suite = test_parts[0]
  test_env.set('TRACKER_EXTRACT_DEADLINE', '5')
  test(test_name, python,
    args: [
      meson.current_source_dir() / 'test_misbehavior_generic.py',
      rules,
      data,
    ],
    env: test_env,
    protocol: test_protocol,
    suite: ['extractor', test_suite])
endforeach
