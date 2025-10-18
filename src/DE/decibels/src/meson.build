gnome = import('gnome')

sources = [
  'application.ts',
  'drag-overlay.ts',
  'empty.ts',
  'error.ts',
  'header.ts',
  'main.ts',
  'mpris.ts',
  'playback-rate-button.ts',
  'player.ts',
  'stream.ts',
  'volume-button.ts',
  'waveform-bar-chart.ts',
  'waveform-generator.ts',
  'waveform-scale.ts',
  'window.ts',
]

outputs = []
foreach source : sources
    outputs += source.replace('.ts', '.js')
endforeach

typescript = custom_target(
  'typescript-compile',
  input: files(sources),
  build_by_default: true,
  command: [ tsc, '--project', tsconfig, '--outDir', '@OUTDIR@' ],
  output: outputs,
)

src_res = gnome.compile_resources(
 application_id + '.src',
 configure_file(
  	input: 'org.gnome.Decibels.src.gresource.xml.in',
  	output: '@BASENAME@',
  	configuration: conf
  ),
  dependencies: typescript,
  gresource_bundle: true,
  install: true,
  install_dir: decibels_pkgdatadir,
)

app_launcher = configure_file(
  input: 'org.gnome.Decibels.in',
  output: application_id,
  configuration: conf,
  install: true,
  install_dir: decibels_bindir
)

run_target(
  'devel',
  command: [gjs_console, '-m', app_launcher],
  depends: [src_res, data_res]
)
