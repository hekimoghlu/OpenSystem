project('decibels',
          version: '49.0',
    meson_version: '>= 0.62.0',
  default_options: [ 'warning_level=2', 'werror=false', ],
)

if get_option('profile') == 'development'
  profile = '.Devel'
  name_suffix = ' (Development)'
  vcs_tag = run_command('git', 'rev-parse', '--short', 'HEAD', check: true).stdout().strip()
  if vcs_tag == ''
    version_suffix = '-devel'
  else
    version_suffix = '-@0@'.format (vcs_tag)
  endif
else
  profile = ''
  name_suffix = ''
  version_suffix = ''
endif

application_id = 'org.gnome.Decibels@0@'.format(profile)

i18n = import('i18n')
gnome = import('gnome')

decibels_prefix = get_option('prefix')
decibels_bindir = decibels_prefix / get_option('bindir')
decibels_libdir = decibels_prefix / get_option('libdir')
decibels_datadir = decibels_prefix / get_option('datadir')
decibels_pkgdatadir = decibels_datadir / application_id

gjs_dep = dependency('gjs-1.0', version: '>= 1.54.0')
gjs_console = gjs_dep.get_variable(pkgconfig: 'gjs_console')
libadwaita = dependency('libadwaita-1', version: '>= 1.8.alpha')

tsc = find_program('tsc', required: true)

conf = configuration_data()
conf.set('GJS', gjs_console)
conf.set('PACKAGE_VERSION', '@0@@1@'.format(meson.project_version(), version_suffix))
conf.set('APP_ID', application_id)
conf.set('bindir', decibels_bindir)
conf.set('prefix', decibels_prefix)
conf.set('libdir', decibels_libdir)
conf.set('datadir', decibels_datadir)
conf.set('resource_path', '/org/gnome/Decibels@0@'.format(profile))
conf.set('profile', profile)

tsconfig = files('tsconfig.json')

subdir('data')
subdir('src')
subdir('po')

gnome.post_install(
    gtk_update_icon_cache: true,
  update_desktop_database: true,
)
