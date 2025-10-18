if not get_option('feature-text')
  error('-Dgtk=true requires -Dfeature-text=true')
endif

foundry_gtk_include_directories = [include_directories('.')]

foundry_gtk_sources = files([
  'foundry-gtk-init.c',
])

foundry_gtk_headers = files([
  'foundry-gtk.h',
  'foundry-gtk-init.h',
])

foundry_gtk_private_sources = [
  gnome.compile_resources('foundry-gtk-resources', 'foundry-gtk.gresource.xml',
    c_name: '_foundry_gtk',
  ),
]

foundry_gtk_private_sources += files([
  'foundry-gtk-plugin.c',
  'foundry-pango.c',
])

foundry_gtk_deps = [
  cc.find_library('m', required: false),
  gsv_dep,
  gtk_dep,
  vte_dep,
]

if get_option('commonmark')
  foundry_gtk_deps += [dependency('libcmark', version: '>= 0.29.0')]
  config_h.set10('HAVE_CMARK', true)
endif

if get_option('webkit')
  foundry_gtk_deps += [dependency('webkitgtk-6.0')]
  config_h.set10('HAVE_WEBKIT', true)
endif

subdir('contrib')
subdir('editing')
subdir('markup')
subdir('platform')
subdir('shortcuts')
subdir('terminal')
subdir('tweaks')
