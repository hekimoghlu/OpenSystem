manuals_sources = [
  'main.c',
  'manuals-application.c',
  'manuals-bundle-dialog.c',
  'manuals-install-button.c',
  'manuals-model-manager.c',
  'manuals-path-bar.c',
  'manuals-path-button.c',
  'manuals-path-element.c',
  'manuals-path-model.c',
  'manuals-search-entry.c',
  'manuals-search-row.c',
  'manuals-tab.c',
  'manuals-tag.c',
  'manuals-tree-expander.c',
  'manuals-window.c',
  'manuals-wrapped-model.c',
]

manuals_deps = [
  adw_dep,
  dex_dep,
  flatpak_dep,
  foundry_dep,
  glib_dep,
  gom_dep,
  gtk_dep,
  libpanel_dep,
  webkit_dep,
]

manuals_sources += gnome.compile_resources('manuals-resources',
  'manuals.gresource.xml',
  c_name: 'manuals'
)

manuals = executable('manuals', manuals_sources,
  dependencies: manuals_deps,
       install: true,
)
