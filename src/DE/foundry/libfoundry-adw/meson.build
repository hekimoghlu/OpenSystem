foundry_adw_include_directories = [include_directories('.')]

foundry_adw_sources = files([
  'foundry-adw-init.c',
  'foundry-page.c',
  'foundry-panel.c',
  'foundry-panel-bar.c',
  'foundry-search-dialog.c',
  'foundry-splitable-page.c',
  'foundry-tree-expander.c',
  'foundry-workspace.c',
  'foundry-workspace-addin.c',
])

foundry_adw_headers = files([
  'foundry-adw.h',
  'foundry-adw-init.h',
  'foundry-page.h',
  'foundry-panel.h',
  'foundry-panel-bar.h',
  'foundry-search-dialog.h',
  'foundry-splitable-page.h',
  'foundry-tree-expander.h',
  'foundry-workspace.h',
  'foundry-workspace-addin.h',
])

foundry_adw_private_sources = files([
  'foundry-action-responder.c',
  'foundry-action-responder-group.c',
  'foundry-frame.c',
  'foundry-multi-reaction.c',
  'foundry-panel-button.c',
  'foundry-property-reaction.c',
  'foundry-reaction.c',
  'foundry-responder.c',
  'foundry-signal-responder.c',
  'foundry-workspace-child.c',
])

foundry_adw_private_sources += [
  gnome.compile_resources('foundry-adw-resources', 'foundry-adw.gresource.xml',
    c_name: '_foundry_adw',
  ),
]

foundry_adw_deps = [
  dependency('libadwaita-1', version: '>= 1.7'),
  dependency('libpanel-1', version: '>= 1.8'),
]
