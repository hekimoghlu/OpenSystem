foundry_sources += files([
  'foundry-cli-command.c',
  'foundry-cli-command-tree.c',
  'foundry-command-line.c',
  'foundry-dbus-service.c',
])

foundry_private_sources += files([
  'foundry-cli-builtin-build.c',
  'foundry-cli-builtin-config-list.c',
  'foundry-cli-builtin-config-switch.c',
  'foundry-cli-builtin-dependencies-list.c',
  'foundry-cli-builtin-dependencies-update.c',
  'foundry-cli-builtin-deploy.c',
  'foundry-cli-builtin-devenv.c',
  'foundry-cli-builtin-device-list.c',
  'foundry-cli-builtin-device-switch.c',
  'foundry-cli-builtin-diagnose.c',
  'foundry-cli-builtin-enter.c',
  'foundry-cli-builtin-guess-language.c',
  'foundry-cli-builtin-init.c',
  'foundry-cli-builtin-pipeline-flags.c',
  'foundry-cli-builtin-pipeline-info.c',
  'foundry-cli-builtin-pipeline-invalidate.c',
  'foundry-cli-builtin-pipeline-targets.c',
  'foundry-cli-builtin-pipeline-which.c',
  'foundry-cli-builtin-run.c',
  'foundry-cli-builtin-sdk-install.c',
  'foundry-cli-builtin-sdk-list.c',
  'foundry-cli-builtin-sdk-shell.c',
  'foundry-cli-builtin-sdk-switch.c',
  'foundry-cli-builtin-sdk-which.c',
  'foundry-cli-builtin-search.c',
  'foundry-cli-builtin-settings-get.c',
  'foundry-cli-builtin-settings-set.c',
  'foundry-cli-builtin-shell.c',
  'foundry-cli-builtin-show.c',
  'foundry-cli-builtin-test-list.c',
  'foundry-cli-builtin-test-run.c',

  'foundry-command-line-input.c',
  'foundry-command-line-local.c',
  'foundry-command-line-remote.c',
])

if get_option('feature-git')
  foundry_private_sources += files([
    'foundry-cli-builtin-clone.c',
  ])
endif

if get_option('feature-docs')
  foundry_private_sources += files([
    'foundry-cli-builtin-doc-bundle-install.c',
    'foundry-cli-builtin-doc-bundle-list.c',
    'foundry-cli-builtin-doc-query.c',
  ])
endif

if get_option('feature-llm')
  foundry_private_sources += files([
    'foundry-cli-builtin-llm-complete.c',
    'foundry-cli-builtin-llm-list-models.c',
    'foundry-cli-builtin-llm-list-tools.c',
  ])
endif

if get_option('feature-lsp')
  foundry_private_sources += files([
    'foundry-cli-builtin-lsp-list.c',
    'foundry-cli-builtin-lsp-prefer.c',
    'foundry-cli-builtin-lsp-run.c',
  ])
endif

if get_option('feature-templates')
  foundry_private_sources += files([
    'foundry-cli-builtin-template-create.c',
    'foundry-cli-builtin-template-list.c',
  ])
endif

if get_option('feature-vcs')
  foundry_private_sources += files([
    'foundry-cli-builtin-vcs-blame.c',
    'foundry-cli-builtin-vcs-fetch.c',
    'foundry-cli-builtin-vcs-ignored.c',
    'foundry-cli-builtin-vcs-list.c',
    'foundry-cli-builtin-vcs-list-branches.c',
    'foundry-cli-builtin-vcs-list-files.c',
    'foundry-cli-builtin-vcs-list-remotes.c',
    'foundry-cli-builtin-vcs-list-tags.c',
    'foundry-cli-builtin-vcs-log.c',
    'foundry-cli-builtin-vcs-switch.c',
  ])
endif

foundry_headers += files([
  'foundry-cli-command.h',
  'foundry-cli-command-tree.h',
  'foundry-command-line.h',
  'foundry-dbus-service.h',
])

foundry_include_directories += [include_directories('.')]
