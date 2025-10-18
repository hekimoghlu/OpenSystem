if not get_option('feature-text')
  error('-Dfeature-lsp=true requires -Dfeature-text=true')
endif

libfoundry_config_conf.set10('FOUNDRY_FEATURE_LSP', true)

foundry_sources += files([
  'foundry-lsp-client.c',
  'foundry-lsp-completion-proposal.c',
  'foundry-lsp-completion-provider.c',
  'foundry-lsp-manager.c',
  'foundry-lsp-provider.c',
  'foundry-lsp-server.c',
  'foundry-plugin-lsp-provider.c',
])

foundry_private_sources += files([
  'foundry-lsp-capabilities.c',
  'foundry-lsp-completion-results.c',
  'foundry-plugin-lsp-server.c',
])

foundry_headers += files([
  'foundry-lsp-client.h',
  'foundry-lsp-completion-proposal.h',
  'foundry-lsp-completion-provider.h',
  'foundry-lsp-manager.h',
  'foundry-lsp-provider.h',
  'foundry-lsp-server.h',
  'foundry-plugin-lsp-provider.h',
])

foundry_include_directories += [include_directories('.')]
