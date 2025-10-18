libfoundry_config_conf.set10('FOUNDRY_FEATURE_TEMPLATES', true)

foundry_sources += files([
  'foundry-code-template.c',
  'foundry-project-template.c',
  'foundry-template.c',
  'foundry-template-manager.c',
  'foundry-template-output.c',
  'foundry-template-provider.c',
])

foundry_private_sources += files([
  'foundry-internal-template.c',
  'foundry-template-locator.c',
])

foundry_headers += files([
  'foundry-code-template.h',
  'foundry-project-template.h',
  'foundry-template.h',
  'foundry-template-manager.h',
  'foundry-template-output.h',
  'foundry-template-provider.h',
])

foundry_include_directories += [include_directories('.')]

foundry_deps += [
  dependency('template-glib-1.0'),
]
