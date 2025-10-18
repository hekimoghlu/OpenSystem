# TODO: it would be great to clean this up by making plugins an array
#       of plugin names: {features: []} requirements.

fallback_plugins = [
  'buildconfig',
  'content-types',
  'host-sdk',
  'internal-tweaks',
  'local-device',
  'no-sdk',
  'plugin-tweaks',
  'tty-auth',
]

if get_option('feature-flatpak')
  fallback_plugins += ['flatpak']
endif

if get_option('feature-git')
  fallback_plugins += ['git']
endif

if get_option('feature-lsp')
  fallback_plugins += ['lsp-bridge']
endif

if get_option('feature-text')
  fallback_plugins += ['hover-bridge', 'gsettings', 'language-defaults']
endif

if get_option('feature-vcs')
  fallback_plugins += ['no-vcs']
endif

if get_option('feature-llm')
  fallback_plugins += ['llm-build']
endif

plugins = [
  'autotools',
  'buildstream',
  'cargo',
  'cmake',
  'codesearch',
  'codespell',
  'deviced',
  'doap',
  'dub',
  'eslint',
  'file-search',
  'flake8',
  'gdiagnose',
  'gettext',
  'go',
  'gradle',
  'jhbuild',
  'make',
  'maven',
  'meson',
  'npm',
  'phpize',
  'podman',
  'qemu',
  'shellcheck',
  'sarif',
  'waf',
]

doc_plugins = [
  'devhelp',
]

llm_plugins = [
  'ollama',
]

lsp_plugins = [
  'astro-ls',
  'bash-language-server',
  'blueprint',
  'clangd',
  'elixir-ls',
  'glsl-language-server',
  'gopls',
  'intelephense',
  'jdtls',
  'jedi-language-server',
  'lua-language-server',
  'mesonlsp',
  'pylsp',
  'pyrefly',
  'ruff',
  'rust-analyzer',
  'serve-d',
  'sourcekit-lsp',
  'ts-language-server',
  'ty',
  'vala-language-server',
  'vhdl-language-server',
  'zls',
]

text_plugins = [
  'ctags',
  'editorconfig',
  'modelines',
]

template_plugins = [
  'make-templates',
  'meson-templates',
  'simple-templates',
]

if get_option('feature-debugger') and get_option('feature-dap')
  plugins += ['gdb', 'lldb']
endif

foundry_plugin_manifest_xml = ''

foreach plugin: fallback_plugins
  subdir(join_paths('fallbacks', plugin))
  foundry_plugin_manifest_xml += '    <file alias="@0@/@0@.plugin">fallbacks/@0@/@0@.plugin</file>\n'.format(plugin)
  config_h.set10('HAVE_PLUGIN_@0@'.format(plugin.replace('-','_').to_upper()), true)
endforeach

foreach plugin: plugins
  if get_option('plugin-@0@'.format(plugin))
    subdir(plugin)
    foundry_plugin_manifest_xml += '    <file>@0@/@0@.plugin</file>\n'.format(plugin)
    config_h.set10('HAVE_PLUGIN_@0@'.format(plugin.replace('-','_').to_upper()), true)
  endif
endforeach

foreach plugin: doc_plugins
  if get_option('plugin-@0@'.format(plugin))
    if get_option('feature-docs')
      subdir(plugin)
      foundry_plugin_manifest_xml += '    <file>@0@/@0@.plugin</file>\n'.format(plugin)
      config_h.set10('HAVE_PLUGIN_@0@'.format(plugin.replace('-','_').to_upper()), true)
    endif
  endif
endforeach

foreach plugin: llm_plugins
  if get_option('plugin-@0@'.format(plugin))
    if get_option('feature-llm')
      subdir(plugin)
      foundry_plugin_manifest_xml += '    <file>@0@/@0@.plugin</file>\n'.format(plugin)
      config_h.set10('HAVE_PLUGIN_@0@'.format(plugin.replace('-','_').to_upper()), true)
    endif
  endif
endforeach

foreach plugin: lsp_plugins
  if get_option('plugin-@0@'.format(plugin))
    if get_option('feature-lsp')
      subdir(plugin)
      foundry_plugin_manifest_xml += '    <file>@0@/@0@.plugin</file>\n'.format(plugin)
      config_h.set10('HAVE_PLUGIN_@0@'.format(plugin.replace('-','_').to_upper()), true)
    endif
  endif
endforeach

foreach plugin: text_plugins
  if get_option('plugin-@0@'.format(plugin))
    if get_option('feature-text')
      subdir(plugin)
      foundry_plugin_manifest_xml += '    <file>@0@/@0@.plugin</file>\n'.format(plugin)
      config_h.set10('HAVE_PLUGIN_@0@'.format(plugin.replace('-','_').to_upper()), true)
    endif
  endif
endforeach

foreach plugin: template_plugins
  if get_option('plugin-@0@'.format(plugin))
    if get_option('feature-templates')
      subdir(plugin)
      foundry_plugin_manifest_xml += '    <file>@0@/@0@.plugin</file>\n'.format(plugin)
      config_h.set10('HAVE_PLUGIN_@0@'.format(plugin.replace('-','_').to_upper()), true)
    endif
  endif
endforeach

foundry_plugin_manifests_data = configuration_data()
foundry_plugin_manifests_data.set('XML', foundry_plugin_manifest_xml)
foundry_plugins_gresource_xml = configure_file(
          input: 'plugins.gresource.xml.in',
         output: 'foundry-plugins.gresource.xml',
  configuration: foundry_plugin_manifests_data)

foundry_private_sources += [
  gnome.compile_resources(
    'foundry-plugins-resources',
    foundry_plugins_gresource_xml,
    c_name: '_foundry_plugins',
  )
]

foundry_include_directories += [include_directories('.')]

if get_option('gtk')
  foundry_gtk_plugin_manifest_xml = ''

  gtk_plugins = {
    'spellcheck': {},
    'word-completion': {},
  }

  foreach plugin, params: gtk_plugins
    if get_option('plugin-@0@'.format(plugin))
      features = params.get('options', [])
      has_features = true
      foreach feature: features
        if not get_option(feature)
          has_features = false
        endif
      endforeach
      if has_features
        subdir(plugin)
        foundry_gtk_plugin_manifest_xml += '    <file>@0@/@0@.plugin</file>\n'.format(plugin)
      endif
    endif
  endforeach

  foundry_gtk_plugin_manifests_data = configuration_data()
  foundry_gtk_plugin_manifests_data.set('XML', foundry_gtk_plugin_manifest_xml)
  foundry_gtk_plugins_gresource_xml = configure_file(
            input: 'plugins.gresource.xml.in',
           output: 'foundry-gtk-plugins.gresource.xml',
    configuration: foundry_gtk_plugin_manifests_data)

  foundry_gtk_private_sources += [
    gnome.compile_resources(
      'foundry-gtk-plugins-resources',
      foundry_gtk_plugins_gresource_xml,
      c_name: '_foundry_gtk_plugins',
    )
  ]

  foundry_gtk_include_directories += [include_directories('.')]
endif
