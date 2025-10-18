if enable_nautilus
  shared_module(
    'papers-document-properties',
    sources: 'pps-nautilus-extension.c',
    include_directories: top_inc,
    dependencies: [libppsdocument_dep, libnautilus_extension_dep],
    install: true,
    install_dir: nautilus_extension_dir,
    gnu_symbol_visibility: 'hidden',
  )
endif
