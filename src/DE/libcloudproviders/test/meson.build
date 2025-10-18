installed = get_option('installed-tests')
executable('testcloudprovidersserver', 'testcloudprovidersserver.c',
  dependencies: libcloudproviders_dep,
  install: installed,
)
executable('testcloudprovidersclient', 'testcloudprovidersclient.c',
  dependencies: libcloudproviders_dep,
  install: installed,
)

configure_file(input : 'icon.svg',
               output : 'icon.svg',
               copy : true)

if installed
  service_conf = configuration_data ()
  service_conf.set ('bindir', join_paths (prefix, bindir))
  configure_file(
    input: 'org.freedesktop.CloudProviders.ServerExample.service.in',
    output: 'org.freedesktop.CloudProviders.ServerExample.service',
    configuration: service_conf,
    install_dir: servicedir,
  )
  applications = join_paths (datadir, 'applications')
  cloudprovidersdata = join_paths (datadir, 'cloud-providers')
  install_data('org.freedesktop.CloudProviders.ServerExample.desktop', install_dir: applications)
  install_data('org.freedesktop.CloudProviders.ServerExample.ini', install_dir: cloudprovidersdata)
endif
