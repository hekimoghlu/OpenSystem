# Copyright 2023 Endless OS Foundation, LLC
# SPDX-License-Identifier: LGPL-2.1-or-later

project('gnome-app-list',
  license: 'LGPL-2.1-or-later',
  version: '3.0',
)

prog_python = find_program('python3')
bash = find_program('bash', required : false)
have_bash = bash.found()

app_lists = files(
  'data/flathub-apps.txt',
  'data/gnome-apps.txt',
  'data/other-apps.txt',
)

swcatalogdir = get_option('datadir') / 'swcatalog' / 'xml'

generated_app_list = custom_target(
  'org.gnome.App-list.xml',
  output : 'org.gnome.App-list.xml',
  input : 'scripts/xml_generator.py',
  command : [prog_python, '@INPUT@', app_lists],
  install : true,
  install_dir: swcatalogdir,
)

# Validation. We can’t use `appstreamcli validate` because it doesn’t handle
# merged components properly — see https://github.com/ximion/appstream/issues/455
xmllint = find_program('xmllint', required: false)
if xmllint.found()
  test(
    'xmllint AppStream files',
    xmllint,
    args: ['--nonet', '--noout'] + [generated_app_list.full_path()],
  )
endif

subdir('tests')