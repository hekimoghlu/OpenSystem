# docs

# input: install_datadir, website_custom_cmd, python3
# output: can_parse_and_validate, build_translations

# xsltproc is required by website_custom_cmd html.
xsltproc = find_program('xsltproc', required: true)

build_translations = get_option('build-translations')
msgfmt = find_program('msgfmt', required: build_translations)
itstool = find_program('itstool', required: build_translations)

xmllint = find_program('xmllint', required: false)
can_parse_and_validate = xmllint.found()
validate = get_option('validation') ? 'true' : 'false'
allow_network_access = get_option('allow-network-access') ? 'true' : 'false'

relax_ng_schema_file = meson.current_source_dir() / 'schema.rng'
stylesheet_file = meson.current_source_dir() / 'style.xsl'

# Check if xmllint can be used.
if xmllint.found()
  can_parse_and_validate = run_command(
    python3, website_custom_cmd, 'xmllint',
    validate,
    allow_network_access,
    relax_ng_schema_file,
    meson.current_source_dir() / 'can_use_xmllint.xml',
    meson.current_build_dir() / 'can_use_xmllint.stamp',
    check: false,
  ).returncode() == 0
  if not can_parse_and_validate
    # The DocBook V5.0 package is called docbook5-xml in Ubuntu,
    # docbook5-schemas in Fedora. It may have other names in other distros.
    warning('Can\'t validate XML file.\n' +
      'xmllint does not support Relax NG schemas and DocBook V5.0.\n' +
      'DocBook V5.0 support may require docbook5-xml, docbook5-schemas or a similar package.'
    )
  endif
endif

# index.withimg is index.docbook with images.
# index.withimg shall not be found by the translation module damned-lies.
# https://gitlab.gnome.org/GNOME/gnomemm-website/-/issues/4
xml_files = [
  'index.withimg', # Must be the first element in the list
  'books.xml',
  'bugs.xml',
  'commercial_support.xml',
  'developers.xml',
  'documentation.xml',
  'download.xml',
  'extra.xml',
  'faq.xml',
  'git.xml',
  'license.xml',
  'mailinglist.xml',
  'main.xml',
  'news.xml',
]
xml_C_files = []
foreach f : xml_files
  xml_C_files += 'C' / f
endforeach

if build_translations
  # Get a list of translated languages.
  language_obj = run_command(
    python3, website_custom_cmd, 'get_languages',
    meson.current_source_dir(),
    check: false,
  )
  if language_obj.stderr() != ''
    warning(language_obj.stderr())
  endif
  languages = language_obj.stdout().split()
else
  languages = []
endif

if can_parse_and_validate
  # Parse and possibly validate the C locale's version of the DocBook.
  custom_target('C-xmllint',
    input: [relax_ng_schema_file] + xml_C_files,
    output: 'C_xmllint.stamp',
    command: [
      python3, website_custom_cmd, 'xmllint',
      validate,
      allow_network_access,
      '@INPUT0@',
      '@INPUT1@',
      '@OUTPUT@'
    ],
    build_by_default: true,
  )
endif

publishing_targets = []

# Create an html version of the C locale's version of the DocBook.
publishing_targets += custom_target('en_html_index.html',
  input: [stylesheet_file] + xml_C_files,
  output: 'en_html.stamp',
  command: [
    python3, website_custom_cmd, 'html',
    allow_network_access,
    '@INPUT0@',
    '@INPUT1@',
    meson.current_build_dir() / 'html' / 'en',
    '@OUTPUT@',
  ],
  build_by_default: true,
  install: false,
)

# Install the C locale's XML files.
install_data(xml_C_files,
  install_dir: install_datadir / 'help' / 'C' / meson.project_name(),
  install_tag: 'doc',
)

# gnome.yelp() is not used. It does not return anything, such as a custom target
# object. It's not possible to guarantee that translated html files are created
# *after* the translated xml files.
# Meson module i18n is not used. It requires a different directory structure
# with all .po files in the same directory. Probably it also has other
# problematic restrictions.

foreach language : languages
  # Create translated XML files.
  translated_xml_target = custom_target(language + '_xml_index.docbook',
    input: [language / language + '.po'] + xml_C_files,
    output: language + '_xml.stamp',
    command: [
      python3, website_custom_cmd, 'translate_xml',
      '@INPUT0@',
      meson.current_source_dir() / 'C',
      meson.current_build_dir() / language, # Absolute path
      '@OUTPUT@',
      xml_files,
    ],
    build_by_default: true,
    install: false,
  )

  # Create an html version of the translated version of the DocBook.
  publishing_targets += custom_target(language + '_html_index.html',
    input: [stylesheet_file, translated_xml_target],
    output: language + '_html.stamp',
    command: [
      python3, website_custom_cmd, 'html',
      allow_network_access,
      '@INPUT0@',
      meson.current_build_dir() / language / xml_files[0],
      meson.current_build_dir() / 'html' / language,
      '@OUTPUT@',
    ],
    build_by_default: true,
    install: false,
  )

  # Install the translated XML files.
  meson.add_install_script(
    python3, website_custom_cmd, 'install',
    meson.current_build_dir() / language,
    install_datadir / 'help' / language / meson.project_name(),
    xml_files,
    install_tag: 'doc',
  )
endforeach

# Publish the website at www.gtkmm.org.
run_target('publish',
  command: [
    python3, website_custom_cmd, 'publish',
  ],
  depends: publishing_targets,
)
