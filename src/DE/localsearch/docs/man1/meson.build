daemon_manpages = [
    ['localsearch-3', 1],
    ['localsearch-writeback-3', 1],
]

cli_manpages = [
  ['localsearch', 1],
  ['localsearch-extract', 1],
  ['localsearch-index', 1],
  ['localsearch-info', 1],
  ['localsearch-inhibit', 1],
  ['localsearch-reset', 1],
  ['localsearch-search', 1],
  ['localsearch-status', 1],
  ['localsearch-tag', 1],
]

foreach m : daemon_manpages + cli_manpages
  name = m[0]
  section = m[1]

  manpage = '@0@.@1@'.format(name, section)

  xml = custom_target(manpage,
    command: [a2x,
              '--attribute=author=The LocalSearch developers',
              '--attribute=manversion=@0@'.format(meson.project_version()),
              '--attribute=manmanual=LocalSearch manual',
              '--doctype=manpage',
              '--format=manpage',
              '--no-xmllint',
              '--xsltproc-opts=' + ' '.join([
                '--stringparam', 'man.authors.section.enabled', '0',
              ]),
              '-D', '@OUTDIR@',
              '@INPUT@'],
    input: manpage + '.txt',
    output: manpage,
    install: true,
    install_dir: get_option('mandir') / 'man@0@'.format(section),
  )
endforeach
