# This file is part of gnomemm-website.

project('gnomemm-website',
  version: '4.0.0',
  meson_version: '>= 0.60.0', # required for install_tag keyword argument.
)

python3 = find_program('python3', version: '>=3.7')

# Installation directories are relative to {prefix}.
install_prefix = get_option('prefix')
install_datadir = get_option('datadir')

# Source and build root directories of the current (sub)project.
project_source_root = meson.project_source_root()
project_build_root = meson.project_build_root()

script_dir = project_source_root / 'tools'
website_custom_cmd = script_dir / 'website-custom-cmd.py'
extra_dist_cmd = script_dir / 'extra-dist-cmd.py'

# Copy some files to the build directory.
# (fs.copyfile() would be useful, if there were also a fs.copydir().)
copy_obj = run_command(
  python3, website_custom_cmd, 'copy_files_and_dirs',
  project_source_root / 'docs' / 'html',
  project_build_root / 'docs' / 'html',
  'index.html',
  'images',
  'style',
  check: false,
)
if copy_obj.stdout() != ''
  warning(copy_obj.stdout())
endif

subdir('docs')

# Don't distribute these files and directories.
dont_distribute = [
  'gnomemm-website.doap',
  '.gitlab-ci.yml',
]
# Modify the contents of the distribution directory.
meson.add_dist_script(
  python3, extra_dist_cmd,
  project_source_root,
  project_build_root,
  dont_distribute,
)

# Print a summary.
validate = get_option('validation') and can_parse_and_validate
explain_val = ''
if get_option('validation') and not validate
  explain_val = ' (requires xmllint with Relax NG and DocBook V5.0 support)'
endif

summary = [
  '',
  '------',
  meson.project_name() + ' ' + meson.project_version(),
  '',
  '      XML validation: @0@@1@'.format(validate, explain_val),
  'Allow network access: @0@'.format(get_option('allow-network-access')),
  '  Build translations: @0@'.format(build_translations),
  'Directories:',
  '              prefix: @0@'.format(install_prefix),
  '             datadir: @0@'.format(install_prefix / install_datadir),
  '------'
]

message('\n'.join(summary))
