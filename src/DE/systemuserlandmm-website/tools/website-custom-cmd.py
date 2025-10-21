#!/usr/bin/env python3

# External command, intended to be called in meson.build

#                           argv[1]   argv[2:]
# tutorial-custom-cmd.py <subcommand> <xxx>...

import os
import sys
import subprocess
import shutil
from pathlib import Path

subcommand = sys.argv[1]

# Called from run_command()
def copy_files_and_dirs():
  #   argv[2]      argv[3]       argv[4:]
  # <input_dir> <output_dir> <files_and_dirs>...

  input_dir = sys.argv[2]
  output_dir = sys.argv[3]
  files_and_dirs =  sys.argv[4:]

  for f in files_and_dirs:
    input_path = os.path.join(input_dir, f)
    output_path = os.path.join(output_dir, f)
    if os.path.isfile(input_path):
      # Create the destination directory, if it does not exist.
      os.makedirs(os.path.dirname(output_path), exist_ok=True)
      # shutil.copy2() copies timestamps and some other file metadata.
      shutil.copy2(input_path, output_path)
    elif os.path.isdir(input_path):
      # Create the destination directory, if it does not exist.
      os.makedirs(output_path, exist_ok=True)
      shutil.copytree(input_path, output_path,
                      copy_function=shutil.copy2, dirs_exist_ok=True)
    else:
      # Ignore non-existent files and directories.
      print(input_path, 'not found, not copied.')
  return 0

# Called from run_command()
def get_languages():
  #   argv[2]
  # <input_dir>

  input_dir = sys.argv[2]

  # Syntax of the LINGUAS file is documented here:
  # https://www.gnu.org/software/gettext/manual/html_node/po_002fLINGUAS.html
  linguas_file = os.path.join(input_dir, 'LINGUAS')
  try:
    with open(linguas_file, encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
          print(line, end=' ')
  except (FileNotFoundError, PermissionError):
    print('Could not find file LINGUAS in', input_dir, file=sys.stderr)
    return 1
  return 0

# Called from custom_target()
def xmllint():
  #   argv[2]         argv[3]                argv[4]            argv[5]            argv[6]
  # <validate> <allow_network_access> <relax_ng_schema_file> <input_xml_file> <stamp_file_path>

  validate = sys.argv[2]
  allow_network_access = sys.argv[3]
  relax_ng_schema_file = sys.argv[4]
  input_xml_file = sys.argv[5]
  stamp_file_path = sys.argv[6]

  # schematron_schema = 'http://docbook.org/xml/5.0/sch/docbook.sch'

  # Validation against the Schematron schema does not work on Ubuntu 21.04:
  # file:///usr/share/xml/docbook/schema/schematron/5.0/docbook.sch:6: element rule:
  #   Schemas parser error : Failed to compile context expression db:firstterm[@linkend]
  # .....
  # Schematron schema http://docbook.org/xml/5.0/sch/docbook.sch failed to compile

  cmd = [
    'xmllint',
    '--noout',
    '--noent',
    '--xinclude',
  ]
  if validate == 'true':
    cmd += [
      '--relaxng', relax_ng_schema_file,
      #'--schematron', schematron_schema,
    ]
  if allow_network_access != 'true':
    cmd += ['--nonet']
  cmd += [input_xml_file]
  result = subprocess.run(cmd)
  if result.returncode:
    return result.returncode

  Path(stamp_file_path).touch(exist_ok=True)
  return 0

# Called from custom_target()
def translate_xml():
  #      argv[2]         argv[3]          argv[4]          argv[5]          argv[6:]
  # <input_po_file> <input_xml_dir> <output_xml_dir> <stamp_file_path> <input_xml_files>...

  input_po_file = sys.argv[2]
  input_xml_dir = sys.argv[3]
  output_xml_dir = sys.argv[4] # Absolute path
  stamp_file_path = sys.argv[5]
  input_xml_files = sys.argv[6:]

  # Create the destination directory, if it does not exist.
  os.makedirs(output_xml_dir, exist_ok=True)

  # Create an .mo file.
  language = os.path.splitext(os.path.basename(input_po_file))[0]
  mo_file = os.path.join(output_xml_dir, language + '.mo')
  cmd = [
    'msgfmt',
    '-o', mo_file,
    input_po_file,
  ]
  result = subprocess.run(cmd)
  if result.returncode:
    return result.returncode

  # Create translated XML files.
  cmd = [
    'itstool',
    '-m', mo_file,
    '-o', output_xml_dir,
  ] + input_xml_files
  result = subprocess.run(cmd, cwd=input_xml_dir)
  if result.returncode:
    return result.returncode

  Path(stamp_file_path).touch(exist_ok=True)
  return 0

# Called from custom_target()
def html():
  #        argv[2]              argv[3]          argv[4]            argv[5]          argv[6]
  # <allow_network_access> <stylesheet_file> <input_xml_file> <output_html_dir> <stamp_file_path>

  allow_network_access = sys.argv[2]
  stylesheet_file = sys.argv[3]
  input_xml_file = sys.argv[4]
  output_html_dir = sys.argv[5]
  stamp_file_path = sys.argv[6]

  # Remove old files and create the destination directory.
  shutil.rmtree(output_html_dir, ignore_errors=True)
  os.makedirs(output_html_dir, exist_ok=True)

  cmd = [
    'xsltproc',
    '-o', output_html_dir + '/',
    '--xinclude',
  ]
  if allow_network_access != 'true':
    cmd += ['--nonet']
  cmd += [
    stylesheet_file,
    input_xml_file,
  ]
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
  # xsltproc prints the names of all written files. Don't print those lines.
  for line in result.stdout.splitlines():
    if not line.startswith('Writing '):
      print(line)

  if result.returncode:
    return result.returncode

  # Remove root.html.
  root_html_file = os.path.join(output_html_dir, 'root.html')
  if os.path.isfile(root_html_file):
    os.remove(root_html_file)

  Path(stamp_file_path).touch(exist_ok=True)
  return 0

# Called from meson.add_install_script()
def install():
  #      argv[2]           argv[3]              argv[4:]
  # <input_xml_dir> <rel_install_xml_dir> <input_xml_files>...

  # <rel_install_xml_dir> is the installation directory, relative to {prefix}.
  input_xml_dir = sys.argv[2]
  rel_install_xml_dir = sys.argv[3]
  input_xml_files =  sys.argv[4:]
  destdir_xmldir = os.path.join(os.getenv('MESON_INSTALL_DESTDIR_PREFIX'), rel_install_xml_dir)

  # Create the installation directory, if it does not exist.
  os.makedirs(destdir_xmldir, exist_ok=True)

  for f in input_xml_files:
    # shutil.copy2() copies timestamps and some other file metadata.
    shutil.copy2(os.path.join(input_xml_dir, f), destdir_xmldir)
  return 0

# Called from run_target()
def publish():
  import glob

  # Use environment variable GTKMM_WEB_USER if it exists, else USER
  user_name = os.getenv('GTKMM_WEB_USER', os.getenv('USER'))

  os.chdir(os.path.join(os.getenv('MESON_BUILD_ROOT'), 'docs'))
  html_dirs = glob.glob(os.path.join('html', '*'))
  web_path = '/home/murrayc/gtkmm.org/'

  # rsync -avz --rsh ssh --cvs-exclude  html/.htaccess html/* $$USER@www.gtkmm.org:$(web_path)
  cmd = [
    'rsync',
    '-avz', '--rsh', 'ssh', '--cvs-exclude',
    'html/.htaccess',
  ] + html_dirs + [
    user_name + '@www.gtkmm.org:' + web_path,
  ]
  #return subprocess.run(cmd).returncode
  print('The "publish" target is disabled.')
  print('To enable it, edit', sys.argv[0])
  print(' '.join(cmd))
  return 0

# ----- Main -----
if subcommand == 'copy_files_and_dirs':
  sys.exit(copy_files_and_dirs())
if subcommand == 'get_languages':
  sys.exit(get_languages())
if subcommand == 'xmllint':
  sys.exit(xmllint())
if subcommand == 'translate_xml':
  sys.exit(translate_xml())
if subcommand == 'html':
  sys.exit(html())
if subcommand == 'install':
  sys.exit(install())
if subcommand == 'publish':
  sys.exit(publish())
print(sys.argv[0], ': illegal subcommand,', subcommand)
sys.exit(1)
