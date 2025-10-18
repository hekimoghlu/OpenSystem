/* foundry-os-info.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "line-reader-private.h"
#include "foundry-os-info.h"

static char *
get_os_info_from_data (const char *data,
                       gsize       data_len,
                       const char *key_name)
{
  gsize key_len = strlen (key_name);
  LineReader reader;
  const char *line;
  gsize line_len;

  line_reader_init (&reader, (char *)data, data_len);

  while ((line = line_reader_next (&reader, &line_len)))
    {
      if (g_str_has_prefix (line, key_name) && line[key_len] == '=')
        {
          const char *begin = line + key_len + 1;
          const char *end = line + line_len;
          g_autofree char *quoted = g_strndup (begin, end - begin);

          return g_shell_unquote (quoted, NULL);
        }
    }

  return NULL;
}

char *
foundry_get_os_info (const char *key_name)
{
  static gsize initialized;
  static char *os_release_data;
  static gsize os_release_len;

  if (g_once_init_enter (&initialized))
    {
      if (g_file_test ("/.flatpak-info", G_FILE_TEST_EXISTS))
        g_file_get_contents ("/var/run/host/os-release", &os_release_data, &os_release_len, NULL);
      g_once_init_leave (&initialized, TRUE);
    }

  if (os_release_data)
    return get_os_info_from_data (os_release_data, os_release_len, key_name);

  return g_get_os_info (key_name);
}
