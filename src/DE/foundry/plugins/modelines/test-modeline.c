/* test-modeline.c
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

#include "line-reader-private.h"
#include "modeline.c"

int
main (int argc,
      char *argv[])
{
  for (int i = 1; i < argc; i++)
    {
      g_autofree char *contents = NULL;
      g_autoptr(GError) error = NULL;
      g_autoptr(Modeline) m = NULL;
      LineReader reader;
      const char *first = NULL;
      const char *last = NULL;
      gsize len;
      gsize line_len;
      char *line;

      if (!g_file_get_contents (argv[i], &contents, &len, &error))
        g_error ("%s", error->message);

      if (!g_utf8_validate (contents, len, NULL))
        g_error ("Only UTF-8 is supported from test program");

      line_reader_init (&reader, contents, len);

      while ((line = line_reader_next (&reader, &line_len)))
        {
          line[line_len] = 0;

          if (first == NULL)
            first = line;

          if (line_len > 0)
            last = line;
        }

      if ((m = modeline_parse (first)) || (m = modeline_parse (last)))
        {
          g_print ("%s\n", m->editor);

          if (m->settings != NULL)
            {
              for (guint j = 0; m->settings[j]; j++)
                g_print ("  %s\n", m->settings[j]);
            }
        }
    }

  return 0;
}
