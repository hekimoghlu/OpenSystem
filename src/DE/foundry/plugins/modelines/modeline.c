/* modeline.c
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

#include "modeline.h"

static Modeline *
modeline_new (const char *editor)
{
  Modeline *self = g_new0 (Modeline, 1);
  self->editor = g_strdup (editor);
  return self;
}

void
modeline_free (Modeline *self)
{
  g_clear_pointer (&self->editor, g_free);
  g_clear_pointer (&self->settings, g_strfreev);
  g_free (self);
}

static void
modeline_add_setting (Modeline   *self,
                      const char *key,
                      const char *value)
{
  self->settings = g_environ_setenv (self->settings, key, value, TRUE);
}

static void
parse_key_value_list (Modeline   *self,
                      const char *text,
                      const char *sep,
                      gboolean    has_eq)
{
  g_auto(GStrv) tokens = g_strsplit (text, sep, 0);

  for (char **t = tokens; *t; t++)
    {
      g_autofree char *key = NULL;
      char *val = NULL;

      if (has_eq)
        {
          key = g_strstrip (g_strdup (*t));
          val = strchr (key, '=');

          if (*key && val == NULL)
            {
              modeline_add_setting (self, key, "");
              continue;
            }
        }
      else
        {
          key = g_strstrip (g_strdup (*t));
          val = strchr (key, ':');

          if (!val)
            val = strchr (key, ' ');
        }

      if (val)
        {
          *(val++) = '\0';
          val = g_strstrip (val);
          key = g_strstrip (key);

          if (*key != 0)
            modeline_add_setting (self, key, val);
        }
    }
}

Modeline *
modeline_parse (const char *line)
{
  const char *vim_match;
  const char *emacs_start;
  const char *kate_match;

  if (line == NULL || line[0] == 0)
    return NULL;

  /* Vim: vim: set ts=4 sw=4 et: */
  if ((vim_match = strstr (line, "vim:")) || (vim_match = strstr (line, "vi:")))
    {
      const char *start = strstr (vim_match, "set");
      const char *end = strrchr (line, ':');

      if (start)
        start += strlen ("set");

      if (start && end > start)
        {
          g_autofree char *sub = g_strndup (start, end - start);
          Modeline *m = modeline_new ("vim");
          parse_key_value_list (m, sub, " ", TRUE);
          return m;
        }
    }

  /* Emacs and JOE: -*- mode: python; tab-width: 4 -*- */
  if ((emacs_start = strstr (line, "-*-")))
    {
      const char *end = strstr (emacs_start + 3, "-*-");

      if (end)
        {
          g_autofree char *sub = g_strndup (emacs_start + 3, end - (emacs_start + 3));
          Modeline *m = modeline_new (strstr (sub, "nano-") ? "nano" : "emacs");
          parse_key_value_list (m, sub, ";", FALSE);
          return m;
        }
    }

  /* Kate: kate: indent-mode normal; tab-width 4; */
  if ((kate_match = strstr (line, "kate:")))
    {
      Modeline *m = modeline_new ("kate");
      kate_match += 5;
      parse_key_value_list (m, kate_match, ";", FALSE);
      return m;
    }

  /* Geany: geany_encoding=UTF-8 */
  if (strstr (line, "geany_"))
    {
      Modeline *m = modeline_new ("geany");
      parse_key_value_list (m, line, " ", TRUE);
      return m;
    }

  return NULL;
}

static Modeline *
modeline_copy (Modeline *modeline)
{
  if (modeline != NULL)
    {
      Modeline *copy = g_new0 (Modeline, 1);
      copy->editor = g_strdup (modeline->editor);
      copy->settings = g_strdupv (modeline->settings);
      return copy;
    }

  return NULL;
}

G_DEFINE_BOXED_TYPE (Modeline, modeline, modeline_copy, modeline_free)
