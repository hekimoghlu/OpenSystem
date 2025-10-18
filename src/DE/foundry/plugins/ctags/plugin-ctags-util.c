/* plugin-ctags-util.c
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

#include "plugin-ctags-util.h"

static const char *c_languages[] = { ".c", ".h", ".cc", ".hh", ".cpp", ".hpp", ".cxx", ".hxx", NULL };
static const char *vala_languages[] = { ".vala", NULL };
static const char *python_languages[] = { ".py", NULL };
static const char *js_languages[] = { ".js", NULL };
static const char *ruby_languages[] = { ".rb", NULL };
static const char *html_languages[] = { ".html", ".htm", ".tmpl", ".css", ".js", NULL };
static const char *languages[] = { "c", "chdr", "cpp", "cpphdr", "vala", "python", "js", "html", "ruby", NULL };

gboolean
plugin_ctags_is_indexable (const char *name)
{
  const char *dot;

  if (name == NULL)
    return FALSE;

  if (!(dot = strrchr (name, '.')))
    return FALSE;

  /* TODO: Probably better to hash and look that up instead */

  return g_strv_contains (c_languages, dot) ||
         g_strv_contains (vala_languages, dot) ||
         g_strv_contains (python_languages, dot) ||
         g_strv_contains (js_languages, dot) ||
         g_strv_contains (ruby_languages, dot) ||
         g_strv_contains (html_languages, dot);
}

gboolean
plugin_ctags_is_known_language_id (const char *language_id)
{
  if (language_id == NULL)
    return FALSE;

  return g_strv_contains (languages, language_id);
}
