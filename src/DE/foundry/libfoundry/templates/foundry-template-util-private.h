/* templates.h
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

#pragma once

#include <tmpl-glib.h>

#include "foundry-util.h"

G_BEGIN_DECLS

static inline TmplExpr *
expr_new_from_bytes (GBytes  *bytes,
                     GError **error)
{
  gsize size = 0;
  const char *data = g_bytes_get_data (bytes, &size);
  g_autofree char *copy = g_strndup (data, size);

  return tmpl_expr_from_string (copy, error);
}

static inline gboolean
template_parse_bytes (TmplTemplate  *template,
                      GBytes        *bytes,
                      GError       **error)
{
  gsize size = 0;
  const char *data = g_bytes_get_data (bytes, &size);
  g_autofree char *copy = g_strndup (data, size);

  return tmpl_template_parse_string (template, copy, error);
}

static inline char *
capitalize (const gchar *input)
{
  gunichar c;
  GString *str;

  if (input == NULL)
    return NULL;

  if (*input == 0)
    return g_strdup ("");

  c = g_utf8_get_char (input);
  if (g_unichar_isupper (c))
    return g_strdup (input);

  str = g_string_new (NULL);
  input = g_utf8_next_char (input);
  g_string_append_unichar (str, g_unichar_toupper (c));
  if (*input)
    g_string_append (str, input);

  return g_string_free (str, FALSE);
}

static inline char *
camelize (const char *input)
{
  gboolean next_is_upper = TRUE;
  gboolean skip = FALSE;
  GString *str;

  if (input == NULL)
    return NULL;

  if (!strchr (input, '_') && !strchr (input, ' ') && !strchr (input, '-'))
    return capitalize (input);

  str = g_string_new (NULL);

	for (; *input; input = g_utf8_next_char (input))
    {
      gunichar c = g_utf8_get_char (input);

      switch (c)
      {
      case '_':
      case '-':
      case ' ':
        next_is_upper = TRUE;
        skip = TRUE;
        break;

      default:
        break;
      }

      if (skip)
        {
          skip = FALSE;
          continue;
        }

      if (next_is_upper)
        {
          c = g_unichar_toupper (c);
          next_is_upper = FALSE;
        }
      else
        c = g_unichar_tolower (c);

      g_string_append_unichar (str, c);
    }

  if (g_str_has_suffix (str->str, "Private"))
    g_string_truncate (str, str->len - strlen ("Private"));

  return g_string_free (str, FALSE);
}

static inline char *
functify (const gchar *input)
{
  gunichar last = 0;
  GString *str;

  if (input == NULL)
    return NULL;

  str = g_string_new (NULL);

  for (; *input; input = g_utf8_next_char (input))
    {
      gunichar c = g_utf8_get_char (input);
      gunichar n = g_utf8_get_char (g_utf8_next_char (input));

      if (last)
        {
          if ((g_unichar_islower (last) && g_unichar_isupper (c)) ||
              (g_unichar_isupper (c) && g_unichar_islower (n)))
            g_string_append_c (str, '_');
        }

      if ((c == ' ') || (c == '-'))
        c = '_';

      g_string_append_unichar (str, g_unichar_tolower (c));

      last = c;
    }

  if (g_str_has_suffix (str->str, "_private") ||
      g_str_has_suffix (str->str, "_PRIVATE"))
    g_string_truncate (str, str->len - strlen ("_private"));

  return g_string_free (str, FALSE);
}

static inline void
scope_take_string (TmplScope  *scope,
                   const char *name,
                   char       *value)
{
  tmpl_scope_set_string (scope, name, value);
  g_free (value);
}

static inline void
add_to_scope (TmplScope  *scope,
              const char *pattern)
{
  g_autofree char *key = NULL;
  const char *val;

  g_assert (scope != NULL);
  g_assert (pattern != NULL);

  val = strchr (pattern, '=');

  /* If it is just "FOO" then set "FOO" to True */
  if (val == NULL)
    {
      tmpl_scope_set_boolean (scope, pattern, TRUE);
      return;
    }

  key = g_strndup (pattern, val - pattern);
  val++;

  /* If simple key=value, set the bool/string */
  if (strstr (val, "{{") == NULL)
    {
      if (foundry_str_equal0 (val, "false"))
        tmpl_scope_set_boolean (scope, key, FALSE);
      else if (foundry_str_equal0 (val, "true"))
        tmpl_scope_set_boolean (scope, key, TRUE);
      else
        tmpl_scope_set_string (scope, key, val);

      return;
    }

  /* More complex, we have a template to expand from scope */
  {
    g_autoptr(TmplTemplate) template = tmpl_template_new (NULL);
    g_autoptr(GError) error = NULL;
    g_autofree char *expanded = NULL;

    if (!tmpl_template_parse_string (template, val, &error))
      {
        g_warning ("Failed to parse template %s: %s",
                   val, error->message);
        return;
      }

    if (!(expanded = tmpl_template_expand_string (template, scope, &error)))
      {
        g_warning ("Failed to expand template %s: %s",
                   val, error->message);
        return;
      }

    tmpl_scope_set_string (scope, key, expanded);
  }
}

/* Based on gtkbuilderscope.c */
static inline char *
type_name_mangle (const char *name,
                  gboolean     split_first_cap)
{
  GString *symbol_name = g_string_new ("");
  int i;

  for (i = 0; name[i] != '\0'; i++)
    {
      /* skip if uppercase, first or previous is uppercase */
      if ((name[i] == g_ascii_toupper (name[i]) &&
             ((i > 0 && name[i-1] != g_ascii_toupper (name[i-1])) ||
              (i == 1 && name[0] == g_ascii_toupper (name[0]) && split_first_cap))) ||
           (i > 2 && name[i]  == g_ascii_toupper (name[i]) &&
           name[i-1] == g_ascii_toupper (name[i-1]) &&
           name[i-2] == g_ascii_toupper (name[i-2])))
        g_string_append_c (symbol_name, '_');
      g_string_append_c (symbol_name, g_ascii_tolower (name[i]));
    }

  return g_string_free (symbol_name, FALSE);
}

static inline void
add_simple_scope (TmplScope *scope)
{
  g_autoptr(GDateTime) now = g_date_time_new_now_local ();

  scope_take_string (scope, "year", g_date_time_format (now, "%Y"));
  scope_take_string (scope, "YEAR", g_date_time_format (now, "%Y"));
  tmpl_scope_set_string (scope, "author", g_get_real_name () ? g_get_real_name () : g_get_user_name ());
  tmpl_scope_set_string (scope, "gnome_sdk_version", GNOME_SDK_VERSION);
  tmpl_scope_set_string (scope, "freedesktop_sdk_version", FREEDESKTOP_SDK_VERSION);
}

G_END_DECLS
