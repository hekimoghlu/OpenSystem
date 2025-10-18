/* foundry-tweak-info.c
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

#include "foundry-tweak-info-private.h"

static char *
expand_string (const char         *input,
               const char * const *environment)
{
  const char *iter;
  GString *str;

  if (input == NULL)
    return NULL;

  if (environment == NULL)
    return g_strdup (input);

  iter = strchr (input, '@');

  if (iter == NULL)
    return g_strdup (input);

  str = g_string_new (input);

  for (guint i = 0; environment[i]; i++)
    {
      const char *eq = strchr (environment[i], '=');
      g_autofree char *key = g_strndup (environment[i], eq - environment[i]);
      g_autofree char *template = g_strdup_printf ("@%s@", key);

      if (eq == NULL)
        continue;

      g_string_replace (str, template, eq+1, 0);
    }

  /* Fixup GSettings path if need be */
  while (strstr (str->str, "//"))
    g_string_replace (str, "//", "/", 0);

  return g_string_free (str, FALSE);
}

FoundryTweakInfo *
foundry_tweak_info_expand (const FoundryTweakInfo *info,
                           const char * const     *environment)
{
  FoundryTweakInfo *copy;

  g_return_val_if_fail (info != NULL, NULL);

  copy = g_atomic_rc_box_new0 (FoundryTweakInfo);
  copy->type = info->type;
  copy->flags = info->flags;
  copy->subpath = expand_string (info->subpath, environment);
  copy->title = expand_string (info->title, environment);
  copy->subtitle = expand_string (info->subtitle, environment);
  copy->icon_name = expand_string (info->icon_name, environment);
  copy->display_hint = g_strdup (info->display_hint);
  copy->sort_key = expand_string (info->sort_key, environment);
  copy->section = expand_string (info->section, environment);

  if (info->source)
    {
      copy->source = g_memdup2 (info->source, sizeof *info->source);

      switch (copy->source->type)
        {
        case FOUNDRY_TWEAK_SOURCE_TYPE_SETTING:
          copy->source->setting.schema_id = expand_string (copy->source->setting.schema_id, environment);
          copy->source->setting.path = expand_string (copy->source->setting.path, environment);
          copy->source->setting.key = expand_string (copy->source->setting.key, environment);
          break;

        case FOUNDRY_TWEAK_SOURCE_TYPE_CALLBACK:
          break;

        default:
          break;
        }
    }

  return copy;
}

static void
foundry_tweak_info_finalize (gpointer data)
{
  FoundryTweakInfo *info = data;

  g_free ((gpointer)info->subpath);
  g_free ((gpointer)info->title);
  g_free ((gpointer)info->subtitle);
  g_free ((gpointer)info->icon_name);
  g_free ((gpointer)info->display_hint);
  g_free ((gpointer)info->sort_key);
  g_free ((gpointer)info->section);

  if (info->source != NULL &&
      info->source->type == FOUNDRY_TWEAK_SOURCE_TYPE_SETTING)
    {
      g_free ((gpointer)info->source->setting.schema_id);
      g_free ((gpointer)info->source->setting.path);
      g_free ((gpointer)info->source->setting.key);
    }

  g_free (info->source);
}

void
foundry_tweak_info_unref (FoundryTweakInfo *info)
{
  g_atomic_rc_box_release_full (info, foundry_tweak_info_finalize);
}

FoundryTweakInfo *
foundry_tweak_info_ref (FoundryTweakInfo *self)
{
  return g_atomic_rc_box_acquire (self);
}
