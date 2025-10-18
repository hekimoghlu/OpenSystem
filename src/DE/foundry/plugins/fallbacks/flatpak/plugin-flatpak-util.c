/* plugin-flatpak-util.c
 *
 * Copyright 2016-2024 Christian Hergert <chergert@redhat.com>
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

#include <string.h>

#include "plugin-flatpak-util.h"

static gboolean
_plugin_flatpak_split_id (const char  *str,
                          char       **id,
                          char       **arch,
                          char       **branch)
{
  g_auto(GStrv) parts = g_strsplit (str, "/", 0);
  guint i = 0;

  if (id)
    *id = NULL;

  if (arch)
    *arch = NULL;

  if (branch)
    *branch = NULL;

  if (parts[i] != NULL)
    {
      if (id != NULL)
        *id = g_strdup (parts[i]);
    }
  else
    {
      /* we require at least a runtime/app ID */
      return FALSE;
    }

  i++;

  if (parts[i] != NULL)
    {
      if (arch != NULL)
        *arch = g_strdup (parts[i]);
    }
  else
    return TRUE;

  i++;

  if (parts[i] != NULL)
    {
      if (branch != NULL && !foundry_str_empty0 (parts[i]))
        *branch = g_strdup (parts[i]);
    }

  return TRUE;
}

gboolean
plugin_flatpak_split_id (const char  *str,
                         char       **id,
                         char       **arch,
                         char       **branch)
{
  if (g_str_has_prefix (str, "runtime/"))
    str += strlen ("runtime/");
  else if (g_str_has_prefix (str, "app/"))
    str += strlen ("app/");

  return _plugin_flatpak_split_id (str, id, arch, branch);
}

char *
plugin_flatpak_get_repo_dir (FoundryContext *context)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  return foundry_context_cache_filename (context, "flatpak", "repo", NULL);
}

char *
plugin_flatpak_get_staging_dir (FoundryBuildPipeline *pipeline)
{
  g_autofree char *branch = NULL;
  g_autofree char *name = NULL;
  g_autofree char *arch = NULL;
  g_autoptr(FoundryContext) context = NULL;

  g_return_val_if_fail (FOUNDRY_IS_BUILD_PIPELINE (pipeline), NULL);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (pipeline));

#ifdef FOUNDRY_FEATURE_VCS
  {
    g_autoptr(FoundryVcsManager) vcs_manager = foundry_context_dup_vcs_manager (context);
    g_autoptr(FoundryVcs) vcs = foundry_vcs_manager_dup_vcs (vcs_manager);

    branch = foundry_vcs_dup_branch_name (vcs);
  }
#else
  branch = g_strdup ("unversioned");
#endif

  arch = foundry_build_pipeline_dup_arch (pipeline);
  name = g_strdup_printf ("%s-%s", arch, branch);

  g_strdelimit (name, G_DIR_SEPARATOR_S, '-');

  return foundry_context_cache_filename (context, "flatpak", "staging", name, NULL);
}

static DexFuture *
parse_a11y_result (DexFuture *completed,
                   gpointer   user_data)
{
  g_autofree char *stdout_buf = NULL;
  g_autoptr(GVariant) variant = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *a11y_bus = NULL;

  stdout_buf = dex_await_string (dex_ref (completed), NULL);

  if (!(variant = g_variant_parse (G_VARIANT_TYPE ("(s)"), stdout_buf, NULL, NULL, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  g_variant_take_ref (variant);
  g_variant_get (variant, "(s)", &a11y_bus, NULL);

  return dex_future_new_take_string (g_steal_pointer (&a11y_bus));
}

DexFuture *
plugin_flatpak_get_a11y_bus (void)
{
  static DexFuture *future;

  if (future == NULL)
    {
      g_autoptr(FoundryProcessLauncher) launcher = NULL;
      g_autoptr(GSubprocess) subprocess = NULL;
      g_autoptr(GError) error = NULL;

      launcher = foundry_process_launcher_new ();
      foundry_process_launcher_push_host (launcher);
      foundry_process_launcher_append_args (launcher,
                                            FOUNDRY_STRV_INIT ("gdbus",
                                                               "call",
                                                               "--session",
                                                               "--dest=org.a11y.Bus",
                                                               "--object-path=/org/a11y/bus",
                                                               "--method=org.a11y.Bus.GetAddress"));

      if (!(subprocess = foundry_process_launcher_spawn_with_flags (launcher, G_SUBPROCESS_FLAGS_STDOUT_PIPE, &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      future = dex_future_then (foundry_subprocess_communicate_utf8 (subprocess, NULL),
                                parse_a11y_result, NULL, NULL);
    }

  return dex_ref (future);
}

gboolean
plugin_flatpak_parse_a11y_bus (const char  *address,
                               char       **unix_path,
                               char       **address_suffix)
{
  const char *skip;
  const char *a11y_bus_suffix;
  char *a11y_bus_path;

  g_return_val_if_fail (address != NULL, FALSE);
  g_return_val_if_fail (unix_path != NULL, FALSE);
  g_return_val_if_fail (address_suffix != NULL, FALSE);

  *unix_path = NULL;
  *address_suffix = NULL;

  if (!g_str_has_prefix (address, "unix:path="))
    return FALSE;

  skip = address + strlen ("unix:path=");

  if ((a11y_bus_suffix = strchr (skip, ',')))
    a11y_bus_path = g_strndup (skip, a11y_bus_suffix - skip);
  else
    a11y_bus_path = g_strdup (skip);

  *unix_path = g_steal_pointer (&a11y_bus_path);
  *address_suffix = g_strdup (a11y_bus_suffix);

  return TRUE;
}

char *
plugin_flatpak_uri_to_filename (const char *uri)
{
  GString *s;
  const char *p;

  s = g_string_new ("");

  for (p = uri; *p != 0; p++)
    {
      if (*p == '/' || *p == ':')
        {
          while (p[1] == '/' || p[1] == ':')
            p++;
          g_string_append_c (s, '_');
        }
      else
        {
          g_string_append_c (s, *p);
        }
  }

  return g_string_free (s, FALSE);
}

char *
plugin_flatpak_dup_state_dir (FoundryContext *context)
{
  g_autoptr(FoundrySettings) settings = foundry_context_load_settings (context, "app.devsuite.foundry.flatpak", NULL);
  g_autofree char *state_dir = foundry_settings_get_string (settings, "state-dir");

  if (foundry_str_empty0 (state_dir))
    return foundry_context_cache_filename (context, "flatpak-builder", NULL);

  foundry_path_expand_inplace (&state_dir);

  return g_steal_pointer (&state_dir);
}
