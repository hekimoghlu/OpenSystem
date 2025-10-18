/* plugin-flatpak.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include <flatpak.h>

#include "foundry-util-private.h"

#include "plugin-flatpak.h"

static DexFuture *g_installations;

static char *plugin_flatpak_dup_private_installation_dir (FoundryContext *context);

static FlatpakQueryFlags
adjust_flags (FoundryContext    *context,
              FlatpakQueryFlags  flags)
{
  if (!foundry_context_network_allowed (context))
    return flags | FLATPAK_QUERY_FLAGS_ONLY_CACHED;

  return flags;
}

static DexFuture *
plugin_flatpak_load_installations_fiber (gpointer user_data)
{
  FlatpakInstallation *installation;
  GPtrArray *ar = g_ptr_array_new_with_free_func (g_object_unref);

  if ((installation = dex_await_object (plugin_flatpak_installation_new_system (), NULL)))
    {
      g_autoptr(GFile) file = flatpak_installation_get_path (installation);

      g_debug ("Found system Flatpak installation at `%s`", g_file_peek_path (file));
      g_ptr_array_add (ar, installation);
    }
  else
    {
      g_debug ("Failed to locate system Flatpak installation");
    }

  if ((installation = dex_await_object (plugin_flatpak_installation_new_user (), NULL)))
    {
      g_autoptr(GFile) file = flatpak_installation_get_path (installation);

      g_debug ("Found user Flatpak installation at `%s`", g_file_peek_path (file));
      g_ptr_array_add (ar, installation);
    }
  else
    {
      g_debug ("Failed to locate user Flatpak installation");
    }

  return dex_future_new_take_boxed (G_TYPE_PTR_ARRAY, g_steal_pointer (&ar));
}

DexFuture *
plugin_flatpak_load_installations (void)
{
  if (g_once_init_enter (&g_installations))
    g_once_init_leave (&g_installations,
                       dex_scheduler_spawn (NULL, 0,
                                            plugin_flatpak_load_installations_fiber,
                                            NULL, NULL));

  return dex_ref (g_installations);
}

static DexFuture *
plugin_flatpak_installation_new_system_fiber (gpointer user_data)
{
  FlatpakInstallation *installation;
  GError *error = NULL;

  if (!(installation = flatpak_installation_new_system (NULL, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));
  else
    return dex_future_new_take_object (g_steal_pointer (&installation));
}

DexFuture *
plugin_flatpak_installation_new_system (void)
{
  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              plugin_flatpak_installation_new_system_fiber,
                              NULL, NULL);
}

static DexFuture *
plugin_flatpak_installation_new_user_fiber (gpointer user_data)
{
  FlatpakInstallation *installation;
  GError *error = NULL;

  /* If we're running inside of Flatpak, what we really want is the
   * one on the host (generally at .local/share/flatpak).
   */
  if (!_foundry_in_container ())
    {
      installation = flatpak_installation_new_user (NULL, &error);
    }
  else
    {
      g_autoptr(GFile) file = g_file_new_build_filename (g_get_home_dir (), ".local", "share", "flatpak", NULL);
      installation = flatpak_installation_new_for_path (file, TRUE, NULL, &error);
    }

  if (installation == NULL)
    return dex_future_new_for_error (g_steal_pointer (&error));
  else
    return dex_future_new_take_object (g_steal_pointer (&installation));
}

DexFuture *
plugin_flatpak_installation_new_user (void)
{
  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              plugin_flatpak_installation_new_user_fiber,
                              NULL, NULL);
}

static DexFuture *
plugin_flatpak_installation_new_for_path_fiber (GFile    *file,
                                                gboolean  is_user)
{
  FlatpakInstallation *installation;
  GError *error = NULL;

  if (!(installation = flatpak_installation_new_for_path (file, is_user, NULL, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));
  else
    return dex_future_new_take_object (g_steal_pointer (&installation));
}

DexFuture *
plugin_flatpak_installation_new_for_path (GFile    *path,
                                          gboolean  user)
{
  dex_return_error_if_fail (G_IS_FILE (path));

  return foundry_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                  G_CALLBACK (plugin_flatpak_installation_new_for_path_fiber),
                                  2,
                                  G_TYPE_FILE, path,
                                  G_TYPE_BOOLEAN, !!user);
}

DexFuture *
plugin_flatpak_installation_new_private (FoundryContext *context)
{
  g_autofree char *path = NULL;
  g_autoptr(GFile) file = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_CONTEXT (context));

  path = plugin_flatpak_dup_private_installation_dir (context);
  file = g_file_new_for_path (path);

  return plugin_flatpak_installation_new_for_path (file, TRUE);
}

typedef struct _ListRefs
{
  FlatpakInstallation *installation;
  FlatpakRemote       *remote;
  FlatpakQueryFlags    flags;
} ListRefs;

static void
list_refs_finalize (gpointer data)
{
  ListRefs *state = data;

  g_clear_object (&state->installation);
  g_clear_object (&state->remote);
}

static void
list_refs_unref (ListRefs *state)
{
  g_atomic_rc_box_release_full (state, list_refs_finalize);
}

static ListRefs *
list_refs_ref (ListRefs *state)
{
  return g_atomic_rc_box_acquire (state);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (ListRefs, list_refs_unref)

static DexFuture *
plugin_flatpak_installation_list_refs_cb (gpointer user_data)
{
  ListRefs *state = user_data;
  g_autofree char *display_name = NULL;
  g_autoptr(GPtrArray) remotes = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GPtrArray) all_refs = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (state != NULL);
  g_assert (FLATPAK_IS_INSTALLATION (state->installation));

  display_name = g_strdup (flatpak_installation_get_display_name (state->installation));

  g_debug ("Listing all refs from `%s` with flags 0x%x",
           display_name, state->flags);

  if (!(remotes = flatpak_installation_list_remotes (state->installation, NULL, &error)))
    {
      g_debug ("Failed to list remotes on installation `%s`: %s",
               display_name, error->message);
      return dex_future_new_for_error (g_steal_pointer (&error));
    }

  all_refs = g_ptr_array_new_with_free_func (g_object_unref);

  for (guint i = 0; i < remotes->len; i++)
    {
      g_autoptr(GPtrArray) refs = NULL;
      FlatpakRemote *remote = g_ptr_array_index (remotes, i);
      const char *name = flatpak_remote_get_name (remote);

      refs = flatpak_installation_list_remote_refs_sync_full (state->installation, name, state->flags, NULL, &error);

      if (refs == NULL)
        {
          g_debug ("Failed to list remote refs in installation `%s` from remote `%s`: %s",
                   display_name, name, error->message);
          g_clear_error (&error);
          continue;
        }

      g_debug ("`%s` remote `%s` contains %u refs",
               display_name, name, refs->len);

      for (guint j = 0; j < refs->len; j++)
        {
          FlatpakRef *ref = g_ptr_array_index (refs, j);

          g_ptr_array_add (all_refs, g_object_ref (ref));
        }
    }

  return dex_future_new_take_boxed (G_TYPE_PTR_ARRAY, g_steal_pointer (&all_refs));
}

DexFuture *
plugin_flatpak_installation_list_refs (FoundryContext      *context,
                                       FlatpakInstallation *installation,
                                       FlatpakQueryFlags    flags)
{
  g_autoptr(ListRefs) state = NULL;

  dex_return_error_if_fail (FLATPAK_IS_INSTALLATION (installation));

  state = g_atomic_rc_box_new0 (ListRefs);
  state->installation = g_object_ref (installation);
  state->flags = adjust_flags (context, flags);

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              plugin_flatpak_installation_list_refs_cb,
                              list_refs_ref (state),
                              (GDestroyNotify) list_refs_unref);
}

static DexFuture *
plugin_flatpak_installation_list_refs_for_remote_cb (gpointer user_data)
{
  ListRefs *state = user_data;
  g_autoptr(GPtrArray) refs = NULL;
  g_autoptr(GError) error = NULL;
  const char *name;

  g_assert (state != NULL);
  g_assert (FLATPAK_IS_INSTALLATION (state->installation));
  g_assert (FLATPAK_IS_REMOTE (state->remote));

  name = flatpak_remote_get_name (state->remote);
  refs = flatpak_installation_list_remote_refs_sync_full (state->installation, name, state->flags, NULL, &error);

  if (error != NULL)
    return dex_future_new_for_error (g_steal_pointer (&error));
  else
    return dex_future_new_take_boxed (G_TYPE_PTR_ARRAY, g_steal_pointer (&refs));
}

DexFuture *
plugin_flatpak_installation_list_refs_for_remote (FoundryContext      *context,
                                                  FlatpakInstallation *installation,
                                                  FlatpakRemote       *remote,
                                                  FlatpakQueryFlags    flags)
{
  g_autoptr(ListRefs) state = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_CONTEXT (context));
  dex_return_error_if_fail (FLATPAK_IS_INSTALLATION (installation));
  dex_return_error_if_fail (FLATPAK_IS_REMOTE (remote));

  state = g_atomic_rc_box_new0 (ListRefs);
  state->installation = g_object_ref (installation);
  state->remote = g_object_ref (remote);
  state->flags = adjust_flags (context, flags);

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              plugin_flatpak_installation_list_refs_for_remote_cb,
                              list_refs_ref (state),
                              (GDestroyNotify) list_refs_unref);
}

static DexFuture *
plugin_flatpak_installation_list_installed_refs_fiber (gpointer user_data)
{
  ListRefs *state = user_data;
  g_autoptr(GPtrArray) refs = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (state != NULL);
  g_assert (FLATPAK_IS_INSTALLATION (state->installation));

  if (!(refs = flatpak_installation_list_installed_refs (state->installation, NULL, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_boxed (G_TYPE_PTR_ARRAY, g_steal_pointer (&refs));
}

DexFuture *
plugin_flatpak_installation_list_installed_refs (FoundryContext      *context,
                                                 FlatpakInstallation *installation,
                                                 FlatpakQueryFlags    flags)
{
  g_autoptr(ListRefs) state = NULL;

  dex_return_error_if_fail (FLATPAK_IS_INSTALLATION (installation));

  state = g_atomic_rc_box_new0 (ListRefs);
  state->installation = g_object_ref (installation);
  state->flags = adjust_flags (context, flags);

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              plugin_flatpak_installation_list_installed_refs_fiber,
                              list_refs_ref (state),
                              (GDestroyNotify) list_refs_unref);
}

gboolean
plugin_flatpak_ref_can_be_sdk (FlatpakRef *ref)
{
  const char *name = flatpak_ref_get_name (ref);

  if (g_str_has_suffix (name, ".Debug"))
    return FALSE;

  if (g_str_has_suffix (name, ".Sources"))
    return FALSE;

  if (g_str_has_suffix (name, ".Locale"))
    return FALSE;

  if (g_str_has_suffix (name, ".Docs"))
    return FALSE;

  if (g_str_has_suffix (name, ".Var"))
    return FALSE;

  if (g_str_has_prefix (name, "org.gtk.Gtk3theme."))
    return FALSE;

  if (strstr (name, ".KStyle.") != NULL ||
      strstr (name, ".WaylandDecoration.") != NULL ||
      strstr (name, ".PlatformTheme.") != NULL ||
      strstr (name, ".Icontheme") != NULL)
    return FALSE;

  if (g_str_has_suffix (name, ".openh264") ||
      g_str_has_suffix (name, ".ffmpeg-full") ||
      g_str_has_suffix (name, ".GL.default"))
    return FALSE;

  if (strstr (name, ".GL.nvidia") != NULL ||
      strstr (name, ".GL32.nvidia") != NULL)
    return FALSE;

  if (flatpak_ref_get_kind (ref) == FLATPAK_REF_KIND_RUNTIME)
    return TRUE;

  if (flatpak_ref_get_kind (ref) == FLATPAK_REF_KIND_APP)
    {
      if (g_str_has_suffix (name, ".BaseApp"))
        return TRUE;
    }

  return FALSE;
}

static DexFuture *
plugin_flatpak_find_ref_cb (DexFuture *completed,
                            gpointer   user_data)
{
  const char *id = user_data;
  g_autoptr(GPtrArray) refs = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (id != NULL);

  if (!(refs = dex_await_boxed (dex_ref (completed), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  for (guint i = 0; i < refs->len; i++)
    {
      FlatpakRef *ref = g_ptr_array_index (refs, i);
      g_autofree char *ref_id = g_strdup_printf ("%s/%s/%s",
                                                 flatpak_ref_get_name (ref),
                                                 flatpak_ref_get_arch (ref),
                                                 flatpak_ref_get_branch (ref));

      if (foundry_str_equal0 (ref_id, id))
        return dex_future_new_take_object (g_object_ref (ref));
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Cannot find runtime %s", id);
}

DexFuture *
plugin_flatpak_find_ref (FoundryContext      *context,
                         FlatpakInstallation *installation,
                         const char          *runtime,
                         const char          *arch,
                         const char          *runtime_version)
{
  dex_return_error_if_fail (FOUNDRY_IS_CONTEXT (context));
  dex_return_error_if_fail (FLATPAK_IS_INSTALLATION (installation));
  dex_return_error_if_fail (runtime != NULL);
  dex_return_error_if_fail (runtime_version != NULL);

  if (arch == NULL)
    arch = flatpak_get_default_arch ();

  return dex_future_then (plugin_flatpak_installation_list_refs (context, installation, 0),
                          plugin_flatpak_find_ref_cb,
                          g_strdup_printf ("%s/%s/%s",
                                           runtime,
                                           arch,
                                           runtime_version),
                          g_free);
}

/* Must be called by fiber */
FlatpakRemote *
plugin_flatpak_find_remote (FoundryContext      *context,
                            FlatpakInstallation *installation,
                            FlatpakRef          *ref)
{

  g_autoptr(GPtrArray) remotes = NULL;
  g_autoptr(GError) error = NULL;

  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);
  g_return_val_if_fail (FLATPAK_IS_INSTALLATION (installation), NULL);
  g_return_val_if_fail (FLATPAK_IS_REF (ref), NULL);

  if (!(remotes = flatpak_installation_list_remotes (installation, NULL, NULL)))
    return NULL;

  for (guint i = 0; i < remotes->len; i++)
    {
      FlatpakRemote *remote = g_ptr_array_index (remotes, i);
      g_autoptr(GPtrArray) refs = NULL;

      if ((refs = dex_await_boxed (plugin_flatpak_installation_list_refs_for_remote (context, installation, remote, 0), NULL)))
        {
          for (guint j = 0; j < refs->len; j++)
            {
              FlatpakRef *item = g_ptr_array_index (refs, j);

              if (foundry_str_equal0 (flatpak_ref_get_name (ref),
                                      flatpak_ref_get_name (item)) &&
                  foundry_str_equal0 (flatpak_ref_get_arch (ref),
                                      flatpak_ref_get_arch (item)) &&
                  foundry_str_equal0 (flatpak_ref_get_branch (ref),
                                      flatpak_ref_get_branch (item)))
                return g_object_ref (remote);
            }
        }
    }

  return NULL;
}

gboolean
plugin_flatpak_ref_matches (FlatpakRef *ref,
                            const char *name,
                            const char *arch,
                            const char *branch)
{
  if (ref == NULL || name == NULL || arch == NULL || branch == NULL)
    return FALSE;

  return g_strcmp0 (name, flatpak_ref_get_name (ref)) == 0 &&
         g_strcmp0 (arch, flatpak_ref_get_arch (ref)) == 0 &&
         g_strcmp0 (branch, flatpak_ref_get_branch (ref)) == 0;
}

static char *
plugin_flatpak_dup_private_installation_dir (FoundryContext *context)
{
  g_autoptr(FoundrySettings) settings = NULL;
  g_autofree char *path = NULL;

  g_assert (FOUNDRY_IS_CONTEXT (context));

  settings = foundry_settings_new (context, "app.devsuite.foundry.flatpak");
  path = foundry_settings_get_string (settings, "private-installation");

  if (foundry_str_empty0 (path))
    {
      g_autofree char *projects_dir = foundry_dup_projects_directory ();
      g_autofree char *installation_dir = g_build_filename (projects_dir, ".foundry-flatpak", NULL);

      g_set_str (&path, installation_dir);
    }

  foundry_path_expand_inplace (&path);

  return g_steal_pointer (&path);
}

void
plugin_flatpak_apply_config_dir (FoundryContext         *context,
                                 FoundryProcessLauncher *launcher)
{
  g_autofree char *install_dir = NULL;
  g_autofree char *etc_dir = NULL;

  g_return_if_fail (FOUNDRY_IS_CONTEXT (context));
  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));

  if (_foundry_in_container ())
    {
      g_autofree char *user_dir = NULL;

      user_dir = g_build_filename (g_get_home_dir (), ".local", "share", "flatpak", NULL);
      foundry_process_launcher_setenv (launcher, "FLATPAK_USER_DIR", user_dir);
      foundry_process_launcher_setenv (launcher, "XDG_RUNTIME_DIR", g_get_user_runtime_dir ());
    }

  install_dir = plugin_flatpak_dup_private_installation_dir (context);
  etc_dir = g_build_filename (install_dir, "etc", NULL);

  foundry_process_launcher_setenv (launcher, "FLATPAK_CONFIG_DIR", etc_dir);
}
