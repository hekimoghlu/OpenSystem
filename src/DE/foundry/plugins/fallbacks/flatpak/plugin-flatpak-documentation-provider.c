/* plugin-flatpak-documentation-provider.c
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

#include "plugin-flatpak.h"
#include "plugin-flatpak-documentation-bundle.h"
#include "plugin-flatpak-documentation-provider.h"

struct _PluginFlatpakDocumentationProvider
{
  FoundryDocumentationProvider  parent_instance;
  GListStore                   *roots;
  GHashTable                   *monitors;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakDocumentationProvider, plugin_flatpak_documentation_provider, FOUNDRY_TYPE_DOCUMENTATION_PROVIDER)

static char *
rewrite_path (const char *path)
{
  const char *beginptr;

  g_assert (path != NULL);

  if (!g_str_has_suffix (path, "/active") && (beginptr = strrchr (path, '/')))
    {
      GString *str = g_string_new (path);

      g_string_truncate (str, beginptr - path);
      g_string_append (str, "/active");

      return g_string_free (str, FALSE);
    }

  return g_strdup (path);
}

static DexFuture *
plugin_flatpak_documentation_provider_update_installation (PluginFlatpakDocumentationProvider *self,
                                                           FlatpakInstallation                *installation)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) roots = NULL;
  g_autoptr(GPtrArray) refs = NULL;
  g_autoptr(GError) error = NULL;
  const char *display_name;

  g_assert (PLUGIN_IS_FLATPAK_DOCUMENTATION_PROVIDER (self));
  g_assert (FLATPAK_IS_INSTALLATION (installation));

  display_name = flatpak_installation_get_display_name (installation);
  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  g_debug ("Updating documentation for installation `%s`", display_name);

  if (!(refs = dex_await_boxed (plugin_flatpak_installation_list_installed_refs  (context, installation, FLATPAK_QUERY_FLAGS_NONE), &error)))
    {
      g_debug ("Failed to list installed refs for `%s`: %s",
               display_name, error->message);
      return dex_future_new_for_error (g_steal_pointer (&error));
    }

  roots = g_ptr_array_new_with_free_func (g_object_unref);

  for (guint i = 0; i < refs->len; i++)
    {
      FlatpakInstalledRef *ref = g_ptr_array_index (refs, i);
      FlatpakRefKind kind = flatpak_ref_get_kind (FLATPAK_REF (ref));
      g_autoptr(FoundryDocumentationRoot) root = NULL;
      g_autoptr(GListStore) directories = g_list_store_new (G_TYPE_FILE);
      g_autoptr(GFile) doc_dir = NULL;
      g_autoptr(GFile) gtk_doc_dir = NULL;
      g_autofree char *identifier = NULL;
      g_autoptr(GIcon) icon = NULL;
      g_autofree char *title = NULL;
      g_autofree char *deploy_dir = NULL;
      const char *name;
      const char *branch;

      name = flatpak_ref_get_name (FLATPAK_REF (ref));
      branch = flatpak_ref_get_branch (FLATPAK_REF (ref));

      if (kind != FLATPAK_REF_KIND_RUNTIME || !g_str_has_suffix (name, ".Docs"))
        continue;

      deploy_dir = rewrite_path (flatpak_installed_ref_get_deploy_dir (ref));
      doc_dir = g_file_new_build_filename (deploy_dir, "files", "doc", NULL);
      gtk_doc_dir = g_file_new_build_filename (deploy_dir, "files", "gtk-doc", "html", NULL);

      g_list_store_append (directories, doc_dir);
      g_list_store_append (directories, gtk_doc_dir);

      if (g_str_equal (name, "org.gnome.Sdk.Docs"))
        name = "GNOME";
      else if (g_str_equal (name, "org.freedesktop.Sdk.Docs"))
        name = "FreeDesktop";

      if (g_str_equal (branch, "master"))
        branch = "Nightly";

      identifier = g_strdup_printf ("flatpak:%s/%s/%s",
                                    flatpak_ref_get_name (FLATPAK_REF (ref)),
                                    flatpak_ref_get_arch (FLATPAK_REF (ref)),
                                    flatpak_ref_get_branch (FLATPAK_REF (ref)));

      title = g_strdup_printf ("%s %s", name, branch);
      root = foundry_documentation_root_new (identifier,
                                             title,
                                             flatpak_ref_get_branch (FLATPAK_REF (ref)),
                                             icon,
                                             G_LIST_MODEL (directories));

      g_debug ("`%s` contained documentation from runtime `%s/%s/%s`",
               display_name,
               flatpak_ref_get_name (FLATPAK_REF (ref)),
               flatpak_ref_get_arch (FLATPAK_REF (ref)),
               flatpak_ref_get_branch (FLATPAK_REF (ref)));

      g_ptr_array_add (roots, g_steal_pointer (&root));
    }

  g_debug ("Installation `%s` contained %u roots",
           display_name, roots->len);

  if (self->roots != NULL)
    g_list_store_splice (self->roots,
                         0,
                         g_list_model_get_n_items (G_LIST_MODEL (self->roots)),
                         roots->pdata,
                         roots->len);


  return dex_future_new_true ();
}

static void
plugin_flatpak_documentation_provider_update (PluginFlatpakDocumentationProvider *self,
                                              GFile                              *file,
                                              GFile                              *other_file,
                                              GFileMonitorEvent                   event,
                                              GFileMonitor                       *source)
{
  GHashTableIter iter;
  gpointer key, value;

  g_assert (PLUGIN_IS_FLATPAK_DOCUMENTATION_PROVIDER (self));
  g_assert (G_IS_FILE (file));
  g_assert (!other_file || G_IS_FILE (other_file));
  g_assert (G_IS_FILE_MONITOR (source));

  if (self->monitors == NULL || self->roots == NULL)
    return;

  g_hash_table_iter_init (&iter, self->monitors);
  while (g_hash_table_iter_next (&iter, &key, &value))
    {
      FlatpakInstallation *installation = key;
      GFileMonitor *monitor = value;

      if (source == monitor)
        {
          dex_future_disown (foundry_scheduler_spawn (NULL, 0,
                                                      G_CALLBACK (plugin_flatpak_documentation_provider_update_installation),
                                                      2,
                                                      PLUGIN_TYPE_FLATPAK_DOCUMENTATION_PROVIDER, self,
                                                      FLATPAK_TYPE_INSTALLATION, installation));
          break;
        }
    }
}

static DexFuture *
plugin_flatpak_documentation_provider_load_fiber (gpointer user_data)
{
  PluginFlatpakDocumentationProvider *self = user_data;
  g_autoptr(GPtrArray) installations = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_FLATPAK_DOCUMENTATION_PROVIDER (self));

  if (!(installations = dex_await_boxed (plugin_flatpak_load_installations (), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < installations->len; i++)
    {
      FlatpakInstallation *installation = g_ptr_array_index (installations, i);
      g_autoptr(GFileMonitor) monitor = flatpak_installation_create_monitor (installation, NULL, NULL);

      if (monitor == NULL)
        continue;

      g_signal_connect_object (monitor,
                               "changed",
                               G_CALLBACK (plugin_flatpak_documentation_provider_update),
                               self,
                               G_CONNECT_SWAPPED);

      g_hash_table_insert (self->monitors,
                           g_object_ref (installation),
                           g_object_ref (monitor));

      g_ptr_array_add (futures,
                       foundry_scheduler_spawn (NULL, 0,
                                                G_CALLBACK (plugin_flatpak_documentation_provider_update_installation),
                                                2,
                                                PLUGIN_TYPE_FLATPAK_DOCUMENTATION_PROVIDER, self,
                                                FLATPAK_TYPE_INSTALLATION, installation));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  return dex_future_new_true ();
}

static DexFuture *
plugin_flatpak_documentation_provider_load (FoundryDocumentationProvider *provider)
{
  PluginFlatpakDocumentationProvider *self = (PluginFlatpakDocumentationProvider *)provider;

  g_assert (PLUGIN_IS_FLATPAK_DOCUMENTATION_PROVIDER (self));

  self->monitors = g_hash_table_new_full (NULL, NULL, g_object_unref, g_object_unref);
  self->roots = g_list_store_new (FOUNDRY_TYPE_DOCUMENTATION_ROOT);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_documentation_provider_load_fiber,
                              g_object_ref (provider),
                              g_object_unref);
}

static DexFuture *
plugin_flatpak_documentation_provider_unload (FoundryDocumentationProvider *provider)
{
  PluginFlatpakDocumentationProvider *self = (PluginFlatpakDocumentationProvider *)provider;

  g_assert (PLUGIN_IS_FLATPAK_DOCUMENTATION_PROVIDER (self));

  g_clear_pointer (&self->monitors, g_hash_table_unref);
  g_clear_object (&self->roots);

  return dex_future_new_true ();
}

static GListModel *
plugin_flatpak_documentation_provider_list_roots (FoundryDocumentationProvider *provider)
{
  return g_object_ref (G_LIST_MODEL (PLUGIN_FLATPAK_DOCUMENTATION_PROVIDER (provider)->roots));
}

static gboolean
match_ref (gconstpointer a,
           gconstpointer b)
{
  FlatpakRef *ref_a = FLATPAK_REF (a);
  FlatpakRef *ref_b = FLATPAK_REF (b);

  return g_strcmp0 (flatpak_ref_get_name (ref_a),
                    flatpak_ref_get_name (ref_b)) == 0 &&
         g_strcmp0 (flatpak_ref_get_arch (ref_a),
                    flatpak_ref_get_arch (ref_b)) == 0 &&
         g_strcmp0 (flatpak_ref_get_branch (ref_a),
                    flatpak_ref_get_branch (ref_b)) == 0;
}

static DexFuture *
plugin_flatpak_documentation_provider_list_bundles_fiber (gpointer data)
{
  PluginFlatpakDocumentationProvider *self = data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GPtrArray) installations = NULL;
  g_autoptr(GPtrArray) futures = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (PLUGIN_IS_FLATPAK_DOCUMENTATION_PROVIDER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  store = g_list_store_new (FOUNDRY_TYPE_DOCUMENTATION_BUNDLE);

  if (!(installations = dex_await_boxed (plugin_flatpak_load_installations (), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  futures = g_ptr_array_new_with_free_func (dex_unref);

  for (guint i = 0; i < installations->len; i++)
    {
      FlatpakInstallation *installation = g_ptr_array_index (installations, i);

      g_ptr_array_insert (futures, i, plugin_flatpak_installation_list_refs (context, installation, FLATPAK_QUERY_FLAGS_NONE));
      g_ptr_array_insert (futures, i*2+1, plugin_flatpak_installation_list_installed_refs (context, installation, FLATPAK_QUERY_FLAGS_NONE));
    }

  if (futures->len)
    dex_await (dex_future_allv ((DexFuture **)futures->pdata, futures->len), NULL);

  g_assert (futures->len == (installations->len * 2));

  for (guint i = 0; i < installations->len; i++)
    {
      FlatpakInstallation *installation = g_ptr_array_index (installations, i);
      g_autoptr(GPtrArray) refs = dex_await_boxed (dex_ref (g_ptr_array_index (futures, i)), NULL);
      g_autoptr(GPtrArray) installed = dex_await_boxed (dex_ref (g_ptr_array_index (futures, installations->len + i)), NULL);

      if (refs == NULL)
        continue;

      for (guint j = 0; j < refs->len; j++)
        {
          FlatpakRef *ref = g_ptr_array_index (refs, j);
          FlatpakRefKind kind = flatpak_ref_get_kind (ref);
          const char *name = flatpak_ref_get_name (FLATPAK_REF (ref));
          g_autoptr(FoundryDocumentationBundle) bundle = NULL;
          gboolean is_installed = FALSE;
          guint pos;

          if (kind != FLATPAK_REF_KIND_RUNTIME ||
              !g_str_has_suffix (name, ".Docs") ||
              !g_str_equal (flatpak_get_default_arch (), flatpak_ref_get_arch (ref)))
            continue;

          if (installed)
            is_installed = g_ptr_array_find_with_equal_func (installed, ref, match_ref, &pos);

          bundle = plugin_flatpak_documentation_bundle_new (context, installation, ref, is_installed);

          g_list_store_append (store, bundle);
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

static DexFuture *
plugin_flatpak_documentation_provider_list_bundles (FoundryDocumentationProvider *provider)
{
  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_documentation_provider_list_bundles_fiber,
                              g_object_ref (provider),
                              g_object_unref);
}

static void
plugin_flatpak_documentation_provider_class_init (PluginFlatpakDocumentationProviderClass *klass)
{
  FoundryDocumentationProviderClass *documentation_provider_class = FOUNDRY_DOCUMENTATION_PROVIDER_CLASS (klass);

  documentation_provider_class->load = plugin_flatpak_documentation_provider_load;
  documentation_provider_class->unload = plugin_flatpak_documentation_provider_unload;
  documentation_provider_class->list_roots = plugin_flatpak_documentation_provider_list_roots;
  documentation_provider_class->list_bundles = plugin_flatpak_documentation_provider_list_bundles;
}

static void
plugin_flatpak_documentation_provider_init (PluginFlatpakDocumentationProvider *self)
{
}
