/* plugin-host-documentation-provider.c
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

#include "plugin-host-documentation-provider.h"
#include "plugin-host-sdk.h"

struct _PluginHostDocumentationProvider
{
  FoundryDocumentationProvider parent_instance;
  GListStore *roots;
};

G_DEFINE_FINAL_TYPE (PluginHostDocumentationProvider, plugin_host_documentation_provider, FOUNDRY_TYPE_DOCUMENTATION_PROVIDER)

static DexFuture *
plugin_host_documentation_provider_load_fiber (gpointer user_data)
{
  PluginHostDocumentationProvider *self = user_data;
  g_autoptr(FoundrySdkManager) sdk_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundrySdk) sdk = NULL;

  g_assert (PLUGIN_IS_HOST_DOCUMENTATION_PROVIDER (self));

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))) &&
      (sdk_manager = foundry_context_dup_sdk_manager (context)) &&
      dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (sdk_manager)), NULL) &&
      (sdk = dex_await_object (foundry_sdk_manager_find_by_id (sdk_manager, "host"), NULL)) &&
      PLUGIN_IS_HOST_SDK (sdk))
    {
      g_autofree char *os_name = foundry_get_os_info (G_OS_INFO_KEY_NAME);
      g_autofree char *os_icon = foundry_get_os_info ("LOGO");
      g_autoptr(FoundryDocumentationRoot) root = NULL;
      g_autoptr(GListStore) directories = g_list_store_new (G_TYPE_FILE);
      g_autoptr(GIcon) icon = g_themed_icon_new (os_icon ? os_icon : "go-home-symbolic");
      g_autofree char *doc = plugin_host_sdk_build_filename (PLUGIN_HOST_SDK (sdk), "usr", "share", "doc", NULL);
      g_autofree char *gtk_doc = plugin_host_sdk_build_filename (PLUGIN_HOST_SDK (sdk), "usr", "share", "gtk-doc", "html", NULL);
      g_autofree char *devhelp = plugin_host_sdk_build_filename (PLUGIN_HOST_SDK (sdk), "usr", "share", "devhelp", "books", NULL);
      g_autofree char *user_doc = g_build_filename (g_get_user_data_dir (), "doc", NULL);
      g_autofree char *user_gtk_doc = g_build_filename (g_get_user_data_dir (), "gtk-doc", "html", NULL);
      g_autofree char *user_devhelp = g_build_filename (g_get_user_data_dir (), "devhelp", "books", NULL);
      g_autoptr(GFile) doc_file = g_file_new_for_path (doc);
      g_autoptr(GFile) gtk_doc_file = g_file_new_for_path (gtk_doc);
      g_autoptr(GFile) devhelp_file = g_file_new_for_path (devhelp);
      g_autoptr(GFile) user_doc_file = g_file_new_for_path (user_doc);
      g_autoptr(GFile) user_gtk_doc_file = g_file_new_for_path (user_gtk_doc);
      g_autoptr(GFile) user_devhelp_file = g_file_new_for_path (user_devhelp);
      g_autoptr(GFileEnumerator) enumerator = NULL;
      g_autoptr(GError) error = NULL;

      g_debug ("Discovered documentation directory at \"%s\"", doc);
      g_debug ("Discovered documentation directory at \"%s\"", gtk_doc);
      g_debug ("Discovered documentation directory at \"%s\"", devhelp);
      g_debug ("Discovered documentation directory at \"%s\"", user_doc);
      g_debug ("Discovered documentation directory at \"%s\"", user_gtk_doc);
      g_debug ("Discovered documentation directory at \"%s\"", user_devhelp);

      g_list_store_append (directories, doc_file);
      g_list_store_append (directories, gtk_doc_file);
      g_list_store_append (directories, devhelp_file);
      g_list_store_append (directories, user_doc_file);
      g_list_store_append (directories, user_gtk_doc_file);
      g_list_store_append (directories, user_devhelp_file);

      /* On some systems like Debian, the documentation is in a subdirectory */
      enumerator = dex_await_object (dex_file_enumerate_children (doc_file,
                                                                  G_FILE_ATTRIBUTE_STANDARD_NAME","
                                                                  G_FILE_ATTRIBUTE_STANDARD_TYPE",",
                                                                  G_FILE_QUERY_INFO_NOFOLLOW_SYMLINKS,
                                                                  G_PRIORITY_DEFAULT),
                                     &error);

      if (enumerator != NULL)
        {
          for (;;)
            {
              g_autolist(GFileInfo) files = NULL;

              if (!(files = dex_await_boxed (dex_file_enumerator_next_files (enumerator,
                                                                             100,
                                                                             G_PRIORITY_DEFAULT),
                                             NULL)))
                break;

              for (const GList *iter = files; iter; iter = iter->next)
                {
                  GFileInfo *info = iter->data;

                  if (g_file_info_get_file_type (info) == G_FILE_TYPE_DIRECTORY)
                    {
                      g_autoptr(GFile) child = g_file_get_child (doc_file, g_file_info_get_name (info));

                      g_list_store_append (directories, child);
                    }
                }
            }
        }

      root = foundry_documentation_root_new ("host", os_name, NULL, icon, G_LIST_MODEL (directories));

      g_list_store_append (self->roots, root);
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_host_documentation_provider_load (FoundryDocumentationProvider *provider)
{
  PluginHostDocumentationProvider *self = (PluginHostDocumentationProvider *)provider;

  g_assert (PLUGIN_IS_HOST_DOCUMENTATION_PROVIDER (self));

  self->roots = g_list_store_new (FOUNDRY_TYPE_DOCUMENTATION_ROOT);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_host_documentation_provider_load_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
plugin_host_documentation_provider_unload (FoundryDocumentationProvider *provider)
{
  PluginHostDocumentationProvider *self = (PluginHostDocumentationProvider *)provider;

  g_assert (PLUGIN_IS_HOST_DOCUMENTATION_PROVIDER (self));

  g_clear_object (&self->roots);

  return dex_future_new_true ();
}

static GListModel *
plugin_host_documentation_provider_list_roots (FoundryDocumentationProvider *provider)
{
  PluginHostDocumentationProvider *self = (PluginHostDocumentationProvider *)provider;

  g_assert (FOUNDRY_IS_DOCUMENTATION_PROVIDER (self));

  return g_object_ref (G_LIST_MODEL (self->roots));
}

static void
plugin_host_documentation_provider_class_init (PluginHostDocumentationProviderClass *klass)
{
  FoundryDocumentationProviderClass *provider_class = FOUNDRY_DOCUMENTATION_PROVIDER_CLASS (klass);

  provider_class->load = plugin_host_documentation_provider_load;
  provider_class->unload = plugin_host_documentation_provider_unload;
  provider_class->list_roots = plugin_host_documentation_provider_list_roots;
}

static void
plugin_host_documentation_provider_init (PluginHostDocumentationProvider *self)
{
}

