/* foundry-gtk-init.c
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

#include "foundry-gtk-init.h"
#include "foundry-gtk-model-manager-private.h"
#include "foundry-gtk-resources.h"
#include "foundry-menu-manager.h"
#include "foundry-shortcut-manager.h"
#include "foundry-source-buffer.h"
#include "foundry-source-buffer-provider-private.h"

#include "foundry-gtk-plugins-resources.h"

static char *
get_resource_path (PeasPluginInfo *plugin_info,
                   const char     *suffix)
{
  return g_strdup_printf ("/app/devsuite/foundry/plugins/%s/%s",
                          peas_plugin_info_get_module_name (plugin_info),
                          suffix ? suffix : "");
}

static gboolean
has_resource (const char *path)
{
  return g_resources_get_info (path, 0, NULL, NULL, NULL);
}

static void
load_snippets (void)
{
  GtkSourceSnippetManager *manager;
  char **search_path;
  gsize len;

  manager = gtk_source_snippet_manager_get_default ();
  search_path = g_strdupv ((char **)gtk_source_snippet_manager_get_search_path (manager));
  len = g_strv_length (search_path);
  search_path = g_realloc_n (search_path, len + 2, sizeof (char **));
  search_path[len++] = g_strdup ("resource:///app/devsuite/foundry/snippets/");
  search_path[len] = NULL;
  gtk_source_snippet_manager_set_search_path (manager, (const char * const *)search_path);
  g_strfreev (search_path);
}

static void
foundry_gtk_load_plugin_cb (PeasEngine     *engine,
                            PeasPluginInfo *plugin_info,
                            gpointer        user_data)
{
  g_autofree char *menus_ui = NULL;
  g_autofree char *style_css = NULL;
  g_autofree char *resource_dir = NULL;

  g_assert (PEAS_IS_ENGINE (engine));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));

  resource_dir = get_resource_path (plugin_info, NULL);
  menus_ui = get_resource_path (plugin_info, "gtk/menus.ui");
  style_css = get_resource_path (plugin_info, "gtk/style.css");

  if (has_resource (style_css))
    {
      g_autoptr(GtkCssProvider) provider = gtk_css_provider_new ();

      gtk_css_provider_load_from_resource (provider, style_css);
      g_object_set_data_full (G_OBJECT (plugin_info),
                              "GTK_CSS_PROVIDER",
                              g_object_ref (provider),
                              g_object_unref);
      gtk_style_context_add_provider_for_display (gdk_display_get_default (),
                                                  GTK_STYLE_PROVIDER (provider),
                                                  GTK_STYLE_PROVIDER_PRIORITY_USER + 1);
    }

  if (has_resource (menus_ui))
    {
      g_autoptr(GError) error = NULL;
      FoundryMenuManager *menu_manager = foundry_menu_manager_get_default ();
      guint merge_id = foundry_menu_manager_add_resource (menu_manager, menus_ui, &error);

      if (error != NULL)
        g_warning ("Failed to parse `%s` from plugin `%s`: %s",
                   menus_ui,
                   peas_plugin_info_get_module_name (plugin_info),
                   error->message);
      else
        g_object_set_data (G_OBJECT (plugin_info),
                           "FOUNDRY_MENU_MERGE_ID",
                           GUINT_TO_POINTER (merge_id));
    }

  foundry_shortcut_manager_add_resources (resource_dir);
}

static void
foundry_gtk_unload_plugin_cb (PeasEngine     *engine,
                              PeasPluginInfo *plugin_info,
                              gpointer        user_data)
{
  g_autofree char *resource_dir = NULL;
  GtkCssProvider *provider;
  guint merge_id;

  g_assert (PEAS_IS_ENGINE (engine));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));

  resource_dir = get_resource_path (plugin_info, NULL);

  if ((provider = g_object_get_data (G_OBJECT (plugin_info), "GTK_CSS_PROVIDER")))
    {
      gtk_style_context_remove_provider_for_display (gdk_display_get_default (),
                                                     GTK_STYLE_PROVIDER (provider));
      g_object_set_data (G_OBJECT (plugin_info), "GTK_CSS_PROVIDER", NULL);
    }

  if ((merge_id = GPOINTER_TO_UINT (g_object_get_data (G_OBJECT (plugin_info), "FOUNDRY_MENU_MERGE_ID"))))
    {
      FoundryMenuManager *menu_manager = foundry_menu_manager_get_default ();
      foundry_menu_manager_remove (menu_manager, merge_id);
    }

  foundry_shortcut_manager_remove_resources (resource_dir);
}

static void
_foundry_gtk_init_once (void)
{
  PeasEngine *engine = peas_engine_get_default ();
  g_autoptr(FoundryModelManager) model_manager = NULL;
  g_auto(GStrv) loaded_plugins = NULL;
  GdkDisplay *display;

  g_resources_register (_foundry_gtk_get_resource ());
  g_resources_register (_foundry_gtk_plugins_get_resource ());

  dex_future_disown (foundry_init ());

  g_type_ensure (FOUNDRY_TYPE_SOURCE_BUFFER);
  g_type_ensure (FOUNDRY_TYPE_SOURCE_BUFFER_PROVIDER);

  if (!(display = gdk_display_get_default ()))
    {
      g_debug ("No GDK display, skipping full initialization");
      return;
    }

  model_manager = g_object_new (FOUNDRY_TYPE_GTK_MODEL_MANAGER, NULL);
  foundry_model_manager_set_default (model_manager);

  gtk_icon_theme_add_resource_path (gtk_icon_theme_get_for_display (display),
                                    "/app/devsuite/foundry/icons");

  load_snippets ();

  g_signal_connect (engine,
                    "load-plugin",
                    G_CALLBACK (foundry_gtk_load_plugin_cb),
                    NULL);
  g_signal_connect_after (engine,
                          "unload-plugin",
                          G_CALLBACK (foundry_gtk_unload_plugin_cb),
                          NULL);

  if ((loaded_plugins = peas_engine_dup_loaded_plugins (engine)))
    {
      for (guint i = 0; loaded_plugins[i]; i++)
        {
          PeasPluginInfo *plugin_info = peas_engine_get_plugin_info (engine, loaded_plugins[i]);
          foundry_gtk_load_plugin_cb (engine, plugin_info, NULL);
        }
    }
}

void
foundry_gtk_init (void)
{
  static gsize initialized;

  if (g_once_init_enter (&initialized))
    {
      _foundry_gtk_init_once ();
      g_once_init_leave (&initialized, TRUE);
    }
}
