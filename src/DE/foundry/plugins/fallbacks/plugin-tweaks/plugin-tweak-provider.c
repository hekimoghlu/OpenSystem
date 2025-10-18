/* plugin-tweak-provider.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <glib/gi18n-lib.h>

#include <libpeas.h>

#include "plugin-tweak-provider.h"

struct _PluginTweakProvider
{
  FoundryTweakProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginTweakProvider, plugin_tweak_provider, FOUNDRY_TYPE_TWEAK_PROVIDER)

static const FoundryTweakInfo top_page_info[] = {
  {
    .type = FOUNDRY_TWEAK_TYPE_GROUP,
    .subpath = "/plugins/",
    .title = N_("Plugins"),
    .icon_name = "plugin-symbolic",
    .display_hint = "menu",
    .section = "-plugins",
    .sort_key = "050-010",
  },
};

static const char *
category_title (const char *category)
{
  g_assert (category != NULL);

  if (foundry_str_equal0 (category, "diagnostics"))
    return N_("Diagnostics");

  if (foundry_str_equal0 (category, "lsp"))
    return N_("Language Servers");

  if (foundry_str_equal0 (category, "search"))
    return N_("Search");

  if (foundry_str_equal0 (category, "buildsystem"))
    return N_("Build Systems");

  if (foundry_str_equal0 (category, "sdk"))
    return N_("SDKs");

  if (foundry_str_equal0 (category, "templates"))
    return N_("Templates");

  if (foundry_str_equal0 (category, "doc"))
    return N_("Documentation");

  if (foundry_str_equal0 (category, "llm"))
    return N_("Language Models");

  if (foundry_str_equal0 (category, "vcs"))
    return N_("Version Control");

  if (foundry_str_equal0 (category, "device"))
    return N_("Devices");

  if (foundry_str_equal0 (category, "completion"))
    return N_("Completion");

  if (foundry_str_equal0 (category, "project"))
    return N_("Projects");

  if (foundry_str_equal0 (category, "other"))
    return N_("Other");

  return NULL;
}

static void
plugin_tweak_provider_update (FoundryInputSwitch   *input,
                              gpointer              ignored,
                              FoundryPluginManager *manager)
{
  PeasPluginInfo *plugin_info;
  gboolean disabled;

  g_assert (FOUNDRY_IS_INPUT_SWITCH (input));
  g_assert (FOUNDRY_IS_PLUGIN_MANAGER (manager));

  plugin_info = g_object_get_data (G_OBJECT (input), "PLUGIN_INFO");
  disabled = foundry_input_switch_get_value (input) == FALSE;

  g_assert (plugin_info != NULL);
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));

  foundry_plugin_manager_set_disabled (manager, plugin_info, disabled);
}

static FoundryInput *
plugin_tweak_provider_callback (const FoundryTweakInfo *info,
                                const char             *path,
                                FoundryContext         *context)
{
  g_autofree char *changed_signal = NULL;
  FoundryPluginManager *manager;
  PeasPluginInfo *plugin_info;
  FoundryInput *input;
  PeasEngine *engine;
  const char *module_name;

  if ((module_name = strrchr (info->subpath, '/')))
    module_name++;

  if (module_name == NULL)
    return NULL;

  engine = peas_engine_get_default ();
  changed_signal = g_strdup_printf ("changed::%s", module_name);

  if (!(plugin_info = peas_engine_get_plugin_info (engine, module_name)))
    return NULL;

  manager = foundry_plugin_manager_get_default ();
  input = foundry_input_switch_new (info->title,
                                    info->subtitle,
                                    NULL,
                                    !foundry_plugin_manager_get_disabled (manager, plugin_info));
  g_object_set_data_full (G_OBJECT (input),
                          "PLUGIN_INFO",
                          g_object_ref (plugin_info),
                          g_object_unref);
  g_signal_connect_object (input,
                           "notify::value",
                           G_CALLBACK (plugin_tweak_provider_update),
                           manager,
                           0);
  g_signal_connect_object (manager,
                           changed_signal,
                           G_CALLBACK (plugin_tweak_provider_update),
                           input,
                           G_CONNECT_SWAPPED);
  return input;
}

static DexFuture *
plugin_tweak_provider_load (FoundryTweakProvider *provider)
{
  static const FoundryTweakSource source = {
    .type = FOUNDRY_TWEAK_SOURCE_TYPE_CALLBACK,
    .callback.callback = plugin_tweak_provider_callback,
  };

  g_autoptr(GStringChunk) strings = NULL;
  g_autoptr(GHashTable) seen = NULL;
  g_autoptr(GArray) items = NULL;
  PeasEngine *engine;
  guint n_items;

  dex_return_error_if_fail (PLUGIN_IS_TWEAK_PROVIDER (provider));

  foundry_tweak_provider_register (provider,
                                   GETTEXT_PACKAGE,
                                   "/app",
                                   top_page_info,
                                   G_N_ELEMENTS (top_page_info),
                                   NULL);

  engine = peas_engine_get_default ();
  n_items = g_list_model_get_n_items (G_LIST_MODEL (engine));

  seen = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
  strings = g_string_chunk_new (4096);
  items = g_array_new (FALSE, FALSE, sizeof (FoundryTweakInfo));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(PeasPluginInfo) plugin_info = g_list_model_get_item (G_LIST_MODEL (engine), i);
      const char *category = peas_plugin_info_get_external_data (plugin_info, "Category");
      const char *name = peas_plugin_info_get_name (plugin_info);
      const char *module_name = peas_plugin_info_get_module_name (plugin_info);
      const char *description = peas_plugin_info_get_description (plugin_info);
      g_autofree char *subpath = NULL;
      g_autofree char *subpath_group = NULL;
      g_autofree char *sort_key = NULL;

      if (peas_plugin_info_is_hidden (plugin_info))
        continue;

      if (category == NULL)
        category = "other";

      subpath_group = g_strdup_printf ("%s/plugins", category);
      subpath = g_strdup_printf ("%s/plugins/%s", category, module_name);

      if (!g_hash_table_contains (seen, category))
        {
          const char *ctitle = category_title (category);

          if (ctitle)
            ctitle = g_string_chunk_insert_const (strings, ctitle);

          g_array_append_val (items, ((const FoundryTweakInfo) {
            .type = FOUNDRY_TWEAK_TYPE_GROUP,
            .subpath = g_string_chunk_insert_const (strings, category),
            .title = ctitle,
            .sort_key = g_string_chunk_insert_const (strings, category),
            .display_hint = "page",
            .icon_name = "plugin-symbolic",
          }));
          g_array_append_val (items, ((const FoundryTweakInfo) {
            .type = FOUNDRY_TWEAK_TYPE_GROUP,
            .subpath = g_string_chunk_insert_const (strings, subpath_group),
          }));
          g_hash_table_replace (seen, g_strdup (category), NULL);
        }

      sort_key = g_strdup_printf ("%s-%s", category, name);

      g_array_append_val (items, ((const FoundryTweakInfo) {
        .type = FOUNDRY_TWEAK_TYPE_SWITCH,
        .subpath = g_string_chunk_insert_const (strings, subpath),
        .title = g_string_chunk_insert_const (strings, name),
        .subtitle = g_string_chunk_insert_const (strings, description ?: ""),
        .sort_key = g_string_chunk_insert_const (strings, sort_key),
        .source = (gpointer)&source,
      }));
    }

  if (items->len > 0)
    foundry_tweak_provider_register (provider,
                                     GETTEXT_PACKAGE,
                                     "/app/plugins",
                                     (gpointer)items->data,
                                     items->len,
                                     NULL);

  return dex_future_new_true ();
}

static void
plugin_tweak_provider_class_init (PluginTweakProviderClass *klass)
{
  FoundryTweakProviderClass *provider_class = FOUNDRY_TWEAK_PROVIDER_CLASS (klass);

  provider_class->load = plugin_tweak_provider_load;
}

static void
plugin_tweak_provider_init (PluginTweakProvider *self)
{
}
