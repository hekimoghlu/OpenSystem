/* foundry-shortcut-manager.c
 *
 * Copyright 2022-2025 Christian Hergert <chergert@redhat.com>
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

#include <gtk/gtk.h>

#include "foundry-menu-manager.h"
#include "foundry-shortcut-bundle-private.h"
#include "foundry-shortcut-manager.h"
#include "foundry-shortcut-provider.h"

struct _FoundryShortcutManager
{
  FoundryService parent_instance;

  /* Holds [plugin_models,internal_models] so that plugin models take
   * priority over the others.
   */
  GListStore *toplevel;

  /* Holds the bundle for user shortcut overrides. */
  FoundryShortcutBundle *user_bundle;

  /* Holds bundles loaded from plugins, more recently loaded plugins
   * towards the head of the list.
   *
   * Plugins loaded dynamically could change ordering here, which might
   * be something we want to address someday. In practice, it doesn't
   * happen very often and people restart applications often.
   */
  GListStore *plugin_models;

  /* A flattened list model we proxy through our interface */
  GtkFlattenListModel *flatten;

  /* Extension set of FoundryShortcutProvider */
  FoundryExtensionSet *providers;
  GListStore *providers_models;

  /* Used to track action-name -> accel mappings */
  FoundryShortcutObserver *observer;

  /* Idle GSource to update triggers with new values from a user
   * bundle having changed.
   */
  guint update_triggers_source;
};

static GType
foundry_shortcut_manager_get_item_type (GListModel *model)
{
  return GTK_TYPE_SHORTCUT;
}

static guint
foundry_shortcut_manager_get_n_items (GListModel *model)
{
  FoundryShortcutManager *self = FOUNDRY_SHORTCUT_MANAGER (model);

  if (self->flatten)
    return g_list_model_get_n_items (G_LIST_MODEL (self->flatten));

  return 0;
}

static gpointer
foundry_shortcut_manager_get_item (GListModel *model,
                               guint       position)
{
  FoundryShortcutManager *self = FOUNDRY_SHORTCUT_MANAGER (model);
  GtkShortcut *ret = NULL;

  if (self->flatten)
    ret = g_list_model_get_item (G_LIST_MODEL (self->flatten), position);

  return ret;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_shortcut_manager_get_item_type;
  iface->get_n_items = foundry_shortcut_manager_get_n_items;
  iface->get_item = foundry_shortcut_manager_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryShortcutManager, foundry_shortcut_manager, FOUNDRY_TYPE_SERVICE,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GListStore *plugin_models;
static FoundryShortcutBundle *user_bundle;
static guint update_menu_source;

static gboolean
update_menus_cb (GHashTable *id_to_trigger)
{
  FoundryMenuManager *menu_manager;
  const char * const *menu_ids;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (id_to_trigger != NULL);

  update_menu_source = 0;

  menu_manager = foundry_menu_manager_get_default ();
  menu_ids = foundry_menu_manager_get_menu_ids (menu_manager);

  for (guint i = 0; menu_ids[i]; i++)
    {
      GMenu *menu = foundry_menu_manager_get_menu_by_id (menu_manager, menu_ids[i]);
      guint n_items = g_menu_model_get_n_items (G_MENU_MODEL (menu));

      for (guint j = 0; j < n_items; j++)
        {
          g_autofree char *shortcut_id = NULL;
          g_autofree char *accel = NULL;
          g_autofree char *original_accel = NULL;
          GtkShortcutTrigger *trigger;

          g_menu_model_get_item_attribute (G_MENU_MODEL (menu), j, "id", "s", &shortcut_id);

          if (shortcut_id == NULL)
            continue;

          g_menu_model_get_item_attribute (G_MENU_MODEL (menu), j, "accel", "s", &accel);
          g_menu_model_get_item_attribute (G_MENU_MODEL (menu), j, "original-accel", "s", &original_accel);

          if ((trigger = g_hash_table_lookup (id_to_trigger, shortcut_id)))
            {
              g_autofree char *new_accel = gtk_shortcut_trigger_to_string (trigger);

              if (foundry_str_equal0 (new_accel, "never"))
                new_accel[0] = 0;

              /* Save original accel for re-use later */
              if (original_accel == NULL && accel != NULL)
                foundry_menu_manager_set_attribute_string (menu_manager, menu, j, "original-accel", accel);

              foundry_menu_manager_set_attribute_string (menu_manager, menu, j, "accel", new_accel);
            }
          else
            {
              if (original_accel != NULL && !foundry_str_equal0 (accel, original_accel))
                foundry_menu_manager_set_attribute_string (menu_manager, menu, j, "accel", original_accel);
            }
        }
    }

  return G_SOURCE_REMOVE;
}

static GListModel *
get_internal_shortcuts (void)
{
  static GtkFlattenListModel *flatten;

  if (flatten == NULL)
    {
      static const char *names[] = { "libfoundry-gtk", };
      GListStore *internal_models;

      internal_models = g_list_store_new (G_TYPE_LIST_MODEL);

      for (guint i = 0; i < G_N_ELEMENTS (names); i++)
        {
          g_autoptr(FoundryShortcutBundle) bundle = foundry_shortcut_bundle_new ();
          g_autofree char *uri = g_strdup_printf ("resource:///app/devsuite/%s/gtk/keybindings.json", names[i]);
          g_autoptr(GFile) file = g_file_new_for_uri (uri);
          g_autoptr(GError) error = NULL;

          if (!g_file_query_exists (file, NULL))
            continue;

          if (!foundry_shortcut_bundle_parse (bundle, file, &error))
            g_critical ("Failed to parse %s: %s", uri, error->message);
          else
            g_list_store_append (internal_models, bundle);
        }

      flatten = gtk_flatten_list_model_new (G_LIST_MODEL (internal_models));
    }

  return G_LIST_MODEL (flatten);
}

static void
override_bundles (GListModel *model,
                  GHashTable *id_to_trigger)
{
  guint n_items;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (G_IS_LIST_MODEL (model));
  g_assert (id_to_trigger != NULL);

  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryShortcutBundle) bundle = g_list_model_get_item (model, i);

      g_assert (bundle != NULL);
      g_assert (FOUNDRY_IS_SHORTCUT_BUNDLE (bundle));

      foundry_shortcut_bundle_override_triggers (bundle, id_to_trigger);
    }
}

static gboolean
foundry_shortcut_manager_update_overrides (FoundryShortcutManager *self)
{
  g_autoptr(GHashTable) id_to_trigger = NULL;
  GListModel *model;
  guint n_items;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SHORTCUT_MANAGER (self));

  self->update_triggers_source = 0;

  if (user_bundle == NULL)
    FOUNDRY_RETURN (G_SOURCE_REMOVE);

  model = G_LIST_MODEL (user_bundle);
  n_items = g_list_model_get_n_items (model);
  id_to_trigger = g_hash_table_new_full (g_str_hash,
                                         g_str_equal,
                                         g_free,
                                         g_object_unref);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(GtkShortcut) shortcut = g_list_model_get_item (model, i);
      FoundryShortcut *state = g_object_get_data (G_OBJECT (shortcut), "FOUNDRY_SHORTCUT");

      g_assert (GTK_IS_SHORTCUT (shortcut));
      g_assert (state != NULL);

      if (state->override != NULL && state->trigger != NULL)
        g_hash_table_insert (id_to_trigger,
                             g_strdup (state->override),
                             g_object_ref (state->trigger));
    }

  override_bundles (G_LIST_MODEL (self->plugin_models), id_to_trigger);

  if (update_menu_source == 0)
    update_menu_source = g_idle_add_full (G_PRIORITY_LOW,
                                          (GSourceFunc)update_menus_cb,
                                          g_hash_table_ref (id_to_trigger),
                                          (GDestroyNotify)g_hash_table_unref);

  FOUNDRY_RETURN (G_SOURCE_REMOVE);
}

static void
foundry_shortcut_manager_user_items_changed_cb (FoundryShortcutManager *self,
                                                guint                   position,
                                                guint                   removed,
                                                guint                   added,
                                                FoundryShortcutBundle      *bundle)
{
  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SHORTCUT_MANAGER (self));
  g_assert (FOUNDRY_IS_SHORTCUT_BUNDLE (bundle));

  if (self->update_triggers_source == 0)
    self->update_triggers_source =
      g_idle_add_full (G_PRIORITY_LOW,
                       (GSourceFunc) foundry_shortcut_manager_update_overrides,
                       self, NULL);

  FOUNDRY_EXIT;
}

static void
foundry_shortcut_manager_items_changed_cb (FoundryShortcutManager *self,
                                           guint                   position,
                                           guint                   removed,
                                           guint                   added,
                                           GListModel             *model)
{
  g_assert (FOUNDRY_IS_SHORTCUT_MANAGER (self));
  g_assert (G_IS_LIST_MODEL (model));

  g_list_model_items_changed (G_LIST_MODEL (self), position, removed, added);
}

static void
on_provider_added_cb (FoundryExtensionSet *set,
                      PeasPluginInfo      *plugin_info,
                      GObject             *exten,
                      gpointer             user_data)
{
  FoundryShortcutProvider *provider = (FoundryShortcutProvider *)exten;
  FoundryShortcutManager *self = user_data;
  g_autoptr(GListModel) model = NULL;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_EXTENSION_SET (set));
  g_assert (plugin_info != NULL);
  g_assert (FOUNDRY_IS_SHORTCUT_PROVIDER (provider));

  if ((model = foundry_shortcut_provider_list_shortcuts (provider)))
    {
      FOUNDRY_TRACE_MSG ("Adding shortcut model for %s with %d items",
                         peas_plugin_info_get_module_name (plugin_info),
                         g_list_model_get_n_items (model));
      g_object_set_data_full (G_OBJECT (provider),
                              "SHORTCUTS_MODEL",
                              g_object_ref (model),
                              g_object_unref);
      g_list_store_append (self->providers_models, model);
    }

  FOUNDRY_EXIT;
}

static void
on_provider_removed_cb (FoundryExtensionSet *set,
                        PeasPluginInfo      *plugin_info,
                        GObject             *exten,
                        gpointer             user_data)
{
  FoundryShortcutProvider *provider = (FoundryShortcutProvider *)exten;
  FoundryShortcutManager *self = user_data;
  GListModel *model;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_EXTENSION_SET (set));
  g_assert (plugin_info != NULL);
  g_assert (FOUNDRY_IS_SHORTCUT_PROVIDER (provider));

  if (self->providers_models == NULL)
    FOUNDRY_EXIT;

  if ((model = g_object_get_data (G_OBJECT (provider), "SHORTCUTS_MODEL")))
    {
      guint n_items;

      g_assert (G_IS_LIST_MODEL (model));

      n_items = g_list_model_get_n_items (G_LIST_MODEL (self->providers_models));

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(GListModel) item = g_list_model_get_item (G_LIST_MODEL (self->providers_models), i);

          if (item == model)
            {
              g_list_store_remove (self->providers_models, i);
              FOUNDRY_EXIT;
            }
        }
    }

  FOUNDRY_EXIT;
}

static DexFuture *
foundry_shortcut_manager_start (FoundryService *service)
{
  FoundryShortcutManager *self = (FoundryShortcutManager *)service;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (FOUNDRY_IS_SHORTCUT_MANAGER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  self->providers = foundry_extension_set_new (context,
                                               peas_engine_get_default (),
                                               FOUNDRY_TYPE_SHORTCUT_PROVIDER,
                                               NULL, NULL, NULL);
  g_signal_connect (self->providers,
                    "extension-added",
                    G_CALLBACK (on_provider_added_cb),
                    self);
  g_signal_connect (self->providers,
                    "extension-removed",
                    G_CALLBACK (on_provider_removed_cb),
                    self);
  foundry_extension_set_foreach_by_priority (self->providers,
                                             on_provider_added_cb,
                                             self);

  return dex_future_new_true ();
}

static DexFuture *
foundry_shortcut_manager_stop (FoundryService *service)
{
  FoundryShortcutManager *self = (FoundryShortcutManager *)service;

  g_clear_handle_id (&self->update_triggers_source, g_source_remove);

  g_clear_object (&self->observer);
  g_clear_object (&self->providers);
  g_clear_object (&self->providers_models);
  g_clear_object (&self->plugin_models);
  g_clear_object (&self->user_bundle);
  g_clear_object (&self->toplevel);
  g_clear_object (&self->flatten);

  return dex_future_new_true ();
}

static void
foundry_shortcut_manager_class_init (FoundryShortcutManagerClass *klass)
{
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  service_class->start = foundry_shortcut_manager_start;
  service_class->stop = foundry_shortcut_manager_stop;

  g_type_ensure (FOUNDRY_TYPE_SHORTCUT_PROVIDER);
}

static void
foundry_shortcut_manager_init (FoundryShortcutManager *self)
{
  GtkFlattenListModel *flatten;

  if (plugin_models == NULL)
    plugin_models = g_list_store_new (G_TYPE_LIST_MODEL);

  self->toplevel = g_list_store_new (G_TYPE_LIST_MODEL);

  if (user_bundle == NULL)
    {
      g_autoptr(GFile) user_file = NULL;

      user_file = g_file_new_build_filename (g_get_user_config_dir (),
                                             "gnome-builder",
                                             "keybindings.json",
                                             NULL);
      user_bundle = foundry_shortcut_bundle_new_for_user (user_file);
    }

  /* We monitor the user-bundle for changes so that we can
   * handle any sort of override by replacing the trigger
   * of the original shortcut.
   */
  g_signal_connect_object (user_bundle,
                           "items-changed",
                           G_CALLBACK (foundry_shortcut_manager_user_items_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  /* Setup user shortcuts at highest priority */
  g_list_store_append (self->toplevel, user_bundle);

  /* Then add providers implemented by plugins */
  self->providers_models = g_list_store_new (G_TYPE_LIST_MODEL);
  flatten = gtk_flatten_list_model_new (g_object_ref (G_LIST_MODEL (self->providers_models)));
  g_list_store_append (self->toplevel, flatten);
  g_object_unref (flatten);

  /* Then add keybindings.json found within plugin resources */
  self->plugin_models = g_object_ref (plugin_models);
  flatten = gtk_flatten_list_model_new (g_object_ref (G_LIST_MODEL (self->plugin_models)));
  g_list_store_append (self->toplevel, flatten);
  g_object_unref (flatten);

  /* Then attach our internal plugins */
  g_list_store_append (self->toplevel, get_internal_shortcuts ());

  /* And finally flatten the whole thing into a shortcut list */
  self->flatten = gtk_flatten_list_model_new (g_object_ref (G_LIST_MODEL (self->toplevel)));
  g_signal_connect_object (self->flatten,
                           "items-changed",
                           G_CALLBACK (foundry_shortcut_manager_items_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  /* Build a "compiled" shortcut map to make remappings easier */
  self->observer = foundry_shortcut_observer_new (G_LIST_MODEL (self->flatten));

  /* Apply user-bundle overrides to plugin models/etc */
  foundry_shortcut_manager_update_overrides (self);
}

/**
 * foundry_shortcut_manager_from_context:
 * @context: a [class@Foundry.Context]
 *
 * Gets the shortcut manager for the contenxt
 *
 * Returns: (transfer full):
 */
FoundryShortcutManager *
foundry_shortcut_manager_from_context (FoundryContext *context)
{
  g_return_val_if_fail (FOUNDRY_IS_MAIN_THREAD (), NULL);
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  return foundry_context_dup_service_typed (context, FOUNDRY_TYPE_SHORTCUT_MANAGER);
}

void
foundry_shortcut_manager_add_resources (const char *resource_path)
{
  g_autoptr(GFile) keybindings_json = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *keybindings_json_path = NULL;
  g_autoptr(FoundryShortcutBundle) bundle = NULL;

  g_return_if_fail (resource_path != NULL);

  keybindings_json_path = g_build_filename (resource_path, "gtk", "keybindings.json", NULL);

  if (g_str_has_prefix (resource_path, "resource://"))
    keybindings_json = g_file_new_for_uri (keybindings_json_path);
  else
    keybindings_json = g_file_new_for_path (keybindings_json_path);

  if (!g_file_query_exists (keybindings_json, NULL))
    return;

  bundle = foundry_shortcut_bundle_new ();

  if (!foundry_shortcut_bundle_parse (bundle, keybindings_json, &error))
    {
      g_warning ("Failed to parse %s: %s", resource_path, error->message);
      return;
    }

  g_object_set_data_full (G_OBJECT (bundle),
                          "RESOURCE_PATH",
                          g_strdup (resource_path),
                          g_free);

  if (plugin_models == NULL)
    plugin_models = g_list_store_new (G_TYPE_LIST_MODEL);

  g_list_store_append (plugin_models, bundle);
}

void
foundry_shortcut_manager_remove_resources (const char *resource_path)
{
  guint n_items;

  g_return_if_fail (resource_path != NULL);

  if (plugin_models == NULL)
    return;

  n_items = g_list_model_get_n_items (G_LIST_MODEL (plugin_models));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryShortcutBundle) bundle = g_list_model_get_item (G_LIST_MODEL (plugin_models), i);

      if (g_strcmp0 (resource_path, g_object_get_data (G_OBJECT (bundle), "RESOURCE_PATH")) == 0)
        {
          g_list_store_remove (plugin_models, i);
          return;
        }
    }
}

/**
 * foundry_shortcut_manager_get_observer:
 * @self: a [class@FoundryGtk.ShortcutManager]
 *
 * Returns: (transfer none):
 */
FoundryShortcutObserver *
foundry_shortcut_manager_get_observer (FoundryShortcutManager *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SHORTCUT_MANAGER (self), NULL);

  return self->observer;
}

/**
 * foundry_shortcut_manager_reset_user:
 *
 * Reset user modified shortcuts.
 */
void
foundry_shortcut_manager_reset_user (void)
{
  g_autoptr(GFile) file = g_file_new_build_filename (g_get_user_config_dir (),
                                                     "gnome-builder",
                                                     "keybindings.json",
                                                     NULL);
  g_file_delete (file, NULL, NULL);
}

/**
 * foundry_shortcut_manager_get_user_bundle:
 *
 * Returns: (transfer none):
 */
FoundryShortcutBundle *
foundry_shortcut_manager_get_user_bundle (void)
{
  return user_bundle;
}
