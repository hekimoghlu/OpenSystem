/* foundry-template-manager.c
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

#include <libpeas.h>

#include "foundry-context.h"
#include "foundry-debug.h"
#include "foundry-model-manager.h"
#include "foundry-template.h"
#include "foundry-template-manager.h"
#include "foundry-template-provider.h"
#include "foundry-util-private.h"

struct _FoundryTemplateManager
{
  GObject           parent_instance;
  PeasExtensionSet *addins;
};

struct _FoundryTemplateManagerClass
{
  GObjectClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryTemplateManager, foundry_template_manager, G_TYPE_OBJECT)

static void
foundry_template_manager_constructed (GObject *object)
{
  FoundryTemplateManager *self = (FoundryTemplateManager *)object;

  G_OBJECT_CLASS (foundry_template_manager_parent_class)->constructed (object);

  self->addins = peas_extension_set_new (peas_engine_get_default (),
                                         FOUNDRY_TYPE_TEMPLATE_PROVIDER,
                                         NULL);
}

static void
foundry_template_manager_dispose (GObject *object)
{
  FoundryTemplateManager *self = (FoundryTemplateManager *)object;

  g_clear_object (&self->addins);

  G_OBJECT_CLASS (foundry_template_manager_parent_class)->dispose (object);
}

static void
foundry_template_manager_class_init (FoundryTemplateManagerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = foundry_template_manager_constructed;
  object_class->dispose = foundry_template_manager_dispose;
}

static void
foundry_template_manager_init (FoundryTemplateManager *self)
{
}

/**
 * foundry_template_manager_list_code_templates:
 * @self: a [class@Foundry.TemplateManager]
 * @context: (nullable): a [class@Foundry.Context] or %NULL
 *
 * Queries all [class@Foundry.TemplateProvider] for available
 * [class@Foundry.CodeTemplate].
 *
 * The resulting module may not be fully populated by all providers
 * by time it resolves. You may await the completion of all providers
 * by awaiting [func@Foundry.list_model_await] for the completion
 * of all providers.
 *
 * This allows the consumer to get a dynamically populating list model
 * for user interfaces without delay.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.CodeTemplate]
 */
DexFuture *
foundry_template_manager_list_code_templates (FoundryTemplateManager *self,
                                              FoundryContext         *context)
{
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_TEMPLATE_MANAGER (self));
  dex_return_error_if_fail (!context || FOUNDRY_IS_CONTEXT (context));

  if (self->addins == NULL || g_list_model_get_n_items (G_LIST_MODEL (self->addins)) == 0)
    return foundry_future_new_not_supported ();

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTemplateProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_template_provider_list_code_templates (provider, context));
    }

  return _foundry_flatten_list_model_new_from_futures (futures);
}

/**
 * foundry_template_manager_list_project_templates:
 * @self: a [class@Foundry.TemplateManager]
 *
 * Queries all [class@Foundry.TemplateProvider] for available
 * [class@Foundry.ProjectTemplate].
 *
 * The resulting module may not be fully populated by all providers
 * by time it resolves. You may await the completion of all providers
 * by awaiting [func@Foundry.list_model_await] for the completion
 * of all providers.
 *
 * This allows the consumer to get a dynamically populating list model
 * for user interfaces without delay.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.ProjectTemplate]
 */
DexFuture *
foundry_template_manager_list_project_templates (FoundryTemplateManager *self)
{
  g_autoptr(GPtrArray) futures = NULL;
  guint n_items;

  dex_return_error_if_fail (FOUNDRY_IS_TEMPLATE_MANAGER (self));

  if (self->addins == NULL || g_list_model_get_n_items (G_LIST_MODEL (self->addins)) == 0)
    return foundry_future_new_not_supported ();

  futures = g_ptr_array_new_with_free_func (dex_unref);
  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->addins));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryTemplateProvider) provider = g_list_model_get_item (G_LIST_MODEL (self->addins), i);

      g_ptr_array_add (futures, foundry_template_provider_list_project_templates (provider));
    }

  return _foundry_flatten_list_model_new_from_futures (futures);
}

typedef struct _ListTemplates
{
  FoundryTemplateManager *self;
  FoundryContext *context;
} ListTemplates;

static void
list_templates_free (ListTemplates *state)
{
  g_clear_object (&state->self);
  g_clear_object (&state->context);
  g_free (state);
}

static DexFuture *
foundry_template_manager_list_templates_fiber (gpointer data)
{
  ListTemplates *state = data;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GListModel) projects = NULL;
  g_autoptr(GListModel) codes = NULL;
  g_autoptr(GPtrArray) futures = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_TEMPLATE_MANAGER (state->self));
  g_assert (!state->context || FOUNDRY_IS_CONTEXT (state->context));

  store = g_list_store_new (G_TYPE_LIST_MODEL);

  if ((projects = dex_await_object (foundry_template_manager_list_project_templates (state->self), NULL)))
    g_list_store_append (store, projects);

  if ((codes = dex_await_object (foundry_template_manager_list_code_templates (state->self, state->context), NULL)))
    g_list_store_append (store, codes);

  dex_await (dex_future_all (foundry_list_model_await (projects),
                             foundry_list_model_await (codes),
                             NULL),
             NULL);

  return dex_future_new_take_object (foundry_flatten_list_model_new (G_LIST_MODEL (g_steal_pointer (&store))));
}

/**
 * foundry_template_manager_list_templates:
 * @self: a [class@Foundry.TemplateManager]
 * @context: (nullable): a [class@Foundry.Context] or %NULL
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.Template].
 */
DexFuture *
foundry_template_manager_list_templates (FoundryTemplateManager *self,
                                         FoundryContext         *context)
{
  g_autoptr(GPtrArray) futures = NULL;
  ListTemplates *state;

  dex_return_error_if_fail (FOUNDRY_IS_TEMPLATE_MANAGER (self));
  dex_return_error_if_fail (!context || FOUNDRY_IS_CONTEXT (context));

  if (self->addins == NULL)
    return foundry_future_new_not_supported ();

  state = g_new0 (ListTemplates, 1);
  state->self = g_object_ref (self);
  state->context = context ? g_object_ref (context) : NULL;

  return dex_scheduler_spawn (NULL, 0,
                              foundry_template_manager_list_templates_fiber,
                              state,
                              (GDestroyNotify) list_templates_free);
}

FoundryTemplateManager *
foundry_template_manager_new (void)
{
  return g_object_new (FOUNDRY_TYPE_TEMPLATE_MANAGER, NULL);
}

static DexFuture *
foundry_template_manager_filter_template (DexFuture *completed,
                                          gpointer   user_data)
{
  const char *template_id = user_data;
  g_autoptr(GListModel) model = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (template_id != NULL);

  if ((model = dex_await_object (dex_ref (completed), &error)))
    {
      guint n_items = g_list_model_get_n_items (model);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryTemplate) template = g_list_model_get_item (model, i);
          g_autofree char *id = foundry_template_dup_id (template);

          if (g_strcmp0 (id, template_id) == 0)
            return dex_future_new_take_object (g_steal_pointer (&template));
        }
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

/**
 * foundry_template_manager_find_template:
 * @self: a [class@Foundry.TemplateManager]
 * @context: (nullable): a [class@Foundry.Context] or %NULL
 * @template_id:
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.Template] or rejects with error.
 */
DexFuture *
foundry_template_manager_find_template (FoundryTemplateManager *self,
                                        FoundryContext         *context,
                                        const char             *template_id)
{
  dex_return_error_if_fail (FOUNDRY_IS_TEMPLATE_MANAGER (self));
  dex_return_error_if_fail (!context || FOUNDRY_IS_CONTEXT (context));
  dex_return_error_if_fail (template_id != NULL);

  return dex_future_finally (foundry_template_manager_list_templates (self, context),
                             foundry_template_manager_filter_template,
                             g_strdup (template_id),
                             g_free);
}
