/* foundry-search-dialog.c
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

#include "foundry-search-dialog.h"

#define DELAY_MSEC 100

struct _FoundrySearchDialog
{
  AdwDialog           parent_instance;

  FoundryContext     *context;
  char               *search_text;

  GtkStack           *stack;
  GtkListView        *list_view;
  GtkSingleSelection *selection;

  guint               update_source;
  guint               stamp;
};

enum {
  PROP_0,
  PROP_CONTEXT,
  PROP_SEARCH_TEXT,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundrySearchDialog, foundry_search_dialog, ADW_TYPE_DIALOG)

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_search_dialog_update_fiber (FoundrySearchDialog  *self,
                                    FoundrySearchManager *search_manager,
                                    FoundrySearchRequest *request,
                                    guint                 stamp)
{
  g_autoptr(GListModel) results = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_SEARCH_DIALOG (self));
  g_assert (FOUNDRY_IS_SEARCH_MANAGER (search_manager));
  g_assert (FOUNDRY_IS_SEARCH_REQUEST (request));

  if (self->stamp != stamp)
    return dex_future_new_true ();

  if (!(results = dex_await_object (foundry_search_manager_search (search_manager, request), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (self->stamp != stamp)
    return dex_future_new_true ();

  gtk_single_selection_set_model (self->selection, results);

  if (g_list_model_get_n_items (G_LIST_MODEL (results)) > 0)
    gtk_stack_set_visible_child_name (self->stack, "results");
  else
    gtk_stack_set_visible_child_name (self->stack, "empty");

  return dex_future_new_true ();
}

static gboolean
foundry_search_dialog_do_update (gpointer data)
{
  FoundrySearchDialog *self = data;

  g_assert (FOUNDRY_IS_SEARCH_DIALOG (self));
  g_assert (!self->context || FOUNDRY_IS_CONTEXT (self->context));

  self->stamp++;

  g_clear_handle_id (&self->update_source, g_source_remove);

  if (self->context == NULL)
    return G_SOURCE_REMOVE;

  if (foundry_str_empty0 (self->search_text))
    {
      gtk_single_selection_set_model (self->selection, NULL);
      gtk_stack_set_visible_child_name (self->stack, "empty");
    }
  else
    {
      g_autoptr(FoundrySearchManager) search_manager = foundry_context_dup_search_manager (self->context);
      g_autoptr(FoundrySearchRequest) request = foundry_search_request_new (self->context, self->search_text);
      DexFuture *future;

      future = foundry_scheduler_spawn (NULL, 0,
                                        G_CALLBACK (foundry_search_dialog_update_fiber),
                                        4,
                                        FOUNDRY_TYPE_SEARCH_DIALOG, self,
                                        FOUNDRY_TYPE_SEARCH_MANAGER, search_manager,
                                        FOUNDRY_TYPE_SEARCH_REQUEST, request,
                                        G_TYPE_UINT, self->stamp);

      dex_future_disown (future);
    }

  return G_SOURCE_REMOVE;
}

static void
foundry_search_dialog_queue_update (FoundrySearchDialog *self)
{
  g_assert (FOUNDRY_IS_SEARCH_DIALOG (self));

  if (self->update_source == 0)
    self->update_source = g_timeout_add_full (G_PRIORITY_LOW,
                                              DELAY_MSEC,
                                              foundry_search_dialog_do_update,
                                              self, NULL);
}

static void
foundry_search_dialog_set_search_text (FoundrySearchDialog *self,
                                       const char          *search_text)
{
  g_assert (FOUNDRY_IS_SEARCH_DIALOG (self));

  if (search_text != NULL && search_text[0] == 0)
    search_text = NULL;

  if (g_set_str (&self->search_text, search_text))
    {
      foundry_search_dialog_queue_update (self);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SEARCH_TEXT]);
    }
}

static void
foundry_search_dialog_dispose (GObject *object)
{
  FoundrySearchDialog *self = (FoundrySearchDialog *)object;

  gtk_widget_dispose_template (GTK_WIDGET (self), FOUNDRY_TYPE_SEARCH_DIALOG);

  g_clear_pointer (&self->search_text, g_free);
  g_clear_handle_id (&self->update_source, g_source_remove);
  g_clear_object (&self->context);

  G_OBJECT_CLASS (foundry_search_dialog_parent_class)->dispose (object);
}

static void
foundry_search_dialog_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundrySearchDialog *self = FOUNDRY_SEARCH_DIALOG (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      g_value_take_object (value, foundry_search_dialog_dup_context (self));
      break;

    case PROP_SEARCH_TEXT:
      g_value_set_string (value, self->search_text);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_search_dialog_set_property (GObject      *object,
                                    guint         prop_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  FoundrySearchDialog *self = FOUNDRY_SEARCH_DIALOG (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      foundry_search_dialog_set_context (self, g_value_get_object (value));
      break;

    case PROP_SEARCH_TEXT:
      foundry_search_dialog_set_search_text (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_search_dialog_class_init (FoundrySearchDialogClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = foundry_search_dialog_dispose;
  object_class->get_property = foundry_search_dialog_get_property;
  object_class->set_property = foundry_search_dialog_set_property;

  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/foundry-adw/ui/foundry-search-dialog.ui");
  gtk_widget_class_bind_template_child (widget_class, FoundrySearchDialog, selection);
  gtk_widget_class_bind_template_child (widget_class, FoundrySearchDialog, stack);
  gtk_widget_class_bind_template_child (widget_class, FoundrySearchDialog, list_view);

  properties[PROP_CONTEXT] =
    g_param_spec_object ("context", NULL, NULL,
                         FOUNDRY_TYPE_CONTEXT,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SEARCH_TEXT] =
    g_param_spec_string ("search-text", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_search_dialog_init (FoundrySearchDialog *self)
{
  gtk_widget_init_template (GTK_WIDGET (self));
}

GtkWidget *
foundry_search_dialog_new (void)
{
  return g_object_new (FOUNDRY_TYPE_SEARCH_DIALOG, NULL);
}

/**
 * foundry_search_dialog_dup_context:
 * @self: a [class@FoundryAdw.SearchDialog]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryContext *
foundry_search_dialog_dup_context (FoundrySearchDialog *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SEARCH_DIALOG (self), NULL);

  return self->context ? g_object_ref (self->context) : NULL;
}

void
foundry_search_dialog_set_context (FoundrySearchDialog *self,
                                   FoundryContext      *context)
{
  g_return_if_fail (FOUNDRY_IS_SEARCH_DIALOG (self));
  g_return_if_fail (!context || FOUNDRY_IS_CONTEXT (context));

  if (g_set_object (&self->context, context))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CONTEXT]);
}
