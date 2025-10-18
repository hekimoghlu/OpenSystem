/* foundry-source-completion-provider.c
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

#include <gtksourceview/gtksource.h>

#include "foundry-source-buffer-private.h"
#include "foundry-source-completion-proposal-private.h"
#include "foundry-source-completion-provider-private.h"
#include "foundry-source-completion-request-private.h"

struct _FoundrySourceCompletionProvider
{
  GObject                    parent_instance;
  FoundryCompletionProvider *provider;
};

enum {
  PROP_0,
  PROP_PROVIDER,
  N_PROPS
};

static gpointer
map_completion_result (gpointer item,
                       gpointer user_data)
{
  g_autoptr(FoundryCompletionProposal) proposal = item;

  return foundry_source_completion_proposal_new (proposal);
}

static DexFuture *
map_completion_results (DexFuture *completed,
                        gpointer   user_data)
{
  g_autoptr(GListModel) model = dex_await_object (dex_ref (completed), NULL);

  return dex_future_new_take_object (gtk_map_list_model_new (g_steal_pointer (&model),
                                                             map_completion_result,
                                                             NULL, NULL));
}

static void
foundry_source_completion_provider_populate_async (GtkSourceCompletionProvider *provider,
                                                   GtkSourceCompletionContext  *context,
                                                   GCancellable                *cancellable,
                                                   GAsyncReadyCallback          callback,
                                                   gpointer                     user_data)
{
  FoundrySourceCompletionProvider *self = (FoundrySourceCompletionProvider *)provider;
  g_autoptr(FoundryCompletionRequest) request = NULL;
  g_autoptr(DexAsyncResult) result = NULL;
  g_autoptr(DexFuture) future = NULL;

  g_assert (FOUNDRY_IS_SOURCE_COMPLETION_PROVIDER (self));
  g_assert (GTK_SOURCE_IS_COMPLETION_CONTEXT (context));
  g_assert (FOUNDRY_IS_COMPLETION_PROVIDER (self->provider));
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  request = foundry_source_completion_request_new (context);
  future = foundry_completion_provider_complete (self->provider, request);

  result = dex_async_result_new (provider, cancellable, callback, user_data);
  dex_async_result_await (result,
                          dex_future_then (g_steal_pointer (&future),
                                           map_completion_results,
                                           NULL, NULL));
}

static GListModel *
foundry_source_completion_provider_populate_finish (GtkSourceCompletionProvider  *provider,
                                                    GAsyncResult                 *result,
                                                    GError                      **error)
{
  FoundrySourceCompletionProvider *self = (FoundrySourceCompletionProvider *)provider;
  g_autoptr(GListModel) model = NULL;
  g_autoptr(GError) local_error = NULL;

  g_assert (FOUNDRY_IS_SOURCE_COMPLETION_PROVIDER (self));
  g_assert (DEX_IS_ASYNC_RESULT (result));

  if ((model = dex_async_result_propagate_pointer (DEX_ASYNC_RESULT (result), &local_error)))
    g_debug ("%s populated with %u proposals",
             G_OBJECT_TYPE_NAME (self->provider),
             g_list_model_get_n_items (model));
  else
    g_debug ("%s failed to populate with error \"%s\"",
             G_OBJECT_TYPE_NAME (self->provider),
             local_error->message);

  if (local_error != NULL)
    g_propagate_error (error, g_steal_pointer (&local_error));

  return g_steal_pointer (&model);
}

static void
foundry_source_completion_provider_display (GtkSourceCompletionProvider *provider,
                                            GtkSourceCompletionContext  *context,
                                            GtkSourceCompletionProposal *proposal,
                                            GtkSourceCompletionCell     *cell)
{
  FoundryCompletionProposal *wrapped;

  g_assert (FOUNDRY_IS_SOURCE_COMPLETION_PROVIDER (provider));
  g_assert (GTK_SOURCE_IS_COMPLETION_CONTEXT (context));
  g_assert (FOUNDRY_IS_SOURCE_COMPLETION_PROPOSAL (proposal));
  g_assert (GTK_SOURCE_IS_COMPLETION_CELL (cell));

  if (!(wrapped = foundry_source_completion_proposal_get_proposal (FOUNDRY_SOURCE_COMPLETION_PROPOSAL (proposal))))
    return;

  switch (gtk_source_completion_cell_get_column (cell))
    {
    case GTK_SOURCE_COMPLETION_COLUMN_TYPED_TEXT:
      {
        const char *word = gtk_source_completion_context_get_word (context);
        g_autofree char *str = foundry_completion_proposal_dup_typed_text (wrapped);

        if (word && word[0] && str)
          {
            g_autoptr(PangoAttrList) attrs = gtk_source_completion_fuzzy_highlight (str, word);
            gtk_source_completion_cell_set_text_with_attributes (cell, str, attrs);
          }
        else
          {
            gtk_source_completion_cell_set_text (cell, str);
          }

        break;
      }

    case GTK_SOURCE_COMPLETION_COLUMN_ICON:
      {
        g_autoptr(GIcon) icon = foundry_completion_proposal_dup_icon (wrapped);
        gtk_source_completion_cell_set_gicon (cell, icon);
        break;
      }

    case GTK_SOURCE_COMPLETION_COLUMN_COMMENT:
      {
        g_autofree char *str = foundry_completion_proposal_dup_comment (wrapped);

        gtk_source_completion_cell_set_text (cell, str);
      }
      break;

    case GTK_SOURCE_COMPLETION_COLUMN_AFTER:
      {
        g_autofree char *str = foundry_completion_proposal_dup_after (wrapped);

        gtk_source_completion_cell_set_text (cell, str);
      }
      break;

    case GTK_SOURCE_COMPLETION_COLUMN_BEFORE:
      break;

    case GTK_SOURCE_COMPLETION_COLUMN_DETAILS:
      {
        g_autofree char *str = foundry_completion_proposal_dup_details (wrapped);

        gtk_source_completion_cell_set_text (cell, str);
      }
      break;

    default:
      break;
    }
}

static gboolean
foundry_source_completion_provider_is_trigger (GtkSourceCompletionProvider *provider,
                                               const GtkTextIter           *iter,
                                               gunichar                     ch)
{
  FoundrySourceCompletionProvider *self = FOUNDRY_SOURCE_COMPLETION_PROVIDER (provider);
  FoundrySourceBuffer *buffer;
  FoundryTextIter translated;

  buffer = FOUNDRY_SOURCE_BUFFER (gtk_text_iter_get_buffer (iter));
  _foundry_source_buffer_init_iter (buffer, &translated, iter);

  return foundry_completion_provider_is_trigger (self->provider, &translated, ch);
}

typedef struct _Refilter
{
  GtkSourceCompletionProvider *provider;
  GtkSourceCompletionContext  *context;
} Refilter;

static void
refilter_free (Refilter *state)
{
  g_clear_object (&state->context);
  g_clear_object (&state->provider);
  g_free (state);
}

static DexFuture *
foundry_source_completion_provider_refilter_cb (DexFuture *completed,
                                                gpointer   user_data)
{
  Refilter *state = user_data;
  g_autoptr(GListModel) model = NULL;
  g_autoptr(GtkMapListModel) mapped = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_SOURCE_COMPLETION_PROVIDER (state->provider));
  g_assert (GTK_SOURCE_IS_COMPLETION_CONTEXT (state->context));

  if ((model = dex_await_object (dex_ref (completed), NULL)))
    mapped = gtk_map_list_model_new (g_object_ref (model), map_completion_result, NULL, NULL);

  gtk_source_completion_context_set_proposals_for_provider (state->context,
                                                            state->provider,
                                                            G_LIST_MODEL (mapped));

  return dex_future_new_true ();
}

static void
foundry_source_completion_provider_refilter (GtkSourceCompletionProvider *provider,
                                             GtkSourceCompletionContext  *context,
                                             GListModel                  *model)
{
  FoundrySourceCompletionProvider *self = (FoundrySourceCompletionProvider *)provider;
  g_autoptr(FoundryCompletionRequest) request = NULL;
  Refilter *state;

  g_assert (FOUNDRY_IS_SOURCE_COMPLETION_PROVIDER (self));
  g_assert (GTK_SOURCE_IS_COMPLETION_CONTEXT (context));
  g_assert (G_IS_LIST_MODEL (model));

  if (GTK_IS_MAP_LIST_MODEL (model))
    model = gtk_map_list_model_get_model (GTK_MAP_LIST_MODEL (model));

  request = foundry_source_completion_request_new (context);

  state = g_new0 (Refilter, 1);
  state->context = g_object_ref (context);
  state->provider = g_object_ref (provider);

  dex_future_disown (dex_future_finally (foundry_completion_provider_refilter (self->provider, request, model),
                                         foundry_source_completion_provider_refilter_cb,
                                         state,
                                         (GDestroyNotify) refilter_free));
}

static void
foundry_source_completion_provider_activate (GtkSourceCompletionProvider *provider,
                                             GtkSourceCompletionContext  *context,
                                             GtkSourceCompletionProposal *proposal)
{
  FoundryCompletionProposal *wrapped;
  g_autofree char *snippet_text = NULL;
  g_autofree char *typed_text = NULL;
  GtkSourceBuffer *buffer;
  GtkSourceView *view;
  GtkTextIter begin;
  GtkTextIter end;

  g_assert (GTK_SOURCE_IS_COMPLETION_PROVIDER (provider));
  g_assert (GTK_SOURCE_IS_COMPLETION_CONTEXT (context));
  g_assert (FOUNDRY_IS_SOURCE_COMPLETION_PROPOSAL (proposal));

  view = gtk_source_completion_context_get_view (context);
  buffer = gtk_source_completion_context_get_buffer (context);

  wrapped = foundry_source_completion_proposal_get_proposal (FOUNDRY_SOURCE_COMPLETION_PROPOSAL (proposal));

  gtk_source_completion_context_get_bounds (context, &begin, &end);
  gtk_text_buffer_delete (GTK_TEXT_BUFFER (buffer), &begin, &end);

  if ((snippet_text = foundry_completion_proposal_dup_snippet_text (wrapped)))
    {
      g_autoptr(GtkSourceSnippet) snippet = NULL;

      if ((snippet = gtk_source_snippet_new (snippet_text, NULL)))
        {
          gtk_source_view_push_snippet (view, snippet, &begin);
          return;
        }
    }

  if ((typed_text = foundry_completion_proposal_dup_typed_text (wrapped)))
    {
      gtk_text_buffer_insert (GTK_TEXT_BUFFER (buffer), &begin, typed_text, -1);
      return;
    }
}

static void
completion_provider_iface_init (GtkSourceCompletionProviderInterface *iface)
{
  iface->populate_async = foundry_source_completion_provider_populate_async;
  iface->populate_finish = foundry_source_completion_provider_populate_finish;
  iface->display = foundry_source_completion_provider_display;
  iface->is_trigger = foundry_source_completion_provider_is_trigger;
  iface->refilter = foundry_source_completion_provider_refilter;
  iface->activate = foundry_source_completion_provider_activate;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundrySourceCompletionProvider, foundry_source_completion_provider, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (GTK_SOURCE_TYPE_COMPLETION_PROVIDER, completion_provider_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_source_completion_provider_dispose (GObject *object)
{
  FoundrySourceCompletionProvider *self = (FoundrySourceCompletionProvider *)object;

  g_clear_object (&self->provider);

  G_OBJECT_CLASS (foundry_source_completion_provider_parent_class)->dispose (object);
}

static void
foundry_source_completion_provider_get_property (GObject    *object,
                                                 guint       prop_id,
                                                 GValue     *value,
                                                 GParamSpec *pspec)
{
  FoundrySourceCompletionProvider *self = FOUNDRY_SOURCE_COMPLETION_PROVIDER (object);

  switch (prop_id)
    {
    case PROP_PROVIDER:
      g_value_set_object (value, self->provider);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_source_completion_provider_set_property (GObject      *object,
                                                 guint         prop_id,
                                                 const GValue *value,
                                                 GParamSpec   *pspec)
{
  FoundrySourceCompletionProvider *self = FOUNDRY_SOURCE_COMPLETION_PROVIDER (object);

  switch (prop_id)
    {
    case PROP_PROVIDER:
      self->provider = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_source_completion_provider_class_init (FoundrySourceCompletionProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_source_completion_provider_dispose;
  object_class->get_property = foundry_source_completion_provider_get_property;
  object_class->set_property = foundry_source_completion_provider_set_property;

  properties[PROP_PROVIDER] =
    g_param_spec_object ("provider", NULL, NULL,
                         FOUNDRY_TYPE_COMPLETION_PROVIDER,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_source_completion_provider_init (FoundrySourceCompletionProvider *self)
{
}

/**
 * foundry_source_completion_provider_new:
 *
 * Returns: (transfer full):
 */
GtkSourceCompletionProvider *
foundry_source_completion_provider_new (FoundryCompletionProvider *provider)
{
  g_return_val_if_fail (FOUNDRY_IS_COMPLETION_PROVIDER (provider), NULL);

  return g_object_new (FOUNDRY_TYPE_SOURCE_COMPLETION_PROVIDER,
                       "provider", provider,
                       NULL);
}
