/* foundry-source-hover-provider.c
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

#include "foundry-markup-view.h"
#include "foundry-source-buffer-private.h"
#include "foundry-source-hover-provider-private.h"

struct _FoundrySourceHoverProvider
{
  GObject               parent_instance;
  FoundryHoverProvider *provider;
};

enum {
  PROP_0,
  PROP_PROVIDER,
  N_PROPS
};

static DexFuture *
populate_display (DexFuture *completed,
                  gpointer   user_data)
{
  g_autoptr(GListModel) model = dex_await_object (dex_ref (completed), NULL);
  GtkSourceHoverDisplay *display = user_data;
  guint n_items;

  g_assert (G_IS_LIST_MODEL (model));
  g_assert (GTK_SOURCE_IS_HOVER_DISPLAY (display));

  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryMarkup) markup = g_list_model_get_item (model, i);
      GtkWidget *child = foundry_markup_view_new (markup);

      gtk_source_hover_display_append (display, child);
    }

  return dex_ref (completed);
}

static void
foundry_source_hover_provider_populate_async (GtkSourceHoverProvider *provider,
                                              GtkSourceHoverContext  *context,
                                              GtkSourceHoverDisplay  *display,
                                              GCancellable           *cancellable,
                                              GAsyncReadyCallback     callback,
                                              gpointer                user_data)
{
  FoundrySourceHoverProvider *self = (FoundrySourceHoverProvider *)provider;
  g_autoptr(DexAsyncResult) result = NULL;
  FoundrySourceBuffer *buffer;
  FoundryTextIter location;
  GtkTextIter iter;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SOURCE_HOVER_PROVIDER (self));
  g_assert (GTK_SOURCE_IS_HOVER_CONTEXT (context));
  g_assert (GTK_SOURCE_IS_HOVER_DISPLAY (display));
  g_assert (FOUNDRY_IS_HOVER_PROVIDER (self->provider));
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  gtk_source_hover_context_get_iter (context, &iter);
  buffer = FOUNDRY_SOURCE_BUFFER (gtk_text_iter_get_buffer (&iter));
  _foundry_source_buffer_init_iter (buffer, &location, &iter);

  result = dex_async_result_new (provider, cancellable, callback, user_data);
  dex_async_result_await (result,
                          dex_future_then (foundry_hover_provider_populate (self->provider, &location),
                                           populate_display,
                                           g_object_ref (display),
                                           g_object_unref));
}

static gboolean
foundry_source_hover_provider_populate_finish (GtkSourceHoverProvider  *provider,
                                               GAsyncResult            *result,
                                               GError                 **error)
{
  FoundrySourceHoverProvider *self = (FoundrySourceHoverProvider *)provider;
  g_autoptr(GError) local_error = NULL;
  g_autoptr(GListModel) model = NULL;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_SOURCE_HOVER_PROVIDER (self));
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

  return model != NULL;
}

static void
hover_provider_iface_init (GtkSourceHoverProviderInterface *iface)
{
  iface->populate_async = foundry_source_hover_provider_populate_async;
  iface->populate_finish = foundry_source_hover_provider_populate_finish;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundrySourceHoverProvider, foundry_source_hover_provider, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (GTK_SOURCE_TYPE_HOVER_PROVIDER, hover_provider_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_source_hover_provider_dispose (GObject *object)
{
  FoundrySourceHoverProvider *self = (FoundrySourceHoverProvider *)object;

  g_clear_object (&self->provider);

  G_OBJECT_CLASS (foundry_source_hover_provider_parent_class)->dispose (object);
}

static void
foundry_source_hover_provider_get_property (GObject    *object,
                                            guint       prop_id,
                                            GValue     *value,
                                            GParamSpec *pspec)
{
  FoundrySourceHoverProvider *self = FOUNDRY_SOURCE_HOVER_PROVIDER (object);

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
foundry_source_hover_provider_set_property (GObject      *object,
                                            guint         prop_id,
                                            const GValue *value,
                                            GParamSpec   *pspec)
{
  FoundrySourceHoverProvider *self = FOUNDRY_SOURCE_HOVER_PROVIDER (object);

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
foundry_source_hover_provider_class_init (FoundrySourceHoverProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_source_hover_provider_dispose;
  object_class->get_property = foundry_source_hover_provider_get_property;
  object_class->set_property = foundry_source_hover_provider_set_property;

  properties[PROP_PROVIDER] =
    g_param_spec_object ("provider", NULL, NULL,
                         FOUNDRY_TYPE_HOVER_PROVIDER,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_source_hover_provider_init (FoundrySourceHoverProvider *self)
{
}

/**
 * foundry_source_hover_provider_new:
 *
 * Returns: (transfer full):
 */
GtkSourceHoverProvider *
foundry_source_hover_provider_new (FoundryHoverProvider *provider)
{
  g_return_val_if_fail (FOUNDRY_IS_HOVER_PROVIDER (provider), NULL);

  return g_object_new (FOUNDRY_TYPE_SOURCE_HOVER_PROVIDER,
                       "provider", provider,
                       NULL);
}
