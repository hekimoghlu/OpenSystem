/* foundry-hover-provider.c
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

#include "foundry-hover-provider.h"

typedef struct
{
  GWeakRef document_wr;
  PeasPluginInfo *plugin_info;
} FoundryHoverProviderPrivate;

enum {
  PROP_0,
  PROP_BUFFER,
  PROP_DOCUMENT,
  PROP_PLUGIN_INFO,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryHoverProvider, foundry_hover_provider, FOUNDRY_TYPE_CONTEXTUAL)

static GParamSpec *properties[N_PROPS];

static void
foundry_hover_provider_finalize (GObject *object)
{
  FoundryHoverProvider *self = (FoundryHoverProvider *)object;
  FoundryHoverProviderPrivate *priv = foundry_hover_provider_get_instance_private (self);

  g_weak_ref_clear (&priv->document_wr);
  g_clear_object (&priv->plugin_info);

  G_OBJECT_CLASS (foundry_hover_provider_parent_class)->finalize (object);
}

static void
foundry_hover_provider_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryHoverProvider *self = FOUNDRY_HOVER_PROVIDER (object);
  FoundryHoverProviderPrivate *priv = foundry_hover_provider_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_DOCUMENT:
      g_value_take_object (value, foundry_hover_provider_dup_document (self));
      break;

    case PROP_BUFFER:
      g_value_take_object (value, foundry_hover_provider_dup_buffer (self));
      break;

    case PROP_PLUGIN_INFO:
      g_value_set_object (value, priv->plugin_info);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_hover_provider_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryHoverProvider *self = FOUNDRY_HOVER_PROVIDER (object);
  FoundryHoverProviderPrivate *priv = foundry_hover_provider_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_DOCUMENT:
      g_weak_ref_set (&priv->document_wr, g_value_get_object (value));
      break;

    case PROP_PLUGIN_INFO:
      priv->plugin_info = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_hover_provider_class_init (FoundryHoverProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_hover_provider_finalize;
  object_class->get_property = foundry_hover_provider_get_property;
  object_class->set_property = foundry_hover_provider_set_property;

  properties[PROP_BUFFER] =
    g_param_spec_object ("buffer", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_BUFFER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DOCUMENT] =
    g_param_spec_object ("document", NULL, NULL,
                         FOUNDRY_TYPE_TEXT_DOCUMENT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PLUGIN_INFO] =
    g_param_spec_object ("plugin-info", NULL, NULL,
                         PEAS_TYPE_PLUGIN_INFO,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_hover_provider_init (FoundryHoverProvider *self)
{
  FoundryHoverProviderPrivate *priv = foundry_hover_provider_get_instance_private (self);

  g_weak_ref_init (&priv->document_wr, NULL);
}

/**
 * foundry_hover_provider_dup_buffer:
 * @self: a [class@Foundry.HoverProvider]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryTextBuffer *
foundry_hover_provider_dup_buffer (FoundryHoverProvider *self)
{
  g_autoptr(FoundryTextDocument) document = NULL;

  g_return_val_if_fail (FOUNDRY_IS_HOVER_PROVIDER (self), NULL);

  if ((document = foundry_hover_provider_dup_document (self)))
    return foundry_text_document_dup_buffer (document);

  return NULL;
}

/**
 * foundry_hover_provider_dup_document:
 * @self: a [class@Foundry.HoverProvider]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryTextDocument *
foundry_hover_provider_dup_document (FoundryHoverProvider *self)
{
  FoundryHoverProviderPrivate *priv = foundry_hover_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_HOVER_PROVIDER (self), NULL);

  return g_weak_ref_get (&priv->document_wr);
}

/**
 * foundry_hover_provider_populate:
 * @self: a [class@Foundry.HoverProvider]
 * @location: a [struct@Foundry.TextIter]
 *
 * Returns: (transfer full) (nullable): a [class@Dex.Future] that resolves
 *   to a [iface@Gio.ListModel] of [class@Foundry.Markup].
 */
DexFuture *
foundry_hover_provider_populate (FoundryHoverProvider  *self,
                                 const FoundryTextIter *location)
{
  dex_return_error_if_fail (FOUNDRY_IS_HOVER_PROVIDER (self));

  if (FOUNDRY_HOVER_PROVIDER_GET_CLASS (self)->populate)
    return FOUNDRY_HOVER_PROVIDER_GET_CLASS (self)->populate (self, location);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}

/**
 * foundry_hover_provider_dup_plugin_info:
 * @self: a [class@Foundry.HoverProvider]
 *
 * Returns: (transfer full) (nullable):
 */
PeasPluginInfo *
foundry_hover_provider_dup_plugin_info (FoundryHoverProvider *self)
{
  FoundryHoverProviderPrivate *priv = foundry_hover_provider_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_HOVER_PROVIDER (self), NULL);

  return priv->plugin_info ? g_object_ref (priv->plugin_info) : NULL;
}
