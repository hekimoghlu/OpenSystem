/* foundry-source-view-addin.c
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

#include "foundry-source-view-addin-private.h"

typedef struct
{
  FoundrySourceView *view;
} FoundrySourceViewAddinPrivate;

enum {
  PROP_0,
  PROP_VIEW,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundrySourceViewAddin, foundry_source_view_addin, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_source_view_addin_get_property (GObject    *object,
                                        guint       prop_id,
                                        GValue     *value,
                                        GParamSpec *pspec)
{
  FoundrySourceViewAddin *self = FOUNDRY_SOURCE_VIEW_ADDIN (object);

  switch (prop_id)
    {
    case PROP_VIEW:
      g_value_set_object (value, foundry_source_view_addin_get_view (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_source_view_addin_class_init (FoundrySourceViewAddinClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_source_view_addin_get_property;

  properties[PROP_VIEW] =
    g_param_spec_object ("vie", NULL, NULL,
                         FOUNDRY_TYPE_SOURCE_VIEW,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_source_view_addin_init (FoundrySourceViewAddin *self)
{
}

DexFuture *
foundry_source_view_addin_load (FoundrySourceViewAddin *self,
                                FoundrySourceView      *view)
{
  FoundrySourceViewAddinPrivate *priv = foundry_source_view_addin_get_instance_private (self);

  dex_return_error_if_fail (FOUNDRY_IS_SOURCE_VIEW_ADDIN (self));
  dex_return_error_if_fail (FOUNDRY_IS_SOURCE_VIEW (view));

  priv->view = view;

  if (FOUNDRY_SOURCE_VIEW_ADDIN_GET_CLASS (self)->load)
    return FOUNDRY_SOURCE_VIEW_ADDIN_GET_CLASS (self)->load (self);

  return dex_future_new_true ();
}

DexFuture *
foundry_source_view_addin_unload (FoundrySourceViewAddin *self)
{
  FoundrySourceViewAddinPrivate *priv = foundry_source_view_addin_get_instance_private (self);
  DexFuture *future;

  dex_return_error_if_fail (FOUNDRY_IS_SOURCE_VIEW_ADDIN (self));

  if (FOUNDRY_SOURCE_VIEW_ADDIN_GET_CLASS (self)->unload)
    future = FOUNDRY_SOURCE_VIEW_ADDIN_GET_CLASS (self)->unload (self);
  else
    future = dex_future_new_true ();

  priv->view = NULL;

  return g_steal_pointer (&future);
}

/**
 * foundry_source_view_addin_get_view:
 * @self: a [class@FoundryGtk.SourceViewAddin]
 *
 * Gets teh view for the addin.
 *
 * This will always be %NULL before load has been called and after
 * unload has been called.
 *
 * Returns: (transfer none) (nullable):
 */
FoundrySourceView *
foundry_source_view_addin_get_view (FoundrySourceViewAddin *self)
{
  FoundrySourceViewAddinPrivate *priv = foundry_source_view_addin_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SOURCE_VIEW_ADDIN (self), NULL);

  return priv->view;
}
