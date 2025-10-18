/* foundry-dependency.c
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

#include "foundry-dependency.h"

typedef struct
{
  GWeakRef provider_wr;
} FoundryDependencyPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryDependency, foundry_dependency, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_KIND,
  PROP_NAME,
  PROP_LOCATION,
  PROP_PROVIDER,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_dependency_finalize (GObject *object)
{
  FoundryDependency *self = (FoundryDependency *)object;
  FoundryDependencyPrivate *priv = foundry_dependency_get_instance_private (self);

  g_weak_ref_clear (&priv->provider_wr);

  G_OBJECT_CLASS (foundry_dependency_parent_class)->finalize (object);
}

static void
foundry_dependency_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryDependency *self = FOUNDRY_DEPENDENCY (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_value_take_string (value, foundry_dependency_dup_name (self));
      break;

    case PROP_KIND:
      g_value_take_string (value, foundry_dependency_dup_kind (self));
      break;

    case PROP_LOCATION:
      g_value_take_string (value, foundry_dependency_dup_location (self));
      break;

    case PROP_PROVIDER:
      g_value_take_object (value, foundry_dependency_dup_provider (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_dependency_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryDependency *self = FOUNDRY_DEPENDENCY (object);
  FoundryDependencyPrivate *priv = foundry_dependency_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_PROVIDER:
      g_weak_ref_set (&priv->provider_wr, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_dependency_class_init (FoundryDependencyClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_dependency_finalize;
  object_class->get_property = foundry_dependency_get_property;
  object_class->set_property = foundry_dependency_set_property;

  properties[PROP_KIND] =
    g_param_spec_string ("kind", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_LOCATION] =
    g_param_spec_string ("location", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PROVIDER] =
    g_param_spec_object ("provider", NULL, NULL,
                         FOUNDRY_TYPE_DEPENDENCY_PROVIDER,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_dependency_init (FoundryDependency *self)
{
  FoundryDependencyPrivate *priv = foundry_dependency_get_instance_private (self);

  g_weak_ref_init (&priv->provider_wr, NULL);
}

char *
foundry_dependency_dup_name (FoundryDependency *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEPENDENCY (self), NULL);

  if (FOUNDRY_DEPENDENCY_GET_CLASS (self)->dup_name)
    return FOUNDRY_DEPENDENCY_GET_CLASS (self)->dup_name (self);

  return NULL;
}

char *
foundry_dependency_dup_location (FoundryDependency *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEPENDENCY (self), NULL);

  if (FOUNDRY_DEPENDENCY_GET_CLASS (self)->dup_location)
    return FOUNDRY_DEPENDENCY_GET_CLASS (self)->dup_location (self);

  return NULL;
}

char *
foundry_dependency_dup_kind (FoundryDependency *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEPENDENCY (self), NULL);

  if (FOUNDRY_DEPENDENCY_GET_CLASS (self)->dup_kind)
    return FOUNDRY_DEPENDENCY_GET_CLASS (self)->dup_kind (self);

  return NULL;
}

/**
 * foundry_dependency_dup_provider:
 * @self: a [class@Foundry.Dependency]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryDependencyProvider *
foundry_dependency_dup_provider (FoundryDependency *self)
{
  FoundryDependencyPrivate *priv = foundry_dependency_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_DEPENDENCY (self), NULL);

  return g_weak_ref_get (&priv->provider_wr);
}
