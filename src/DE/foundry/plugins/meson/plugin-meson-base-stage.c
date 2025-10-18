/* plugin-meson-base-stage.c
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

#include "plugin-meson-base-stage.h"

typedef struct
{
  char *builddir;
  char *meson;
  char *ninja;
} PluginMesonBaseStagePrivate;

enum {
  PROP_0,
  PROP_BUILDDIR,
  PROP_MESON,
  PROP_NINJA,
  N_PROPS
};

G_DEFINE_TYPE_WITH_PRIVATE (PluginMesonBaseStage, plugin_meson_base_stage, FOUNDRY_TYPE_BUILD_STAGE)

static GParamSpec *properties[N_PROPS];

static void
plugin_meson_base_stage_finalize (GObject *object)
{
  PluginMesonBaseStage *self = (PluginMesonBaseStage *)object;
  PluginMesonBaseStagePrivate *priv = plugin_meson_base_stage_get_instance_private (self);

  g_clear_pointer (&priv->builddir, g_free);
  g_clear_pointer (&priv->meson, g_free);
  g_clear_pointer (&priv->ninja, g_free);

  G_OBJECT_CLASS (plugin_meson_base_stage_parent_class)->finalize (object);
}

static void
plugin_meson_base_stage_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  PluginMesonBaseStage *self = PLUGIN_MESON_BASE_STAGE (object);

  switch (prop_id)
    {
    case PROP_BUILDDIR:
      g_value_take_string (value, plugin_meson_base_stage_dup_builddir (self));
      break;

    case PROP_MESON:
      g_value_take_string (value, plugin_meson_base_stage_dup_meson (self));
      break;

    case PROP_NINJA:
      g_value_take_string (value, plugin_meson_base_stage_dup_ninja (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_meson_base_stage_set_property (GObject      *object,
                                      guint         prop_id,
                                      const GValue *value,
                                      GParamSpec   *pspec)
{
  PluginMesonBaseStage *self = PLUGIN_MESON_BASE_STAGE (object);
  PluginMesonBaseStagePrivate *priv = plugin_meson_base_stage_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_BUILDDIR:
      priv->builddir = g_value_dup_string (value);
      break;

    case PROP_MESON:
      priv->meson = g_value_dup_string (value);
      break;

    case PROP_NINJA:
      priv->ninja = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_meson_base_stage_class_init (PluginMesonBaseStageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_meson_base_stage_finalize;
  object_class->get_property = plugin_meson_base_stage_get_property;
  object_class->set_property = plugin_meson_base_stage_set_property;

  properties[PROP_BUILDDIR] =
    g_param_spec_string ("builddir", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MESON] =
    g_param_spec_string ("meson", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NINJA] =
    g_param_spec_string ("ninja", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_meson_base_stage_init (PluginMesonBaseStage *self)
{
}

char *
plugin_meson_base_stage_dup_builddir (PluginMesonBaseStage *self)
{
  PluginMesonBaseStagePrivate *priv = plugin_meson_base_stage_get_instance_private (self);

  g_return_val_if_fail (PLUGIN_IS_MESON_BASE_STAGE (self), NULL);

  return g_strdup (priv->builddir);
}

char *
plugin_meson_base_stage_dup_meson (PluginMesonBaseStage *self)
{
  PluginMesonBaseStagePrivate *priv = plugin_meson_base_stage_get_instance_private (self);

  g_return_val_if_fail (PLUGIN_IS_MESON_BASE_STAGE (self), NULL);

  return g_strdup (priv->meson);
}

char *
plugin_meson_base_stage_dup_ninja (PluginMesonBaseStage *self)
{
  PluginMesonBaseStagePrivate *priv = plugin_meson_base_stage_get_instance_private (self);

  g_return_val_if_fail (PLUGIN_IS_MESON_BASE_STAGE (self), NULL);

  return g_strdup (priv->ninja);
}

