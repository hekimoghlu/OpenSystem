/* plugin-meson-build-target.c
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

#include "plugin-meson-build-target.h"

struct _PluginMesonBuildTarget
{
  FoundryBuildTarget parent_instance;
  JsonNode *node;
};

G_DEFINE_FINAL_TYPE (PluginMesonBuildTarget, plugin_meson_build_target, FOUNDRY_TYPE_BUILD_TARGET)

static char *
dup_string_member (JsonNode   *node,
                   const char *name)
{
  const char *str = NULL;

  if (FOUNDRY_JSON_OBJECT_PARSE (node, name, FOUNDRY_JSON_NODE_GET_STRING (&str)))
    return g_strdup (str);

  return NULL;
}

static char *
plugin_meson_build_target_dup_id (FoundryBuildTarget *target)
{
  return dup_string_member (PLUGIN_MESON_BUILD_TARGET (target)->node, "id");
}

static char *
plugin_meson_build_target_dup_title (FoundryBuildTarget *target)
{
  return dup_string_member (PLUGIN_MESON_BUILD_TARGET (target)->node, "name");
}

static void
plugin_meson_build_target_finalize (GObject *object)
{
  PluginMesonBuildTarget *self = (PluginMesonBuildTarget *)object;

  g_clear_pointer (&self->node, json_node_unref);

  G_OBJECT_CLASS (plugin_meson_build_target_parent_class)->finalize (object);
}

static void
plugin_meson_build_target_class_init (PluginMesonBuildTargetClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryBuildTargetClass *build_target_class = FOUNDRY_BUILD_TARGET_CLASS (klass);

  object_class->finalize = plugin_meson_build_target_finalize;

  build_target_class->dup_id = plugin_meson_build_target_dup_id;
  build_target_class->dup_title = plugin_meson_build_target_dup_title;
}

static void
plugin_meson_build_target_init (PluginMesonBuildTarget *self)
{
}

FoundryBuildTarget *
plugin_meson_build_target_new (JsonNode *node)
{
  PluginMesonBuildTarget *self;

  g_return_val_if_fail (node != NULL, NULL);
  g_return_val_if_fail (JSON_NODE_HOLDS_OBJECT (node), NULL);

  self = g_object_new (PLUGIN_TYPE_MESON_BUILD_TARGET, NULL);
  self->node = json_node_ref (node);

  return FOUNDRY_BUILD_TARGET (self);
}
