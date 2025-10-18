/* plugin-meson-base-stage.h
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

#pragma once

#include <foundry.h>

G_BEGIN_DECLS

#define PLUGIN_TYPE_MESON_BASE_STAGE (plugin_meson_base_stage_get_type())

G_DECLARE_DERIVABLE_TYPE (PluginMesonBaseStage, plugin_meson_base_stage, PLUGIN, MESON_BASE_STAGE, FoundryBuildStage)

struct _PluginMesonBaseStageClass
{
  FoundryBuildStageClass parent_class;
};

char *plugin_meson_base_stage_dup_builddir (PluginMesonBaseStage *self);
char *plugin_meson_base_stage_dup_meson    (PluginMesonBaseStage *self);
char *plugin_meson_base_stage_dup_ninja    (PluginMesonBaseStage *self);

G_END_DECLS
