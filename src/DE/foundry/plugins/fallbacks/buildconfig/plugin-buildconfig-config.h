/* plugin-buildconfig-config.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#define PLUGIN_TYPE_BUILDCONFIG_CONFIG (plugin_buildconfig_config_get_type())

G_DECLARE_FINAL_TYPE (PluginBuildconfigConfig, plugin_buildconfig_config, PLUGIN, BUILDCONFIG_CONFIG, FoundryConfig)

FoundryConfig  *plugin_buildconfig_config_new           (FoundryContext          *context,
                                                         GKeyFile                *key_file,
                                                         const char              *group);
char          **plugin_buildconfig_config_dup_prebuild  (PluginBuildconfigConfig *self);
char          **plugin_buildconfig_config_dup_postbuild (PluginBuildconfigConfig *self);

G_END_DECLS
