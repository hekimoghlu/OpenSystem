/* plugin-jhbuild-sdk.h
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

#define PLUGIN_TYPE_JHBUILD_SDK (plugin_jhbuild_sdk_get_type())

G_DECLARE_FINAL_TYPE (PluginJhbuildSdk, plugin_jhbuild_sdk, PLUGIN, JHBUILD_SDK, FoundrySdk)

FoundrySdk *plugin_jhbuild_sdk_new                (FoundryContext   *context,
                                                   const char       *install_prefix,
                                                   const char       *library_dir);
char       *plugin_jhbuild_sdk_dup_install_prefix (PluginJhbuildSdk *self);

G_END_DECLS
