/* plugin-host-sdk.h
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

#define PLUGIN_TYPE_HOST_SDK (plugin_host_sdk_get_type())

G_DECLARE_FINAL_TYPE (PluginHostSdk, plugin_host_sdk, PLUGIN, HOST_SDK, FoundrySdk)

FoundrySdk *plugin_host_sdk_new            (FoundryContext *context);
char       *plugin_host_sdk_build_filename (PluginHostSdk  *self,
                                            const char     *first_element,
                                            ...) G_GNUC_NULL_TERMINATED;

G_END_DECLS
