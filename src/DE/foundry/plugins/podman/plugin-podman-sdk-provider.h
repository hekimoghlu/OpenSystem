/* plugin-podman-sdk-provider.h
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

#define PLUGIN_TYPE_PODMAN_SDK_PROVIDER (plugin_podman_sdk_provider_get_type())

G_DECLARE_FINAL_TYPE (PluginPodmanSdkProvider, plugin_podman_sdk_provider, PLUGIN, PODMAN_SDK_PROVIDER, FoundrySdkProvider)

DexFuture *plugin_podman_sdk_provider_check_version (guint                    major,
                                                     guint                    minor,
                                                     guint                    micro);
DexFuture *plugin_podman_sdk_provider_update        (PluginPodmanSdkProvider *self);
void       plugin_podman_sdk_provider_queue_update  (PluginPodmanSdkProvider *self);

G_END_DECLS
