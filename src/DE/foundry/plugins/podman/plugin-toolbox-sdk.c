/* plugin-toolbox-sdk.c
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

#include "config.h"

#include "plugin-toolbox-sdk.h"

struct _PluginToolboxSdk
{
  PluginPodmanSdk parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginToolboxSdk, plugin_toolbox_sdk, PLUGIN_TYPE_PODMAN_SDK)

static void
plugin_toolbox_sdk_finalize (GObject *object)
{
  G_OBJECT_CLASS (plugin_toolbox_sdk_parent_class)->finalize (object);
}

static void
plugin_toolbox_sdk_class_init (PluginToolboxSdkClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_toolbox_sdk_finalize;
}

static void
plugin_toolbox_sdk_init (PluginToolboxSdk *self)
{
  foundry_sdk_set_kind (FOUNDRY_SDK (self), "toolbox");
}
