/*
 * Copyright (C) 2025, Red Hat Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 *
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */

#include "config-miners.h"

#include "tracker-indexing-tree-methods.h"

#include <gio/gunixmounts.h>

#if !GLIB_CHECK_VERSION (2, 83, 0)
#define g_unix_mount_entry_for g_unix_mount_for
#define g_unix_mount_entry_at g_unix_mount_at
#define g_unix_mount_entry_get_device_path g_unix_mount_get_device_path
#endif

char *
tracker_indexing_tree_get_root_id (TrackerIndexingTree *tree,
                                   GFile               *root,
                                   GUdevClient         *udev_client)
{
	g_autofree char *path = NULL, *expanded = NULL;
	const char *devpath = NULL, *id = NULL;
	GUnixMountEntry *mount = NULL;
	GUdevDevice *udev_device = NULL;

	path = g_file_get_path (root);
	expanded = realpath (path, NULL);

	mount = g_unix_mount_entry_at (expanded ? expanded : path, NULL);

	if (!mount)
		mount = g_unix_mount_entry_for (expanded ? expanded : path, NULL);

	if (mount)
		devpath = g_unix_mount_entry_get_device_path (mount);

	if (devpath) {
		udev_device = g_udev_client_query_by_device_file (udev_client, devpath);

		if (udev_device) {
			id = g_udev_device_get_property (udev_device, "ID_FS_UUID_SUB");
			if (!id)
				id = g_udev_device_get_property (udev_device, "ID_FS_UUID");
		}
	}

	return g_strdup (id);
}
