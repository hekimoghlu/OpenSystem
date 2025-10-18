/*
 * Copyright (C) 2021, Codethink Ltd
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 *
 * Author: Sam Thursfield <sam@afuera.me.uk>
 */

#include <glib.h>
#include <gio/gio.h>

#include "mock-mount.h"
#include "mock-volume.h"

struct _MockMount
{
	GObject parent;

	MockVolumeMonitor *monitor; /* owned by volume monitor */

	/* may be NULL */
	MockVolume        *volume;  /* owned by volume monitor */

	gchar *name;
	GFile *root;
};

static void mock_mount_mount_iface_init (GMountIface *iface);

G_DEFINE_TYPE_EXTENDED (MockMount, mock_mount, G_TYPE_OBJECT, 0,
                        G_IMPLEMENT_INTERFACE (G_TYPE_MOUNT,
                                               mock_mount_mount_iface_init))

static void on_volume_changed (GVolume *volume, gpointer user_data);

static void
mock_mount_finalize (GObject *object)
{
	MockMount *mount = MOCK_MOUNT (object);

	if (mount->volume != NULL) {
		g_signal_handlers_disconnect_by_func (mount->volume, on_volume_changed, mount);
		mock_volume_unset_mount (mount->volume, mount);
	}

	g_free (mount->name);
	g_clear_object (&mount->root);

	G_OBJECT_CLASS (mock_mount_parent_class)->finalize (object);
}

static void
mock_mount_class_init (MockMountClass *klass)
{
	GObjectClass *gobject_class;

	gobject_class = G_OBJECT_CLASS (klass);
	gobject_class->finalize = mock_mount_finalize;
}

static void
mock_mount_init (MockMount *mount)
{
}

static void
emit_changed (MockMount *mount)
{
	g_signal_emit_by_name (mount, "changed");
	g_signal_emit_by_name (mount->monitor, "mount-changed", mount);
}

static void
on_volume_changed (GVolume  *volume,
                   gpointer  user_data)
{
	MockMount *mount = MOCK_MOUNT (user_data);
	emit_changed (mount);
}

MockMount *
mock_mount_new (MockVolumeMonitor *monitor,
                MockVolume        *volume,
                const gchar       *name,
                GFile             *root)
{
	MockMount *mount = NULL;

	mount = g_object_new (MOCK_TYPE_MOUNT, NULL);
	mount->monitor = monitor;
	mount->name = g_strdup (name);
	mount->root = g_object_ref (root);

	/* need to set the volume only when the mount is fully constructed */
	mount->volume = volume;
	if (mount->volume != NULL) {
		mock_volume_set_mount (volume, mount);
		/* this is for piggy backing on the name and icon of the associated volume */
		g_signal_connect (mount->volume, "changed", G_CALLBACK (on_volume_changed), mount);
	}

	return mount;
}

void
mock_mount_unmounted (MockMount *mount)
{
	if (mount->volume != NULL) {
		mock_volume_unset_mount (mount->volume, mount);
		g_signal_handlers_disconnect_by_func (mount->volume, on_volume_changed, mount);
		mount->volume = NULL;
		emit_changed (mount);
	}
}

void
mock_mount_unset_volume (MockMount   *mount,
                         MockVolume  *volume)
{
	if (mount->volume == volume) {
		g_signal_handlers_disconnect_by_func (mount->volume, on_volume_changed, mount);
		mount->volume = NULL;
		emit_changed (mount);
	}
}

void
mock_mount_set_volume (MockMount   *mount,
                       MockVolume  *volume)
{
	if (mount->volume != volume) {
		if (mount->volume != NULL)
			mock_mount_unset_volume (mount, mount->volume);
		mount->volume = volume;
		if (mount->volume != NULL) {
			mock_volume_set_mount (volume, mount);
			/* this is for piggy backing on the name and icon of the associated volume */
			g_signal_connect (mount->volume, "changed", G_CALLBACK (on_volume_changed), mount);
		}
		emit_changed (mount);
  	}
}

static GFile *
mock_mount_get_root (GMount *_mount)
{
	MockMount *mount = MOCK_MOUNT (_mount);
	return g_object_ref (mount->root);
}

static GIcon *
mock_mount_get_icon (GMount *_mount)
{
	return NULL;
}

static GIcon *
mock_mount_get_symbolic_icon (GMount *_mount)
{
	return NULL;
}

static gchar *
mock_mount_get_uuid (GMount *_mount)
{
	return NULL;
}

static gchar *
mock_mount_get_name (GMount *_mount)
{
	MockMount *mount = MOCK_MOUNT (_mount);
	return g_strdup (mount->name);
}

gboolean
mock_mount_has_uuid (MockMount *_mount,
                     const gchar      *uuid)
{
	return FALSE;
}

const gchar *
mock_mount_get_mount_path (MockMount *mount)
{
	return NULL;
}

GUnixMountEntry *
mock_mount_get_mount_entry (MockMount *_mount)
{
	return NULL;
}

static GDrive *
mock_mount_get_drive (GMount *_mount)
{
	MockMount *mount = MOCK_MOUNT (_mount);
	GDrive *drive = NULL;

	if (mount->volume != NULL)
		drive = g_volume_get_drive (G_VOLUME (mount->volume));
	return drive;
}

static GVolume *
mock_mount_get_volume_ (GMount *_mount)
{
	MockMount *mount = MOCK_MOUNT (_mount);
	GVolume *volume = NULL;

	if (mount->volume != NULL)
		volume = G_VOLUME (g_object_ref (mount->volume));
	return volume;
}

static gboolean
mock_mount_can_unmount (GMount *_mount)
{
	return TRUE;
}

static gboolean
mock_mount_can_eject (GMount *_mount)
{
	return FALSE;
}

static const gchar *
mock_mount_get_sort_key (GMount *_mount)
{
	return NULL;
}

static void
mock_mount_mount_iface_init (GMountIface *iface)
{
	iface->get_root = mock_mount_get_root;
	iface->get_name = mock_mount_get_name;
	iface->get_icon = mock_mount_get_icon;
	iface->get_symbolic_icon = mock_mount_get_symbolic_icon;
	iface->get_uuid = mock_mount_get_uuid;
	iface->get_drive = mock_mount_get_drive;
	iface->get_volume = mock_mount_get_volume_;
	iface->can_unmount = mock_mount_can_unmount;
	iface->can_eject = mock_mount_can_eject;
	iface->get_sort_key = mock_mount_get_sort_key;
}

MockVolume *
mock_mount_get_volume (MockMount *mount)
{
	return mount->volume;
}
