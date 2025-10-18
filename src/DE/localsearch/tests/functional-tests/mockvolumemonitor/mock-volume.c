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

#include "mock-volume.h"
#include "mock-mount.h"
#include "mock-drive.h"

struct _MockVolume
{
	GObject parent;

	MockVolumeMonitor *monitor; /* owned by volume monitor */
	MockMount         *mount;   /* owned by volume monitor */
	MockDrive         *drive;   /* owned by volume monitor */

	gchar *name;
	gchar *uuid;
};

static void mock_volume_volume_iface_init (GVolumeIface *iface);

G_DEFINE_TYPE_EXTENDED (MockVolume, mock_volume, G_TYPE_OBJECT, 0,
                        G_IMPLEMENT_INTERFACE (G_TYPE_VOLUME, mock_volume_volume_iface_init))

static void
mock_volume_finalize (GObject *object)
{
	MockVolume *volume = MOCK_VOLUME (object);

	if (volume->mount != NULL) {
		mock_mount_unset_volume (volume->mount, volume);
	}

	if (volume->drive != NULL) {
		mock_drive_unset_volume (volume->drive, volume);
	}

	g_free (volume->name);
	g_free (volume->uuid);

	G_OBJECT_CLASS (mock_volume_parent_class)->finalize (object);
}

static void
mock_volume_class_init (MockVolumeClass *klass)
{
	GObjectClass *gobject_class;

	gobject_class = G_OBJECT_CLASS (klass);
	gobject_class->finalize = mock_volume_finalize;
}

static void
mock_volume_init (MockVolume *volume)
{
}

static void
emit_changed (MockVolume *volume)
{
	g_signal_emit_by_name (volume, "changed");
	g_signal_emit_by_name (volume->monitor, "volume-changed", volume);
}


MockVolume *
mock_volume_new (MockVolumeMonitor   *monitor,
                 MockDrive           *drive,
                 const gchar         *name)
{
	MockVolume *volume;

	volume = g_object_new (MOCK_TYPE_VOLUME, NULL);
	volume->monitor = monitor;

	volume->drive = drive;
	if (drive != NULL)
		mock_drive_set_volume (drive, volume);

	volume->name = g_strdup (name);
	volume->uuid = g_uuid_string_random ();

	return volume;
}

void
mock_volume_removed (MockVolume *volume)
{
	if (volume->mount != NULL) {
		mock_mount_unset_volume (volume->mount, volume);
		volume->mount = NULL;
	}

	if (volume->drive != NULL) {
		mock_drive_unset_volume (volume->drive, volume);
		volume->drive = NULL;
	}
}

void
mock_volume_set_mount (MockVolume *volume,
                       MockMount  *mount)
{
	if (volume->mount != mount) {
		if (volume->mount != NULL)
			mock_mount_unset_volume (volume->mount, volume);

		volume->mount = mount;

		emit_changed (volume);
	}
}

void
mock_volume_unset_mount (MockVolume *volume,
                         MockMount  *mount)
{
	if (volume->mount == mount) {
		volume->mount = NULL;
		emit_changed (volume);
	}
}

void
mock_volume_set_drive (MockVolume *volume,
                       MockDrive  *drive)
{
	if (volume->drive != drive) {
		if (volume->drive != NULL)
			mock_drive_unset_volume (volume->drive, volume);
		volume->drive = drive;
		emit_changed (volume);
	}
}

void
mock_volume_unset_drive (MockVolume *volume,
                         MockDrive  *drive)
{
	if (volume->drive == drive) {
		volume->drive = NULL;
		emit_changed (volume);
	}
}

static GIcon *
mock_volume_get_icon (GVolume *_volume)
{
	return NULL;
}

static GIcon *
mock_volume_get_symbolic_icon (GVolume *_volume)
{
	return NULL;
}

static char *
mock_volume_get_name (GVolume *_volume)
{
	MockVolume *volume = MOCK_VOLUME (_volume);
	return g_strdup (volume->name);
}

static char *
mock_volume_get_uuid (GVolume *_volume)
{
	MockVolume *volume = MOCK_VOLUME (_volume);
	return g_strdup (volume->uuid);
}

static gboolean
mock_volume_can_mount (GVolume *_volume)
{
	return TRUE;
}

static gboolean
mock_volume_can_eject (GVolume *_volume)
{
	return FALSE;
}

static gboolean
mock_volume_should_automount (GVolume *_volume)
{
	return TRUE;
}

static GDrive *
mock_volume_get_drive (GVolume *_volume)
{
	MockVolume *volume = MOCK_VOLUME (_volume);
	GDrive *drive = NULL;

	if (volume->drive != NULL)
		drive = G_DRIVE (g_object_ref (volume->drive));
	return drive;
}

static GMount *
mock_volume_get_mount (GVolume *_volume)
{
	MockVolume *volume = MOCK_VOLUME (_volume);
	GMount *mount = NULL;

	if (volume->mount != NULL)
		mount = G_MOUNT (g_object_ref (volume->mount));
	return mount;
}

static gchar *
mock_volume_get_identifier (GVolume      *_volume,
                            const gchar  *kind)
{
	return g_strdup ("device");
}

static gchar **
mock_volume_enumerate_identifiers (GVolume *_volume)
{
	return NULL;
}

static GFile *
mock_volume_get_activation_root (GVolume *_volume)
{
	return NULL;
}

static const gchar *
mock_volume_get_sort_key (GVolume *_volume)
{
	return NULL;
}

static void
mock_volume_volume_iface_init (GVolumeIface *iface)
{
	iface->get_name = mock_volume_get_name;
	iface->get_icon = mock_volume_get_icon;
	iface->get_symbolic_icon = mock_volume_get_symbolic_icon;
	iface->get_uuid = mock_volume_get_uuid;
	iface->get_drive = mock_volume_get_drive;
	iface->get_mount = mock_volume_get_mount;
	iface->can_mount = mock_volume_can_mount;
	iface->can_eject = mock_volume_can_eject;
	iface->should_automount = mock_volume_should_automount;
	iface->get_activation_root = mock_volume_get_activation_root;
	iface->enumerate_identifiers = mock_volume_enumerate_identifiers;
	iface->get_identifier = mock_volume_get_identifier;

	iface->get_sort_key = mock_volume_get_sort_key;
}
