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

#include "mock-drive.h"

struct _MockDrive
{
	GObject parent;

	MockVolumeMonitor  *monitor; /* owned by volume monitor */
	GList              *volumes; /* entries in list are owned by volume monitor */

	const gchar *name;
	gboolean removable;
};

static void mock_drive_drive_iface_init (GDriveIface *iface);

G_DEFINE_TYPE_EXTENDED (MockDrive, mock_drive, G_TYPE_OBJECT, 0,
	                      G_IMPLEMENT_INTERFACE (G_TYPE_DRIVE, mock_drive_drive_iface_init))

static void
mock_drive_finalize (GObject *object)
{
	MockDrive *drive = MOCK_DRIVE (object);
	GList *l;

	for (l = drive->volumes; l != NULL; l = l->next) {
		MockVolume *volume = l->data;
		mock_volume_unset_drive (volume, drive);
	}

	G_OBJECT_CLASS (mock_drive_parent_class)->finalize (object);
}

static void
mock_drive_class_init (MockDriveClass *klass)
{
	GObjectClass *gobject_class;

	gobject_class = G_OBJECT_CLASS (klass);
	gobject_class->finalize = mock_drive_finalize;
}

static void
mock_drive_init (MockDrive *drive)
{
}

static void
emit_changed (MockDrive *drive)
{
	g_signal_emit_by_name (drive, "changed");
	g_signal_emit_by_name (drive->monitor, "drive-changed", drive);
}


MockDrive *
mock_drive_new (MockVolumeMonitor *monitor,
                const gchar       *name,
                gboolean           removable)
{
	MockDrive *drive;

	drive = g_object_new (MOCK_TYPE_DRIVE, NULL);
	drive->monitor = monitor;
	drive->name = g_strdup (name);
	drive->removable = removable;

	return drive;
}

void
mock_drive_disconnected (MockDrive *drive)
{
	GList *l, *volumes;

	volumes = drive->volumes;
	drive->volumes = NULL;
	for (l = volumes; l != NULL; l = l->next) {
		MockVolume *volume = l->data;
		mock_volume_unset_drive (volume, drive);
	}
	g_list_free (volumes);
}

void
mock_drive_set_volume (MockDrive  *drive,
                       MockVolume *volume)
{
	if (g_list_find (drive->volumes, volume) == NULL) {
		drive->volumes = g_list_prepend (drive->volumes, volume);
		emit_changed (drive);
	}
}

void
mock_drive_unset_volume (MockDrive  *drive,
	                               MockVolume *volume)
{
	GList *l;
	l = g_list_find (drive->volumes, volume);
	if (l != NULL)
	  {
	    drive->volumes = g_list_delete_link (drive->volumes, l);
	    emit_changed (drive);
	  }
}

static GIcon *
mock_drive_get_icon (GDrive *_drive)
{
	return NULL;
}

static GIcon *
mock_drive_get_symbolic_icon (GDrive *_drive)
{
	return NULL;
}

static char *
mock_drive_get_name (GDrive *_drive)
{
	MockDrive *drive = MOCK_DRIVE (_drive);
	return g_strdup (drive->name);
}

static GList *
mock_drive_get_volumes (GDrive *_drive)
{
	MockDrive *drive = MOCK_DRIVE (_drive);
	GList *l;
	l = g_list_copy (drive->volumes);
	g_list_foreach (l, (GFunc) g_object_ref, NULL);
	return l;
}

static gboolean
mock_drive_has_volumes (GDrive *_drive)
{
	MockDrive *drive = MOCK_DRIVE (_drive);
	gboolean res;
	res = drive->volumes != NULL;
	return res;
}

static gboolean
mock_drive_is_removable (GDrive *_drive)
{
	MockDrive *drive = MOCK_DRIVE (_drive);

	return drive->removable;
}

static gboolean
mock_drive_is_media_removable (GDrive *_drive)
{
	MockDrive *drive = MOCK_DRIVE (_drive);

	return drive->removable;
}

static gboolean
mock_drive_has_media (GDrive *_drive)
{
	return TRUE;
}

static gboolean
mock_drive_is_media_check_automatic (GDrive *_drive)
{
	return TRUE;
}

static gboolean
mock_drive_can_eject (GDrive *_drive)
{
	return FALSE;
}

static gboolean
mock_drive_can_poll_for_media (GDrive *_drive)
{
	return FALSE;
}

static gboolean
mock_drive_can_start (GDrive *_drive)
{
	return FALSE;
}

static gboolean
mock_drive_can_start_degraded (GDrive *_drive)
{
	return FALSE;
}

static gboolean
mock_drive_can_stop (GDrive *_drive)
{
	return FALSE;
}

static GDriveStartStopType
mock_drive_get_start_stop_type (GDrive *_drive)
{
	return G_DRIVE_START_STOP_TYPE_SHUTDOWN;
}

/* ---------------------------------------------------------------------------------------------------- */

static char *
mock_drive_get_identifier (GDrive      *_drive,
                           const gchar *kind)
{
	return NULL;
}

static gchar **
mock_drive_enumerate_identifiers (GDrive *_drive)
{
	return NULL;
}

static const gchar *
mock_drive_get_sort_key (GDrive *_drive)
{
	return NULL;
}

static void
mock_drive_drive_iface_init (GDriveIface *iface)
{
	iface->get_name = mock_drive_get_name;
	iface->get_icon = mock_drive_get_icon;
	iface->get_symbolic_icon = mock_drive_get_symbolic_icon;
	iface->has_volumes = mock_drive_has_volumes;
	iface->get_volumes = mock_drive_get_volumes;
	iface->is_removable = mock_drive_is_removable;
	iface->is_media_removable = mock_drive_is_media_removable;
	iface->has_media = mock_drive_has_media;
	iface->is_media_check_automatic = mock_drive_is_media_check_automatic;
	iface->can_eject = mock_drive_can_eject;
	iface->can_poll_for_media = mock_drive_can_poll_for_media;
	iface->get_identifier = mock_drive_get_identifier;
	iface->enumerate_identifiers = mock_drive_enumerate_identifiers;
	iface->get_start_stop_type = mock_drive_get_start_stop_type;
	iface->can_start = mock_drive_can_start;
	iface->can_start_degraded = mock_drive_can_start_degraded;
	iface->can_stop = mock_drive_can_stop;
	iface->get_sort_key = mock_drive_get_sort_key;
}
