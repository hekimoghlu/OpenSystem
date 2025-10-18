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

#include "mock-volume-monitor.h"
#include "mock-drive.h"
#include "mock-volume.h"
#include "mock-mount.h"

typedef enum {
	MOUNT_FLAG_NONE = 0,
	MOUNT_FLAG_NON_REMOVABLE = 1 << 0,
} MountFlags;

typedef enum {
	UNMOUNT_FLAG_NONE = 0,
	UNMOUNT_FLAG_EMULATE_BUSY = 1 << 0,
} UnmountFlags;

struct _MockVolumeMonitor {
	GNativeVolumeMonitor parent;

	GDBusNodeInfo *node_info;
	guint bus_token;
	guint object_token;

	GList *drives;
	GList *volumes;
	GList *mounts;

	gint counter;
};

G_DEFINE_TYPE (MockVolumeMonitor, mock_volume_monitor, G_TYPE_NATIVE_VOLUME_MONITOR);


static void
add_mock_mount (MockVolumeMonitor *self,
                const gchar       *uri,
                uint32_t           flags)
{
	int id = self->counter ++;
	g_autofree gchar *drive_name;
	g_autofree gchar *volume_name;
	g_autofree gchar *mount_name;
	MockDrive *drive;
	MockVolume *volume;
	MockMount *mount;
	g_autoptr(GFile) root = NULL;

	g_message ("Adding mock mount at '%s'", uri);

	drive_name = g_strdup_printf ("MockDrive%i", id);
	volume_name = g_strdup_printf ("MockVolume%i", id);
	mount_name = g_strdup_printf ("MockMount%i", id);

	root = g_file_new_for_uri (uri);

	drive = mock_drive_new (self, drive_name,
	                        (flags & MOUNT_FLAG_NON_REMOVABLE) == 0);
	volume = mock_volume_new (self, drive, volume_name);
	mount = mock_mount_new (self, volume, mount_name, root);

	self->drives = g_list_prepend (self->drives, drive);
	self->volumes = g_list_prepend (self->volumes, volume);
	self->mounts = g_list_prepend (self->mounts, mount);

	g_signal_emit_by_name (self, "drive-connected", drive);
	g_signal_emit_by_name (self, "volume-added", volume);
	g_signal_emit_by_name (self, "mount-added", mount);
}

static gint
match_mount_by_root (gconstpointer a, gconstpointer b)
{
	GMount *mount = G_MOUNT (a);
	GFile *root = G_FILE (b);

	return !g_file_equal (g_mount_get_root (mount), root);
}

static void
remove_mock_mount (MockVolumeMonitor *self,
                   const gchar       *uri,
                   UnmountFlags       flags)
{
	GList *node;
	g_autoptr(GDrive) drive = NULL;
	g_autoptr(GVolume) volume = NULL;
	g_autoptr(MockMount) mount = NULL;
	g_autoptr(GFile) root = NULL;

	g_message ("Removing mock mount at '%s'", uri);

	root = g_file_new_for_uri (uri);
	node = g_list_find_custom (self->mounts, root, match_mount_by_root);

	if (!node) {
		g_warning ("No mount found with root %s", uri);
		return;
	}

	volume = g_mount_get_volume (G_MOUNT (node->data));
	drive = g_volume_get_drive (volume);

	g_signal_emit_by_name (node->data, "pre-unmount");
	g_signal_emit_by_name (self, "mount-pre-unmount", node->data);

	if ((flags & UNMOUNT_FLAG_EMULATE_BUSY) != 0) {
		return;
	}

	mount = MOCK_MOUNT (node->data);
	mock_mount_unmounted (mount);
	mock_volume_removed (MOCK_VOLUME (volume));
	mock_drive_disconnected (MOCK_DRIVE (drive));

	g_signal_emit_by_name (self, "mount-removed", mount);
	g_signal_emit_by_name (self, "volume-removed", volume);
	g_signal_emit_by_name (self, "drive-disconnected", drive);

	self->drives = g_list_remove (self->drives, drive);
	self->volumes = g_list_remove (self->volumes, volume);
	self->mounts = g_list_remove (self->mounts, mount);
}


/* DBus interface
 * --------------
 *
 * Used by tests to control the MockVolumeMonitor object.
 */

#define BUS_NAME "org.freedesktop.Tracker3.MockVolumeMonitor"
#define BUS_PATH "/org/freedesktop/Tracker3/MockVolumeMonitor"

static const gchar dbus_xml[] =
	"<node>"
	"  <interface name='org.freedesktop.Tracker3.MockVolumeMonitor'>"
	"    <method name='AddMount'>"
	"      <arg type='s' name='path' direction='in' />"
	"      <arg type='u' name='flags' direction='in' />"
	"    </method>"
	"    <method name='RemoveMount'>"
	"      <arg type='s' name='path' direction='in' />"
	"      <arg type='u' name='flags' direction='in' />"
	"    </method>"
	"  </interface>"
	"</node>";


static void
on_dbus_method_call (GDBusConnection       *connection,
                     const gchar           *sender,
                     const gchar           *object_path,
                     const gchar           *interface_name,
                     const gchar           *method_name,
                     GVariant              *parameters,
                     GDBusMethodInvocation *invocation,
                     gpointer               user_data)
{
	MockVolumeMonitor *self = MOCK_VOLUME_MONITOR (user_data);

	if (g_strcmp0 (method_name, "AddMount") == 0) {
		g_autofree gchar *uri = NULL;
		uint32_t flags;

		g_variant_get (parameters, "(su)", &uri, &flags);

		add_mock_mount (self, uri, flags);

		g_dbus_method_invocation_return_value (invocation, NULL);
	} else if (g_strcmp0 (method_name, "RemoveMount") == 0) {
		g_autofree gchar *uri = NULL;
		uint32_t flags;

		g_variant_get (parameters, "(su)", &uri, &flags);

		remove_mock_mount (self, uri, flags);

		g_dbus_method_invocation_return_value (invocation, NULL);
	} else {
		g_dbus_method_invocation_return_error (invocation,
		                                       G_DBUS_ERROR,
		                                       G_DBUS_ERROR_UNKNOWN_METHOD,
		                                       "Unknown method on DBus interface: '%s'", method_name);
	}

}

static void
on_bus_acquired (GDBusConnection *connection, const gchar *name, gpointer user_data)
{
	MockVolumeMonitor *self = MOCK_VOLUME_MONITOR (user_data);
	GError *error = NULL;

	g_message ("Publishing DBus object at " BUS_PATH);

	self->node_info= g_dbus_node_info_new_for_xml (dbus_xml, &error);
	g_assert_no_error (error);

	self->object_token =
		g_dbus_connection_register_object (connection,
		                                   BUS_PATH,
		                                   self->node_info->interfaces[0],
		                                   &(GDBusInterfaceVTable) { on_dbus_method_call, NULL, NULL },
		                                   self,
		                                   NULL,
		                                   &error);
}

static void
on_bus_name_acquired (GDBusConnection *connection,
                      const gchar *name,
                      gpointer user_data)
{
	g_message ("Acquired name: %s", name);
}

static void
on_bus_name_lost (GDBusConnection *connection,
                  const gchar *name,
                  gpointer user_data)
{
	MockVolumeMonitor *self = MOCK_VOLUME_MONITOR (user_data);
	g_message ("Lost name: %s", name);

	g_dbus_connection_unregister_object (connection, self->object_token);
	self->object_token = 0;
}

/* GVolumeMonitor implementation
 * -----------------------------
 *
 * We inject fake mounts into current process by replacing the usual
 * GNativeVolumeMonitor implementation with this.
 */

static gboolean
is_supported (void)
{
	g_debug (__func__);
	return TRUE;
}

static GList *
get_connected_drives (GVolumeMonitor *volume_monitor)
{
	MockVolumeMonitor *self = MOCK_VOLUME_MONITOR (volume_monitor);
	return self->drives;
}

static GList *
get_volumes (GVolumeMonitor *volume_monitor)
{
	MockVolumeMonitor *self = MOCK_VOLUME_MONITOR (volume_monitor);
	return self->volumes;
}

static GList *
get_mounts (GVolumeMonitor *volume_monitor)
{
	MockVolumeMonitor *self = MOCK_VOLUME_MONITOR (volume_monitor);
	return self->mounts;
}

static GVolume *
get_volume_for_uuid (GVolumeMonitor *volume_monitor,
                     const char     *uuid)
{
	return NULL;

}

static GMount *
get_mount_for_uuid (GVolumeMonitor *volume_monitor,
                    const char     *uuid)
{
	return NULL;
}

static GVolume *
adopt_orphan_mount (GMount         *mount,
                    GVolumeMonitor *volume_monitor)
{
	return NULL;
}

void
drive_eject_button (GVolumeMonitor *volume_monitor,
                    GDrive         *drive)
{
}

void
drive_stop_button (GVolumeMonitor *volume_monitor,
                   GDrive         *drive)
{
}


static void mock_volume_monitor_init (MockVolumeMonitor *self) {
	self->bus_token = g_bus_own_name (G_BUS_TYPE_SESSION, BUS_NAME, G_BUS_NAME_OWNER_FLAGS_NONE,
	                                  on_bus_acquired,
	                                  on_bus_name_acquired,
	                                  on_bus_name_lost,
	                                  g_object_ref (self),
	                                  g_object_unref);
}

static void mock_volume_monitor_class_init (MockVolumeMonitorClass *klass) {
	GVolumeMonitorClass *monitor_class = G_VOLUME_MONITOR_CLASS (klass);

	monitor_class->is_supported = is_supported;
	monitor_class->get_connected_drives = get_connected_drives;
	monitor_class->get_volumes = get_volumes;
	monitor_class->get_mounts = get_mounts;
	monitor_class->get_volume_for_uuid = get_volume_for_uuid;
	monitor_class->get_mount_for_uuid = get_mount_for_uuid;
	monitor_class->adopt_orphan_mount = adopt_orphan_mount;
	monitor_class->drive_eject_button = drive_eject_button;
	monitor_class->drive_stop_button = drive_stop_button;
}

void g_io_module_load (GIOModule *module) {
	g_debug (__func__);

	g_type_module_use (G_TYPE_MODULE (module));

	g_io_extension_point_implement (
	    G_NATIVE_VOLUME_MONITOR_EXTENSION_POINT_NAME,
	    MOCK_TYPE_VOLUME_MONITOR, "mockvolumemonitor", 0);
}

void g_io_module_unload (GIOModule *module) {
}
