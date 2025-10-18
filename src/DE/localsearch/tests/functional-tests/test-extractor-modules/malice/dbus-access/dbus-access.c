#include <tracker-common.h>

#include <gio/gio.h>

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	TrackerResource *resource;
	g_autoptr (GDBusConnection) conn = NULL;
	g_autoptr (GError) dbus_error = NULL;

	/* Attempt to open a dbus connection */
	conn = g_bus_get_sync (G_BUS_TYPE_SESSION, NULL, &dbus_error);
	if (conn || !g_error_matches (dbus_error, G_IO_ERROR, G_IO_ERROR_PERMISSION_DENIED))
		goto fail;

	return TRUE;

 fail:
	/* Hint unexpected success with a fail:// resource */
	resource = tracker_resource_new ("fail://");
	tracker_resource_add_uri (resource, "rdf:type", "rdfs:Resource");
	tracker_extract_info_set_resource (info, resource);
	return TRUE;
}
