#include <tracker-common.h>

#include <fcntl.h>

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	TrackerResource *resource;
	g_autoptr (GError) inner_error = NULL;
	int wait_status;
	gboolean retval;

	/* Check that child processes are also constrained */
	if (g_spawn_command_line_sync ("/bin/true",
	                               NULL, NULL, &wait_status,
	                               NULL)) {
		retval = g_spawn_check_wait_status (wait_status, &inner_error);
		if (retval)
			goto fail;
		if (!inner_error)
			goto fail;
	}

	return TRUE;
 fail:
	/* Hint unexpected success with a fail:// resource */
	resource = tracker_resource_new ("fail://");
	tracker_resource_add_uri (resource, "rdf:type", "rdfs:Resource");
	tracker_extract_info_set_resource (info, resource);
	return TRUE;
}
