#include <tracker-common.h>

#include <fcntl.h>

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	TrackerResource *resource;
	g_autofree gchar *path = NULL;
	int fd;

	path = g_file_get_path (tracker_extract_info_get_file (info));
	/* Attempt to truncate the file */
	fd = open (path, O_RDONLY | O_TRUNC);
	if (fd >= 0)
		goto fail;

	return TRUE;
 fail:
	/* Hint unexpected success with a fail:// resource */
	resource = tracker_resource_new ("fail://");
	tracker_resource_add_uri (resource, "rdf:type", "rdfs:Resource");
	tracker_extract_info_set_resource (info, resource);
	return TRUE;
}
