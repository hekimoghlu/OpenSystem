#include <tracker-common.h>

#include <fcntl.h>

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	TrackerResource *resource;
	g_autofree gchar *home_parent = NULL;
	int fd;

	/* Attempt to read files from disallowed locations */
	fd = open ("/proc/cmdline", O_RDONLY);
	if (fd >= 0)
		goto fail;

	fd = open ("/etc/motd", O_RDONLY);
	if (fd >= 0)
		goto fail;

	home_parent = g_path_get_dirname (g_get_home_dir ());
	fd = open (home_parent, O_RDONLY | O_DIRECTORY);
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
