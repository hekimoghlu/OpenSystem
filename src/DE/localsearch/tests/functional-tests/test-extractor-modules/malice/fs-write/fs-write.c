#include <tracker-common.h>

#include <fcntl.h>

static int
try_open (const gchar *path)
{
	int fd;

	fd = open (path, O_RDWR);
	if (fd >= 0)
		return fd;

	fd = open (path, O_WRONLY);
	if (fd >= 0)
		return fd;

	return -1;
}

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	TrackerResource *resource;
	g_autofree gchar *tmpfile = NULL, *file = NULL;
	int fd;

	/* Attempt to open files with write permissions */
	tmpfile = g_build_filename (g_get_tmp_dir (), "bwahaha.txt", NULL);
	fd = try_open (tmpfile);
	if (fd >= 0)
		goto fail;
	if (g_file_test (tmpfile, G_FILE_TEST_EXISTS))
		return fd;

	/* Attempt to open files with write permissions */
	file = g_file_get_path (tracker_extract_info_get_file (info));
	fd = try_open (file);
	if (fd >= 0)
		goto fail;

	return TRUE;
 fail:
	close (fd);
	/* Hint unexpected success with a fail:// resource */
	resource = tracker_resource_new ("fail://");
	tracker_resource_add_uri (resource, "rdf:type", "rdfs:Resource");
	tracker_extract_info_set_resource (info, resource);
	return TRUE;
}
