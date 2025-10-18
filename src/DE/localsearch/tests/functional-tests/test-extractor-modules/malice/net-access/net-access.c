#include <tracker-common.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	TrackerResource *resource;
	int fd;

	/* Try to get sockets of different families/types */
	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd >= 0)
		goto fail;

	fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd >= 0)
		goto fail;

	fd = socket(AF_INET, SOCK_DGRAM, 0);
	if (fd >= 0)
		goto fail;

	fd = socket(AF_INET6, SOCK_STREAM, 0);
	if (fd >= 0)
		goto fail;

	fd = socket(AF_INET6, SOCK_DGRAM, 0);
	if (fd >= 0)
		goto fail;

	fd = socket(AF_NETLINK, SOCK_STREAM, 0);
	if (fd >= 0)
		goto fail;

	return TRUE;

 fail:
	/* Hint unexpected success with a fail:// resource */
	resource = tracker_resource_new ("fail://");
	tracker_resource_add_uri (resource, "rdf:type", "rdfs:Resource");
	tracker_extract_info_set_resource (info, resource);
	close (fd);
	return TRUE;
}
