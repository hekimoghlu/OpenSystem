/*
 * Copyright (C) 2008, Nokia <ivan.frade@nokia.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include "config-miners.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

#include <glib.h>
#include <glib/gstdio.h>

#include <gio/gio.h>

#include <tracker-common.h>

#include "utils/tracker-extract.h"

#include "tracker-main.h"
#include "tracker-extract.h"
#include "tracker-read.h"

static gchar *
get_file_content (GFile   *file,
                  gsize    n_bytes,
                  GError **error)
{
	gchar *text, *uri, *path;
	int fd;

	uri = g_file_get_uri (file);

	/* Get filename from URI */
	path = g_file_get_path (file);

	fd = tracker_file_open_fd (path);

	if (fd == -1) {
		g_set_error (error, TRACKER_EXTRACT_ERROR, TRACKER_EXTRACT_ERROR_IO_ERROR,
		             "Could not open file '%s': %s", uri, g_strerror (errno));
		g_free (uri);
		g_free (path);
		return NULL;
	}

	g_debug ("  Starting to read '%s' up to %" G_GSIZE_FORMAT " bytes...",
	         uri, n_bytes);

	/* Read up to n_bytes from stream. Output is always, always valid UTF-8,
	 * this function closes the FD.
	 */
	text = tracker_read_text_from_fd (fd, n_bytes, error);
	g_free (uri);
	g_free (path);

	return text;
}

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	g_autoptr (TrackerResource) metadata = NULL;
	g_autofree char *content = NULL, *resource_uri = NULL;
	GError *inner_error = NULL;

	resource_uri = tracker_extract_info_get_content_id (info, NULL);
	metadata = tracker_resource_new (resource_uri);
	tracker_resource_add_uri (metadata, "rdf:type", "nfo:PlainTextDocument");

	content = get_file_content (tracker_extract_info_get_file (info),
	                            tracker_extract_info_get_max_text (info),
	                            &inner_error);

	if (inner_error) {
		/* An error occurred, perhaps the file was deleted. */
		g_propagate_prefixed_error (error, inner_error, "Could not open:");
		return FALSE;
	}

	if (content)
		tracker_resource_set_string (metadata, "nie:plainTextContent", content);

	tracker_extract_info_set_resource (info, metadata);

	return TRUE;
}
