/*
 * Copyright (C) 2012, Red Hat, Inc.
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

#include <glib.h>
#include <gmodule.h>

#include <libgxps/gxps.h>

#include <tracker-common.h>

#include "utils/tracker-extract.h"

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	TrackerResource *resource;
	GXPSDocument *document;
	GXPSFile *xps_file;
	GFile *file;
	gchar *filename, *resource_uri;
	GError *inner_error = NULL;

	file = tracker_extract_info_get_file (info);
	xps_file = gxps_file_new (file, &inner_error);
	filename = g_file_get_path (file);

	if (inner_error != NULL) {
		g_propagate_prefixed_error (error, inner_error, "Unable to open: ");
		g_free (filename);
		return FALSE;
	}

	document = gxps_file_get_document (xps_file, 0, &inner_error);
	g_object_unref (xps_file);

	if (inner_error != NULL) {
		g_propagate_prefixed_error (error, inner_error, "Unable to read: ");
		g_free (filename);
		return FALSE;
	}

	resource_uri = tracker_extract_info_get_content_id (info, NULL);
	resource = tracker_resource_new (resource_uri);
	tracker_resource_add_uri (resource, "rdf:type", "nfo:PaginatedTextDocument");
	tracker_resource_set_int64 (resource, "nfo:pageCount", gxps_document_get_n_pages (document));
	g_free (resource_uri);

	g_object_unref (document);
	g_free (filename);

	tracker_extract_info_set_resource (info, resource);
	g_object_unref (resource);

	return TRUE;
}
