/*
 * Copyright (C) 2025, Red Hat Inc.
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
 *
 * Author: Nieves Montero <nmontero@redhat.com>
 */

#include "config-miners.h"

#include <stdio.h>
#include <stdlib.h>
#include <webp/demux.h>
#include <webp/decode.h>
#include <tracker-common.h>
#include "tracker-extract.h"
#include "tracker-guarantee.h"
#include "tracker-exif.h"
#include "tracker-iptc.h"
#include "tracker-xmp.h"
#include "tracker-resource-helpers.h"

#define BUFFER_SIZE (256 * 1024)

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	GFile *file;
	gboolean success = FALSE;
	g_autoptr (GFileInputStream) stream = NULL;
	g_autoptr (GBytes) bytes = NULL;
	g_autoptr (TrackerResource) metadata = NULL;
	g_autofree char *uri = NULL;
	g_autofree char *resource_uri = NULL;
	WebPDemuxer *demux = NULL;
	WebPData webp_data;
	WebPDemuxState demux_state;
	uint32_t width, height, flags;
	WebPChunkIterator chunk_iter;

	file = tracker_extract_info_get_file (info);
	uri = g_file_get_uri (file);

	stream = g_file_read (file, NULL, error);
	if (!stream)
		goto cleanup;

	bytes = g_input_stream_read_bytes (G_INPUT_STREAM (stream),
	                                   BUFFER_SIZE,
	                                   NULL,
	                                   error);
	if (!bytes)
		goto cleanup;

	webp_data = (WebPData) {
		g_bytes_get_data (bytes, NULL),
		g_bytes_get_size (bytes)
	};

	demux = WebPDemuxPartial (&webp_data, &demux_state);

	if (!demux ||
	    (demux_state != WEBP_DEMUX_PARSED_HEADER && demux_state != WEBP_DEMUX_DONE)) {
		g_set_error (error,
		             G_IO_ERROR,
		             G_IO_ERROR_INVALID_DATA,
		             "WebP header not found in the first 64KB");
		goto cleanup;
	}

	width = WebPDemuxGetI (demux, WEBP_FF_CANVAS_WIDTH);
	height = WebPDemuxGetI (demux, WEBP_FF_CANVAS_HEIGHT);
	flags = WebPDemuxGetI (demux, WEBP_FF_FORMAT_FLAGS);

	resource_uri = tracker_extract_info_get_content_id (info, NULL);
	metadata = tracker_resource_new (resource_uri);
	tracker_resource_add_uri (metadata, "rdf:type", "nfo:Image");
	tracker_resource_add_uri (metadata, "rdf:type", "nmm:Photo");
	tracker_resource_set_int64 (metadata, "nfo:width", width);
	tracker_resource_set_int64 (metadata, "nfo:height", height);

	if ((flags & EXIF_FLAG) && WebPDemuxGetChunk (demux, "EXIF", 1, &chunk_iter)) {
		TrackerExifData *exif;

		exif = tracker_exif_new (chunk_iter.chunk.bytes, chunk_iter.chunk.size, uri);

		if (exif) {
			tracker_exif_apply_to_resource (metadata, exif);
			tracker_exif_free (exif);
		}

		WebPDemuxReleaseChunkIterator (&chunk_iter);
	}

	if ((flags & XMP_FLAG) && WebPDemuxGetChunk (demux, "XMP ", 1, &chunk_iter)) {
		TrackerXmpData *xmp;

		xmp = tracker_xmp_new ((const gchar *) chunk_iter.chunk.bytes, chunk_iter.chunk.size, uri);

		if (xmp) {
			tracker_xmp_apply_to_resource (metadata, xmp);
			tracker_xmp_free (xmp);
		}

		WebPDemuxReleaseChunkIterator (&chunk_iter);
	}

	tracker_extract_info_set_resource (info, metadata);
	success = TRUE;

cleanup:

	g_clear_pointer (&demux, WebPDemuxDelete);

	return success;
}
