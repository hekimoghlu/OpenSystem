/*
 * Copyright (C) 2021, Red Hat Inc.
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
 * Author: Carlos Garnacho <carlosg@gnome.org>
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

#include <libcue.h>

#include <tracker-common.h>

#include "utils/tracker-extract.h"

#include "tracker-main.h"
#include "tracker-extract.h"

typedef struct {
	guint8 length;
	guint8 ext_attr_length;
	guint8 extent[8];
	guint8 size[8];
	guint8 date[7];
	guint8 flags;
	guint8 file_unit_size;
	guint8 interleave;
	guint8 volume_sequence_number[4];
	guint8 name_length;
	guint8 name[0];
} PsDiscDirectoryRecord;

typedef struct {
	guint8 minute;
	guint8 second;
	guint8 frame;
} PsDiscTime;

typedef struct {
	guint8 sync[12];
	guint8 header[12];
	guint8 content[2048];
	guint8 crc[280];
} PsDiscMode1Frame;

#define PS_DISC_FRAMES_PER_SECOND 75
#define PS_DISC_FRAME_SIZE 2352
#define PS_DISC_FRAME_HEADER_SIZE 12
#define PS_DISC_FRAME_CONTENT_SIZE 2048

#define PS_DISC_TIME_TO_EXTENT(time)	  \
	((time->minute * 60 + time->second - 2) * PS_DISC_FRAMES_PER_SECOND + time->frame)
#define PS_DISC_EXTENT_TO_TIME(extent, time)	  \
	G_STMT_START { \
		gint32 block = GINT32_FROM_LE (*((gint32 *) extent)); \
		block += 2 * PS_DISC_FRAMES_PER_SECOND; \
		(time)->minute = block / (60 * PS_DISC_FRAMES_PER_SECOND); \
		block = block - (time)->minute * (60 * PS_DISC_FRAMES_PER_SECOND); \
		(time)->second = block / PS_DISC_FRAMES_PER_SECOND; \
		(time)->frame = block - (time)->second * PS_DISC_FRAMES_PER_SECOND; \
	} G_STMT_END
#define PS_DISC_TIME_INC(time)	\
	G_STMT_START { \
		time->frame++; \
		if (time->frame < PS_DISC_FRAMES_PER_SECOND) { \
			time->frame = 0; \
			time->second++; \
			if (time->second >= 60) { \
				time->second = 0; \
				time->minute++; \
			} \
		} \
	} G_STMT_END

static TrackerResource *
build_basic_resource (TrackerExtractInfo *info,
                      GFile              *cue,
                      GFile              *image)
{
	TrackerResource *metadata, *child;
	g_autofree char *resource_uri = NULL;
	gchar *uri;

	resource_uri = tracker_extract_info_get_content_id (info, NULL);
	metadata = tracker_resource_new (resource_uri);
	tracker_resource_add_uri (metadata, "rdf:type", "nfo:GameImage");
	tracker_resource_set_string (metadata, "nie:mimeType", "application/x-cue");

	uri = g_file_get_uri (cue);
	tracker_resource_add_uri (metadata, "nie:isStoredAs", uri);
	g_free (uri);

	/* In addition to the cue file, link the information element to the data file */
	uri = g_file_get_uri (image);
	child = tracker_resource_new (uri);
	tracker_resource_add_uri (child, "rdf:type", "nfo:FileDataObject");
	tracker_resource_set_uri (child, "nie:interpretedAs", resource_uri);
	tracker_resource_set_take_relation (metadata, "nie:isStoredAs", child);
	g_free (uri);

	return metadata;
}

static gboolean
ps_disc_read_frame (const guchar     *data,
                    gsize             length,
                    const PsDiscTime *time,
                    PsDiscMode1Frame *frame /* out */)
{
	gint extent;
	gsize offset;

	extent = PS_DISC_TIME_TO_EXTENT (time);

	if (extent < 0)
		return FALSE;

	offset = (gsize) extent * PS_DISC_FRAME_SIZE;
	if (offset + PS_DISC_FRAME_SIZE > length)
		return FALSE;

	*frame = *((PsDiscMode1Frame*) &data[offset]);

	return TRUE;
}

static guint8 *
ps_disc_read_directory (const guchar *data,
                        gsize         length,
                        PsDiscTime   *time /* out */)
{
	guint8 *buf;
	gint extent;
	gsize offset;

	extent = PS_DISC_TIME_TO_EXTENT (time);
	if (extent < 0)
		return NULL;
	offset = extent * PS_DISC_FRAME_SIZE + PS_DISC_FRAME_HEADER_SIZE + 12;
	if (offset + PS_DISC_FRAME_CONTENT_SIZE > length)
		return NULL;

	buf = g_new0 (guint8, 4096);
	memcpy (buf, &data[offset], PS_DISC_FRAME_CONTENT_SIZE);

	PS_DISC_TIME_INC (time);
	extent = PS_DISC_TIME_TO_EXTENT (time);
	if (extent < 0) {
		g_free (buf);
		return NULL;
	}
	offset = extent * PS_DISC_FRAME_SIZE + PS_DISC_FRAME_HEADER_SIZE + 12;
	if (offset + PS_DISC_FRAME_CONTENT_SIZE > length) {
		g_free (buf);
		return NULL;
	}

	memcpy (&buf[PS_DISC_FRAME_CONTENT_SIZE], &data[offset], PS_DISC_FRAME_CONTENT_SIZE);

	return buf;
}

static gboolean
ps_disc_get_file (const guint8 *dir,
                  gsize         length,
                  const gchar  *filename,
                  PsDiscTime   *time /* out */)
{
	PsDiscDirectoryRecord *record;
	gsize pos = 0;

	while (pos < length) {
		record = (PsDiscDirectoryRecord *) &dir[pos];
		if (record->length == 0)
			break;

		pos += record->length;

		if (record->flags & 0x2) {
			/* This is a directory, we only look for files in the root directory though */
			continue;
		}

		if (g_ascii_strncasecmp (&record->name[0], filename, strlen (filename)) == 0) {
			/* File is found */
			if (time)
				PS_DISC_EXTENT_TO_TIME(record->extent, time);
			return TRUE;
		}
	}

	return FALSE;
}

static gboolean
ps_disc_lookup_executable_filename (const gchar *content,
                                    const gchar *prefix,
                                    gsize        buffer_len,
                                    gchar       *exe_buffer /* out */)
{
	const gchar *ptr = content;
	gint i;

	if (!ptr || !prefix || strncmp (ptr, prefix, strlen (prefix)) != 0)
		return FALSE;

	ptr += strlen (prefix);

	/* Skip slashes. */
	while (*ptr == '\\' || *ptr == '/')
		ptr++;

	strncpy (exe_buffer, ptr, buffer_len);
	exe_buffer[buffer_len - 1] = '\0';

	/* Keep only the first line. */
	for (i = 0; i < buffer_len; i++) {
		if (exe_buffer[i] == '\r' || exe_buffer[i] == '\n') {
			exe_buffer[i] = '\0';
			break;
		}
	}

	return TRUE;
}

static gboolean
check_is_playstation_image (const guchar *data,
                            gsize         length)
{
	gchar exe_buffer[256] = { 0 };
	PsDiscDirectoryRecord *record;
	PsDiscTime time = { 0, 2, 0x10 };
	guint8 *buf;
	gchar *ptr;
	PsDiscMode1Frame frame;
	gboolean is_ps = FALSE;

	if (!ps_disc_read_frame (data, length, &time, &frame))
		return FALSE;

	/* Skip head and sub, and go to the root directory record */
	record = (PsDiscDirectoryRecord *) (frame.content + 156);
	PS_DISC_EXTENT_TO_TIME (record->extent, &time);

	buf = ps_disc_read_directory (data, length, &time);
	if (!buf)
		return FALSE;

	/* Check that SYSTEM.CNF exists */
	if (ps_disc_get_file (buf, 4096, "SYSTEM.CNF;1", &time)) {
		g_debug ("SYSTEM.CNF found, looking for executable");

		if (!ps_disc_read_frame (data, length, &time, &frame))
			goto out;

		/* Look of "BOOT = cdrom:"  */
		if (ps_disc_lookup_executable_filename ((gchar *) frame.content,
		                                        "BOOT = cdrom:",
		                                        G_N_ELEMENTS (exe_buffer),
		                                        exe_buffer)) {
			if (ps_disc_get_file (buf, 4096, exe_buffer, NULL)) {
				g_debug ("Executable '%s' found", exe_buffer);
				is_ps = TRUE;
				goto out;
			}
		}

		/* Look of "cdrom:" */
		ptr = strstr ((gchar *) frame.content, "cdrom:");
		if (ptr &&
		    ps_disc_lookup_executable_filename ((gchar *) ptr,
		                                        "cdrom:",
		                                        G_N_ELEMENTS (exe_buffer),
		                                        exe_buffer)) {
			if (ps_disc_get_file (buf, 4096, exe_buffer, NULL)) {
				g_debug ("Executable '%s' found", exe_buffer);
				is_ps = TRUE;
				goto out;
			}
		}

		goto out;
	}

	/* Look for the default PSX.EXE executable. */
	if (ps_disc_get_file (buf, 4096, "PSX.EXE;1", NULL)) {
		g_debug ("PSX.EXE found");
		is_ps = TRUE;
		goto out;
	}

	/* SYSTEM.CNF and PSX.EXE not found. */
	is_ps = FALSE;

 out:
	g_free (buf);

	return is_ps;
}

static GMappedFile *
try_open_mapped_file (const gchar  *image_path,
                      GFile        *cue,
                      GFile       **image,
                      GError      **error)
{
	GMappedFile *mapped_file;
	gchar *path, *basename, *dirname;

	/* 1st. Try the image path untouched */
	mapped_file = g_mapped_file_new (image_path, FALSE, NULL);
	if (mapped_file) {
		g_debug ("Found bin file '%s'", image_path);
		*image = g_file_new_for_path (image_path);
		return mapped_file;
	}

	/* 2nd. Try the image basename, relative to the cue file */
	basename = g_path_get_basename (image_path);
	dirname = g_path_get_dirname (g_file_peek_path (cue));
	path = g_build_filename (dirname, basename, NULL);

	mapped_file = g_mapped_file_new (path,
	                                 FALSE,
	                                 error);
	if (mapped_file) {
		*image = g_file_new_for_path (path);
		g_debug ("Found bin file '%s'", path);
	} else {
		g_debug ("No matching bin file found for '%s'", image_path);
	}

	g_free (path);
	g_free (basename);
	g_free (dirname);

	return mapped_file;
}

static TrackerResource *
get_playstation_image_data (const gchar         *image_path,
                            TrackerExtractInfo  *info,
                            GFile               *cue,
                            GError             **error)
{
	GMappedFile *mapped_file;
	GBytes *bytes;
	const guchar *content;
	gsize length;
	TrackerResource *metadata = NULL;
	GFile *image;

	mapped_file = try_open_mapped_file (image_path, cue, &image, error);
	if (!mapped_file)
		return NULL;

	bytes = g_mapped_file_get_bytes (mapped_file);
	content = g_bytes_get_data (bytes, &length);

	if (check_is_playstation_image (content, length)) {
		g_debug ("Image is a Playstation game");
		metadata = build_basic_resource (info, cue, image);
	}

	g_bytes_unref (bytes);
	g_mapped_file_unref (mapped_file);
	g_object_unref (image);

	return metadata;
}

static TrackerResource *
get_turbografx_image_data (const gchar         *image_path,
                           TrackerExtractInfo  *info,
                           GFile               *cue,
                           GError             **error)
{
	GMappedFile *mapped_file;
	GBytes *bytes;
	const gchar *content;
	gsize length;
	TrackerResource *metadata = NULL;
	GFile *image;
	gsize magic_position = 0x81c90;
	const gchar *magic_string = "PC Engine CD-ROM SYSTEM";
	gint magic_len = strlen (magic_string);

	mapped_file = try_open_mapped_file (image_path, cue, &image, error);
	if (!mapped_file)
		return NULL;

	bytes = g_mapped_file_get_bytes (mapped_file);
	content = g_bytes_get_data (bytes, &length);

	if (magic_position < length - magic_len &&
	    strncmp (&content[magic_position], magic_string, magic_len) == 0) {
		g_debug ("Image is a Turbografx game");
		metadata = build_basic_resource (info, cue, image);
	}

	g_bytes_unref (bytes);
	g_mapped_file_unref (mapped_file);
	g_object_unref (image);

	return metadata;
}

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	g_autoptr (TrackerResource) metadata = NULL;
	GError *inner_error = NULL;
	GFile *file;
	g_autofree char *buffer = NULL;
	Track *track;
	Cd *cd = NULL;

	file = tracker_extract_info_get_file (info);

	if (!g_file_load_contents (file, NULL, &buffer, NULL, NULL, &inner_error))
		goto error;

	cd = cue_parse_string (buffer);
	if (!cd) {
		g_set_error (&inner_error,
		             G_IO_ERROR,
		             G_IO_ERROR_INVALID_DATA,
		             "Not a CUE sheet");
		goto error;
	}

	track = cd_get_track (cd, 1);
	if (!track) {
		g_set_error (&inner_error,
		             G_IO_ERROR,
		             G_IO_ERROR_INVALID_DATA,
		             "No first track");
		goto error;
	}

	/* Playstation 1 check. One data track either mode1 or mode2 raw */
	if (track_get_mode (track) == MODE_MODE1_RAW ||
	    track_get_mode (track) == MODE_MODE2_RAW) {
		g_debug ("Checking whether image is a Playstation game");
		metadata = get_playstation_image_data (track_get_filename (track),
		                                       info,
		                                       file,
		                                       &inner_error);
		if (metadata || inner_error)
			goto out;
	}

	/* Turbografx check. Data is in track 2, must be mode1(-raw) */
	if (cd_get_ntrack (cd) >= 2) {
		track = cd_get_track (cd, 2);

		if (track_get_mode (track) == MODE_MODE1 ||
		    track_get_mode (track) == MODE_MODE1_RAW) {
			g_debug ("Checking whether image is a Turbografx game");
			metadata = get_turbografx_image_data (track_get_filename (track),
			                                      info,
			                                      file,
			                                      &inner_error);
			if (metadata || inner_error)
				goto out;
		}
	}

	g_debug ("CUE file not recognized");
	g_clear_pointer (&cd, cd_delete);
	return TRUE;

out:
	if (metadata) {
		tracker_extract_info_set_resource (info, metadata);
		g_clear_pointer (&cd, cd_delete);
		return TRUE;
	}

error:
	g_clear_pointer (&cd, cd_delete);

	if (inner_error)
		g_propagate_error (error, inner_error);

	return FALSE;
}
