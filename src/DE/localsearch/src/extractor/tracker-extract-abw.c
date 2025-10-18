/*
 * Copyright (C) 2007, Jamie McCracken <jamiemcc@gnome.org>
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

#include <tracker-common.h>

#include "utils/tracker-extract.h"

#define BUFFER_SIZE (8 * 1024)

typedef struct AbwParserData AbwParserData;
typedef enum {
	ABW_PARSER_TAG_UNHANDLED,
	ABW_PARSER_TAG_TITLE,
	ABW_PARSER_TAG_SUBJECT,
	ABW_PARSER_TAG_CREATOR,
	ABW_PARSER_TAG_KEYWORDS,
	ABW_PARSER_TAG_DESCRIPTION,
	ABW_PARSER_TAG_GENERATOR
} AbwParserTag;

struct AbwParserData {
	TrackerResource *resource;
	GString *content;
	gchar *uri;

	guint cur_tag;
	guint in_text       : 1;
	guint has_title     : 1;
	guint has_subject   : 1;
	guint has_comment   : 1;
	guint has_generator : 1;
};

static void
abw_parser_start_elem (GMarkupParseContext *context,
                       const gchar         *element_name,
                       const gchar        **attribute_names,
                       const gchar        **attribute_values,
                       gpointer             user_data,
                       GError             **error)
{
	AbwParserData *data = user_data;

	if (g_strcmp0 (element_name, "m") == 0 &&
	    g_strcmp0 (attribute_names[0], "key") == 0) {
		if (g_strcmp0 (attribute_values[0], "dc.title") == 0) {
			data->cur_tag = ABW_PARSER_TAG_TITLE;
		} else if (g_strcmp0 (attribute_values[0], "dc.subject") == 0) {
			data->cur_tag = ABW_PARSER_TAG_SUBJECT;
		} else if (g_strcmp0 (attribute_values[0], "dc.creator") == 0) {
			data->cur_tag = ABW_PARSER_TAG_CREATOR;
		} else if (g_strcmp0 (attribute_values[0], "abiword.keywords") == 0) {
			data->cur_tag = ABW_PARSER_TAG_KEYWORDS;
		} else if (g_strcmp0 (attribute_values[0], "dc.description") == 0) {
			data->cur_tag = ABW_PARSER_TAG_DESCRIPTION;
		} else if (g_strcmp0 (attribute_values[0], "abiword.generator") == 0) {
			data->cur_tag = ABW_PARSER_TAG_GENERATOR;
		}
	} else if (g_strcmp0 (element_name, "section") == 0) {
		data->in_text = TRUE;
	}
}

static void
abw_parser_text (GMarkupParseContext *context,
                 const gchar         *text,
                 gsize                text_len,
                 gpointer             user_data,
                 GError             **error)
{
	AbwParserData *data = user_data;
	gchar *str;

	str = g_strndup (text, text_len);

	switch (data->cur_tag) {
	case ABW_PARSER_TAG_TITLE:
		if (data->has_title) {
			g_warning ("Avoiding additional title (%s) in Abiword document '%s'",
			           str, data->uri);
		} else {
			data->has_title = TRUE;
			tracker_resource_set_string (data->resource, "nie:title", str);
		}
		break;
	case ABW_PARSER_TAG_SUBJECT:
		if (data->has_subject) {
			g_warning ("Avoiding additional subject (%s) in Abiword document '%s'",
			           str, data->uri);
		} else {
			data->has_subject = TRUE;
			tracker_resource_set_string (data->resource, "nie:subject", str);
		}
		break;
	case ABW_PARSER_TAG_CREATOR: {
		TrackerResource *creator;
		creator = tracker_extract_new_contact (str);
		tracker_resource_set_relation (data->resource, "nco:creator", creator);
		g_object_unref (creator);

		break;
	}
	case ABW_PARSER_TAG_DESCRIPTION:
		if (data->has_comment) {
			g_warning ("Avoiding additional comment (%s) in Abiword document '%s'",
			           str, data->uri);
		} else {
			data->has_comment = TRUE;
			tracker_resource_set_string (data->resource, "nie:comment", str);
		}
		break;
	case ABW_PARSER_TAG_GENERATOR:
		if (data->has_generator) {
			g_warning ("Avoiding additional generator (%s) in Abiword document '%s'",
			           str, data->uri);
		} else {
			data->has_generator = TRUE;
			tracker_resource_set_string (data->resource, "nie:generator", str);
		}
		break;
	case ABW_PARSER_TAG_KEYWORDS:
	{
		char *lasts, *keyword;

		for (keyword = strtok_r (str, ",; ", &lasts); keyword;
		     keyword = strtok_r (NULL, ",; ", &lasts)) {
			tracker_resource_add_string (data->resource, "nie:keyword", keyword);
		}
	}
		break;
	default:
		break;
	}

	if (data->in_text) {
		if (G_UNLIKELY (!data->content)) {
			data->content = g_string_new ("");
		}

		g_string_append_len (data->content, text, text_len);
	}

	data->cur_tag = ABW_PARSER_TAG_UNHANDLED;
	g_free (str);
}

static GMarkupParser parser = {
	abw_parser_start_elem,
	NULL,
	abw_parser_text,
	NULL, NULL
};

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	g_autoptr (GInputStream) stream = NULL, buffered_stream = NULL, converter_stream = NULL;
	GInputStream *read_stream;
	g_autoptr (GMappedFile) mapped_file = NULL;
	g_autoptr (GFile) file = NULL;
	g_autoptr (GMarkupParseContext) context = NULL;
	g_autoptr (TrackerResource) resource = NULL;
	g_autoptr (GBytes) bytes = NULL;
	g_autoptr (GString) content = NULL;
	g_autofree char *resource_uri = NULL, *uri = NULL, *path = NULL;
	AbwParserData data = { 0 };
	const char *buffer;

	file = g_object_ref (tracker_extract_info_get_file (info));
	path = g_file_get_path (file);

	stream = G_INPUT_STREAM (g_file_read (file, NULL, error));
	if (!stream)
		return FALSE;

	buffered_stream = g_buffered_input_stream_new_sized (stream, BUFFER_SIZE);

	if (g_buffered_input_stream_fill (G_BUFFERED_INPUT_STREAM (buffered_stream),
					  BUFFER_SIZE, NULL, error) == -1)
		return FALSE;

	buffer = g_buffered_input_stream_peek_buffer (G_BUFFERED_INPUT_STREAM (buffered_stream), NULL);
	if (buffer[0] == '<') {
		/* Uncompressed XML */
		read_stream = buffered_stream;
	} else {
		g_autoptr (GZlibDecompressor) decompressor = NULL;

		/* Compressed data(?) */
		decompressor = g_zlib_decompressor_new (G_ZLIB_COMPRESSOR_FORMAT_GZIP);
		converter_stream = g_converter_input_stream_new (buffered_stream,
								 G_CONVERTER (decompressor));
		read_stream = converter_stream;
	}

	resource_uri = tracker_extract_info_get_content_id (info, NULL);
	resource = tracker_resource_new (resource_uri);
	uri = g_file_get_uri (file);
	content = g_string_new ("");

	data.uri = uri;
	data.resource = resource;
	data.content = content;

	tracker_resource_add_uri (data.resource, "rdf:type", "nfo:Document");
	context = g_markup_parse_context_new (&parser, 0, &data, NULL);

	while ((bytes = g_input_stream_read_bytes (read_stream, BUFFER_SIZE, NULL, error)) != NULL) {
		if (g_bytes_get_size (bytes) == 0)
			break;

		if (!g_markup_parse_context_parse (context,
						   g_bytes_get_data (bytes, NULL),
						   g_bytes_get_size (bytes),
						   error))
			return FALSE;

		g_clear_pointer (&bytes, g_bytes_unref);
	}

	if (!g_markup_parse_context_end_parse (context, error))
		return FALSE;

	if (content->len > 0)
		tracker_resource_set_string (data.resource, "nie:plainTextContent", content->str);

	tracker_extract_info_set_resource (info, resource);

	return TRUE;
}
