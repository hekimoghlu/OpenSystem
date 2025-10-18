/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 8; tab-width: 8 -*- */
/*
 * GData Client
 * Copyright (C) Philip Withnall 2009–2010 <philip@tecnocode.co.uk>
 *
 * GData Client is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * GData Client is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GData Client.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * SECTION:gdata-media-content
 * @short_description: Media RSS content element
 * @stability: Stable
 * @include: gdata/media/gdata-media-content.h
 *
 * #GDataMediaContent represents a "content" element from the
 * <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
 *
 * The class only implements parsing, not XML output, at the moment.
 *
 * Since: 0.4.0
 */

#include <glib.h>
#include <libxml/parser.h>

#include "gdata-media-content.h"
#include "gdata-download-stream.h"
#include "gdata-parsable.h"
#include "gdata-parser.h"
#include "gdata-media-enums.h"
#include "gdata-private.h"

static void gdata_media_content_finalize (GObject *object);
static void gdata_media_content_get_property (GObject *object, guint property_id, GValue *value, GParamSpec *pspec);
static gboolean pre_parse_xml (GDataParsable *parsable, xmlDoc *doc, xmlNode *root_node, gpointer user_data, GError **error);
static void get_namespaces (GDataParsable *parsable, GHashTable *namespaces);

struct _GDataMediaContentPrivate {
	gchar *uri;
	gsize filesize;
	gchar *content_type;
	GDataMediaMedium medium;
	gboolean is_default;
	GDataMediaExpression expression;
	gint64 duration;
	guint height;
	guint width;
	/* TODO: implement other properties from the Media RSS standard */
};

enum {
	PROP_URI = 1,
	PROP_FILESIZE,
	PROP_CONTENT_TYPE,
	PROP_MEDIUM,
	PROP_IS_DEFAULT,
	PROP_EXPRESSION,
	PROP_DURATION,
	PROP_HEIGHT,
	PROP_WIDTH
};

G_DEFINE_TYPE_WITH_PRIVATE (GDataMediaContent, gdata_media_content, GDATA_TYPE_PARSABLE)

static void
gdata_media_content_class_init (GDataMediaContentClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
	GDataParsableClass *parsable_class = GDATA_PARSABLE_CLASS (klass);

	gobject_class->get_property = gdata_media_content_get_property;
	gobject_class->finalize = gdata_media_content_finalize;

	parsable_class->pre_parse_xml = pre_parse_xml;
	parsable_class->get_namespaces = get_namespaces;
	parsable_class->element_name = "content";
	parsable_class->element_namespace = "media";

	/**
	 * GDataMediaContent:uri:
	 *
	 * The direct URI to the media object.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_URI,
	                                 g_param_spec_string ("uri",
	                                                      "URI", "The direct URI to the media object.",
	                                                      NULL,
	                                                      G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataMediaContent:filesize:
	 *
	 * The number of bytes of the media object.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_FILESIZE,
	                                 g_param_spec_ulong ("filesize",
	                                                     "Filesize", "The number of bytes of the media object.",
	                                                     0, G_MAXULONG, 0,
	                                                     G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataMediaContent:content-type:
	 *
	 * The standard MIME type of the object.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_CONTENT_TYPE,
	                                 g_param_spec_string ("content-type",
	                                                      "Content type", "The standard MIME type of the object.",
	                                                      NULL,
	                                                      G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataMediaContent:medium:
	 *
	 * The type of object, complementing #GDataMediaContent:content-type. It allows the consuming application to make simpler decisions between
	 * different content objects, based on whether they're a video or audio stream, for example.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_MEDIUM,
	                                 g_param_spec_enum ("medium",
	                                                    "Medium", "The type of object.",
	                                                    GDATA_TYPE_MEDIA_MEDIUM, GDATA_MEDIA_UNKNOWN,
	                                                    G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataMediaContent:is-default:
	 *
	 * Determines if this is the default content for the media group. There should only be one default object per media group.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_IS_DEFAULT,
	                                 g_param_spec_boolean ("is-default",
	                                                       "Default?", "Determines if this is the default content for the media group.",
	                                                       FALSE,
	                                                       G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataMediaContent:expression:
	 *
	 * Determines if the object is a sample or the full version of the object, or even if it is a continuous stream.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_EXPRESSION,
	                                 g_param_spec_enum ("expression",
	                                                    "Expression", "Determines if the object is a sample or the full version of the object.",
	                                                    GDATA_TYPE_MEDIA_EXPRESSION, GDATA_MEDIA_EXPRESSION_FULL,
	                                                    G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataMediaContent:duration:
	 *
	 * The number of seconds for which the media object plays.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_DURATION,
	                                 g_param_spec_int64 ("duration",
	                                                     "Duration", "The number of seconds for which the media object plays.",
	                                                     0, G_MAXINT64, 0,
	                                                     G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataMediaContent:height:
	 *
	 * The height of the media object.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_HEIGHT,
	                                 g_param_spec_uint ("height",
	                                                    "Height", "The height of the media object.",
	                                                    0, G_MAXUINT, 0,
	                                                    G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataMediaContent:width:
	 *
	 * The width of the media object.
	 *
	 * For more information, see the <ulink type="http" url="http://video.search.yahoo.com/mrss">Media RSS specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_WIDTH,
	                                 g_param_spec_uint ("width",
	                                                    "Width", "The width of the media object.",
	                                                    0, G_MAXUINT, 0,
	                                                    G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
}

static void
gdata_media_content_init (GDataMediaContent *self)
{
	self->priv = gdata_media_content_get_instance_private (self);
}

static void
gdata_media_content_finalize (GObject *object)
{
	GDataMediaContentPrivate *priv = GDATA_MEDIA_CONTENT (object)->priv;

	g_free (priv->uri);
	g_free (priv->content_type);

	/* Chain up to the parent class */
	G_OBJECT_CLASS (gdata_media_content_parent_class)->finalize (object);
}

static void
gdata_media_content_get_property (GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
	GDataMediaContentPrivate *priv = GDATA_MEDIA_CONTENT (object)->priv;

	switch (property_id) {
		case PROP_URI:
			g_value_set_string (value, priv->uri);
			break;
		case PROP_FILESIZE:
			g_value_set_ulong (value, priv->filesize);
			break;
		case PROP_CONTENT_TYPE:
			g_value_set_string (value, priv->content_type);
			break;
		case PROP_MEDIUM:
			g_value_set_enum (value, priv->medium);
			break;
		case PROP_IS_DEFAULT:
			g_value_set_boolean (value, priv->is_default);
			break;
		case PROP_EXPRESSION:
			g_value_set_enum (value, priv->expression);
			break;
		case PROP_DURATION:
			g_value_set_int64 (value, priv->duration);
			break;
		case PROP_HEIGHT:
			g_value_set_uint (value, priv->height);
			break;
		case PROP_WIDTH:
			g_value_set_uint (value, priv->width);
			break;
		default:
			/* We don't have any other property... */
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
			break;
	}
}

static gboolean
pre_parse_xml (GDataParsable *parsable, xmlDoc *doc, xmlNode *root_node, gpointer user_data, GError **error)
{
	GDataMediaContentPrivate *priv = GDATA_MEDIA_CONTENT (parsable)->priv;
	xmlChar *uri, *expression, *medium, *duration, *filesize, *height, *width;
	gboolean is_default_bool;
	GDataMediaExpression expression_enum;
	GDataMediaMedium medium_enum;
	guint height_uint, width_uint;
	gint64 duration_int64;
	gulong filesize_ulong;

	/* Parse isDefault */
	if (gdata_parser_boolean_from_property (root_node, "isDefault", &is_default_bool, 0, error) == FALSE)
		return FALSE;

	/* Parse expression */
	expression = xmlGetProp (root_node, (xmlChar*) "expression");
	if (expression == NULL || xmlStrcmp (expression, (xmlChar*) "full") == 0) {
		expression_enum = GDATA_MEDIA_EXPRESSION_FULL;
	} else if (xmlStrcmp (expression, (xmlChar*) "sample") == 0) {
		expression_enum = GDATA_MEDIA_EXPRESSION_SAMPLE;
	} else if (xmlStrcmp (expression, (xmlChar*) "nonstop") == 0) {
		expression_enum = GDATA_MEDIA_EXPRESSION_NONSTOP;
	} else {
		gdata_parser_error_unknown_property_value (root_node, "expression", (gchar*) expression, error);
		xmlFree (expression);
		return FALSE;
	}
	xmlFree (expression);

	/* Parse medium */
	medium = xmlGetProp (root_node, (xmlChar*) "medium");
	if (medium == NULL) {
		medium_enum = GDATA_MEDIA_UNKNOWN;
	} else if (xmlStrcmp (medium, (xmlChar*) "image") == 0) {
		medium_enum = GDATA_MEDIA_IMAGE;
	} else if (xmlStrcmp (medium, (xmlChar*) "audio") == 0) {
		medium_enum = GDATA_MEDIA_AUDIO;
	} else if (xmlStrcmp (medium, (xmlChar*) "video") == 0) {
		medium_enum = GDATA_MEDIA_VIDEO;
	} else if (xmlStrcmp (medium, (xmlChar*) "document") == 0) {
		medium_enum = GDATA_MEDIA_DOCUMENT;
	} else if (xmlStrcmp (medium, (xmlChar*) "executable") == 0) {
		medium_enum = GDATA_MEDIA_EXECUTABLE;
	} else {
		gdata_parser_error_unknown_property_value (root_node, "medium", (gchar*) medium, error);
		xmlFree (medium);
		return FALSE;
	}
	xmlFree (medium);

	/* Parse duration */
	duration = xmlGetProp (root_node, (xmlChar*) "duration");
	duration_int64 = (duration == NULL) ? 0 : g_ascii_strtoll ((gchar*) duration, NULL, 10);
	xmlFree (duration);

	/* Parse filesize */
	filesize = xmlGetProp (root_node, (xmlChar*) "fileSize");
	filesize_ulong = (filesize == NULL) ? 0 : g_ascii_strtoull ((gchar*) filesize, NULL, 10);
	xmlFree (filesize);

	/* Parse height and width */
	height = xmlGetProp (root_node, (xmlChar*) "height");
	height_uint = (height == NULL) ? 0 : g_ascii_strtoull ((gchar*) height, NULL, 10);
	xmlFree (height);

	width = xmlGetProp (root_node, (xmlChar*) "width");
	width_uint = (width == NULL) ? 0 : g_ascii_strtoull ((gchar*) width, NULL, 10);
	xmlFree (width);

	/* Other properties */
	uri = xmlGetProp (root_node, (xmlChar*) "url");
	if (uri != NULL && *uri == '\0') {
		xmlFree (uri);
		return gdata_parser_error_required_property_missing (root_node, "url", error);
	}

	priv->uri = (gchar*) uri;
	priv->filesize = filesize_ulong;
	priv->content_type = (gchar*) xmlGetProp (root_node, (xmlChar*) "type");
	priv->medium = medium_enum;
	priv->is_default = is_default_bool;
	priv->expression = expression_enum;
	priv->duration = duration_int64;
	priv->height = height_uint;
	priv->width = width_uint;

	return TRUE;
}

static void
get_namespaces (GDataParsable *parsable, GHashTable *namespaces)
{
	g_hash_table_insert (namespaces, (gchar*) "media", (gchar*) "http://search.yahoo.com/mrss/");
}

/**
 * gdata_media_content_get_uri:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:uri property.
 *
 * Return value: the content's URI
 *
 * Since: 0.4.0
 */
const gchar *
gdata_media_content_get_uri (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), NULL);
	return self->priv->uri;
}

/**
 * gdata_media_content_get_filesize:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:filesize property.
 *
 * Return value: the number of bytes in the content, or <code class="literal">0</code>
 *
 * Since: 0.4.0
 */
gsize
gdata_media_content_get_filesize (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), 0);
	return self->priv->filesize;
}

/**
 * gdata_media_content_get_content_type:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:content-type property.
 *
 * Return value: the content's content (MIME) type, or %NULL
 *
 * Since: 0.4.0
 */
const gchar *
gdata_media_content_get_content_type (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), NULL);
	return self->priv->content_type;
}

/**
 * gdata_media_content_get_medium:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:medium property.
 *
 * Return value: the type of the content, or %GDATA_MEDIA_UNKNOWN
 *
 * Since: 0.4.0
 */
GDataMediaMedium
gdata_media_content_get_medium (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), GDATA_MEDIA_UNKNOWN);
	return self->priv->medium;
}

/**
 * gdata_media_content_is_default:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:is-default property.
 *
 * Return value: %TRUE if the #GDataMediaContent is the default content for the media group, %FALSE otherwise
 *
 * Since: 0.4.0
 */
gboolean
gdata_media_content_is_default (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), FALSE);
	return self->priv->is_default;
}

/**
 * gdata_media_content_get_expression:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:expression property.
 *
 * Return value: the content's expression, or %GDATA_MEDIA_EXPRESSION_FULL
 *
 * Since: 0.4.0
 */
GDataMediaExpression
gdata_media_content_get_expression (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), GDATA_MEDIA_EXPRESSION_FULL);
	return self->priv->expression;
}

/**
 * gdata_media_content_get_duration:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:duration property.
 *
 * Return value: the content's duration in seconds, or <code class="literal">0</code>
 *
 * Since: 0.4.0
 */
gint64
gdata_media_content_get_duration (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), 0);
	return self->priv->duration;
}

/**
 * gdata_media_content_get_height:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:height property.
 *
 * Return value: the content's height in pixels, or <code class="literal">0</code>
 *
 * Since: 0.4.0
 */
guint
gdata_media_content_get_height (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), 0);
	return self->priv->height;
}

/**
 * gdata_media_content_get_width:
 * @self: a #GDataMediaContent
 *
 * Gets the #GDataMediaContent:width property.
 *
 * Return value: the content's width in pixels, or <code class="literal">0</code>
 *
 * Since: 0.4.0
 */
guint
gdata_media_content_get_width (GDataMediaContent *self)
{
	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), 0);
	return self->priv->width;
}

/**
 * gdata_media_content_download:
 * @self: a #GDataMediaContent
 * @service: the #GDataService
 * @cancellable: (allow-none): a #GCancellable for the entire download stream, or %NULL
 * @error: a #GError, or %NULL
 *
 * Downloads and returns a #GDataDownloadStream allowing the content represented by @self to be read.
 *
 * To get the content type of the downloaded data, gdata_download_stream_get_content_type() can be called on the returned #GDataDownloadStream.
 * Calling gdata_download_stream_get_content_length() on the stream will not return a meaningful result, however, as the stream is encoded in chunks,
 * rather than by content length.
 *
 * In order to cancel the download, a #GCancellable passed in to @cancellable must be cancelled using g_cancellable_cancel(). Cancelling the individual
 * #GInputStream operations on the #GDataDownloadStream will not cancel the entire download; merely the read or close operation in question. See the
 * #GDataDownloadStream:cancellable for more details.
 *
 * Return value: (transfer full): a #GDataDownloadStream to download the content with, or %NULL; unref with g_object_unref()
 *
 * Since: 0.8.0
 */
GDataDownloadStream *
gdata_media_content_download (GDataMediaContent *self, GDataService *service, GCancellable *cancellable, GError **error)
{
	const gchar *src_uri;

	g_return_val_if_fail (GDATA_IS_MEDIA_CONTENT (self), NULL);
	g_return_val_if_fail (GDATA_IS_SERVICE (service), NULL);
	g_return_val_if_fail (cancellable == NULL || G_IS_CANCELLABLE (cancellable), NULL);
	g_return_val_if_fail (error == NULL || *error == NULL, NULL);

	/* We keep a GError in the argument list so that we can add authentication errors, etc., in future if necessary */

	/* Get the download URI and create a stream for it */
	src_uri = gdata_media_content_get_uri (self);
	return GDATA_DOWNLOAD_STREAM (gdata_download_stream_new (service, NULL, src_uri, cancellable));
}
