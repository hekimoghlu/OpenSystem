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
 * SECTION:gdata-link
 * @short_description: Atom link element
 * @stability: Stable
 * @include: gdata/atom/gdata-link.h
 *
 * #GDataLink represents a "link" element from the
 * <ulink type="http" url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php">Atom specification</ulink>.
 *
 * Since: 0.4.0
 */

#include <glib.h>
#include <libxml/parser.h>
#include <string.h>

#include "gdata-link.h"
#include "gdata-parsable.h"
#include "gdata-parser.h"
#include "gdata-comparable.h"

static void gdata_link_comparable_init (GDataComparableIface *iface);
static void gdata_link_finalize (GObject *object);
static void gdata_link_get_property (GObject *object, guint property_id, GValue *value, GParamSpec *pspec);
static void gdata_link_set_property (GObject *object, guint property_id, const GValue *value, GParamSpec *pspec);
static gboolean pre_parse_xml (GDataParsable *parsable, xmlDoc *doc, xmlNode *root_node, gpointer user_data, GError **error);
static void pre_get_xml (GDataParsable *parsable, GString *xml_string);

struct _GDataLinkPrivate {
	gchar *uri;
	gchar *relation_type;
	gchar *content_type;
	gchar *language;
	gchar *title;
	gint length;
};

enum {
	PROP_URI = 1,
	PROP_RELATION_TYPE,
	PROP_CONTENT_TYPE,
	PROP_LANGUAGE,
	PROP_TITLE,
	PROP_LENGTH
};

G_DEFINE_TYPE_WITH_CODE (GDataLink, gdata_link, GDATA_TYPE_PARSABLE,
                         G_ADD_PRIVATE (GDataLink)
                         G_IMPLEMENT_INTERFACE (GDATA_TYPE_COMPARABLE, gdata_link_comparable_init))

static void
gdata_link_class_init (GDataLinkClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
	GDataParsableClass *parsable_class = GDATA_PARSABLE_CLASS (klass);

	gobject_class->set_property = gdata_link_set_property;
	gobject_class->get_property = gdata_link_get_property;
	gobject_class->finalize = gdata_link_finalize;

	parsable_class->pre_parse_xml = pre_parse_xml;
	parsable_class->pre_get_xml = pre_get_xml;
	parsable_class->element_name = "link";

	/**
	 * GDataLink:uri:
	 *
	 * The link's IRI.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php#element.link">Atom specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_URI,
	                                 g_param_spec_string ("uri",
	                                                      "URI", "The link's IRI.",
	                                                      NULL,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataLink:relation-type:
	 *
	 * The link relation type.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php#element.link">Atom specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_RELATION_TYPE,
	                                 g_param_spec_string ("relation-type",
	                                                      "Relation type", "The link relation type.",
	                                                      GDATA_LINK_ALTERNATE,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataLink:content-type:
	 *
	 * An advisory media type: it is a hint about the type of the representation that is expected to be returned when the value
	 * of the #GDataLink:uri property is dereferenced.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php#element.link">Atom specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_CONTENT_TYPE,
	                                 g_param_spec_string ("content-type",
	                                                      "Content type", "An advisory media type.",
	                                                      NULL,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataLink:language:
	 *
	 * Describes the language of the resource pointed to by the #GDataLink:uri property.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php#element.link">Atom specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_LANGUAGE,
	                                 g_param_spec_string ("language",
	                                                      "Language", "Describes the language of the resource pointed to by the uri property.",
	                                                      NULL,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataLink:title:
	 *
	 * Conveys human-readable information about the link.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php#element.link">Atom specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_TITLE,
	                                 g_param_spec_string ("title",
	                                                      "Title", "Conveys human-readable information about the link.",
	                                                      NULL,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataLink:length:
	 *
	 * Indicates an advisory length of the linked content in octets. <code class="literal">-1</code> means the length is unspecified.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php#element.link">Atom specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_LENGTH,
	                                 g_param_spec_int ("length",
	                                                   "Length", "Indicates an advisory length of the linked content in octets.",
	                                                   -1, G_MAXINT, -1,
	                                                   G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static gint
compare_with (GDataComparable *self, GDataComparable *other)
{
	GDataLinkPrivate *a = ((GDataLink*) self)->priv, *b = ((GDataLink*) other)->priv;

	if (g_strcmp0 (a->uri, b->uri) == 0 && g_strcmp0 (a->relation_type, b->relation_type) == 0)
		return 0;
	return 1;
}

static void
gdata_link_comparable_init (GDataComparableIface *iface)
{
	iface->compare_with = compare_with;
}

static void
gdata_link_init (GDataLink *self)
{
	self->priv = gdata_link_get_instance_private (self);
	self->priv->length = -1;
	self->priv->relation_type = g_strdup (GDATA_LINK_ALTERNATE);
}

static void
gdata_link_finalize (GObject *object)
{
	GDataLinkPrivate *priv = GDATA_LINK (object)->priv;

	g_free (priv->uri);
	g_free (priv->relation_type);
	g_free (priv->content_type);
	g_free (priv->language);
	g_free (priv->title);

	/* Chain up to the parent class */
	G_OBJECT_CLASS (gdata_link_parent_class)->finalize (object);
}

static void
gdata_link_get_property (GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
	GDataLinkPrivate *priv = GDATA_LINK (object)->priv;

	switch (property_id) {
		case PROP_URI:
			g_value_set_string (value, priv->uri);
			break;
		case PROP_RELATION_TYPE:
			g_value_set_string (value, priv->relation_type);
			break;
		case PROP_CONTENT_TYPE:
			g_value_set_string (value, priv->content_type);
			break;
		case PROP_LANGUAGE:
			g_value_set_string (value, priv->language);
			break;
		case PROP_TITLE:
			g_value_set_string (value, priv->title);
			break;
		case PROP_LENGTH:
			g_value_set_int (value, priv->length);
			break;
		default:
			/* We don't have any other property... */
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
			break;
	}
}

static void
gdata_link_set_property (GObject *object, guint property_id, const GValue *value, GParamSpec *pspec)
{
	GDataLink *self = GDATA_LINK (object);

	switch (property_id) {
		case PROP_URI:
			gdata_link_set_uri (self, g_value_get_string (value));
			break;
		case PROP_RELATION_TYPE:
			gdata_link_set_relation_type (self, g_value_get_string (value));
			break;
		case PROP_CONTENT_TYPE:
			gdata_link_set_content_type (self, g_value_get_string (value));
			break;
		case PROP_LANGUAGE:
			gdata_link_set_language (self, g_value_get_string (value));
			break;
		case PROP_TITLE:
			gdata_link_set_title (self, g_value_get_string (value));
			break;
		case PROP_LENGTH:
			gdata_link_set_length (self, g_value_get_int (value));
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
	xmlChar *uri, *relation_type, *content_type, *language, *length;
	GDataLink *self = GDATA_LINK (parsable);

	/* href */
	uri = xmlGetProp (root_node, (xmlChar*) "href");
	if (uri == NULL || *uri == '\0') {
		xmlFree (uri);
		return gdata_parser_error_required_property_missing (root_node, "href", error);
	}

	/* rel */
	relation_type = xmlGetProp (root_node, (xmlChar*) "rel");
	if (relation_type != NULL && *relation_type == '\0') {
		xmlFree (uri);
		xmlFree (relation_type);
		return gdata_parser_error_required_property_missing (root_node, "rel", error);
	}

	/* type */
	content_type = xmlGetProp (root_node, (xmlChar*) "type");
	if (content_type != NULL && *content_type == '\0') {
		xmlFree (uri);
		xmlFree (relation_type);
		xmlFree (content_type);
		return gdata_parser_error_required_property_missing (root_node, "type", error);
	}

	/* hreflang */
	language = xmlGetProp (root_node, (xmlChar*) "hreflang");
	if (language != NULL && *language == '\0') {
		xmlFree (uri);
		xmlFree (relation_type);
		xmlFree (content_type);
		xmlFree (language);
		return gdata_parser_error_required_property_missing (root_node, "hreflang", error);
	}

	self->priv->uri = (gchar*) uri;
	gdata_link_set_relation_type (self, (const gchar*) relation_type);
	xmlFree (relation_type);
	self->priv->content_type = (gchar*) content_type;
	self->priv->language = (gchar*) language;
	self->priv->title = (gchar*) xmlGetProp (root_node, (xmlChar*) "title");

	length = xmlGetProp (root_node, (xmlChar*) "length");
	if (length == NULL)
		self->priv->length = -1;
	else
		self->priv->length = g_ascii_strtoull ((gchar*) length, NULL, 10);
	xmlFree (length);

	return TRUE;
}

static void
pre_get_xml (GDataParsable *parsable, GString *xml_string)
{
	GDataLinkPrivate *priv = GDATA_LINK (parsable)->priv;

	gdata_parser_string_append_escaped (xml_string, " href='", priv->uri, "'");

	if (priv->title != NULL)
		gdata_parser_string_append_escaped (xml_string, " title='", priv->title, "'");
	if (priv->relation_type != NULL)
		gdata_parser_string_append_escaped (xml_string, " rel='", priv->relation_type, "'");
	if (priv->content_type != NULL)
		gdata_parser_string_append_escaped (xml_string, " type='", priv->content_type, "'");
	if (priv->language != NULL)
		gdata_parser_string_append_escaped (xml_string, " hreflang='", priv->language, "'");
	if (priv->length != -1)
		g_string_append_printf (xml_string, " length='%i'", priv->length);
}

/**
 * gdata_link_new:
 * @uri: the link's IRI
 * @relation_type: (allow-none): the relationship of the link to the current document, or %NULL
 *
 * Creates a new #GDataLink. More information is available in the <ulink type="http"
 * url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php#element.link">Atom specification</ulink>.
 *
 * @uri must be non-%NULL and non-empty. @relation_type must be %NULL or non-empty.
 *
 * Return value: a new #GDataLink, or %NULL; unref with g_object_unref()
 */
GDataLink *
gdata_link_new (const gchar *uri, const gchar *relation_type)
{
	g_return_val_if_fail (uri != NULL && *uri != '\0', NULL);
	g_return_val_if_fail (relation_type == NULL || *relation_type != '\0', NULL);

	return g_object_new (GDATA_TYPE_LINK, "uri", uri, "relation-type", relation_type, NULL);
}

/**
 * gdata_link_get_uri:
 * @self: a #GDataLink
 *
 * Gets the #GDataLink:uri property. The return value is guaranteed to be a valid IRI, as
 * specified by the Atom protocol. Common relationship values such as <literal>alternate</literal>
 * are returned as <literal>http://www.iana.org/assignments/relation/alternate</literal>.
 *
 * For more information, see the <ulink type="http" url="http://www.atomenabled.org/developers/syndication/atom-format-spec.php#rel_attribute">
 * Atom specification</ulink>.
 *
 * Return value: the link's URI
 *
 * Since: 0.4.0
 */
const gchar *
gdata_link_get_uri (GDataLink *self)
{
	g_return_val_if_fail (GDATA_IS_LINK (self), NULL);
	return self->priv->uri;
}

/**
 * gdata_link_set_uri:
 * @self: a #GDataLink
 * @uri: the new URI for the link
 *
 * Sets the #GDataLink:uri property to @uri. @uri must be non-%NULL and non-empty.
 *
 * Since: 0.4.0
 */
void
gdata_link_set_uri (GDataLink *self, const gchar *uri)
{
	g_return_if_fail (GDATA_IS_LINK (self));
	g_return_if_fail (uri != NULL && *uri != '\0');

	g_free (self->priv->uri);
	self->priv->uri = g_strdup (uri);
	g_object_notify (G_OBJECT (self), "uri");
}

/**
 * gdata_link_get_relation_type:
 * @self: a #GDataLink
 *
 * Gets the #GDataLink:relation-type property. If the relation type is non-%NULL, it will be non-empty.
 *
 * Return value: (nullable): the link's relation type
 *
 * Since: 0.4.0
 */
const gchar *
gdata_link_get_relation_type (GDataLink *self)
{
	g_return_val_if_fail (GDATA_IS_LINK (self), NULL);
	return self->priv->relation_type;
}

/**
 * gdata_link_set_relation_type:
 * @self: a #GDataLink
 * @relation_type: (allow-none): the new relation type for the link, or %NULL
 *
 * Sets the #GDataLink:relation-type property to @relation_type. If @relation_type is one of the standard Atom relation types,
 * use one of the defined relation type values, instead of a static string. e.g. %GDATA_LINK_EDIT or %GDATA_LINK_SELF.
 *
 * Set @relation_type to %NULL to unset the property in the link.
 *
 * Since: 0.4.0
 */
void
gdata_link_set_relation_type (GDataLink *self, const gchar *relation_type)
{
	g_return_if_fail (GDATA_IS_LINK (self));
	g_return_if_fail (relation_type == NULL || *relation_type != '\0');

	/* If the relation type is unset, use the default "alternate" relation type. If it's set, and isn't an IRI, turn it into an IRI
	 * by appending it to "http://www.iana.org/assignments/relation/". If it's set and is an IRI, just use the IRI.
	 * See: http://www.atomenabled.org/developers/syndication/atom-format-spec.php#rel_attribute
	 */
	g_free (self->priv->relation_type);
	if (relation_type == NULL)
		self->priv->relation_type = g_strdup (GDATA_LINK_ALTERNATE);
	else if (strchr ((char*) relation_type, ':') == NULL)
		self->priv->relation_type = g_strconcat ("http://www.iana.org/assignments/relation/", (const gchar*) relation_type, NULL);
	else
		self->priv->relation_type = g_strdup ((gchar*) relation_type);

	g_object_notify (G_OBJECT (self), "relation-type");
}

/**
 * gdata_link_get_content_type:
 * @self: a #GDataLink
 *
 * Gets the #GDataLink:content-type property. If the content type is non-%NULL, it will be non-empty.
 *
 * Return value: (nullable): the link's content type, or %NULL
 *
 * Since: 0.4.0
 */
const gchar *
gdata_link_get_content_type (GDataLink *self)
{
	g_return_val_if_fail (GDATA_IS_LINK (self), NULL);
	return self->priv->content_type;
}

/**
 * gdata_link_set_content_type:
 * @self: a #GDataLink
 * @content_type: (allow-none): the new content type for the link, or %NULL
 *
 * Sets the #GDataLink:content-type property to @content_type. @content_type must be %NULL or non-empty.
 *
 * Set @content_type to %NULL to unset the property in the link.
 *
 * Since: 0.4.0
 */
void
gdata_link_set_content_type (GDataLink *self, const gchar *content_type)
{
	g_return_if_fail (GDATA_IS_LINK (self));
	g_return_if_fail (content_type == NULL || *content_type != '\0');

	g_free (self->priv->content_type);
	self->priv->content_type = g_strdup (content_type);
	g_object_notify (G_OBJECT (self), "content-type");
}

/**
 * gdata_link_get_language:
 * @self: a #GDataLink
 *
 * Gets the #GDataLink:language property. If the language is non-%NULL, it will be non-empty.
 *
 * Return value: (nullable): the link's language, or %NULL
 *
 * Since: 0.4.0
 */
const gchar *
gdata_link_get_language (GDataLink *self)
{
	g_return_val_if_fail (GDATA_IS_LINK (self), NULL);
	return self->priv->language;
}

/**
 * gdata_link_set_language:
 * @self: a #GDataLink
 * @language: (allow-none): the new language for the link, or %NULL
 *
 * Sets the #GDataLink:language property to @language. @language must be %NULL or non-empty.
 *
 * Set @language to %NULL to unset the property in the link.
 *
 * Since: 0.4.0
 */
void
gdata_link_set_language (GDataLink *self, const gchar *language)
{
	g_return_if_fail (GDATA_IS_LINK (self));
	g_return_if_fail (language == NULL || *language != '\0');

	g_free (self->priv->language);
	self->priv->language = g_strdup (language);
	g_object_notify (G_OBJECT (self), "language");
}

/**
 * gdata_link_get_title:
 * @self: a #GDataLink
 *
 * Gets the #GDataLink:title property.
 *
 * Return value: the link's title, or %NULL
 *
 * Since: 0.4.0
 */
const gchar *
gdata_link_get_title (GDataLink *self)
{
	g_return_val_if_fail (GDATA_IS_LINK (self), NULL);
	return self->priv->title;
}

/**
 * gdata_link_set_title:
 * @self: a #GDataLink
 * @title: (allow-none): the new title for the link, or %NULL
 *
 * Sets the #GDataLink:title property to @title.
 *
 * Set @title to %NULL to unset the property in the link.
 *
 * Since: 0.4.0
 */
void
gdata_link_set_title (GDataLink *self, const gchar *title)
{
	g_return_if_fail (GDATA_IS_LINK (self));

	g_free (self->priv->title);
	self->priv->title = g_strdup (title);
	g_object_notify (G_OBJECT (self), "title");
}

/**
 * gdata_link_get_length:
 * @self: a #GDataLink
 *
 * Gets the #GDataLink:length property.
 *
 * Return value: the link's length, or <code class="literal">-1</code>
 *
 * Since: 0.4.0
 */
gint
gdata_link_get_length (GDataLink *self)
{
	g_return_val_if_fail (GDATA_IS_LINK (self), -1);
	return self->priv->length;
}

/**
 * gdata_link_set_length:
 * @self: a #GDataLink
 * @length: the new length for the link, or <code class="literal">-1</code>
 *
 * Sets the #GDataLink:length property to @length.
 *
 * Set @length to <code class="literal">-1</code> to unset the property in the link.
 *
 * Since: 0.4.0
 */
void
gdata_link_set_length (GDataLink *self, gint length)
{
	g_return_if_fail (GDATA_IS_LINK (self));
	g_return_if_fail (length >= -1);

	self->priv->length = length;
	g_object_notify (G_OBJECT (self), "length");
}
