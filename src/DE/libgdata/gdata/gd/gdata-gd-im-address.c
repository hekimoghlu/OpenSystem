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
 * SECTION:gdata-gd-im-address
 * @short_description: GData instant messaging address element
 * @stability: Stable
 * @include: gdata/gd/gdata-gd-im-address.h
 *
 * #GDataGDIMAddress represents an "im" element from the
 * <ulink type="http" url="http://code.google.com/apis/gdata/docs/2.0/elements.html#gdIm">GData specification</ulink>.
 *
 * Since: 0.4.0
 */

#include <glib.h>
#include <libxml/parser.h>

#include "gdata-gd-im-address.h"
#include "gdata-parsable.h"
#include "gdata-parser.h"
#include "gdata-comparable.h"

static void gdata_gd_im_address_comparable_init (GDataComparableIface *iface);
static void gdata_gd_im_address_finalize (GObject *object);
static void gdata_gd_im_address_get_property (GObject *object, guint property_id, GValue *value, GParamSpec *pspec);
static void gdata_gd_im_address_set_property (GObject *object, guint property_id, const GValue *value, GParamSpec *pspec);
static gboolean pre_parse_xml (GDataParsable *parsable, xmlDoc *doc, xmlNode *root_node, gpointer user_data, GError **error);
static void pre_get_xml (GDataParsable *parsable, GString *xml_string);
static void get_namespaces (GDataParsable *parsable, GHashTable *namespaces);

struct _GDataGDIMAddressPrivate {
	gchar *address;
	gchar *protocol;
	gchar *relation_type;
	gchar *label;
	gboolean is_primary;
};

enum {
	PROP_ADDRESS = 1,
	PROP_PROTOCOL,
	PROP_RELATION_TYPE,
	PROP_LABEL,
	PROP_IS_PRIMARY
};

G_DEFINE_TYPE_WITH_CODE (GDataGDIMAddress, gdata_gd_im_address, GDATA_TYPE_PARSABLE,
                         G_ADD_PRIVATE (GDataGDIMAddress)
                         G_IMPLEMENT_INTERFACE (GDATA_TYPE_COMPARABLE, gdata_gd_im_address_comparable_init))

static void
gdata_gd_im_address_class_init (GDataGDIMAddressClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
	GDataParsableClass *parsable_class = GDATA_PARSABLE_CLASS (klass);

	gobject_class->get_property = gdata_gd_im_address_get_property;
	gobject_class->set_property = gdata_gd_im_address_set_property;
	gobject_class->finalize = gdata_gd_im_address_finalize;

	parsable_class->pre_parse_xml = pre_parse_xml;
	parsable_class->pre_get_xml = pre_get_xml;
	parsable_class->get_namespaces = get_namespaces;
	parsable_class->element_name = "im";
	parsable_class->element_namespace = "gd";

	/**
	 * GDataGDIMAddress:address:
	 *
	 * The IM address itself.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://code.google.com/apis/gdata/docs/2.0/elements.html#gdIm">GData specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_ADDRESS,
	                                 g_param_spec_string ("address",
	                                                      "Address", "The IM address itself.",
	                                                      NULL,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataGDIMAddress:protocol:
	 *
	 * Identifies the IM network. For example: %GDATA_GD_IM_PROTOCOL_JABBER or %GDATA_GD_IM_PROTOCOL_GOOGLE_TALK.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://code.google.com/apis/gdata/docs/2.0/elements.html#gdIm">GData specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_PROTOCOL,
	                                 g_param_spec_string ("protocol",
	                                                      "Protocol", "Identifies the IM network.",
	                                                      NULL,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataGDIMAddress:relation-type:
	 *
	 * A programmatic value that identifies the type of IM address. For example: %GDATA_GD_IM_ADDRESS_HOME or %GDATA_GD_IM_ADDRESS_WORK.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://code.google.com/apis/gdata/docs/2.0/elements.html#gdIm">GData specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_RELATION_TYPE,
	                                 g_param_spec_string ("relation-type",
	                                                      "Relation type", "A programmatic value that identifies the type of IM address.",
	                                                      NULL,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataGDIMAddress:label:
	 *
	 * A simple string value used to name this IM address. It allows UIs to display a label such as "Work", "Personal", "Preferred", etc.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://code.google.com/apis/gdata/docs/2.0/elements.html#gdIm">GData specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_LABEL,
	                                 g_param_spec_string ("label",
	                                                      "Label", "A simple string value used to name this IM address.",
	                                                      NULL,
	                                                      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	/**
	 * GDataGDIMAddress:is-primary:
	 *
	 * Indicates which IM address out of a group is primary.
	 *
	 * For more information, see the
	 * <ulink type="http" url="http://code.google.com/apis/gdata/docs/2.0/elements.html#gdIm">GData specification</ulink>.
	 *
	 * Since: 0.4.0
	 */
	g_object_class_install_property (gobject_class, PROP_IS_PRIMARY,
	                                 g_param_spec_boolean ("is-primary",
	                                                       "Primary?", "Indicates which IM address out of a group is primary.",
	                                                       FALSE,
	                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static gint
compare_with (GDataComparable *self, GDataComparable *other)
{
	GDataGDIMAddressPrivate *a = ((GDataGDIMAddress*) self)->priv, *b = ((GDataGDIMAddress*) other)->priv;

	if (g_strcmp0 (a->address, b->address) == 0 && g_strcmp0 (a->protocol, b->protocol) == 0)
		return 0;
	return 1;
}

static void
gdata_gd_im_address_comparable_init (GDataComparableIface *iface)
{
	iface->compare_with = compare_with;
}

static void
gdata_gd_im_address_init (GDataGDIMAddress *self)
{
	self->priv = gdata_gd_im_address_get_instance_private (self);
}

static void
gdata_gd_im_address_finalize (GObject *object)
{
	GDataGDIMAddressPrivate *priv = GDATA_GD_IM_ADDRESS (object)->priv;

	g_free (priv->address);
	g_free (priv->protocol);
	g_free (priv->relation_type);
	g_free (priv->label);

	/* Chain up to the parent class */
	G_OBJECT_CLASS (gdata_gd_im_address_parent_class)->finalize (object);
}

static void
gdata_gd_im_address_get_property (GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
	GDataGDIMAddressPrivate *priv = GDATA_GD_IM_ADDRESS (object)->priv;

	switch (property_id) {
		case PROP_ADDRESS:
			g_value_set_string (value, priv->address);
			break;
		case PROP_PROTOCOL:
			g_value_set_string (value, priv->protocol);
			break;
		case PROP_RELATION_TYPE:
			g_value_set_string (value, priv->relation_type);
			break;
		case PROP_LABEL:
			g_value_set_string (value, priv->label);
			break;
		case PROP_IS_PRIMARY:
			g_value_set_boolean (value, priv->is_primary);
			break;
		default:
			/* We don't have any other property... */
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
			break;
	}
}

static void
gdata_gd_im_address_set_property (GObject *object, guint property_id, const GValue *value, GParamSpec *pspec)
{
	GDataGDIMAddress *self = GDATA_GD_IM_ADDRESS (object);

	switch (property_id) {
		case PROP_ADDRESS:
			gdata_gd_im_address_set_address (self, g_value_get_string (value));
			break;
		case PROP_PROTOCOL:
			gdata_gd_im_address_set_protocol (self, g_value_get_string (value));
			break;
		case PROP_RELATION_TYPE:
			gdata_gd_im_address_set_relation_type (self, g_value_get_string (value));
			break;
		case PROP_LABEL:
			gdata_gd_im_address_set_label (self, g_value_get_string (value));
			break;
		case PROP_IS_PRIMARY:
			gdata_gd_im_address_set_is_primary (self, g_value_get_boolean (value));
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
	xmlChar *address, *rel;
	gboolean primary_bool;
	GDataGDIMAddressPrivate *priv = GDATA_GD_IM_ADDRESS (parsable)->priv;

	/* Is it the primary IM address? */
	if (gdata_parser_boolean_from_property (root_node, "primary", &primary_bool, 0, error) == FALSE)
		return FALSE;

	address = xmlGetProp (root_node, (xmlChar*) "address");
	if (address == NULL || *address == '\0') {
		xmlFree (address);
		return gdata_parser_error_required_property_missing (root_node, "address", error);
	}

	rel = xmlGetProp (root_node, (xmlChar*) "rel");
	if (rel != NULL && *rel == '\0') {
		xmlFree (address);
		xmlFree (rel);
		return gdata_parser_error_required_property_missing (root_node, "rel", error);
	}

	priv->address = (gchar*) address;
	priv->protocol = (gchar*) xmlGetProp (root_node, (xmlChar*) "protocol");
	priv->relation_type = (gchar*) rel;
	priv->label = (gchar*) xmlGetProp (root_node, (xmlChar*) "label");
	priv->is_primary = primary_bool;

	return TRUE;
}

static void
pre_get_xml (GDataParsable *parsable, GString *xml_string)
{
	GDataGDIMAddressPrivate *priv = GDATA_GD_IM_ADDRESS (parsable)->priv;

	gdata_parser_string_append_escaped (xml_string, " address='", priv->address, "'");
	if (priv->protocol != NULL)
		gdata_parser_string_append_escaped (xml_string, " protocol='", priv->protocol, "'");

	if (priv->relation_type != NULL)
		gdata_parser_string_append_escaped (xml_string, " rel='", priv->relation_type, "'");
	if (priv->label != NULL)
		gdata_parser_string_append_escaped (xml_string, " label='", priv->label, "'");

	if (priv->is_primary == TRUE)
		g_string_append (xml_string, " primary='true'");
	else
		g_string_append (xml_string, " primary='false'");
}

static void
get_namespaces (GDataParsable *parsable, GHashTable *namespaces)
{
	g_hash_table_insert (namespaces, (gchar*) "gd", (gchar*) "http://schemas.google.com/g/2005");
}

/**
 * gdata_gd_im_address_new:
 * @address: the IM address
 * @protocol: (allow-none): a URI identifying the IM protocol, or %NULL
 * @relation_type: (allow-none): the relationship between the IM address and its owner, or %NULL
 * @label: (allow-none): a human-readable label for the IM address, or %NULL
 * @is_primary: %TRUE if this IM address is its owner's primary address, %FALSE otherwise
 *
 * Creates a new #GDataGDIMAddress. More information is available in the <ulink type="http"
 * url="http://code.google.com/apis/gdata/docs/2.0/elements.html#gdIm">GData specification</ulink>.
 *
 * Return value: a new #GDataGDIMAddress, or %NULL; unref with g_object_unref()
 *
 * Since: 0.2.0
 */
GDataGDIMAddress *
gdata_gd_im_address_new (const gchar *address, const gchar *protocol, const gchar *relation_type, const gchar *label, gboolean is_primary)
{
	g_return_val_if_fail (address != NULL && *address != '\0', NULL);
	g_return_val_if_fail (protocol == NULL || *protocol != '\0', NULL);
	g_return_val_if_fail (relation_type == NULL || *relation_type != '\0', NULL);
	return g_object_new (GDATA_TYPE_GD_IM_ADDRESS, "address", address, "protocol", protocol, "relation-type", relation_type,
	                     "label", label, "is-primary", is_primary, NULL);
}

/**
 * gdata_gd_im_address_get_address:
 * @self: a #GDataGDIMAddress
 *
 * Gets the #GDataGDIMAddress:address property.
 *
 * Return value: the IM address itself, or %NULL
 *
 * Since: 0.4.0
 */
const gchar *
gdata_gd_im_address_get_address (GDataGDIMAddress *self)
{
	g_return_val_if_fail (GDATA_IS_GD_IM_ADDRESS (self), NULL);
	return self->priv->address;
}

/**
 * gdata_gd_im_address_set_address:
 * @self: a #GDataGDIMAddress
 * @address: the new IM address
 *
 * Sets the #GDataGDIMAddress:address property to @address.
 *
 * Since: 0.4.0
 */
void
gdata_gd_im_address_set_address (GDataGDIMAddress *self, const gchar *address)
{
	g_return_if_fail (GDATA_IS_GD_IM_ADDRESS (self));
	g_return_if_fail (address != NULL && *address != '\0');

	g_free (self->priv->address);
	self->priv->address = g_strdup (address);
	g_object_notify (G_OBJECT (self), "address");
}

/**
 * gdata_gd_im_address_get_protocol:
 * @self: a #GDataGDIMAddress
 *
 * Gets the #GDataGDIMAddress:protocol property.
 *
 * Return value: the IM address' protocol, or %NULL
 *
 * Since: 0.4.0
 */
const gchar *
gdata_gd_im_address_get_protocol (GDataGDIMAddress *self)
{
	g_return_val_if_fail (GDATA_IS_GD_IM_ADDRESS (self), NULL);
	return self->priv->protocol;
}

/**
 * gdata_gd_im_address_set_protocol:
 * @self: a #GDataGDIMAddress
 * @protocol: (allow-none): the new IM protocol, or %NULL
 *
 * Sets the #GDataGDIMAddress:protocol property to @protocol.
 *
 * Since: 0.4.0
 */
void
gdata_gd_im_address_set_protocol (GDataGDIMAddress *self, const gchar *protocol)
{
	g_return_if_fail (GDATA_IS_GD_IM_ADDRESS (self));
	g_return_if_fail (protocol == NULL || *protocol != '\0');

	g_free (self->priv->protocol);
	self->priv->protocol = g_strdup (protocol);
	g_object_notify (G_OBJECT (self), "protocol");
}

/**
 * gdata_gd_im_address_get_relation_type:
 * @self: a #GDataGDIMAddress
 *
 * Gets the #GDataGDIMAddress:relation-type property.
 *
 * Return value: the IM address' relation type, or %NULL
 *
 * Since: 0.4.0
 */
const gchar *
gdata_gd_im_address_get_relation_type (GDataGDIMAddress *self)
{
	g_return_val_if_fail (GDATA_IS_GD_IM_ADDRESS (self), NULL);
	return self->priv->relation_type;
}

/**
 * gdata_gd_im_address_set_relation_type:
 * @self: a #GDataGDIMAddress
 * @relation_type: (allow-none): the new relation type for the im_address, or %NULL
 *
 * Sets the #GDataGDIMAddress:relation-type property to @relation_type.
 *
 * Set @relation_type to %NULL to unset the property in the IM address.
 *
 * Since: 0.4.0
 */
void
gdata_gd_im_address_set_relation_type (GDataGDIMAddress *self, const gchar *relation_type)
{
	g_return_if_fail (GDATA_IS_GD_IM_ADDRESS (self));
	g_return_if_fail (relation_type == NULL || *relation_type != '\0');

	g_free (self->priv->relation_type);
	self->priv->relation_type = g_strdup (relation_type);
	g_object_notify (G_OBJECT (self), "relation-type");
}

/**
 * gdata_gd_im_address_get_label:
 * @self: a #GDataGDIMAddress
 *
 * Gets the #GDataGDIMAddress:label property.
 *
 * Return value: the IM address' label, or %NULL
 *
 * Since: 0.4.0
 */
const gchar *
gdata_gd_im_address_get_label (GDataGDIMAddress *self)
{
	g_return_val_if_fail (GDATA_IS_GD_IM_ADDRESS (self), NULL);
	return self->priv->label;
}

/**
 * gdata_gd_im_address_set_label:
 * @self: a #GDataGDIMAddress
 * @label: (allow-none): the new label for the IM address, or %NULL
 *
 * Sets the #GDataGDIMAddress:label property to @label.
 *
 * Set @label to %NULL to unset the property in the IM address.
 *
 * Since: 0.4.0
 */
void
gdata_gd_im_address_set_label (GDataGDIMAddress *self, const gchar *label)
{
	g_return_if_fail (GDATA_IS_GD_IM_ADDRESS (self));

	g_free (self->priv->label);
	self->priv->label = g_strdup (label);
	g_object_notify (G_OBJECT (self), "label");
}

/**
 * gdata_gd_im_address_is_primary:
 * @self: a #GDataGDIMAddress
 *
 * Gets the #GDataGDIMAddress:is-primary property.
 *
 * Return value: %TRUE if this is the primary IM address, %FALSE otherwise
 *
 * Since: 0.4.0
 */
gboolean
gdata_gd_im_address_is_primary (GDataGDIMAddress *self)
{
	g_return_val_if_fail (GDATA_IS_GD_IM_ADDRESS (self), FALSE);
	return self->priv->is_primary;
}

/**
 * gdata_gd_im_address_set_is_primary:
 * @self: a #GDataGDIMAddress
 * @is_primary: %TRUE if this is the primary IM address, %FALSE otherwise
 *
 * Sets the #GDataGDIMAddress:is-primary property to @is_primary.
 *
 * Since: 0.4.0
 */
void
gdata_gd_im_address_set_is_primary (GDataGDIMAddress *self, gboolean is_primary)
{
	g_return_if_fail (GDATA_IS_GD_IM_ADDRESS (self));

	self->priv->is_primary = is_primary;
	g_object_notify (G_OBJECT (self), "is-primary");
}
