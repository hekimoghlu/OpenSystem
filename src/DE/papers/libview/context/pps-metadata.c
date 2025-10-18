/* pps-metadata.c
 *  this file is part of papers, a gnome document viewer
 *
 * Copyright (C) 2009 Carlos Garcia Campos  <carlosgc@gnome.org>
 *
 * Papers is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Papers is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include <gio/gio.h>
#include <string.h>

#include "pps-file-helpers.h"
#include "pps-metadata.h"

struct _PpsMetadata {
	GObject base;

	GFile *file;
	GHashTable *items;
};

struct _PpsMetadataClass {
	GObjectClass base_class;
};

G_DEFINE_TYPE (PpsMetadata, pps_metadata, G_TYPE_OBJECT)

#define PPS_METADATA_NAMESPACE "metadata::papers"

static void
pps_metadata_finalize (GObject *object)
{
	PpsMetadata *metadata = PPS_METADATA (object);

	g_clear_pointer (&metadata->items, g_hash_table_destroy);
	g_clear_object (&metadata->file);

	G_OBJECT_CLASS (pps_metadata_parent_class)->finalize (object);
}

static void
pps_metadata_init (PpsMetadata *metadata)
{
	metadata->items = g_hash_table_new_full (g_str_hash,
	                                         g_str_equal,
	                                         g_free,
	                                         g_free);
}

static void
pps_metadata_class_init (PpsMetadataClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS (klass);

	gobject_class->finalize = pps_metadata_finalize;
}

static void
pps_metadata_load (PpsMetadata *metadata)
{
	GFileInfo *info;
	gchar **attrs;
	gint i;
	GError *error = NULL;

	info = g_file_query_info (metadata->file, "metadata::*", 0, NULL, &error);
	if (!info) {
		g_warning ("%s", error->message);
		g_error_free (error);

		return;
	}

	if (!g_file_info_has_namespace (info, "metadata")) {
		g_object_unref (info);

		return;
	}

	attrs = g_file_info_list_attributes (info, "metadata");
	for (i = 0; attrs[i]; i++) {
		GFileAttributeType type;
		gpointer value;
		const gchar *key;

		if (!g_str_has_prefix (attrs[i], PPS_METADATA_NAMESPACE))
			continue;

		if (!g_file_info_get_attribute_data (info, attrs[i],
		                                     &type, &value, NULL)) {
			continue;
		}

		key = attrs[i] + strlen (PPS_METADATA_NAMESPACE "::");

		if (type == G_FILE_ATTRIBUTE_TYPE_STRING) {
			g_hash_table_insert (metadata->items,
			                     g_strdup (key),
			                     g_strdup (value));
		}
	}
	g_strfreev (attrs);
	g_object_unref (info);
}

PpsMetadata *
pps_metadata_new (GFile *file)
{
	PpsMetadata *metadata;

	g_return_val_if_fail (G_IS_FILE (file), NULL);

	metadata = PPS_METADATA (g_object_new (PPS_TYPE_METADATA, NULL));
	if (!pps_file_is_temp (file)) {
		metadata->file = g_object_ref (file);
		pps_metadata_load (metadata);
	}

	return metadata;
}

gboolean
pps_metadata_is_empty (PpsMetadata *metadata)
{
	return g_hash_table_size (metadata->items) == 0;
}

/**
 * pps_metadata_get_string:
 * @metadata: a #PpsMetadata
 * @key: The keyword of the metadata
 * @value: (out) (transfer none):  The value of the returned metadata
 *
 * Returns: %TRUE on success, or %FALSE on failure.
 */
gboolean
pps_metadata_get_string (PpsMetadata *metadata,
                         const gchar *key,
                         const gchar **value)
{
	gchar *v;

	v = g_hash_table_lookup (metadata->items, key);
	if (!v)
		return FALSE;

	*value = v;
	return TRUE;
}

static void
metadata_set_callback (GObject *file,
                       GAsyncResult *result,
                       PpsMetadata *metadata)
{
	GError *error = NULL;

	if (!g_file_set_attributes_finish (G_FILE (file), result, NULL, &error)) {
		g_warning ("%s", error->message);
		g_error_free (error);
	}
}

gboolean
pps_metadata_set_string (PpsMetadata *metadata,
                         const gchar *key,
                         const gchar *value)
{
	GFileInfo *info;
	gchar *gio_key;

	g_hash_table_insert (metadata->items, g_strdup (key), g_strdup (value));
	if (!metadata->file)
		return TRUE;

	info = g_file_info_new ();

	gio_key = g_strconcat (PPS_METADATA_NAMESPACE "::", key, NULL);
	if (value) {
		g_file_info_set_attribute_string (info, gio_key, value);
	} else {
		g_file_info_set_attribute (info, gio_key,
		                           G_FILE_ATTRIBUTE_TYPE_INVALID,
		                           NULL);
	}
	g_free (gio_key);

	g_file_set_attributes_async (metadata->file,
	                             info,
	                             0,
	                             G_PRIORITY_DEFAULT,
	                             NULL,
	                             (GAsyncReadyCallback) metadata_set_callback,
	                             metadata);
	g_object_unref (info);

	return TRUE;
}

/**
 * pps_metadata_get_int:
 * @metadata: a #PpsMetadata
 * @key: The keyword of the metadata
 * @value: (out):  The value of the returned metadata
 *
 * Returns: %TRUE on success, or %FALSE on failure.
 */
gboolean
pps_metadata_get_int (PpsMetadata *metadata,
                      const gchar *key,
                      gint *value)
{
	const gchar *string_value;
	gchar *endptr;
	gint int_value;

	if (!pps_metadata_get_string (metadata, key, &string_value))
		return FALSE;

	int_value = g_ascii_strtoull (string_value, &endptr, 0);
	if (int_value == 0 && string_value == endptr)
		return FALSE;

	*value = int_value;
	return TRUE;
}

gboolean
pps_metadata_set_int (PpsMetadata *metadata,
                      const gchar *key,
                      gint value)
{
	gchar string_value[32];

	g_snprintf (string_value, sizeof (string_value), "%d", value);

	return pps_metadata_set_string (metadata, key, string_value);
}

/**
 * pps_metadata_get_double:
 * @metadata: a #PpsMetadata
 * @key: The keyword of the metadata
 * @value: (out):  The value of the returned metadata
 *
 * Returns: %TRUE on success, or %FALSE on failure.
 */
gboolean
pps_metadata_get_double (PpsMetadata *metadata,
                         const gchar *key,
                         gdouble *value)
{
	const gchar *string_value;
	gchar *endptr;
	gdouble double_value;

	if (!pps_metadata_get_string (metadata, key, &string_value))
		return FALSE;

	double_value = g_ascii_strtod (string_value, &endptr);
	if (double_value == 0. && string_value == endptr)
		return FALSE;

	*value = double_value;
	return TRUE;
}

gboolean
pps_metadata_set_double (PpsMetadata *metadata,
                         const gchar *key,
                         gdouble value)
{
	gchar string_value[G_ASCII_DTOSTR_BUF_SIZE];

	g_ascii_dtostr (string_value, G_ASCII_DTOSTR_BUF_SIZE, value);

	return pps_metadata_set_string (metadata, key, string_value);
}

/**
 * pps_metadata_get_boolean:
 * @metadata: a #PpsMetadata
 * @key: The keyword of the metadata
 * @value: (out):  The value of the returned metadata
 *
 * Returns: %TRUE on success, or %FALSE on failure.
 */
gboolean
pps_metadata_get_boolean (PpsMetadata *metadata,
                          const gchar *key,
                          gboolean *value)
{
	gint int_value;

	if (!pps_metadata_get_int (metadata, key, &int_value))
		return FALSE;

	*value = int_value;
	return TRUE;
}

gboolean
pps_metadata_set_boolean (PpsMetadata *metadata,
                          const gchar *key,
                          gboolean value)
{
	return pps_metadata_set_string (metadata, key, value ? "1" : "0");
}

gboolean
pps_metadata_has_key (PpsMetadata *metadata,
                      const gchar *key)
{
	return g_hash_table_lookup (metadata->items, key) != NULL;
}

gboolean
pps_metadata_is_file_supported (GFile *file)
{
	GFileAttributeInfoList *namespaces;
	gint i;
	gboolean retval = FALSE;

	namespaces = g_file_query_writable_namespaces (file, NULL, NULL);
	if (!namespaces)
		return retval;

	for (i = 0; i < namespaces->n_infos; i++) {
		if (strcmp (namespaces->infos[i].name, "metadata") == 0) {
			retval = TRUE;
			break;
		}
	}

	g_file_attribute_info_list_unref (namespaces);

	return retval;
}
