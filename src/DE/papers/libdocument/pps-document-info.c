// SPDX-License-Identifier: GPL-2.0-or-later
/*
 *  Copyright (C) 2009 Carlos Garcia Campos
 *  Copyright (C) 2004 Marco Pesenti Gritti
 *  Copyright © 2021 Christian Persch
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "pps-document-info.h"
#include "pps-xmp.h"

#ifdef HAVE__NL_MEASUREMENT_MEASUREMENT
#include <langinfo.h>
#endif

#include <glib/gi18n-lib.h>
#include <gtk/gtk.h>

G_DEFINE_BOXED_TYPE (PpsDocumentInfo, pps_document_info, pps_document_info_copy, pps_document_info_free)

/**
 * pps_document_info_new:
 *
 * Returns: (transfer full): a new, empty #PpsDocumentInfo
 */
PpsDocumentInfo *
pps_document_info_new (void)
{
	return g_new0 (PpsDocumentInfo, 1);
}

/**
 * pps_document_info_copy:
 * @info: a #PpsDocumentInfo
 *
 * Returns: (transfer full): a copy of @info
 */
PpsDocumentInfo *
pps_document_info_copy (const PpsDocumentInfo *info)
{
	PpsDocumentInfo *copy;

	g_return_val_if_fail (info != NULL, NULL);

	copy = pps_document_info_new ();

	copy->title = g_strdup (info->title);
	copy->format = g_strdup (info->format);
	copy->author = g_strdup (info->author);
	copy->subject = g_strdup (info->subject);
	copy->keywords = g_strdup (info->keywords);
	copy->security = g_strdup (info->security);
	copy->creator = g_strdup (info->creator);
	copy->producer = g_strdup (info->producer);
	copy->linearized = g_strdup (info->linearized);

	copy->creation_datetime = g_date_time_ref (info->creation_datetime);
	copy->modified_datetime = g_date_time_ref (info->modified_datetime);

	copy->layout = info->layout;
	copy->mode = info->mode;
	copy->ui_hints = info->ui_hints;
	copy->permissions = info->permissions;
	copy->n_pages = info->n_pages;
	copy->license = pps_document_license_copy (info->license);

	copy->fields_mask = info->fields_mask;

	return copy;
}

/**
 * pps_document_info_free:
 * @info: (transfer full): a #PpsDocumentInfo
 *
 * Frees @info.
 */
void
pps_document_info_free (PpsDocumentInfo *info)
{
	if (info == NULL)
		return;

	g_free (info->title);
	g_free (info->format);
	g_free (info->author);
	g_free (info->subject);
	g_free (info->keywords);
	g_free (info->creator);
	g_free (info->producer);
	g_free (info->linearized);
	g_free (info->security);
	pps_document_license_free (info->license);

	g_clear_pointer (&info->creation_datetime, g_date_time_unref);
	g_clear_pointer (&info->modified_datetime, g_date_time_unref);

	g_free (info);
}

/*
 * pps_document_info_take_created_datetime:
 * @info: a #PpsDocumentInfo
 * @datetime: (transfer full): a #GDateTime
 *
 * Sets the #GDateTime for when the document was created.
 */
void
pps_document_info_take_created_datetime (PpsDocumentInfo *info,
                                         GDateTime *datetime)
{
	g_return_if_fail (info != NULL);
	g_clear_pointer (&info->creation_datetime, g_date_time_unref);

	info->creation_datetime = datetime;
	info->fields_mask |= PPS_DOCUMENT_INFO_CREATION_DATETIME;
}

/**
 * pps_document_info_get_created_datetime:
 * @info: a #PpsDocumentInfo
 *
 * Returns: (transfer none) (nullable): a #GDateTime for when the document was created
 */
GDateTime *
pps_document_info_get_created_datetime (const PpsDocumentInfo *info)
{
	g_return_val_if_fail (info != NULL, NULL);
	g_return_val_if_fail (info->fields_mask & PPS_DOCUMENT_INFO_CREATION_DATETIME, NULL);

	return info->creation_datetime;
}

/*
 * pps_document_info_take_modified_datetime:
 * @info: a #PpsDocumentInfo
 * @datetime: (transfer full): a #GDateTime
 *
 * Sets the #GDateTime for when the document was last modified.
 */
void
pps_document_info_take_modified_datetime (PpsDocumentInfo *info,
                                          GDateTime *datetime)
{
	g_return_if_fail (info != NULL);

	g_clear_pointer (&info->modified_datetime, g_date_time_unref);
	info->modified_datetime = datetime;
	info->fields_mask |= PPS_DOCUMENT_INFO_MOD_DATETIME;
}

/**
 * pps_document_info_get_modified_datetime:
 * @info: a #PpsDocumentInfo
 *
 * Returns: (transfer none) (nullable): a #GDateTime for when the document was last modified
 */
GDateTime *
pps_document_info_get_modified_datetime (const PpsDocumentInfo *info)
{
	g_return_val_if_fail (info != NULL, NULL);
	g_return_val_if_fail (info->fields_mask & PPS_DOCUMENT_INFO_MOD_DATETIME, NULL);

	return info->modified_datetime;
}

/*
 * pps_document_info_set_from_xmp:
 * @info: a #PpsDocumentInfo
 * @xmp: a XMP document
 * @size: the size of @xmp in bytes, or -1 if @xmp is a NUL-terminated string
 *
 * Parses the XMP document and sets @info from it.
 *
 * Returns: %TRUE iff @xmp could be successfully parsed as a XMP document
 */
gboolean
pps_document_info_set_from_xmp (PpsDocumentInfo *info,
                                const char *xmp,
                                gssize size)
{
	return pps_xmp_parse (xmp, size != -1 ? size : strlen (xmp), info);
}

static GtkUnit
get_default_user_units (void)
{
	/* Translate to the default units to use for presenting
	 * lengths to the user. Translate to default:inch if you
	 * want inches, otherwise translate to default:mm.
	 * Do *not* translate it to "predefinito:mm", if it
	 * it isn't default:mm or default:inch it will not work
	 */
	gchar *e = _ ("default:mm");

#ifdef HAVE__NL_MEASUREMENT_MEASUREMENT
	gchar *imperial = NULL;

	imperial = nl_langinfo (_NL_MEASUREMENT_MEASUREMENT);
	if (imperial && imperial[0] == 2)
		return GTK_UNIT_INCH; /* imperial */
	if (imperial && imperial[0] == 1)
		return GTK_UNIT_MM; /* metric */
#endif

	if (strcmp (e, "default:mm") == 0)
		return GTK_UNIT_MM;
	if (strcmp (e, "default:inch") == 0)
		return GTK_UNIT_INCH;

	g_warning ("Whoever translated default:mm did so wrongly.\n");

	return GTK_UNIT_MM;
}

static gdouble
get_tolerance (gdouble size)
{
	if (size < 150.0f)
		return 1.5f;
	else if (size >= 150.0f && size <= 600.0f)
		return 2.0f;
	else
		return 3.0f;
}

/**
 * pps_document_info_regular_pager_size:
 * @info: a #PpsDocumentInfo
 *
 * Returns: (transfer full) (nullable): A string represent the regular paper size
 */
char *
pps_document_info_regular_paper_size (const PpsDocumentInfo *info)
{
	GList *paper_sizes, *l;
	gchar *exact_size;
	gchar *str = NULL;
	GtkUnit units;

	g_return_val_if_fail (info->fields_mask & PPS_DOCUMENT_INFO_PAPER_SIZE, NULL);

	units = get_default_user_units ();

	if (units == GTK_UNIT_MM) {
		exact_size = g_strdup_printf (_ ("%.0f × %.0f mm"),
		                              info->paper_width,
		                              info->paper_height);
	} else {
		exact_size = g_strdup_printf (_ ("%.2f × %.2f inch"),
		                              info->paper_width / 25.4f,
		                              info->paper_height / 25.4f);
	}

	paper_sizes = gtk_paper_size_get_paper_sizes (FALSE);

	for (l = paper_sizes; l && l->data; l = g_list_next (l)) {
		GtkPaperSize *size = (GtkPaperSize *) l->data;
		gdouble paper_width;
		gdouble paper_height;
		gdouble width_tolerance;
		gdouble height_tolerance;

		paper_width = gtk_paper_size_get_width (size, GTK_UNIT_MM);
		paper_height = gtk_paper_size_get_height (size, GTK_UNIT_MM);

		width_tolerance = get_tolerance (paper_width);
		height_tolerance = get_tolerance (paper_height);

		if (ABS (info->paper_height - paper_height) <= height_tolerance &&
		    ABS (info->paper_width - paper_width) <= width_tolerance) {
			/* Note to translators: first placeholder is the paper name (eg.
			 * A4), second placeholder is the paper size (eg. 297x210 mm) */
			str = g_strdup_printf (_ ("%s, Portrait (%s)"),
			                       gtk_paper_size_get_display_name (size),
			                       exact_size);
		} else if (ABS (info->paper_width - paper_height) <= height_tolerance &&
		           ABS (info->paper_height - paper_width) <= width_tolerance) {
			/* Note to translators: first placeholder is the paper name (eg.
			 * A4), second placeholder is the paper size (eg. 297x210 mm) */
			str = g_strdup_printf (_ ("%s, Landscape (%s)"),
			                       gtk_paper_size_get_display_name (size),
			                       exact_size);
		}
	}

	g_clear_list (&paper_sizes, (GDestroyNotify) gtk_paper_size_free);

	if (str != NULL) {
		g_free (exact_size);
		return str;
	}

	return exact_size;
}

/**
 * pps_document_info_pages:
 * @info: a #PpsDocumentInfo
 * @pages: (out) (optional):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_pages (const PpsDocumentInfo *info, gint *pages)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_PAPER_SIZE;

	if (has_value && pages)
		*pages = info->n_pages;

	return has_value;
}

/**
 * pps_document_info_permissions:
 * @info: a #PpsDocumentInfo
 * @permissions: (out) (optional):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_permissions (const PpsDocumentInfo *info, PpsDocumentPermissions *permissions)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_PERMISSIONS;

	if (has_value && permissions)
		*permissions = info->permissions;

	return has_value;
}

/**
 * pps_document_info_start_mode:
 * @info: a #PpsDocumentInfo
 * @mode: (out) (optional):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_start_mode (const PpsDocumentInfo *info, PpsDocumentMode *mode)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_START_MODE;

	if (has_value && mode)
		*mode = info->mode;

	return has_value;
}

/**
 * pps_document_info_license:
 * @info: a #PpsDocumentInfo
 * @license: (out) (optional):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_license (const PpsDocumentInfo *info, PpsDocumentLicense **license)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_LICENSE;

	if (has_value && license)
		*license = pps_document_license_copy (info->license);

	return has_value;
}

/**
 * pps_document_info_contains_js:
 * @info: a #PpsDocumentInfo
 * @contains_js: (out) (optional):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_contains_js (const PpsDocumentInfo *info, PpsDocumentContainsJS *contains_js)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_CONTAINS_JS;

	if (has_value && contains_js)
		*contains_js = info->contains_js;

	return has_value;
}

/**
 * pps_document_info_title:
 * @info: a #PpsDocumentInfo
 * @title: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_title (const PpsDocumentInfo *info, gchar **title)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_TITLE;

	if (has_value && title)
		*title = g_strdup (info->title);

	return has_value;
}

/**
 * pps_document_info_format:
 * @info: a #PpsDocumentInfo
 * @format: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_format (const PpsDocumentInfo *info, gchar **format)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_FORMAT;

	if (has_value && format)
		*format = g_strdup (info->format);

	return has_value;
}

/**
 * pps_document_info_author:
 * @info: a #PpsDocumentInfo
 * @author: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_author (const PpsDocumentInfo *info, gchar **author)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_AUTHOR;

	if (has_value && author)
		*author = g_strdup (info->author);

	return has_value;
}

/**
 * pps_document_info_subject:
 * @info: a #PpsDocumentInfo
 * @subject: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_subject (const PpsDocumentInfo *info, gchar **subject)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_SUBJECT;

	if (has_value && subject)
		*subject = g_strdup (info->subject);

	return has_value;
}

/**
 * pps_document_info_keywords:
 * @info: a #PpsDocumentInfo
 * @keywords: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_keywords (const PpsDocumentInfo *info, gchar **keywords)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_KEYWORDS;

	if (has_value && keywords)
		*keywords = g_strdup (info->keywords);

	return has_value;
}

/**
 * pps_document_info_creator:
 * @info: a #PpsDocumentInfo
 * @creator: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_creator (const PpsDocumentInfo *info, gchar **creator)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_CREATOR;

	if (has_value && creator)
		*creator = g_strdup (info->creator);

	return has_value;
}

/**
 * pps_document_info_producer:
 * @info: a #PpsDocumentInfo
 * @producer: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_producer (const PpsDocumentInfo *info, gchar **producer)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_PRODUCER;

	if (has_value && producer)
		*producer = g_strdup (info->producer);

	return has_value;
}

/**
 * pps_document_info_linearized:
 * @info: a #PpsDocumentInfo
 * @linearized: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_linearized (const PpsDocumentInfo *info, gchar **linearized)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_LINEARIZED;

	if (has_value && linearized)
		*linearized = g_strdup (info->linearized);

	return has_value;
}

/**
 * pps_document_info_security:
 * @info: a #PpsDocumentInfo
 * @security: (out) (optional) (transfer full):
 *
 * Returns: %TRUE iff info has this field
 */
gboolean
pps_document_info_security (const PpsDocumentInfo *info, gchar **security)
{
	gboolean has_value = info->fields_mask & PPS_DOCUMENT_INFO_SECURITY;

	if (has_value && security)
		*security = g_strdup (info->security);

	return has_value;
}

/* PpsDocumentLicense */
G_DEFINE_BOXED_TYPE (PpsDocumentLicense, pps_document_license, pps_document_license_copy, pps_document_license_free)

/**
 * pps_document_license_new:
 *
 * Returns: (transfer full): a new, empty #PpsDocumentLicense
 */
PpsDocumentLicense *
pps_document_license_new (void)
{
	return g_new0 (PpsDocumentLicense, 1);
}

/**
 * pps_document_license_copy:
 * @license: (nullable): a #PpsDocumentLicense
 *
 * Returns: (transfer full): a copy of @license, or %NULL
 */
PpsDocumentLicense *
pps_document_license_copy (PpsDocumentLicense *license)
{
	PpsDocumentLicense *new_license;

	if (!license)
		return NULL;

	new_license = pps_document_license_new ();

	if (license->text)
		new_license->text = g_strdup (license->text);
	if (license->uri)
		new_license->uri = g_strdup (license->uri);
	if (license->web_statement)
		new_license->web_statement = g_strdup (license->web_statement);

	return new_license;
}

/**
 * pps_document_license_free:
 * @license: (transfer full): a #PpsDocumentLicense
 *
 * Frees @license.
 */
void
pps_document_license_free (PpsDocumentLicense *license)
{
	if (!license)
		return;

	g_free (license->text);
	g_free (license->uri);
	g_free (license->web_statement);

	g_free (license);
}

/**
 * pps_document_license_get_text:
 * @license: (transfer full): a #PpsDocumentLicense
 *
 * Returns: (transfer none) (nullable): the license text
 */
const gchar *
pps_document_license_get_text (PpsDocumentLicense *license)
{
	return license->text;
}

/**
 * pps_document_license_get_uri:
 * @license: (transfer full): a #PpsDocumentLicense
 *
 * Returns: (transfer none) (nullable): the license URI
 */
const gchar *
pps_document_license_get_uri (PpsDocumentLicense *license)
{
	return license->uri;
}

/**
 * pps_document_license_get_web_statement
 * @license: (transfer full): a #PpsDocumentLicense
 *
 * Returns: (transfer none) (nullable): the license web statement
 */
const gchar *
pps_document_license_get_web_statement (PpsDocumentLicense *license)
{
	return license->web_statement;
}
