/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 8; tab-width: 8 -*- */
/*
 * GData Client
 * Copyright (C) Philip Withnall 2009 <philip@tecnocode.co.uk>
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

#ifndef GDATA_CATEGORY_H
#define GDATA_CATEGORY_H

#include <glib.h>
#include <glib-object.h>

#include <gdata/gdata-parsable.h>

G_BEGIN_DECLS

/**
 * GDATA_CATEGORY_SCHEMA_LABELS:
 *
 * A schema for categories which label the entry they're applied to in some way, such as starring it. The semantics of the various labels
 * (such as %GDATA_CATEGORY_SCHEMA_LABELS_STARRED) are service-specific.
 *
 * Since: 0.11.0
 */
#define GDATA_CATEGORY_SCHEMA_LABELS "http://schemas.google.com/g/2005/labels"

/**
 * GDATA_CATEGORY_SCHEMA_LABELS_SHARED:
 *
 * A term for categories of the %GDATA_CATEGORY_SCHEMA_LABELS schema which labels an entry as being “shared”. The semantics of this label are
 * service-specific, but are obvious for services such as Google Documents.
 *
 * Since: 0.17.0
 */
#define GDATA_CATEGORY_SCHEMA_LABELS_SHARED GDATA_CATEGORY_SCHEMA_LABELS"#shared"

/**
 * GDATA_CATEGORY_SCHEMA_LABELS_STARRED:
 *
 * A term for categories of the %GDATA_CATEGORY_SCHEMA_LABELS schema which labels an entry as being “starred”. The semantics of this label are
 * service-specific, but are obvious for services such as Google Documents.
 *
 * Since: 0.11.0
 */
#define GDATA_CATEGORY_SCHEMA_LABELS_STARRED GDATA_CATEGORY_SCHEMA_LABELS"#starred"

/**
 * GDATA_CATEGORY_SCHEMA_LABELS_VIEWED:
 *
 * A term for categories of the %GDATA_CATEGORY_SCHEMA_LABELS schema which labels an entry as being “viewed”. The semantics of this label are
 * service-specific, but are obvious for services such as Google Documents.
 *
 * Since: 0.17.0
 */
#define GDATA_CATEGORY_SCHEMA_LABELS_VIEWED GDATA_CATEGORY_SCHEMA_LABELS"#viewed"

#define GDATA_TYPE_CATEGORY		(gdata_category_get_type ())
#define GDATA_CATEGORY(o)		(G_TYPE_CHECK_INSTANCE_CAST ((o), GDATA_TYPE_CATEGORY, GDataCategory))
#define GDATA_CATEGORY_CLASS(k)		(G_TYPE_CHECK_CLASS_CAST((k), GDATA_TYPE_CATEGORY, GDataCategoryClass))
#define GDATA_IS_CATEGORY(o)		(G_TYPE_CHECK_INSTANCE_TYPE ((o), GDATA_TYPE_CATEGORY))
#define GDATA_IS_CATEGORY_CLASS(k)	(G_TYPE_CHECK_CLASS_TYPE ((k), GDATA_TYPE_CATEGORY))
#define GDATA_CATEGORY_GET_CLASS(o)	(G_TYPE_INSTANCE_GET_CLASS ((o), GDATA_TYPE_CATEGORY, GDataCategoryClass))

typedef struct _GDataCategoryPrivate	GDataCategoryPrivate;

/**
 * GDataCategory:
 *
 * All the fields in the #GDataCategory structure are private and should never be accessed directly.
 */
typedef struct {
	GDataParsable parent;
	GDataCategoryPrivate *priv;
} GDataCategory;

/**
 * GDataCategoryClass:
 *
 * All the fields in the #GDataCategoryClass structure are private and should never be accessed directly.
 *
 * Since: 0.4.0
 */
typedef struct {
	/*< private >*/
	GDataParsableClass parent;

	/*< private >*/
	/* Padding for future expansion */
	void (*_g_reserved0) (void);
	void (*_g_reserved1) (void);
} GDataCategoryClass;

GType gdata_category_get_type (void) G_GNUC_CONST;

GDataCategory *gdata_category_new (const gchar *term, const gchar *scheme, const gchar *label) G_GNUC_WARN_UNUSED_RESULT G_GNUC_MALLOC;

const gchar *gdata_category_get_term (GDataCategory *self) G_GNUC_PURE;
void gdata_category_set_term (GDataCategory *self, const gchar *term);

const gchar *gdata_category_get_scheme (GDataCategory *self) G_GNUC_PURE;
void gdata_category_set_scheme (GDataCategory *self, const gchar *scheme);

const gchar *gdata_category_get_label (GDataCategory *self) G_GNUC_PURE;
void gdata_category_set_label (GDataCategory *self, const gchar *label);

G_END_DECLS

#endif /* !GDATA_CATEGORY_H */
