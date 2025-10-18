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

#ifndef GDATA_MEDIA_THUMBNAIL_H
#define GDATA_MEDIA_THUMBNAIL_H

#include <glib.h>
#include <glib-object.h>

#include <gdata/gdata-download-stream.h>
#include <gdata/gdata-service.h>
#include <gdata/gdata-parsable.h>

G_BEGIN_DECLS

#define GDATA_TYPE_MEDIA_THUMBNAIL		(gdata_media_thumbnail_get_type ())
#define GDATA_MEDIA_THUMBNAIL(o)		(G_TYPE_CHECK_INSTANCE_CAST ((o), GDATA_TYPE_MEDIA_THUMBNAIL, GDataMediaThumbnail))
#define GDATA_MEDIA_THUMBNAIL_CLASS(k)		(G_TYPE_CHECK_CLASS_CAST((k), GDATA_TYPE_MEDIA_THUMBNAIL, GDataMediaThumbnailClass))
#define GDATA_IS_MEDIA_THUMBNAIL(o)		(G_TYPE_CHECK_INSTANCE_TYPE ((o), GDATA_TYPE_MEDIA_THUMBNAIL))
#define GDATA_IS_MEDIA_THUMBNAIL_CLASS(k)	(G_TYPE_CHECK_CLASS_TYPE ((k), GDATA_TYPE_MEDIA_THUMBNAIL))
#define GDATA_MEDIA_THUMBNAIL_GET_CLASS(o)	(G_TYPE_INSTANCE_GET_CLASS ((o), GDATA_TYPE_MEDIA_THUMBNAIL, GDataMediaThumbnailClass))

typedef struct _GDataMediaThumbnailPrivate	GDataMediaThumbnailPrivate;

/**
 * GDataMediaThumbnail:
 *
 * All the fields in the #GDataMediaThumbnail structure are private and should never be accessed directly.
 */
typedef struct {
	GDataParsable parent;
	GDataMediaThumbnailPrivate *priv;
} GDataMediaThumbnail;

/**
 * GDataMediaThumbnailClass:
 *
 * All the fields in the #GDataMediaThumbnailClass structure are private and should never be accessed directly.
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
} GDataMediaThumbnailClass;

GType gdata_media_thumbnail_get_type (void) G_GNUC_CONST;

const gchar *gdata_media_thumbnail_get_uri (GDataMediaThumbnail *self) G_GNUC_PURE;
guint gdata_media_thumbnail_get_height (GDataMediaThumbnail *self) G_GNUC_PURE;
guint gdata_media_thumbnail_get_width (GDataMediaThumbnail *self) G_GNUC_PURE;
gint64 gdata_media_thumbnail_get_time (GDataMediaThumbnail *self) G_GNUC_PURE;

GDataDownloadStream *gdata_media_thumbnail_download (GDataMediaThumbnail *self, GDataService *service, GCancellable *cancellable,
                                                     GError **error) G_GNUC_WARN_UNUSED_RESULT G_GNUC_MALLOC;

G_END_DECLS

#endif /* !GDATA_MEDIA_THUMBNAIL_H */
