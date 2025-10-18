/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 8; tab-width: 8 -*- */
/*
 * GData Client
 * Copyright (C) Philip Withnall 2009, 2015 <philip@tecnocode.co.uk>
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

#ifndef GDATA_YOUTUBE_QUERY_H
#define GDATA_YOUTUBE_QUERY_H

#include <glib.h>
#include <glib-object.h>

#include <gdata/gdata-query.h>
#include <gdata/gdata-types.h>
#include <gdata/services/youtube/gdata-youtube-enums.h>

G_BEGIN_DECLS

/**
 * GDataYouTubeSafeSearch:
 * @GDATA_YOUTUBE_SAFE_SEARCH_NONE: YouTube will not perform any filtering on the search result set
 * @GDATA_YOUTUBE_SAFE_SEARCH_MODERATE: YouTube will filter some content from search results and, at the least,
 * will filter content that is restricted in your locale
 * @GDATA_YOUTUBE_SAFE_SEARCH_STRICT: YouTube will try to exclude all restricted content from the search result set
 *
 * Safe search levels for removing restricted entries from query results. For more information, see the
 * <ulink type="http" url="https://developers.google.com/youtube/v3/docs/search/list#safeSearch">online documentation</ulink>.
 *
 * Since: 0.3.0
 */
typedef enum {
	GDATA_YOUTUBE_SAFE_SEARCH_NONE = 0,
	GDATA_YOUTUBE_SAFE_SEARCH_MODERATE,
	GDATA_YOUTUBE_SAFE_SEARCH_STRICT
} GDataYouTubeSafeSearch;

/**
 * GDataYouTubeAge:
 * @GDATA_YOUTUBE_AGE_ALL_TIME: retrieve all videos, regardless of the date they were uploaded
 * @GDATA_YOUTUBE_AGE_TODAY: retrieve only videos uploaded in the past day
 * @GDATA_YOUTUBE_AGE_THIS_WEEK: retrieve only videos uploaded in the past week
 * @GDATA_YOUTUBE_AGE_THIS_MONTH: retrieve only videos uploaded in the past month
 *
 * Video ages, allowing queries to be limited to videos uploaded in a recent time period.
 *
 * Since: 0.3.0
 */
typedef enum {
	GDATA_YOUTUBE_AGE_ALL_TIME = 0,
	GDATA_YOUTUBE_AGE_TODAY,
	GDATA_YOUTUBE_AGE_THIS_WEEK,
	GDATA_YOUTUBE_AGE_THIS_MONTH
} GDataYouTubeAge;

/**
 * GDATA_YOUTUBE_LICENSE_CC:
 *
 * Value for #GDataYouTubeQuery:license to restrict search results to only videos which are Creative Commons licensed. Specifically, the license
 * is the Creative Commons Attribution 3.0 Unported license; see the
 * <ulink type="http" url="http://www.google.com/support/youtube/bin/answer.py?hl=en&answer=1284989">YouTube Help</ulink> for more information.
 *
 * Since: 0.11.0
 */
#define GDATA_YOUTUBE_LICENSE_CC "cc"

/**
 * GDATA_YOUTUBE_LICENSE_STANDARD:
 *
 * Value for #GDataYouTubeQuery:license to restrict search results to only videos which are under the standard YouTube license.
 *
 * Since: 0.11.0
 */
#define GDATA_YOUTUBE_LICENSE_STANDARD "youtube"

#define GDATA_TYPE_YOUTUBE_QUERY		(gdata_youtube_query_get_type ())
#define GDATA_YOUTUBE_QUERY(o)			(G_TYPE_CHECK_INSTANCE_CAST ((o), GDATA_TYPE_YOUTUBE_QUERY, GDataYouTubeQuery))
#define GDATA_YOUTUBE_QUERY_CLASS(k)		(G_TYPE_CHECK_CLASS_CAST((k), GDATA_TYPE_YOUTUBE_QUERY, GDataYouTubeQueryClass))
#define GDATA_IS_YOUTUBE_QUERY(o)		(G_TYPE_CHECK_INSTANCE_TYPE ((o), GDATA_TYPE_YOUTUBE_QUERY))
#define GDATA_IS_YOUTUBE_QUERY_CLASS(k)		(G_TYPE_CHECK_CLASS_TYPE ((k), GDATA_TYPE_YOUTUBE_QUERY))
#define GDATA_YOUTUBE_QUERY_GET_CLASS(o)	(G_TYPE_INSTANCE_GET_CLASS ((o), GDATA_TYPE_YOUTUBE_QUERY, GDataYouTubeQueryClass))

typedef struct _GDataYouTubeQueryPrivate	GDataYouTubeQueryPrivate;

/**
 * GDataYouTubeQuery:
 *
 * All the fields in the #GDataYouTubeQuery structure are private and should never be accessed directly.
 *
 * Since: 0.3.0
 */
typedef struct {
	GDataQuery parent;
	GDataYouTubeQueryPrivate *priv;
} GDataYouTubeQuery;

/**
 * GDataYouTubeQueryClass:
 *
 * All the fields in the #GDataYouTubeQueryClass structure are private and should never be accessed directly.
 *
 * Since: 0.3.0
 */
typedef struct {
	/*< private >*/
	GDataQueryClass parent;

	/*< private >*/
	/* Padding for future expansion */
	void (*_g_reserved0) (void);
	void (*_g_reserved1) (void);
} GDataYouTubeQueryClass;

GType gdata_youtube_query_get_type (void) G_GNUC_CONST;

GDataYouTubeQuery *gdata_youtube_query_new (const gchar *q) G_GNUC_WARN_UNUSED_RESULT G_GNUC_MALLOC;

void gdata_youtube_query_get_location (GDataYouTubeQuery *self, gdouble *latitude, gdouble *longitude, gdouble *radius);
void gdata_youtube_query_set_location (GDataYouTubeQuery *self, gdouble latitude, gdouble longitude, gdouble radius);
const gchar *gdata_youtube_query_get_order_by (GDataYouTubeQuery *self) G_GNUC_PURE;
void gdata_youtube_query_set_order_by (GDataYouTubeQuery *self, const gchar *order_by);
const gchar *gdata_youtube_query_get_restriction (GDataYouTubeQuery *self) G_GNUC_PURE;
void gdata_youtube_query_set_restriction (GDataYouTubeQuery *self, const gchar *restriction);
GDataYouTubeSafeSearch gdata_youtube_query_get_safe_search (GDataYouTubeQuery *self) G_GNUC_PURE;
void gdata_youtube_query_set_safe_search (GDataYouTubeQuery *self, GDataYouTubeSafeSearch safe_search);
GDataYouTubeAge gdata_youtube_query_get_age (GDataYouTubeQuery *self) G_GNUC_PURE;
void gdata_youtube_query_set_age (GDataYouTubeQuery *self, GDataYouTubeAge age);
const gchar *gdata_youtube_query_get_license (GDataYouTubeQuery *self) G_GNUC_PURE;
void gdata_youtube_query_set_license (GDataYouTubeQuery *self, const gchar *license);

G_END_DECLS

#endif /* !GDATA_YOUTUBE_QUERY_H */
