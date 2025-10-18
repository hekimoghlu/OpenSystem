/*
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

#ifndef __TRACKERD_EXTRACT_H__
#define __TRACKERD_EXTRACT_H__

#include <gio/gio.h>
#include <tracker-common.h>

#include "utils/tracker-extract.h"

#define TRACKER_TYPE_EXTRACT (tracker_extract_get_type ())
G_DECLARE_FINAL_TYPE (TrackerExtract,
		      tracker_extract,
		      TRACKER, EXTRACT,
		      GObject)

#define TRACKER_EXTRACT_ERROR (tracker_extract_error_quark ())

typedef enum {
	TRACKER_EXTRACT_ERROR_NO_MIMETYPE,
	TRACKER_EXTRACT_ERROR_NO_EXTRACTOR,
	TRACKER_EXTRACT_ERROR_IO_ERROR,
} TrackerExtractError;

GQuark          tracker_extract_error_quark             (void);
TrackerExtract *tracker_extract_new                     (void);

void            tracker_extract_file                    (TrackerExtract         *extract,
                                                         const gchar            *file,
                                                         const gchar            *content_id,
                                                         const gchar            *mimetype,
                                                         GCancellable           *cancellable,
                                                         GAsyncReadyCallback     cb,
                                                         gpointer                user_data);
TrackerExtractInfo *
                tracker_extract_file_finish             (TrackerExtract         *extract,
                                                         GAsyncResult           *res,
                                                         GError                **error);

void            tracker_extract_set_max_text            (TrackerExtract *extract,
                                                         gint            max_text);

TrackerExtractInfo * tracker_extract_file_sync (TrackerExtract  *object,
                                                const gchar     *uri,
                                                const gchar     *content_id,
                                                const gchar     *mimetype,
                                                GError         **error);

#endif /* __TRACKERD_EXTRACT_H__ */
