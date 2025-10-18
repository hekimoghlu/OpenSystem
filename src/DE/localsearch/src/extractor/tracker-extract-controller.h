/*
 * Copyright (C) 2014 - Collabora Ltd.
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

#ifndef __TRACKER_EXTRACT_CONTROLLER_H__
#define __TRACKER_EXTRACT_CONTROLLER_H__

#include <gio/gio.h>

#include "tracker-decorator.h"

G_BEGIN_DECLS

#define TRACKER_TYPE_EXTRACT_CONTROLLER (tracker_extract_controller_get_type ())
G_DECLARE_FINAL_TYPE (TrackerExtractController,
                      tracker_extract_controller,
                      TRACKER, EXTRACT_CONTROLLER,
                      GObject)

TrackerExtractController * tracker_extract_controller_new (TrackerDecorator           *decorator,
                                                           TrackerExtract             *extractor,
                                                           GDBusConnection            *connection,
                                                           TrackerExtractPersistence  *persistence,
                                                           GError                    **error);

G_END_DECLS

#endif /* __TRACKER_EXTRACT_CONTROLLER_H__ */
