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

#ifndef __LIBTRACKER_MINER_STORAGE_H__
#define __LIBTRACKER_MINER_STORAGE_H__

#include <glib-object.h>
#include <gio/gio.h>

G_BEGIN_DECLS

/**
 * TrackerStorageType:
 * @TRACKER_STORAGE_REMOVABLE: Storage is a removable media
 * @TRACKER_STORAGE_OPTICAL: Storage is an optical disc
 *
 * Flags specifying properties of the type of storage.
 *
 * Since: 0.8
 */
typedef enum {
	TRACKER_STORAGE_REMOVABLE = 1 << 0,
	TRACKER_STORAGE_OPTICAL   = 1 << 1
} TrackerStorageType;

/**
 * TRACKER_STORAGE_TYPE_IS_REMOVABLE:
 * @type: Mask of TrackerStorageType flags
 *
 * Check if the given storage type is marked as being removable media.
 *
 * Returns: %TRUE if the storage is marked as removable media, %FALSE otherwise
 *
 * Since: 0.10
 */
#define TRACKER_STORAGE_TYPE_IS_REMOVABLE(type) ((type & TRACKER_STORAGE_REMOVABLE) ? TRUE : FALSE)

/**
 * TRACKER_STORAGE_TYPE_IS_OPTICAL:
 * @type: Mask of TrackerStorageType flags
 *
 * Check if the given storage type is marked as being optical disc
 *
 * Returns: %TRUE if the storage is marked as optical disc, %FALSE otherwise
 *
 * Since: 0.10
 */
#define TRACKER_STORAGE_TYPE_IS_OPTICAL(type) ((type & TRACKER_STORAGE_OPTICAL) ? TRUE : FALSE)

#define TRACKER_TYPE_STORAGE (tracker_storage_get_type ())
G_DECLARE_FINAL_TYPE (TrackerStorage,
                      tracker_storage,
                      TRACKER, STORAGE,
                      GObject)

TrackerStorage *   tracker_storage_new                      (void);
GSList *           tracker_storage_get_device_roots         (TrackerStorage     *storage,
                                                             TrackerStorageType  type,
                                                             gboolean            exact_match);

TrackerStorageType tracker_storage_get_type_for_file (TrackerStorage *storage,
                                                      GFile          *file);

G_END_DECLS

#endif /* __LIBTRACKER_MINER_STORAGE_H__ */
