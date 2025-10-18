/*
 * Copyright (C) 2009%, Nokia <ivan.frade@nokia.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#ifndef __TRACKER_MINER_FS_CONFIG_H__
#define __TRACKER_MINER_FS_CONFIG_H__

#include <glib-object.h>

G_BEGIN_DECLS

#define TRACKER_TYPE_CONFIG (tracker_config_get_type ())
G_DECLARE_FINAL_TYPE (TrackerConfig, tracker_config, TRACKER, CONFIG, GSettings)

TrackerConfig * tracker_config_new (void);

GSList * tracker_config_get_index_recursive_directories (TrackerConfig *config);

GSList * tracker_config_get_index_single_directories (TrackerConfig *config);

G_END_DECLS

#endif /* __TRACKER_MINER_FS_CONFIG_H__ */
