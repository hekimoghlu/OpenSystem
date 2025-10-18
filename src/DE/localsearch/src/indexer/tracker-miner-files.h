/*
 * Copyright (C) 2008, Nokia <ivan.frade@nokia.com>
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

#ifndef __TRACKER_MINER_FS_FILES_H__
#define __TRACKER_MINER_FS_FILES_H__

#include <gio/gio.h>
#include <gudev/gudev.h>

#include "tracker-config.h"

#include "tracker-miner-fs.h"
#include "tracker-monitor.h"
#include "tracker-storage.h"

G_BEGIN_DECLS

#define TRACKER_TYPE_MINER_FILES tracker_miner_files_get_type()
G_DECLARE_FINAL_TYPE (TrackerMinerFiles, tracker_miner_files, TRACKER, MINER_FILES, TrackerMinerFS)

TrackerMiner * tracker_miner_files_new (TrackerSparqlConnection *connection,
                                        TrackerIndexingTree     *indexing_tree,
                                        TrackerMonitor          *monitor);

G_END_DECLS

#endif /* __TRACKER_MINER_FS_FILES_H__ */
