/*
 * Copyright (C) 2009, Nokia <ivan.frade@nokia.com>
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

#ifndef __LIBTRACKER_MINER_MINER_FS_H__
#define __LIBTRACKER_MINER_MINER_FS_H__

#include <glib-object.h>
#include <gio/gio.h>
#include <tinysparql.h>

#include <tracker-common.h>

#include "tracker-indexing-tree.h"
#include "tracker-sparql-buffer.h"

G_BEGIN_DECLS

#define TRACKER_TYPE_MINER_FS tracker_miner_fs_get_type()
G_DECLARE_DERIVABLE_TYPE (TrackerMinerFS, tracker_miner_fs, TRACKER, MINER_FS, TrackerMiner)

struct _TrackerMinerFSClass {
	TrackerMinerClass parent;

	void     (* process_file)             (TrackerMinerFS       *fs,
	                                       GFile                *file,
	                                       GFileInfo            *info,
	                                       TrackerSparqlBuffer  *buffer,
	                                       gboolean              created);
	void     (* finished)                 (TrackerMinerFS       *fs);
	void     (* process_file_attributes)  (TrackerMinerFS       *fs,
	                                       GFile                *file,
	                                       GFileInfo            *info,
	                                       TrackerSparqlBuffer  *buffer);
	void     (* remove_file)              (TrackerMinerFS       *fs,
	                                       GFile                *file,
	                                       TrackerSparqlBuffer  *buffer,
	                                       gboolean              is_dir);
	void     (* remove_children)          (TrackerMinerFS       *fs,
	                                       GFile                *file,
	                                       TrackerSparqlBuffer  *buffer);
	void     (* move_file)                (TrackerMinerFS       *fs,
	                                       GFile                *dest,
	                                       GFile                *source,
	                                       TrackerSparqlBuffer  *buffer,
	                                       gboolean              recursive);
	void (* finish_directory)             (TrackerMinerFS       *fs,
	                                       GFile                *folder,
	                                       TrackerSparqlBuffer  *buffer);
	gchar * (* get_content_identifier)    (TrackerMinerFS       *fs,
					       GFile                *file,
	                                       GFileInfo            *info);
};

/* Properties */
TrackerIndexingTree * tracker_miner_fs_get_indexing_tree     (TrackerMinerFS  *fs);
gdouble               tracker_miner_fs_get_throttle          (TrackerMinerFS  *fs);
void                  tracker_miner_fs_set_throttle          (TrackerMinerFS  *fs,
                                                              gdouble          throttle);

/* URNs */
const gchar * tracker_miner_fs_get_identifier (TrackerMinerFS *miner,
                                               GFile          *file);

/* Progress */
gboolean              tracker_miner_fs_has_items_to_process  (TrackerMinerFS  *fs);

G_END_DECLS

#endif /* __LIBTRACKER_MINER_MINER_FS_H__ */
