/*
 * Copyright (C) 2011, Nokia <ivan.frade@nokia.com>
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
 *
 * Author: Carlos Garnacho  <carlos@lanedo.com>
 */

#ifndef __TRACKER_INDEXING_TREE_H__
#define __TRACKER_INDEXING_TREE_H__

#include <gio/gio.h>
#include "tracker-miner-enums.h"

G_BEGIN_DECLS

#define TRACKER_TYPE_INDEXING_TREE         (tracker_indexing_tree_get_type())
#define TRACKER_INDEXING_TREE(o)           (G_TYPE_CHECK_INSTANCE_CAST ((o), TRACKER_TYPE_INDEXING_TREE, TrackerIndexingTree))
#define TRACKER_INDEXING_TREE_CLASS(c)     (G_TYPE_CHECK_CLASS_CAST ((c),    TRACKER_TYPE_INDEXING_TREE, TrackerIndexingTreeClass))
#define TRACKER_IS_INDEXING_TREE(o)        (G_TYPE_CHECK_INSTANCE_TYPE ((o), TRACKER_TYPE_INDEXING_TREE))
#define TRACKER_IS_INDEXING_TREE_CLASS(c)  (G_TYPE_CHECK_CLASS_TYPE ((c),    TRACKER_TYPE_INDEXING_TREE))
#define TRACKER_INDEXING_TREE_GET_CLASS(o) (G_TYPE_INSTANCE_GET_CLASS ((o),  TRACKER_TYPE_INDEXING_TREE, TrackerIndexingTreeClass))

/**
 * TrackerIndexingTree:
 *
 * Base object used to configure indexing within #TrackerMinerFS items.
 */

typedef struct _TrackerIndexingTree TrackerIndexingTree;

struct _TrackerIndexingTree {
	GObject parent_instance;
	gpointer priv;
};

/**
 * TrackerIndexingTreeClass:
 * @parent_class: parent object class
 * @directory_added: Called when a directory is added.
 * @directory_removed: Called when a directory is removed.
 * @directory_updated: Called when a directory is updated.
 * @child_updated: Called when a file inside a directory is updated.
 * @padding: Reserved for future API improvements.
 *
 * Class for the #TrackerIndexingTree.
 */
typedef struct {
	GObjectClass parent_class;

	void (* directory_added)   (TrackerIndexingTree *indexing_tree,
	                            GFile               *directory);
	void (* directory_removed) (TrackerIndexingTree *indexing_tree,
	                            GFile               *directory);
	void (* directory_updated) (TrackerIndexingTree *indexing_tree,
	                            GFile               *directory);
	void (* child_updated)     (TrackerIndexingTree *indexing_tree,
	                            GFile               *root,
	                            GFile               *child);
	/* <Private> */
	gpointer padding[9];
} TrackerIndexingTreeClass;

G_DEFINE_AUTOPTR_CLEANUP_FUNC (TrackerIndexingTree, g_object_unref)

GType                 tracker_indexing_tree_get_type (void) G_GNUC_CONST;

TrackerIndexingTree * tracker_indexing_tree_new      (void);

void      tracker_indexing_tree_add                  (TrackerIndexingTree   *tree,
                                                      GFile                 *directory,
                                                      TrackerDirectoryFlags  flags);
void      tracker_indexing_tree_remove               (TrackerIndexingTree   *tree,
                                                      GFile                 *directory);
gboolean  tracker_indexing_tree_notify_update        (TrackerIndexingTree   *tree,
                                                      GFile                 *file,
                                                      gboolean               recursive);

void      tracker_indexing_tree_add_filter           (TrackerIndexingTree  *tree,
                                                      TrackerFilterType     filter,
                                                      const gchar          *glob_string);
void      tracker_indexing_tree_clear_filters        (TrackerIndexingTree  *tree,
                                                      TrackerFilterType     type);
gboolean  tracker_indexing_tree_file_matches_filter  (TrackerIndexingTree  *tree,
                                                      TrackerFilterType     type,
                                                      GFile                *file);

gboolean  tracker_indexing_tree_file_is_indexable    (TrackerIndexingTree  *tree,
                                                      GFile                *file,
                                                      GFileInfo            *info);
gboolean  tracker_indexing_tree_parent_is_indexable (TrackerIndexingTree  *tree,
                                                     GFile                *file);

gboolean  tracker_indexing_tree_get_filter_hidden    (TrackerIndexingTree  *tree);
void      tracker_indexing_tree_set_filter_hidden    (TrackerIndexingTree  *tree,
                                                      gboolean              filter_hidden);

GFile *   tracker_indexing_tree_get_root             (TrackerIndexingTree    *tree,
                                                      GFile                  *file,
                                                      const char            **id,
                                                      TrackerDirectoryFlags  *directory_flags);
GFile *   tracker_indexing_tree_get_master_root      (TrackerIndexingTree   *tree);

gboolean  tracker_indexing_tree_file_is_root         (TrackerIndexingTree   *tree,
                                                      GFile                 *file);

GList *   tracker_indexing_tree_list_roots           (TrackerIndexingTree   *tree);

void tracker_indexing_tree_clear_allowed_text_patterns (TrackerIndexingTree *tree);

void tracker_indexing_tree_add_allowed_text_pattern (TrackerIndexingTree *tree,
                                                     const char          *pattern_str);

gboolean tracker_indexing_tree_file_has_allowed_text_extension (TrackerIndexingTree *tree,
                                                                GFile               *file);

void tracker_indexing_tree_update_all (TrackerIndexingTree *tree);

gboolean tracker_indexing_tree_save_config (TrackerIndexingTree  *tree,
                                            GFile                *config,
                                            GError              **error);

gboolean tracker_indexing_tree_check_config (TrackerIndexingTree *tree,
                                             GFile               *config);

G_END_DECLS

#endif /* __TRACKER_INDEXING_TREE_H__ */
