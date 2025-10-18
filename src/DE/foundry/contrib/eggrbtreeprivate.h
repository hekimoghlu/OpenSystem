/* eggrbtree.h
 * Copyright (C) 2000  Red Hat, Inc.,  Jonathan Blandford <jrb@redhat.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 */

/* A Red-Black Tree implementation used specifically by EggTreeView.
 */
#pragma once

#include <glib.h>


G_BEGIN_DECLS


typedef struct _EggRbTree EggRbTree;

typedef void            (* EggRbTreeAugmentFunc)        (EggRbTree               *tree,
                                                         gpointer                 node_augment,
                                                         gpointer                 node,
                                                         gpointer                 left,
                                                         gpointer                 right);

EggRbTree *          egg_rb_tree_new_for_size           (gsize                    element_size,
                                                         gsize                    augment_size,
                                                         EggRbTreeAugmentFunc     augment_func,
                                                         GDestroyNotify           clear_func,
                                                         GDestroyNotify           clear_augment_func);
#define egg_rb_tree_new(type, augment_type, augment_func, clear_func, clear_augment_func) \
  egg_rb_tree_new_for_size (sizeof (type), sizeof (augment_type), (augment_func), (clear_func), (clear_augment_func))

EggRbTree *          egg_rb_tree_ref                    (EggRbTree               *tree);
void                 egg_rb_tree_unref                  (EggRbTree               *tree);

gpointer             egg_rb_tree_get_root               (EggRbTree               *tree);
gpointer             egg_rb_tree_get_first              (EggRbTree               *tree);
gpointer             egg_rb_tree_get_last               (EggRbTree               *tree);

gpointer             egg_rb_tree_node_get_previous      (gpointer                 node);
gpointer             egg_rb_tree_node_get_next          (gpointer                 node);
gpointer             egg_rb_tree_node_get_parent        (gpointer                 node);
gpointer             egg_rb_tree_node_get_left          (gpointer                 node);
gpointer             egg_rb_tree_node_get_right         (gpointer                 node);
EggRbTree *          egg_rb_tree_node_get_tree          (gpointer                 node);
void                 egg_rb_tree_node_mark_dirty        (gpointer                 node);

gpointer             egg_rb_tree_get_augment            (EggRbTree               *tree,
                                                         gpointer                 node);

gpointer             egg_rb_tree_insert_before          (EggRbTree               *tree,
                                                         gpointer                 node);
gpointer             egg_rb_tree_insert_after           (EggRbTree               *tree,
                                                         gpointer                 node);
void                 egg_rb_tree_remove                 (EggRbTree               *tree,
                                                         gpointer                 node);
void                 egg_rb_tree_remove_all             (EggRbTree               *tree);


G_END_DECLS


