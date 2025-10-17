/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef ARCHIVE_RB_H_INCLUDED
#define	ARCHIVE_RB_H_INCLUDED

struct archive_rb_node {
	struct archive_rb_node *rb_nodes[2];
	/*
	 * rb_info contains the two flags and the parent back pointer.
	 * We put the two flags in the low two bits since we know that
	 * rb_node will have an alignment of 4 or 8 bytes.
	 */
	uintptr_t rb_info;
};

#define	ARCHIVE_RB_DIR_LEFT		0
#define	ARCHIVE_RB_DIR_RIGHT		1

#define ARCHIVE_RB_TREE_MIN(T) \
    __archive_rb_tree_iterate((T), NULL, ARCHIVE_RB_DIR_LEFT)
#define ARCHIVE_RB_TREE_MAX(T) \
    __archive_rb_tree_iterate((T), NULL, ARCHIVE_RB_DIR_RIGHT)
#define ARCHIVE_RB_TREE_NEXT(T, N) \
    __archive_rb_tree_iterate((T), (N), ARCHIVE_RB_DIR_RIGHT)
#define ARCHIVE_RB_TREE_PREV(T, N) \
    __archive_rb_tree_iterate((T), (N), ARCHIVE_RB_DIR_LEFT)
#define ARCHIVE_RB_TREE_FOREACH(N, T) \
    for ((N) = ARCHIVE_RB_TREE_MIN(T); (N); \
	(N) = ARCHIVE_RB_TREE_NEXT((T), (N)))
#define ARCHIVE_RB_TREE_FOREACH_REVERSE(N, T) \
    for ((N) = ARCHIVE_RB_TREE_MAX(T); (N); \
	(N) = ARCHIVE_RB_TREE_PREV((T), (N)))
#define ARCHIVE_RB_TREE_FOREACH_SAFE(N, T, S) \
    for ((N) = ARCHIVE_RB_TREE_MIN(T); \
	(N) && ((S) = ARCHIVE_RB_TREE_NEXT((T), (N)), 1); \
	(N) = (S))
#define ARCHIVE_RB_TREE_FOREACH_REVERSE_SAFE(N, T, S) \
    for ((N) = ARCHIVE_RB_TREE_MAX(T); \
        (N) && ((S) = ARCHIVE_RB_TREE_PREV((T), (N)), 1); \
        (N) = (S))

/*
 * archive_rbto_compare_nodes_fn:
 *	return a positive value if the first node < the second node.
 *	return a negative value if the first node > the second node.
 *	return 0 if they are considered same.
 *
 * archive_rbto_compare_key_fn:
 *	return a positive value if the node < the key.
 *	return a negative value if the node > the key.
 *	return 0 if they are considered same.
 */

typedef signed int (*const archive_rbto_compare_nodes_fn)(const struct archive_rb_node *,
    const struct archive_rb_node *);
typedef signed int (*const archive_rbto_compare_key_fn)(const struct archive_rb_node *,
    const void *);

struct archive_rb_tree_ops {
	archive_rbto_compare_nodes_fn rbto_compare_nodes;
	archive_rbto_compare_key_fn rbto_compare_key;
};

struct archive_rb_tree {
	struct archive_rb_node *rbt_root;
	const struct archive_rb_tree_ops *rbt_ops;
};

void	__archive_rb_tree_init(struct archive_rb_tree *,
    const struct archive_rb_tree_ops *);
int	__archive_rb_tree_insert_node(struct archive_rb_tree *,
    struct archive_rb_node *);
struct archive_rb_node	*
	__archive_rb_tree_find_node(struct archive_rb_tree *, const void *);
struct archive_rb_node	*
	__archive_rb_tree_find_node_geq(struct archive_rb_tree *, const void *);
struct archive_rb_node	*
	__archive_rb_tree_find_node_leq(struct archive_rb_tree *, const void *);
void	__archive_rb_tree_remove_node(struct archive_rb_tree *, struct archive_rb_node *);
struct archive_rb_node *
	__archive_rb_tree_iterate(struct archive_rb_tree *,
	struct archive_rb_node *, const unsigned int);

#endif	/* ARCHIVE_RB_H_*/
