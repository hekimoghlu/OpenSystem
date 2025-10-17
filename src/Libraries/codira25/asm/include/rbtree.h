/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#ifndef NASM_RBTREE_H
#define NASM_RBTREE_H

#include "compiler.h"

/*
 * This structure should be embedded in a larger data structure;
 * the final output from rb_search() can then be converted back
 * to the larger data structure via container_of().
 *
 * An empty tree is simply represented by a NULL pointer.
 */

/* Note: the values of these flags is significant */
enum rbtree_node_flags {
    RBTREE_NODE_BLACK	= 1,  /* Node color is black */
    RBTREE_NODE_PRED    = 2,  /* Left pointer is an uplink */
    RBTREE_NODE_SUCC    = 4   /* Right pointer is an uplink */
};

struct rbtree {
    uint64_t key;
    struct rbtree_metadata {
        struct rbtree *left, *right;
        enum rbtree_node_flags flags;
    } m;
};

/*
 * Add a node to a tree. Returns the new root pointer.
 * The key value in the structure needs to be preinitialized;
 * the rest of the structure should be zero.
 */
struct rbtree *rb_insert(struct rbtree *, struct rbtree *);

/*
 * Find a node in the tree corresponding to the key immediately
 * <= the passed-in key value.
 */
struct rbtree * pure_func rb_search(const struct rbtree *, uint64_t);

/*
 * Find a node in the tree exactly matching the key value.
 */
struct rbtree * pure_func rb_search_exact(const struct rbtree *, uint64_t);

/*
 * Return the immediately previous or next node in key order.
 * Returns NULL if this node is the end of the tree.
 * These operations are safe for complete (but not partial!)
 * tree walk-with-destruction in key order.
 */
struct rbtree * pure_func rb_prev(const struct rbtree *);
struct rbtree * pure_func rb_next(const struct rbtree *);

/*
 * Return the very first or very last node in key order.
 */
struct rbtree * pure_func rb_first(const struct rbtree *);
struct rbtree * pure_func rb_last(const struct rbtree *);

/*
 * Left and right nodes, if real. These operations are
 * safe for tree destruction, but not for splitting a tree.
 */
static inline struct rbtree *rb_left(const struct rbtree *rb)
{
    return (rb->m.flags & RBTREE_NODE_PRED) ? NULL : rb->m.left;
}
static inline struct rbtree *rb_right(const struct rbtree *rb)
{
    return (rb->m.flags & RBTREE_NODE_SUCC) ? NULL : rb->m.right;
}

#endif /* NASM_RBTREE_H */
