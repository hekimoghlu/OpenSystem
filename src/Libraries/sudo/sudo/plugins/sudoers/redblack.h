/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#ifndef SUDOERS_REDBLACK_H
#define SUDOERS_REDBLACK_H

enum rbcolor {
    red,
    black
};

enum rbtraversal {
    preorder,
    inorder,
    postorder
};

struct rbnode {
    struct rbnode *left, *right, *parent;
    void *data;
    enum rbcolor color;
};

struct rbtree {
    int (*compar)(const void *, const void *);
    struct rbnode root;
    struct rbnode nil;
};

#define rbapply(t, f, c, o)	rbapply_node((t), (t)->root.left, (f), (c), (o))
#define rbisempty(t)		((t)->root.left == &(t)->nil && (t)->root.right == &(t)->nil)
#define rbfirst(t)		((t)->root.left)
#define rbroot(t)		(&(t)->root)
#define rbnil(t)		(&(t)->nil)

void *rbdelete(struct rbtree *, struct rbnode *);
int rbapply_node(struct rbtree *, struct rbnode *,
	int (*)(void *, void *), void *, enum rbtraversal);
struct rbnode *rbfind(struct rbtree *, void *);
int rbinsert(struct rbtree *, void *, struct rbnode **);
struct rbtree *rbcreate(int (*)(const void *, const void *));
void rbdestroy(struct rbtree *, void (*)(void *));

#endif /* SUDOERS_REDBLACK_H */
