/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
/*
 * Tree search generalized from Knuth (6.2.2) Algorithm T just like
 * the AT&T man page says.
 *
 * The node_t structure is for internal use only, lint doesn't grok it.
 *
 * Written by reading the System V Interface Definition, not the code.
 *
 * Totally public domain.
 */

#include <sys/cdefs.h>
#if 0
#if defined(LIBC_SCCS) && !defined(lint)
__RCSID("$NetBSD: twalk.c,v 1.1 1999/02/22 10:33:16 christos Exp $");
#endif /* LIBC_SCCS and not lint */
#endif
__FBSDID("$FreeBSD: src/lib/libc/stdlib/twalk.c,v 1.5 2003/01/05 02:43:18 tjr Exp $");

#define _SEARCH_PRIVATE
#include <search.h>
#include <stdlib.h>

static void trecurse(const node_t *,
    void (*action)(const void *, VISIT, int), int level);

/* Walk the nodes of a tree */
static void
trecurse(root, action, level)
	const node_t *root;	/* Root of the tree to be walked */
	void (*action)(const void *, VISIT, int);
	int level;
{

	if (root->llink == NULL && root->rlink == NULL)
		(*action)(root, leaf, level);
	else {
		(*action)(root, preorder, level);
		if (root->llink != NULL)
			trecurse(root->llink, action, level + 1);
		(*action)(root, postorder, level);
		if (root->rlink != NULL)
			trecurse(root->rlink, action, level + 1);
		(*action)(root, endorder, level);
	}
}

/* Walk the nodes of a tree */
void
twalk(vroot, action)
	const void *vroot;	/* Root of the tree to be walked */
	void (*action)(const void *, VISIT, int);
{
	if (vroot != NULL && action != NULL)
		trecurse(vroot, action, 0);
}

