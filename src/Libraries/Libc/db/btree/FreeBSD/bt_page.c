/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)bt_page.c	8.3 (Berkeley) 7/14/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/db/btree/bt_page.c,v 1.5 2009/03/02 23:47:18 delphij Exp $");

#include <sys/types.h>

#include <stdio.h>

#include <db.h>
#include "btree.h"

/*
 * __bt_free --
 *	Put a page on the freelist.
 *
 * Parameters:
 *	t:	tree
 *	h:	page to free
 *
 * Returns:
 *	RET_ERROR, RET_SUCCESS
 *
 * Side-effect:
 *	mpool_put's the page.
 */
int
__bt_free(BTREE *t, PAGE *h)
{
	/* Insert the page at the head of the free list. */
	h->prevpg = P_INVALID;
	h->nextpg = t->bt_free;
	t->bt_free = h->pgno;
	F_SET(t, B_METADIRTY);

	/* Make sure the page gets written back. */
	return (mpool_put(t->bt_mp, h, MPOOL_DIRTY));
}

/*
 * __bt_new --
 *	Get a new page, preferably from the freelist.
 *
 * Parameters:
 *	t:	tree
 *	npg:	storage for page number.
 *
 * Returns:
 *	Pointer to a page, NULL on error.
 */
PAGE *
__bt_new(BTREE *t, pgno_t *npg)
{
	PAGE *h;

	if (t->bt_free != P_INVALID &&
	    (h = mpool_get(t->bt_mp, t->bt_free, 0)) != NULL) {
		*npg = t->bt_free;
		t->bt_free = h->nextpg;
		F_SET(t, B_METADIRTY);
		return (h);
	}
	return (mpool_new(t->bt_mp, npg));
}
