/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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
static char sccsid[] = "@(#)bt_search.c	8.8 (Berkeley) 7/31/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/db/btree/bt_search.c,v 1.5 2009/03/03 02:16:12 delphij Exp $");

#include <sys/types.h>

#include <stdio.h>

#include <db.h>
#include "btree.h"

static int __bt_snext(BTREE *, PAGE *, const DBT *, int *);
static int __bt_sprev(BTREE *, PAGE *, const DBT *, int *);

/*
 * __bt_search --
 *	Search a btree for a key.
 *
 * Parameters:
 *	t:	tree to search
 *	key:	key to find
 *	exactp:	pointer to exact match flag
 *
 * Returns:
 *	The EPG for matching record, if any, or the EPG for the location
 *	of the key, if it were inserted into the tree, is entered into
 *	the bt_cur field of the tree.  A pointer to the field is returned.
 */
EPG *
__bt_search(BTREE *t, const DBT *key, int *exactp)
{
	PAGE *h;
	indx_t base, idx, lim;
	pgno_t pg;
	int cmp;

	BT_CLR(t);
	for (pg = P_ROOT;;) {
		if ((h = mpool_get(t->bt_mp, pg, 0)) == NULL)
			return (NULL);

		/* Do a binary search on the current page. */
		t->bt_cur.page = h;
		for (base = 0, lim = NEXTINDEX(h); lim; lim >>= 1) {
			t->bt_cur.index = idx = base + (lim >> 1);
			if ((cmp = __bt_cmp(t, key, &t->bt_cur)) == 0) {
				if (h->flags & P_BLEAF) {
					*exactp = 1;
					return (&t->bt_cur);
				}
				goto next;
			}
			if (cmp > 0) {
				base = idx + 1;
				--lim;
			}
		}

		/*
		 * If it's a leaf page, we're almost done.  If no duplicates
		 * are allowed, or we have an exact match, we're done.  Else,
		 * it's possible that there were matching keys on this page,
		 * which later deleted, and we're on a page with no matches
		 * while there are matches on other pages.  If at the start or
		 * end of a page, check the adjacent page.
		 */
		if (h->flags & P_BLEAF) {
			if (!F_ISSET(t, B_NODUPS)) {
				if (base == 0 &&
				    h->prevpg != P_INVALID &&
				    __bt_sprev(t, h, key, exactp))
					return (&t->bt_cur);
				if (base == NEXTINDEX(h) &&
				    h->nextpg != P_INVALID &&
				    __bt_snext(t, h, key, exactp))
					return (&t->bt_cur);
			}
			*exactp = 0;
			t->bt_cur.index = base;
			return (&t->bt_cur);
		}

		/*
		 * No match found.  Base is the smallest index greater than
		 * key and may be zero or a last + 1 index.  If it's non-zero,
		 * decrement by one, and record the internal page which should
		 * be a parent page for the key.  If a split later occurs, the
		 * inserted page will be to the right of the saved page.
		 */
		idx = base ? base - 1 : base;

next:		BT_PUSH(t, h->pgno, idx);
		pg = GETBINTERNAL(h, idx)->pgno;
		mpool_put(t->bt_mp, h, 0);
	}
}

/*
 * __bt_snext --
 *	Check for an exact match after the key.
 *
 * Parameters:
 *	t:	tree
 *	h:	current page
 *	key:	key
 *	exactp:	pointer to exact match flag
 *
 * Returns:
 *	If an exact match found.
 */
static int
__bt_snext(BTREE *t, PAGE *h, const DBT *key, int *exactp)
{
	EPG e;

	/*
	 * Get the next page.  The key is either an exact
	 * match, or not as good as the one we already have.
	 */
	if ((e.page = mpool_get(t->bt_mp, h->nextpg, 0)) == NULL)
		return (0);
	e.index = 0;
	if (__bt_cmp(t, key, &e) == 0) {
		mpool_put(t->bt_mp, h, 0);
		t->bt_cur = e;
		*exactp = 1;
		return (1);
	}
	mpool_put(t->bt_mp, e.page, 0);
	return (0);
}

/*
 * __bt_sprev --
 *	Check for an exact match before the key.
 *
 * Parameters:
 *	t:	tree
 *	h:	current page
 *	key:	key
 *	exactp:	pointer to exact match flag
 *
 * Returns:
 *	If an exact match found.
 */
static int
__bt_sprev(BTREE *t, PAGE *h, const DBT *key, int *exactp)
{
	EPG e;

	/*
	 * Get the previous page.  The key is either an exact
	 * match, or not as good as the one we already have.
	 */
	if ((e.page = mpool_get(t->bt_mp, h->prevpg, 0)) == NULL)
		return (0);
	e.index = NEXTINDEX(e.page) - 1;
	if (__bt_cmp(t, key, &e) == 0) {
		mpool_put(t->bt_mp, h, 0);
		t->bt_cur = e;
		*exactp = 1;
		return (1);
	}
	mpool_put(t->bt_mp, e.page, 0);
	return (0);
}
