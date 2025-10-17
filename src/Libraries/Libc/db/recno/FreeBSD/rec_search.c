/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
static char sccsid[] = "@(#)rec_search.c	8.4 (Berkeley) 7/14/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/db/recno/rec_search.c,v 1.8 2009/03/04 00:58:04 delphij Exp $");

#include <sys/types.h>

#include <errno.h>
#include <stdio.h>

#include <db.h>
#include "recno.h"

/*
 * __REC_SEARCH -- Search a btree for a key.
 *
 * Parameters:
 *	t:	tree to search
 *	recno:	key to find
 *	op:	search operation
 *
 * Returns:
 *	EPG for matching record, if any, or the EPG for the location of the
 *	key, if it were inserted into the tree.
 *
 * Returns:
 *	The EPG for matching record, if any, or the EPG for the location
 *	of the key, if it were inserted into the tree, is entered into
 *	the bt_cur field of the tree.  A pointer to the field is returned.
 */
EPG *
__rec_search(BTREE *t, recno_t recno, enum SRCHOP op)
{
	indx_t idx;
	PAGE *h;
	EPGNO *parent;
	RINTERNAL *r;
	pgno_t pg;
	indx_t top;
	recno_t total;
	int sverrno;

	BT_CLR(t);
	for (pg = P_ROOT, total = 0;;) {
		if ((h = mpool_get(t->bt_mp, pg, 0)) == NULL)
			goto err;
		if (h->flags & P_RLEAF) {
			t->bt_cur.page = h;
			t->bt_cur.index = recno - total;
			return (&t->bt_cur);
		}
		for (idx = 0, top = NEXTINDEX(h);;) {
			r = GETRINTERNAL(h, idx);
			if (++idx == top || total + r->nrecs > recno)
				break;
			total += r->nrecs;
		}

		BT_PUSH(t, pg, idx - 1);

		pg = r->pgno;
		switch (op) {
		case SDELETE:
			--GETRINTERNAL(h, (idx - 1))->nrecs;
			mpool_put(t->bt_mp, h, MPOOL_DIRTY);
			break;
		case SINSERT:
			++GETRINTERNAL(h, (idx - 1))->nrecs;
			mpool_put(t->bt_mp, h, MPOOL_DIRTY);
			break;
		case SEARCH:
			mpool_put(t->bt_mp, h, 0);
			break;
		}

	}
	/* Try and recover the tree. */
err:	sverrno = errno;
	if (op != SEARCH)
		while  ((parent = BT_POP(t)) != NULL) {
			if ((h = mpool_get(t->bt_mp, parent->pgno, 0)) == NULL)
				break;
			if (op == SINSERT)
				--GETRINTERNAL(h, parent->index)->nrecs;
			else
				++GETRINTERNAL(h, parent->index)->nrecs;
			mpool_put(t->bt_mp, h, MPOOL_DIRTY);
		}
	errno = sverrno;
	return (NULL);
}
