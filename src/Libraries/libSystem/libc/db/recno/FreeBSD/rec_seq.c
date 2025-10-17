/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#include <sys/cdefs.h>
#ifndef lint
/* XXX use __SCCSID */
static char sccsid[] __unused = "@(#)rec_seq.c	8.3 (Berkeley) 7/14/94";
#endif /* not lint */
__FBSDID("$FreeBSD: src/lib/libc/db/recno/rec_seq.c,v 1.8 2009/03/04 00:58:04 delphij Exp $");

#include <sys/types.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <db.h>
#include "recno.h"

/*
 * __REC_SEQ -- Recno sequential scan interface.
 *
 * Parameters:
 *	dbp:	pointer to access method
 *	key:	key for positioning and return value
 *	data:	data return value
 *	flags:	R_CURSOR, R_FIRST, R_LAST, R_NEXT, R_PREV.
 *
 * Returns:
 *	RET_ERROR, RET_SUCCESS or RET_SPECIAL if there's no next key.
 */
int
__rec_seq(const DB *dbp, DBT *key, DBT *data, u_int flags)
{
	BTREE *t;
	EPG *e;
	recno_t nrec;
	int status;

	t = dbp->internal;

	/* Toss any page pinned across calls. */
	if (t->bt_pinned != NULL) {
		mpool_put(t->bt_mp, t->bt_pinned, 0);
		t->bt_pinned = NULL;
	}

	switch(flags) {
	case R_CURSOR:
		if ((nrec = *(recno_t *)key->data) == 0)
			goto einval;
		break;
	case R_NEXT:
		if (F_ISSET(&t->bt_cursor, CURS_INIT)) {
			nrec = t->bt_cursor.rcursor + 1;
			break;
		}
		/* FALLTHROUGH */
	case R_FIRST:
		nrec = 1;
		break;
	case R_PREV:
		if (F_ISSET(&t->bt_cursor, CURS_INIT)) {
			if ((nrec = t->bt_cursor.rcursor - 1) == 0)
				return (RET_SPECIAL);
			break;
		}
		/* FALLTHROUGH */
	case R_LAST:
		if (!F_ISSET(t, R_EOF | R_INMEM) &&
		    t->bt_irec(t, MAX_REC_NUMBER) == RET_ERROR)
			return (RET_ERROR);
		nrec = t->bt_nrecs;
		break;
	default:
einval:		errno = EINVAL;
		return (RET_ERROR);
	}

	if (t->bt_nrecs == 0 || nrec > t->bt_nrecs) {
		if (!F_ISSET(t, R_EOF | R_INMEM) &&
		    (status = t->bt_irec(t, nrec)) != RET_SUCCESS)
			return (status);
		if (t->bt_nrecs == 0 || nrec > t->bt_nrecs)
			return (RET_SPECIAL);
	}

	if ((e = __rec_search(t, nrec - 1, SEARCH)) == NULL)
		return (RET_ERROR);

	F_SET(&t->bt_cursor, CURS_INIT);
	t->bt_cursor.rcursor = nrec;

	status = __rec_ret(t, e, nrec, key, data);
	if (F_ISSET(t, B_DB_LOCK))
		mpool_put(t->bt_mp, e->page, 0);
	else
		t->bt_pinned = e->page;
	return (status);
}
