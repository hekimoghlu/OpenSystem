/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
static char sccsid[] = "@(#)rec_utils.c	8.6 (Berkeley) 7/16/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/db/recno/rec_utils.c,v 1.6 2009/03/05 00:57:01 delphij Exp $");

#include <sys/param.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <db.h>
#include "recno.h"

/*
 * __rec_ret --
 *	Build return data.
 *
 * Parameters:
 *	t:	tree
 *	e:	key/data pair to be returned
 *   nrec:	record number
 *    key:	user's key structure
 *   data:	user's data structure
 *
 * Returns:
 *	RET_SUCCESS, RET_ERROR.
 */
int
__rec_ret(BTREE *t, EPG *e, recno_t nrec, DBT *key, DBT *data)
{
	RLEAF *rl;
	void *p;

	if (key == NULL)
		goto dataonly;

	/* We have to copy the key, it's not on the page. */
	if (sizeof(recno_t) > t->bt_rkey.size) {
		p = realloc(t->bt_rkey.data, sizeof(recno_t));
		if (p == NULL)
			return (RET_ERROR);
		t->bt_rkey.data = p;
		t->bt_rkey.size = sizeof(recno_t);
	}
	memmove(t->bt_rkey.data, &nrec, sizeof(recno_t));
	key->size = sizeof(recno_t);
	key->data = t->bt_rkey.data;

dataonly:
	if (data == NULL)
		return (RET_SUCCESS);

	/*
	 * We must copy big keys/data to make them contigous.  Otherwise,
	 * leave the page pinned and don't copy unless the user specified
	 * concurrent access.
	 */
	rl = GETRLEAF(e->page, e->index);
	if (rl->flags & P_BIGDATA) {
		if (__ovfl_get(t, rl->bytes,
		    &data->size, &t->bt_rdata.data, &t->bt_rdata.size))
			return (RET_ERROR);
		data->data = t->bt_rdata.data;
	} else if (F_ISSET(t, B_DB_LOCK)) {
		/* Use +1 in case the first record retrieved is 0 length. */
		if (rl->dsize + 1 > t->bt_rdata.size) {
			p = realloc(t->bt_rdata.data, rl->dsize + 1);
			if (p == NULL)
				return (RET_ERROR);
			t->bt_rdata.data = p;
			t->bt_rdata.size = rl->dsize + 1;
		}
		memmove(t->bt_rdata.data, rl->bytes, rl->dsize);
		data->size = rl->dsize;
		data->data = t->bt_rdata.data;
	} else {
		data->size = rl->dsize;
		data->data = rl->bytes;
	}
	return (RET_SUCCESS);
}
