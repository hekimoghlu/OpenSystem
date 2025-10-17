/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
static char sccsid[] = "@(#)rec_close.c	8.6 (Berkeley) 8/18/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/db/recno/rec_close.c,v 1.11 2009/03/28 05:45:29 delphij Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/mman.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>
#include "un-namespace.h"

#include <db.h>
#include "recno.h"

/*
 * __REC_CLOSE -- Close a recno tree.
 *
 * Parameters:
 *	dbp:	pointer to access method
 *
 * Returns:
 *	RET_ERROR, RET_SUCCESS
 */
int
__rec_close(DB *dbp)
{
	BTREE *t;
	int status;

	t = dbp->internal;

	/* Toss any page pinned across calls. */
	if (t->bt_pinned != NULL) {
		mpool_put(t->bt_mp, t->bt_pinned, 0);
		t->bt_pinned = NULL;
	}

	if (__rec_sync(dbp, 0) == RET_ERROR)
		return (RET_ERROR);

	/* Committed to closing. */
	status = RET_SUCCESS;
	if (F_ISSET(t, R_MEMMAPPED) && munmap(t->bt_smap, t->bt_msize))
		status = RET_ERROR;

	if (!F_ISSET(t, R_INMEM)) {
		if (F_ISSET(t, R_CLOSEFP)) {
			if (fclose(t->bt_rfp))
				status = RET_ERROR;
		} else {
			if (_close(t->bt_rfd))
				status = RET_ERROR;
		}
	}

	if (__bt_close(dbp) == RET_ERROR)
		status = RET_ERROR;

	return (status);
}

/*
 * __REC_SYNC -- sync the recno tree to disk.
 *
 * Parameters:
 *	dbp:	pointer to access method
 *
 * Returns:
 *	RET_SUCCESS, RET_ERROR.
 */
int
__rec_sync(const DB *dbp, u_int flags)
{
	struct iovec iov[2];
	BTREE *t;
	DBT data, key;
	off_t off;
	recno_t scursor, trec;
	int status;

	t = dbp->internal;

	/* Toss any page pinned across calls. */
	if (t->bt_pinned != NULL) {
		mpool_put(t->bt_mp, t->bt_pinned, 0);
		t->bt_pinned = NULL;
	}

	if (flags == R_RECNOSYNC)
		return (__bt_sync(dbp, 0));

	if (F_ISSET(t, R_RDONLY | R_INMEM) || !F_ISSET(t, R_MODIFIED))
		return (RET_SUCCESS);

	/* Read any remaining records into the tree. */
	if (!F_ISSET(t, R_EOF) && t->bt_irec(t, MAX_REC_NUMBER) == RET_ERROR)
		return (RET_ERROR);

	/* Rewind the file descriptor. */
	if (lseek(t->bt_rfd, (off_t)0, SEEK_SET) != 0)
		return (RET_ERROR);

	/* Save the cursor. */
	scursor = t->bt_cursor.rcursor;

	key.size = sizeof(recno_t);
	key.data = &trec;

	if (F_ISSET(t, R_FIXLEN)) {
		/*
		 * We assume that fixed length records are all fixed length.
		 * Any that aren't are either EINVAL'd or corrected by the
		 * record put code.
		 */
		status = (dbp->seq)(dbp, &key, &data, R_FIRST);
		while (status == RET_SUCCESS) {
			if (_write(t->bt_rfd, data.data, data.size) !=
			    (ssize_t)data.size)
				return (RET_ERROR);
			status = (dbp->seq)(dbp, &key, &data, R_NEXT);
		}
	} else {
		iov[1].iov_base = &t->bt_bval;
		iov[1].iov_len = 1;

		status = (dbp->seq)(dbp, &key, &data, R_FIRST);
		while (status == RET_SUCCESS) {
			iov[0].iov_base = data.data;
			iov[0].iov_len = data.size;
			if (_writev(t->bt_rfd, iov, 2) != (ssize_t)(data.size + 1))
				return (RET_ERROR);
			status = (dbp->seq)(dbp, &key, &data, R_NEXT);
		}
	}

	/* Restore the cursor. */
	t->bt_cursor.rcursor = scursor;

	if (status == RET_ERROR)
		return (RET_ERROR);
	if ((off = lseek(t->bt_rfd, (off_t)0, SEEK_CUR)) == -1)
		return (RET_ERROR);
	if (ftruncate(t->bt_rfd, off))
		return (RET_ERROR);
	F_CLR(t, R_MODIFIED);
	return (RET_SUCCESS);
}

