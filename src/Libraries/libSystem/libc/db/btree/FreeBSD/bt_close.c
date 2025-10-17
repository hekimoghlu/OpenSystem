/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
static char sccsid[] = "@(#)bt_close.c	8.7 (Berkeley) 8/17/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/db/btree/bt_close.c,v 1.10 2009/03/02 23:47:18 delphij Exp $");

#include "namespace.h"
#include <sys/param.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "un-namespace.h"

#include <db.h>
#include "btree.h"

static int bt_meta(BTREE *);

/*
 * BT_CLOSE -- Close a btree.
 *
 * Parameters:
 *	dbp:	pointer to access method
 *
 * Returns:
 *	RET_ERROR, RET_SUCCESS
 */
int
__bt_close(DB *dbp)
{
	BTREE *t;
	int fd;

	t = dbp->internal;

	/* Toss any page pinned across calls. */
	if (t->bt_pinned != NULL) {
		mpool_put(t->bt_mp, t->bt_pinned, 0);
		t->bt_pinned = NULL;
	}

	/* Sync the tree. */
	if (__bt_sync(dbp, 0) == RET_ERROR)
		return (RET_ERROR);

	/* Close the memory pool. */
	if (mpool_close(t->bt_mp) == RET_ERROR)
		return (RET_ERROR);

	/* Free random memory. */
	if (t->bt_cursor.key.data != NULL) {
		free(t->bt_cursor.key.data);
		t->bt_cursor.key.size = 0;
		t->bt_cursor.key.data = NULL;
	}
	if (t->bt_rkey.data) {
		free(t->bt_rkey.data);
		t->bt_rkey.size = 0;
		t->bt_rkey.data = NULL;
	}
	if (t->bt_rdata.data) {
		free(t->bt_rdata.data);
		t->bt_rdata.size = 0;
		t->bt_rdata.data = NULL;
	}

	fd = t->bt_fd;
	free(t);
	free(dbp);
	return (_close(fd) ? RET_ERROR : RET_SUCCESS);
}

/*
 * BT_SYNC -- sync the btree to disk.
 *
 * Parameters:
 *	dbp:	pointer to access method
 *
 * Returns:
 *	RET_SUCCESS, RET_ERROR.
 */
int
__bt_sync(const DB *dbp, u_int flags)
{
	BTREE *t;
	int status;

	t = dbp->internal;

	/* Toss any page pinned across calls. */
	if (t->bt_pinned != NULL) {
		mpool_put(t->bt_mp, t->bt_pinned, 0);
		t->bt_pinned = NULL;
	}

	/* Sync doesn't currently take any flags. */
	if (flags != 0) {
		errno = EINVAL;
		return (RET_ERROR);
	}

	if (F_ISSET(t, B_INMEM | B_RDONLY) || !F_ISSET(t, B_MODIFIED))
		return (RET_SUCCESS);

	if (F_ISSET(t, B_METADIRTY) && bt_meta(t) == RET_ERROR)
		return (RET_ERROR);

	if ((status = mpool_sync(t->bt_mp)) == RET_SUCCESS)
		F_CLR(t, B_MODIFIED);

	return (status);
}

/*
 * BT_META -- write the tree meta data to disk.
 *
 * Parameters:
 *	t:	tree
 *
 * Returns:
 *	RET_ERROR, RET_SUCCESS
 */
static int
bt_meta(BTREE *t)
{
	BTMETA m;
	void *p;

	if ((p = mpool_get(t->bt_mp, P_META, 0)) == NULL)
		return (RET_ERROR);

	/* Fill in metadata. */
	m.magic = BTREEMAGIC;
	m.version = BTREEVERSION;
	m.psize = t->bt_psize;
	m.free = t->bt_free;
	m.nrecs = t->bt_nrecs;
	m.flags = F_ISSET(t, SAVEMETA);

	memmove(p, &m, sizeof(BTMETA));
	mpool_put(t->bt_mp, p, MPOOL_DIRTY);
	return (RET_SUCCESS);
}
