/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
static char sccsid[] = "@(#)rec_open.c	8.10 (Berkeley) 9/1/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/db/recno/rec_open.c,v 1.9 2009/03/04 00:58:04 delphij Exp $");

#include "namespace.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>
#include "un-namespace.h"

#include <db.h>
#include "recno.h"

DB *
__rec_open(const char *fname, int flags, int mode, const RECNOINFO *openinfo,
    int dflags)
{
	BTREE *t;
	BTREEINFO btopeninfo;
	DB *dbp;
	PAGE *h;
	struct stat sb;
	int rfd, sverrno;

	/* Open the user's file -- if this fails, we're done. */
	if (fname != NULL && (rfd = _open(fname, flags, mode)) < 0)
		return (NULL);

	/* Create a btree in memory (backed by disk). */
	dbp = NULL;
	if (openinfo) {
		if (openinfo->flags & ~(R_FIXEDLEN | R_NOKEY | R_SNAPSHOT))
			goto einval;
		btopeninfo.flags = 0;
		btopeninfo.cachesize = openinfo->cachesize;
		btopeninfo.maxkeypage = 0;
		btopeninfo.minkeypage = 0;
		btopeninfo.psize = openinfo->psize;
		btopeninfo.compare = NULL;
		btopeninfo.prefix = NULL;
		btopeninfo.lorder = openinfo->lorder;
		dbp = __bt_open(openinfo->bfname,
		    O_RDWR, S_IRUSR | S_IWUSR, &btopeninfo, dflags);
	} else
		dbp = __bt_open(NULL, O_RDWR, S_IRUSR | S_IWUSR, NULL, dflags);
	if (dbp == NULL)
		goto err;

	/*
	 * Some fields in the tree structure are recno specific.  Fill them
	 * in and make the btree structure look like a recno structure.  We
	 * don't change the bt_ovflsize value, it's close enough and slightly
	 * bigger.
	 */
	t = dbp->internal;
	if (openinfo) {
		if (openinfo->flags & R_FIXEDLEN) {
			F_SET(t, R_FIXLEN);
			t->bt_reclen = openinfo->reclen;
			if (t->bt_reclen == 0)
				goto einval;
		}
		t->bt_bval = openinfo->bval;
	} else
		t->bt_bval = '\n';

	F_SET(t, R_RECNO);
	if (fname == NULL)
		F_SET(t, R_EOF | R_INMEM);
	else
		t->bt_rfd = rfd;

	if (fname != NULL) {
		/*
		 * In 4.4BSD, stat(2) returns true for ISSOCK on pipes.
		 * Unfortunately, that's not portable, so we use lseek
		 * and check the errno values.
		 */
		errno = 0;
		if (lseek(rfd, (off_t)0, SEEK_CUR) == -1 && errno == ESPIPE) {
			switch (flags & O_ACCMODE) {
			case O_RDONLY:
				F_SET(t, R_RDONLY);
				break;
			default:
				goto einval;
			}
slow:			if ((t->bt_rfp = fdopen(rfd, "r")) == NULL)
				goto err;
			F_SET(t, R_CLOSEFP);
			t->bt_irec =
			    F_ISSET(t, R_FIXLEN) ? __rec_fpipe : __rec_vpipe;
		} else {
			switch (flags & O_ACCMODE) {
			case O_RDONLY:
				F_SET(t, R_RDONLY);
				break;
			case O_RDWR:
				break;
			default:
				goto einval;
			}

			if (_fstat(rfd, &sb))
				goto err;
			/*
			 * Kluge -- we'd like to test to see if the file is too
			 * big to mmap.  Since, we don't know what size or type
			 * off_t's or size_t's are, what the largest unsigned
			 * integral type is, or what random insanity the local
			 * C compiler will perpetrate, doing the comparison in
			 * a portable way is flatly impossible.  Hope that mmap
			 * fails if the file is too large.
			 */
			if (sb.st_size == 0)
				F_SET(t, R_EOF);
			else {
#ifdef MMAP_NOT_AVAILABLE
				/*
				 * XXX
				 * Mmap doesn't work correctly on many current
				 * systems.  In particular, it can fail subtly,
				 * with cache coherency problems.  Don't use it
				 * for now.
				 */
				t->bt_msize = sb.st_size;
				if ((t->bt_smap = mmap(NULL, t->bt_msize,
				    PROT_READ, MAP_PRIVATE, rfd,
				    (off_t)0)) == MAP_FAILED)
					goto slow;
				t->bt_cmap = t->bt_smap;
				t->bt_emap = t->bt_smap + sb.st_size;
				t->bt_irec = F_ISSET(t, R_FIXLEN) ?
				    __rec_fmap : __rec_vmap;
				F_SET(t, R_MEMMAPPED);
#else
				goto slow;
#endif
			}
		}
	}

	/* Use the recno routines. */
	dbp->close = __rec_close;
	dbp->del = __rec_delete;
	dbp->fd = __rec_fd;
	dbp->get = __rec_get;
	dbp->put = __rec_put;
	dbp->seq = __rec_seq;
	dbp->sync = __rec_sync;

	/* If the root page was created, reset the flags. */
	if ((h = mpool_get(t->bt_mp, P_ROOT, 0)) == NULL)
		goto err;
	if ((h->flags & P_TYPE) == P_BLEAF) {
		F_CLR(h, P_TYPE);
		F_SET(h, P_RLEAF);
		mpool_put(t->bt_mp, h, MPOOL_DIRTY);
	} else
		mpool_put(t->bt_mp, h, 0);

	if (openinfo && openinfo->flags & R_SNAPSHOT &&
	    !F_ISSET(t, R_EOF | R_INMEM) &&
	    t->bt_irec(t, MAX_REC_NUMBER) == RET_ERROR)
		goto err;
	return (dbp);

einval:	errno = EINVAL;
err:	sverrno = errno;
	if (dbp != NULL)
		(void)__bt_close(dbp);
	if (fname != NULL)
		(void)_close(rfd);
	errno = sverrno;
	return (NULL);
}

int
__rec_fd(const DB *dbp)
{
	BTREE *t;

	t = dbp->internal;

	/* Toss any page pinned across calls. */
	if (t->bt_pinned != NULL) {
		mpool_put(t->bt_mp, t->bt_pinned, 0);
		t->bt_pinned = NULL;
	}

	/* In-memory database can't have a file descriptor. */
	if (F_ISSET(t, R_INMEM)) {
		errno = ENOENT;
		return (-1);
	}
	return (t->bt_rfd);
}
