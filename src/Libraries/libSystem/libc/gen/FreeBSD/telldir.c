/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
static char sccsid[] = "@(#)telldir.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "namespace.h"
#include <sys/param.h>
#include <sys/queue.h>
#include <dirent.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include "un-namespace.h"

#include "libc_private.h"
#include "telldir.h"

/*
 * return a pointer into a directory
 */
long
telldir(DIR *dirp)
{
	struct ddloc *lp;
	long idx;

	if (__isthreaded)
		_pthread_mutex_lock(&dirp->dd_lock);
	LIST_FOREACH(lp, &dirp->dd_td->td_locq, loc_lqe) {
#if __DARWIN_64_BIT_INO_T
		if (lp->loc_seek == dirp->dd_td->seekoff &&
#else /* !__DARWIN_64_BIT_INO_T */
		if (lp->loc_seek == dirp->dd_seek &&
#endif /* __DARWIN_64_BIT_INO_T */
		    lp->loc_loc == dirp->dd_loc)
			break;
	}
	if (lp == NULL) {
		lp = malloc(sizeof(struct ddloc));
		if (lp == NULL) {
			if (__isthreaded)
				_pthread_mutex_unlock(&dirp->dd_lock);
			return (-1);
		}
		lp->loc_index = dirp->dd_td->td_loccnt++;
#if __DARWIN_64_BIT_INO_T
		lp->loc_seek = dirp->dd_td->seekoff;
#else /* !__DARWIN_64_BIT_INO_T */
		lp->loc_seek = dirp->dd_seek;
#endif /* __DARWIN_64_BIT_INO_T */
		lp->loc_loc = dirp->dd_loc;
		LIST_INSERT_HEAD(&dirp->dd_td->td_locq, lp, loc_lqe);
	}
	idx = lp->loc_index;
	if (__isthreaded)
		_pthread_mutex_unlock(&dirp->dd_lock);
	return (idx);
}

/*
 * seek to an entry in a directory.
 * Only values returned by "telldir" should be passed to seekdir.
 */
void
_seekdir(DIR *dirp, long loc)
{
	struct ddloc *lp;
	struct dirent *dp;

	LIST_FOREACH(lp, &dirp->dd_td->td_locq, loc_lqe) {
		if (lp->loc_index == loc)
			break;
	}
	if (lp == NULL)
		return;
#if __DARWIN_64_BIT_INO_T
	if (lp->loc_loc == dirp->dd_loc && lp->loc_seek == dirp->dd_td->seekoff)
#else /* !__DARWIN_64_BIT_INO_T */
	if (lp->loc_loc == dirp->dd_loc && lp->loc_seek == dirp->dd_seek)
#endif /* __DARWIN_64_BIT_INO_T */
		return;

	/* If it's within the same chunk of data, don't bother reloading. */
#if __DARWIN_64_BIT_INO_T
	if (lp->loc_seek == dirp->dd_td->seekoff) {
#else /* !__DARWIN_64_BIT_INO_T */
	if (lp->loc_seek == dirp->dd_seek) {
#endif /* __DARWIN_64_BIT_INO_T */
		/*
		 * If we go back to 0 don't make the next readdir
		 * trigger a call to getdirentries().
		 */
		if (lp->loc_loc == 0)
			dirp->dd_flags |= __DTF_SKIPREAD;
		dirp->dd_loc = lp->loc_loc;
		return;
	}
	(void) lseek(dirp->dd_fd, (off_t)lp->loc_seek, SEEK_SET);
#if __DARWIN_64_BIT_INO_T
	dirp->dd_td->seekoff = lp->loc_seek;
#else /* !__DARWIN_64_BIT_INO_T */
	dirp->dd_seek = lp->loc_seek;
#endif /* __DARWIN_64_BIT_INO_T */
	dirp->dd_loc = 0;
	dirp->dd_flags &= ~(__DTF_SKIPREAD | __DTF_ATEND); /* current contents are invalid */
	while (dirp->dd_loc < lp->loc_loc) {
		dp = _readdir_unlocked(dirp, 0);
		if (dp == NULL)
			break;
	}
}

/*
 * After readdir returns the last entry in a block, a call to telldir
 * returns a location that is after the end of that last entry.
 * However, that location doesn't refer to a valid directory entry.
 * Ideally, the call to telldir would return a location that refers to
 * the first entry in the next block.  That location is not known
 * until the next block is read, so readdir calls this function after
 * fetching a new block to fix any such telldir locations.
 */
void
_fixtelldir(DIR *dirp, long oldseek, long oldloc)
{
	struct ddloc *lp;

	lp = LIST_FIRST(&dirp->dd_td->td_locq);
	if (lp != NULL) {
		if (lp->loc_loc == oldloc &&
		    lp->loc_seek == oldseek) {
#if __DARWIN_64_BIT_INO_T
			lp->loc_seek = dirp->dd_td->seekoff;
#else /* !__DARWIN_64_BIT_INO_T */
			lp->loc_seek = dirp->dd_seek;
#endif /* __DARWIN_64_BIT_INO_T */
			lp->loc_loc = dirp->dd_loc;
		}
	}
}

#ifndef BUILDING_VARIANT
/*
 * Reclaim memory for telldir cookies which weren't used.
 */
void
_reclaim_telldir(DIR *dirp)
{
	struct ddloc *lp;
	struct ddloc *templp;

	lp = LIST_FIRST(&dirp->dd_td->td_locq);
	while (lp != NULL) {
		templp = lp;
		lp = LIST_NEXT(lp, loc_lqe);
		free(templp);
	}
	LIST_INIT(&dirp->dd_td->td_locq);
}
#endif
