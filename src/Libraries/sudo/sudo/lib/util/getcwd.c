/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#include <config.h>

#ifndef HAVE_GETCWD

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>

#include "sudo_compat.h"

#define	ISDOT(dp) \
	(dp->d_name[0] == '.' && (dp->d_name[1] == '\0' || \
	    (dp->d_name[1] == '.' && dp->d_name[2] == '\0')))

#if defined(HAVE_STRUCT_DIRENT_D_NAMLEN) && HAVE_STRUCT_DIRENT_D_NAMLEN
# define NAMLEN(dirent)	(dirent)->d_namlen
#else
# define NAMLEN(dirent)	strlen((dirent)->d_name)
#endif

char *
sudo_getcwd(char *pt, size_t size)
{
	struct dirent *dp;
	DIR *dir = NULL;
	dev_t dev;
	ino_t ino;
	int first;
	char *bpt, *bup;
	struct stat s;
	dev_t root_dev;
	ino_t root_ino;
	size_t ptsize, upsize;
	int save_errno;
	char *ept, *eup, *up;

	/*
	 * If no buffer specified by the user, allocate one as necessary.
	 * If a buffer is specified, the size has to be non-zero.  The path
	 * is built from the end of the buffer backwards.
	 */
	if (pt) {
		ptsize = 0;
		if (!size) {
			errno = EINVAL;
			return NULL;
		}
		ept = pt + size;
	} else {
		if ((pt = malloc(ptsize = 1024 - 4)) == NULL)
			return NULL;
		ept = pt + ptsize;
	}
	bpt = ept - 1;
	*bpt = '\0';

	/*
	 * Allocate bytes (1024 - malloc space) for the string of "../"'s.
	 * Should always be enough (it's 340 levels).  If it's not, allocate
	 * as necessary.  Special * case the first stat, it's ".", not "..".
	 */
	if ((up = malloc(upsize = 1024 - 4)) == NULL)
		goto err;
	eup = up + PATH_MAX;
	bup = up;
	up[0] = '.';
	up[1] = '\0';

	/* Save root values, so know when to stop. */
	if (stat("/", &s))
		goto err;
	root_dev = s.st_dev;
	root_ino = s.st_ino;

	errno = 0;			/* XXX readdir has no error return. */

	for (first = 1;; first = 0) {
		/* Stat the current level. */
		if (lstat(up, &s))
			goto err;

		/* Save current node values. */
		ino = s.st_ino;
		dev = s.st_dev;

		/* Check for reaching root. */
		if (root_dev == dev && root_ino == ino) {
			*--bpt = '/';
			/*
			 * It's unclear that it's a requirement to copy the
			 * path to the beginning of the buffer, but it's always
			 * been that way and stuff would probably break.
			 */
			memcpy(pt, bpt, ept - bpt);
			free(up);
			return pt;
		}

		/*
		 * Build pointer to the parent directory, allocating memory
		 * as necessary.  Max length is 3 for "../", the largest
		 * possible component name, plus a trailing NULL.
		 */
		if (bup + 3  + NAME_MAX + 1 >= eup) {
			char *nup;

			if ((nup = reallocarray(up, upsize, 2)) == NULL)
				goto err;
			upsize *= 2;
			up = nup;
			bup = up;
			eup = up + upsize;
		}
		*bup++ = '.';
		*bup++ = '.';
		*bup = '\0';

		/* Open and stat parent directory. */
		if (!(dir = opendir(up)) || fstat(dirfd(dir), &s))
			goto err;

		/* Add trailing slash for next directory. */
		*bup++ = '/';

		/*
		 * If it's a mount point, have to stat each element because
		 * the inode number in the directory is for the entry in the
		 * parent directory, not the inode number of the mounted file.
		 */
		save_errno = 0;
		if (s.st_dev == dev) {
			for (;;) {
				if (!(dp = readdir(dir)))
					goto notfound;
				if (dp->d_fileno == ino)
					break;
			}
		} else
			for (;;) {
				if (!(dp = readdir(dir)))
					goto notfound;
				if (ISDOT(dp))
					continue;
				memcpy(bup, dp->d_name, NAMLEN(dp) + 1);

				/* Save the first error for later. */
				if (lstat(up, &s)) {
					if (!save_errno)
						save_errno = errno;
					errno = 0;
					continue;
				}
				if (s.st_dev == dev && s.st_ino == ino)
					break;
			}

		/*
		 * Check for length of the current name, preceding slash,
		 * leading slash.
		 */
		if (bpt - pt <= NAMLEN(dp) + (first ? 1 : 2)) {
			size_t len, off;
			char *npt;

			if (!ptsize) {
				errno = ERANGE;
				goto err;
			}
			off = bpt - pt;
			len = ept - bpt;
			if ((npt = reallocarray(pt, ptsize, 2)) == NULL)
				goto err;
			ptsize *= 2;
			pt = npt;
			bpt = pt + off;
			ept = pt + ptsize;
			memcpy(ept - len, bpt, len);
			bpt = ept - len;
		}
		if (!first)
			*--bpt = '/';
		bpt -= NAMLEN(dp);
		memcpy(bpt, dp->d_name, NAMLEN(dp));
		(void)closedir(dir);

		/* Truncate any file name. */
		*bup = '\0';
	}

notfound:
	/*
	 * If readdir set errno, use it, not any saved error; otherwise,
	 * didn't find the current directory in its parent directory, set
	 * errno to ENOENT.
	 */
	if (!errno)
		errno = save_errno ? save_errno : ENOENT;
	/* FALLTHROUGH */
err:
	if (ptsize)
		free(pt);
	if (up)
		free(up);
	if (dir)
		(void)closedir(dir);
	return NULL;
}
#endif /* HAVE_GETCWD */
