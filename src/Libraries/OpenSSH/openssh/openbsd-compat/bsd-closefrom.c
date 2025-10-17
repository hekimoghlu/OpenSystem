/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#include "includes.h"

#if !defined(HAVE_CLOSEFROM) || defined(BROKEN_CLOSEFROM)

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#include <limits.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#ifdef HAVE_DIRENT_H
# include <dirent.h>
# define NAMLEN(dirent) strlen((dirent)->d_name)
#else
# define dirent direct
# define NAMLEN(dirent) (dirent)->d_namlen
# ifdef HAVE_SYS_NDIR_H
#  include <sys/ndir.h>
# endif
# ifdef HAVE_SYS_DIR_H
#  include <sys/dir.h>
# endif
# ifdef HAVE_NDIR_H
#  include <ndir.h>
# endif
#endif
#if defined(HAVE_LIBPROC_H)
# include <libproc.h>
#endif

#ifndef OPEN_MAX
# define OPEN_MAX	256
#endif

#if 0
__unused static const char rcsid[] = "$Sudo: closefrom.c,v 1.11 2006/08/17 15:26:54 millert Exp $";
#endif /* lint */

#ifndef HAVE_FCNTL_CLOSEM
/*
 * Close all file descriptors greater than or equal to lowfd.
 */
static void
closefrom_fallback(int lowfd)
{
	long fd, maxfd;

	/*
	 * Fall back on sysconf() or getdtablesize().  We avoid checking
	 * resource limits since it is possible to open a file descriptor
	 * and then drop the rlimit such that it is below the open fd.
	 */
#ifdef HAVE_SYSCONF
	maxfd = sysconf(_SC_OPEN_MAX);
#else
	maxfd = getdtablesize();
#endif /* HAVE_SYSCONF */
	if (maxfd < 0)
		maxfd = OPEN_MAX;

	for (fd = lowfd; fd < maxfd; fd++)
		(void) close((int) fd);
}
#endif /* HAVE_FCNTL_CLOSEM */

#ifdef HAVE_FCNTL_CLOSEM
void
closefrom(int lowfd)
{
    (void) fcntl(lowfd, F_CLOSEM, 0);
}
#elif defined(HAVE_LIBPROC_H) && defined(HAVE_PROC_PIDINFO)
void
closefrom(int lowfd)
{
	int i, r, sz;
	pid_t pid = getpid();
	struct proc_fdinfo *fdinfo_buf = NULL;

	sz = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, NULL, 0);
	if (sz == 0)
		return; /* no fds, really? */
	else if (sz == -1)
		goto fallback;
	if ((fdinfo_buf = malloc(sz)) == NULL)
		goto fallback;
	r = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, fdinfo_buf, sz);
	if (r < 0 || r > sz)
		goto fallback;
	for (i = 0; i < r / (int)PROC_PIDLISTFD_SIZE; i++) {
		if (fdinfo_buf[i].proc_fd >= lowfd)
			close(fdinfo_buf[i].proc_fd);
	}
	free(fdinfo_buf);
	return;
 fallback:
	free(fdinfo_buf);
	closefrom_fallback(lowfd);
	return;
}
#elif defined(HAVE_DIRFD) && defined(HAVE_PROC_PID)
void
closefrom(int lowfd)
{
    long fd;
    char fdpath[PATH_MAX], *endp;
    struct dirent *dent;
    DIR *dirp;
    int len;

#ifdef HAVE_CLOSE_RANGE
	if (close_range(lowfd, INT_MAX, 0) == 0)
		return;
#endif

    /* Check for a /proc/$$/fd directory. */
    len = snprintf(fdpath, sizeof(fdpath), "/proc/%ld/fd", (long)getpid());
    if (len > 0 && (size_t)len < sizeof(fdpath) && (dirp = opendir(fdpath))) {
	while ((dent = readdir(dirp)) != NULL) {
	    fd = strtol(dent->d_name, &endp, 10);
	    if (dent->d_name != endp && *endp == '\0' &&
		fd >= 0 && fd < INT_MAX && fd >= lowfd && fd != dirfd(dirp))
		(void) close((int) fd);
	}
	(void) closedir(dirp);
	return;
    }
    /* /proc/$$/fd strategy failed, fall back to brute force closure */
    closefrom_fallback(lowfd);
}
#else
void
closefrom(int lowfd)
{
	closefrom_fallback(lowfd);
}
#endif /* !HAVE_FCNTL_CLOSEM */
#endif /* HAVE_CLOSEFROM */
