/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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

__FBSDID("$FreeBSD$");

#ifndef lint
__attribute__((__used__))
static const char sccsid[] = "@(#)ttymsg.c	8.2 (Berkeley) 11/16/93";
#endif

#include <sys/types.h>
#include <sys/uio.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <paths.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "ttymsg.h"

/*
 * Display the contents of a uio structure on a terminal.  Used by wall(1),
 * syslogd(8), and talkd(8).  Forks and finishes in child if write would block,
 * waiting up to tmout seconds.  Returns pointer to error string on unexpected
 * error; string is not newline-terminated.  Various "normal" errors are
 * ignored (exclusive-use, lack of permission, etc.).
 */
const char *
ttymsg(struct iovec *iov, int iovcnt, const char *line, int tmout)
{
	struct iovec localiov[TTYMSG_IOV_MAX];
	ssize_t left, wret;
	int cnt, fd;
	char device[MAXNAMLEN] = _PATH_DEV;
	static char errbuf[1024];
	char *p;
	int forked;

	forked = 0;
	if (iovcnt > (int)(sizeof(localiov) / sizeof(localiov[0])))
		return ("too many iov's (change code in wall/ttymsg.c)");

	strlcat(device, line, sizeof(device));
	p = device + sizeof(_PATH_DEV) - 1;
	if (strncmp(p, "pts/", 4) == 0)
		p += 4;
	if (strchr(p, '/') != NULL) {
		/* A slash is an attempt to break security... */
		(void) snprintf(errbuf, sizeof(errbuf),
		    "Too many '/' in \"%s\"", device);
		return (errbuf);
	}

	/*
	 * open will fail on slip lines or exclusive-use lines
	 * if not running as root; not an error.
	 */
	if ((fd = open(device, O_WRONLY|O_NONBLOCK, 0)) < 0) {
#ifndef __APPLE__
		if (errno == EBUSY || errno == EACCES)
			return (NULL);
#endif /* !__APPLE__ */
		(void) snprintf(errbuf, sizeof(errbuf), "%s: %s", device,
		    strerror(errno));
		return (errbuf);
	}

	for (cnt = 0, left = 0; cnt < iovcnt; ++cnt)
		left += iov[cnt].iov_len;

	for (;;) {
		wret = writev(fd, iov, iovcnt);
		if (wret >= left)
			break;
		if (wret >= 0) {
			left -= wret;
			if (iov != localiov) {
				bcopy(iov, localiov, 
				    iovcnt * sizeof(struct iovec));
				iov = localiov;
			}
			for (cnt = 0; (size_t)wret >= iov->iov_len; ++cnt) {
				wret -= iov->iov_len;
				++iov;
				--iovcnt;
			}
			if (wret) {
				iov->iov_base = (char *)iov->iov_base + wret;
				iov->iov_len -= wret;
			}
			continue;
		}
		if (errno == EWOULDBLOCK) {
			int cpid;

			if (forked) {
				(void) close(fd);
				_exit(1);
			}
			cpid = fork();
			if (cpid < 0) {
				(void) snprintf(errbuf, sizeof(errbuf),
				    "fork: %s", strerror(errno));
				(void) close(fd);
				return (errbuf);
			}
			if (cpid) {	/* parent */
				(void) close(fd);
				return (NULL);
			}
			forked++;
			/* wait at most tmout seconds */
			(void) signal(SIGALRM, SIG_DFL);
			(void) signal(SIGTERM, SIG_DFL); /* XXX */
			(void) sigsetmask(0);
			(void) alarm((u_int)tmout);
			(void) fcntl(fd, F_SETFL, 0);	/* clear O_NONBLOCK */
			continue;
		}
		/*
		 * We get ENODEV on a slip line if we're running as root,
		 * and EIO if the line just went away.
		 */
		if (errno == ENODEV || errno == EIO)
			break;
		(void) close(fd);
		if (forked)
			_exit(1);
		(void) snprintf(errbuf, sizeof(errbuf),
		    "%s: %s", device, strerror(errno));
		return (errbuf);
	}

	(void) close(fd);
	if (forked)
		_exit(0);
	return (NULL);
}
