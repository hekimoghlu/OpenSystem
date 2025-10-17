/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)announce.c	8.3 (Berkeley) 4/28/95";
#endif
__attribute__((__used__))
static const char rcsid[] =
  "$FreeBSD$";
#endif /* not lint */

#include <sys/types.h>
#include <sys/uio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/socket.h>

#include <protocols/talkd.h>

#include <errno.h>
#include <paths.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <vis.h>

#ifdef __APPLE__
#include <util.h>
#else
#include "ttymsg.h"
#endif
#include "extern.h"

/*
 * Announce an invitation to talk.
 */

/*
 * See if the user is accepting messages. If so, announce that
 * a talk is requested.
 */
int
announce(CTL_MSG *request, const char *remote_machine)
{
	char full_tty[32];
	struct stat stbuf;

	(void)snprintf(full_tty, sizeof(full_tty),
	    "%s%s", _PATH_DEV, request->r_tty);
	if (stat(full_tty, &stbuf) < 0 || (stbuf.st_mode&020) == 0)
		return (PERMISSION_DENIED);
	return (print_mesg(request->r_tty, request, remote_machine));
}

#define max(a,b) ( (a) > (b) ? (a) : (b) )
#define N_LINES 5
#define N_CHARS 256

/*
 * Build a block of characters containing the message.
 * It is sent blank filled and in a single block to
 * try to keep the message in one piece if the recipient
 * in vi at the time
 */
int
print_mesg(const char *tty, CTL_MSG *request,
    const char *remote_machine)
{
	struct timeval now;
	time_t clock_sec;
	struct tm *localclock;
	struct iovec iovec;
	char line_buf[N_LINES][N_CHARS];
	int sizes[N_LINES];
	char big_buf[N_LINES*N_CHARS];
	char *bptr, *lptr, *vis_user;
	int i, j, max_size;

	i = 0;
	max_size = 0;
	gettimeofday(&now, NULL);
	clock_sec = now.tv_sec;
	localclock = localtime(&clock_sec);
	(void)snprintf(line_buf[i], N_CHARS, " ");
	sizes[i] = strlen(line_buf[i]);
	max_size = max(max_size, sizes[i]);
	i++;
	(void)snprintf(line_buf[i], N_CHARS,
		"Message from Talk_Daemon@%s at %d:%02d on %d/%.2d/%.2d ...",
		hostname, localclock->tm_hour , localclock->tm_min,
		localclock->tm_year + 1900, localclock->tm_mon + 1,
		localclock->tm_mday);
	sizes[i] = strlen(line_buf[i]);
	max_size = max(max_size, sizes[i]);
	i++;

	vis_user = malloc(strlen(request->l_name) * 4 + 1);
	strvis(vis_user, request->l_name, VIS_CSTYLE);
	(void)snprintf(line_buf[i], N_CHARS,
	    "talk: connection requested by %s@%s", vis_user, remote_machine);
	sizes[i] = strlen(line_buf[i]);
	max_size = max(max_size, sizes[i]);
	i++;
	(void)snprintf(line_buf[i], N_CHARS, "talk: respond with:  talk %s@%s",
	    vis_user, remote_machine);
	sizes[i] = strlen(line_buf[i]);
	max_size = max(max_size, sizes[i]);
	i++;
	(void)snprintf(line_buf[i], N_CHARS, " ");
	sizes[i] = strlen(line_buf[i]);
	max_size = max(max_size, sizes[i]);
	i++;
	bptr = big_buf;
	*bptr++ = '\007'; /* send something to wake them up */
	*bptr++ = '\r';	/* add a \r in case of raw mode */
	*bptr++ = '\n';
	for (i = 0; i < N_LINES; i++) {
		/* copy the line into the big buffer */
		lptr = line_buf[i];
		while (*lptr != '\0')
			*(bptr++) = *(lptr++);
		/* pad out the rest of the lines with blanks */
		for (j = sizes[i]; j < max_size + 2; j++)
			*(bptr++) = ' ';
		*(bptr++) = '\r';	/* add a \r in case of raw mode */
		*(bptr++) = '\n';
	}
	*bptr = '\0';
	iovec.iov_base = big_buf;
	iovec.iov_len = bptr - big_buf;
	/*
	 * we choose a timeout of RING_WAIT-5 seconds so that we don't
	 * stack up processes trying to write messages to a tty
	 * that is permanently blocked.
	 */
	if (ttymsg(&iovec, 1, tty, RING_WAIT - 5) != NULL)
		return (FAILED);

	return (SUCCESS);
}
