/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
/* System library. */

#include <sys_defs.h>
#include <sys/time.h>
#include <signal.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>

 /*
  * Use poll() with fall-back to select(). MacOSX needs this for devices.
  */
#if defined(USE_SYSV_POLL_THEN_SELECT)
#define poll_fd_sysv	poll_fd
#define USE_SYSV_POLL
#define USE_BSD_SELECT
int     poll_fd_bsd(int, int, int, int, int);

 /*
  * Use select() only.
  */
#elif defined(USE_BSD_SELECT)
#define poll_fd_bsd	poll_fd
#undef USE_SYSV_POLL

 /*
  * Use poll() only.
  */
#elif defined(USE_SYSV_POLL)
#define poll_fd_sysv	poll_fd

 /*
  * Sanity check.
  */
#else
#error "specify USE_SYSV_POLL, USE_BSD_SELECT or USE_SYSV_POLL_THEN_SELECT"
#endif

#ifdef USE_SYSV_POLL
#include <poll.h>
#endif

#ifdef USE_SYS_SELECT_H
#include <sys/select.h>
#endif

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

#ifdef USE_BSD_SELECT

/* poll_fd_bsd - block with time_limit until file descriptor is ready */

int     poll_fd_bsd(int fd, int request, int time_limit,
		            int true_res, int false_res)
{
    fd_set  req_fds;
    fd_set *read_fds;
    fd_set *write_fds;
    fd_set  except_fds;
    struct timeval tv;
    struct timeval *tp;
    int     temp_fd = -1;

    /*
     * Sanity checks.
     */
    if (FD_SETSIZE <= fd) {
	if ((temp_fd = dup(fd)) < 0 || temp_fd >= FD_SETSIZE)
	    msg_fatal("descriptor %d does not fit FD_SETSIZE %d", fd, FD_SETSIZE);
	fd = temp_fd;
    }

    /*
     * Use select() so we do not depend on alarm() and on signal() handlers.
     * Restart select() when interrupted by some signal. Some select()
     * implementations reduce the time to wait when interrupted, which is
     * exactly what we want.
     */
    FD_ZERO(&req_fds);
    FD_SET(fd, &req_fds);
    except_fds = req_fds;
    if (request == POLL_FD_READ) {
	read_fds = &req_fds;
	write_fds = 0;
    } else if (request == POLL_FD_WRITE) {
	read_fds = 0;
	write_fds = &req_fds;
    } else {
	msg_panic("poll_fd: bad request %d", request);
    }

    if (time_limit >= 0) {
	tv.tv_usec = 0;
	tv.tv_sec = time_limit;
	tp = &tv;
    } else {
	tp = 0;
    }

    for (;;) {
	switch (select(fd + 1, read_fds, write_fds, &except_fds, tp)) {
	case -1:
	    if (errno != EINTR)
		msg_fatal("select: %m");
	    continue;
	case 0:
	    if (temp_fd != -1)
		(void) close(temp_fd);
	    if (false_res < 0)
		errno = ETIMEDOUT;
	    return (false_res);
	default:
	    if (temp_fd != -1)
		(void) close(temp_fd);
	    return (true_res);
	}
    }
}

#endif

#ifdef USE_SYSV_POLL

#ifdef USE_SYSV_POLL_THEN_SELECT
#define HANDLE_SYSV_POLL_ERROR(fd, req, time_limit, true_res, false_res) \
	return (poll_fd_bsd((fd), (req), (time_limit), (true_res), (false_res)))
#else
#define HANDLE_SYSV_POLL_ERROR(fd, req, time_limit, true_res, false_res) \
	msg_fatal("poll: %m")
#endif

/* poll_fd_sysv - block with time_limit until file descriptor is ready */

int     poll_fd_sysv(int fd, int request, int time_limit,
		             int true_res, int false_res)
{
    struct pollfd pollfd;

    /*
     * System-V poll() is optimal for polling a few descriptors.
     */
#define WAIT_FOR_EVENT	(-1)

    pollfd.fd = fd;
    if (request == POLL_FD_READ) {
	pollfd.events = POLLIN;
    } else if (request == POLL_FD_WRITE) {
	pollfd.events = POLLOUT;
    } else {
	msg_panic("poll_fd: bad request %d", request);
    }

    for (;;) {
	switch (poll(&pollfd, 1, time_limit < 0 ?
		     WAIT_FOR_EVENT : time_limit * 1000)) {
	case -1:
	    if (errno != EINTR)
		HANDLE_SYSV_POLL_ERROR(fd, request, time_limit,
				       true_res, false_res);
	    continue;
	case 0:
	    if (false_res < 0)
		errno = ETIMEDOUT;
	    return (false_res);
	default:
	    if (pollfd.revents & POLLNVAL)
		HANDLE_SYSV_POLL_ERROR(fd, request, time_limit,
				       true_res, false_res);
	    return (true_res);
	}
    }
}

#endif
