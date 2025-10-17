/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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

#include "sys_defs.h"
#include <sys/socket.h>
#include <errno.h>

/* Utility library. */

#include "msg.h"
#include "sane_accept.h"

/* sane_accept - sanitize accept() error returns */

int     sane_accept(int sock, struct sockaddr *sa, SOCKADDR_SIZE *len)
{
    static int accept_ok_errors[] = {
	EAGAIN,
	ECONNREFUSED,
	ECONNRESET,
	EHOSTDOWN,
	EHOSTUNREACH,
	EINTR,
	ENETDOWN,
	ENETUNREACH,
	ENOTCONN,
	EWOULDBLOCK,
	ENOBUFS,			/* HPUX11 */
	ECONNABORTED,
#ifdef EPROTO
	EPROTO,				/* SunOS 5.5.1 */
#endif
	0,
    };
    int     count;
    int     err;
    int     fd;

    /*
     * XXX Solaris 2.4 accept() returns EPIPE when a UNIX-domain client has
     * disconnected in the mean time. From then on, UNIX-domain sockets are
     * hosed beyond recovery. There is no point treating this as a beneficial
     * error result because the program would go into a tight loop.
     * 
     * XXX Solaris 2.5.1 accept() returns EPROTO when a TCP client has
     * disconnected in the mean time. Since there is no connection, it is
     * safe to map the error code onto EAGAIN.
     * 
     * XXX LINUX < 2.1 accept() wakes up before the three-way handshake is
     * complete, so it can fail with ECONNRESET and other "false alarm"
     * indications.
     * 
     * XXX FreeBSD 4.2-STABLE accept() returns ECONNABORTED when a UNIX-domain
     * client has disconnected in the mean time. The data that was sent with
     * connect() write() close() is lost, even though the write() and close()
     * reported successful completion. This was fixed shortly before FreeBSD
     * 4.3.
     * 
     * XXX HP-UX 11 returns ENOBUFS when the client has disconnected in the mean
     * time.
     */
    if ((fd = accept(sock, sa, len)) < 0) {
	for (count = 0; (err = accept_ok_errors[count]) != 0; count++) {
	    if (errno == err) {
		errno = EAGAIN;
		break;
	    }
	}
    }

    /*
     * XXX Solaris select() produces false read events, so that read() blocks
     * forever on a blocking socket, and fails with EAGAIN on a non-blocking
     * socket. Turning on keepalives will fix a blocking socket provided that
     * the kernel's keepalive timer expires before the Postfix watchdog
     * timer.
     * 
     * XXX Work around NAT induced damage by sending a keepalive before an idle
     * connection is expired. This requires that the kernel keepalive timer
     * is set to a short time, like 100s.
     */
    else if (sa && (sa->sa_family == AF_INET
#ifdef HAS_IPV6
		    || sa->sa_family == AF_INET6
#endif
		    )) {
	int     on = 1;

	(void) setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE,
			  (void *) &on, sizeof(on));
    }
    return (fd);
}
