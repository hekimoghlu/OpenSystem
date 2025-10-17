/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
/* System interfaces. */

#include <sys_defs.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

/* Utility library. */

#include "msg.h"
#include "iostuff.h"
#include "listen.h"
#include "sane_accept.h"

/* unix_listen - create UNIX-domain listener */

int     unix_listen(const char *addr, int backlog, int block_mode)
{
#undef sun
    struct sockaddr_un sun;
    ssize_t len = strlen(addr);
    int     sock;

    /*
     * Translate address information to internal form.
     */
    if (len >= sizeof(sun.sun_path))
	msg_fatal("unix-domain name too long: %s", addr);
    memset((void *) &sun, 0, sizeof(sun));
    sun.sun_family = AF_UNIX;
#ifdef HAS_SUN_LEN
    sun.sun_len = len + 1;
#endif
    memcpy(sun.sun_path, addr, len + 1);

    /*
     * Create a listener socket. Do whatever we can so we don't run into
     * trouble when this process is restarted after crash.
     */
    if ((sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)
	msg_fatal("socket: %m");
    if (unlink(addr) < 0 && errno != ENOENT)
	msg_fatal("remove %s: %m", addr);
    if (bind(sock, (struct sockaddr *) &sun, sizeof(sun)) < 0)
	msg_fatal("bind: %s: %m", addr);
#ifdef FCHMOD_UNIX_SOCKETS
    if (fchmod(sock, 0666) < 0)
	msg_fatal("fchmod socket %s: %m", addr);
#else
    if (chmod(addr, 0666) < 0)
	msg_fatal("chmod socket %s: %m", addr);
#endif
    non_blocking(sock, block_mode);
    if (listen(sock, backlog) < 0)
	msg_fatal("listen: %m");
    return (sock);
}

/* unix_accept - accept connection */

int     unix_accept(int fd)
{
    return (sane_accept(fd, (struct sockaddr *) 0, (SOCKADDR_SIZE *) 0));
}
