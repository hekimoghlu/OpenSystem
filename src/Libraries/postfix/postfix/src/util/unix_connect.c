/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#include <errno.h>

/* Utility library. */

#include "msg.h"
#include "iostuff.h"
#include "sane_connect.h"
#include "connect.h"
#include "timed_connect.h"

/* unix_connect - connect to UNIX-domain listener */

int     unix_connect(const char *addr, int block_mode, int timeout)
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
     * Create a client socket.
     */
    if ((sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)
	return (-1);

    /*
     * Timed connect.
     */
    if (timeout > 0) {
	non_blocking(sock, NON_BLOCKING);
	if (timed_connect(sock, (struct sockaddr *) &sun, sizeof(sun), timeout) < 0) {
	    close(sock);
	    return (-1);
	}
	if (block_mode != NON_BLOCKING)
	    non_blocking(sock, block_mode);
	return (sock);
    }

    /*
     * Maybe block until connected.
     */
    else {
	non_blocking(sock, block_mode);
	if (sane_connect(sock, (struct sockaddr *) &sun, sizeof(sun)) < 0
	    && errno != EINPROGRESS) {
	    close(sock);
	    return (-1);
	}
	return (sock);
    }
}
