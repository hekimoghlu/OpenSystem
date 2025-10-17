/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#include <sys/socket.h>
#include <errno.h>

/* Utility library. */

#include "msg.h"
#include "iostuff.h"
#include "sane_connect.h"
#include "timed_connect.h"

/* timed_connect - connect with deadline */

int     timed_connect(int sock, struct sockaddr *sa, int len, int timeout)
{
    int     error;
    SOCKOPT_SIZE error_len;

    /*
     * Sanity check. Just like with timed_wait(), the timeout must be a
     * positive number.
     */
    if (timeout <= 0)
	msg_panic("timed_connect: bad timeout: %d", timeout);

    /*
     * Start the connection, and handle all possible results.
     */
    if (sane_connect(sock, sa, len) == 0)
	return (0);
    if (errno != EINPROGRESS)
	return (-1);

    /*
     * A connection is in progress. Wait for a limited amount of time for
     * something to happen. If nothing happens, report an error.
     */
    if (write_wait(sock, timeout) < 0)
	return (-1);

    /*
     * Something happened. Some Solaris 2 versions have getsockopt() itself
     * return the error, instead of returning it via the parameter list.
     */
    error = 0;
    error_len = sizeof(error);
    if (getsockopt(sock, SOL_SOCKET, SO_ERROR, (void *) &error, &error_len) < 0)
	return (-1);
    if (error) {
	errno = error;
	return (-1);
    }

    /*
     * No problems.
     */
    return (0);
}
