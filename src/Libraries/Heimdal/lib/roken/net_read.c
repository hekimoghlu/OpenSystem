/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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

#include "roken.h"

/*
 * Like read but never return partial data.
 */

#ifndef _WIN32

ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
net_read (rk_socket_t fd, void *buf, size_t nbytes)
{
    char *cbuf = (char *)buf;
    ssize_t count;
    size_t rem = nbytes;

    while (rem > 0) {
	count = read (fd, cbuf, rem);
	if (count < 0) {
	    if (errno == EINTR)
		continue;
	    else
		return count;
	} else if (count == 0) {
	    return count;
	}
	cbuf += count;
	rem -= count;
    }
    return nbytes;
}

#else

ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
net_read(rk_socket_t sock, void *buf, size_t nbytes)
{
    char *cbuf = (char *)buf;
    ssize_t count;
    size_t rem = nbytes;

#ifdef SOCKET_IS_NOT_AN_FD
    int use_read = 0;
#endif

    while (rem > 0) {
#ifdef SOCKET_IS_NOT_AN_FD
	if (use_read)
	    count = _read (sock, cbuf, rem);
	else
	    count = recv (sock, cbuf, rem, 0);

	if (use_read == 0 &&
	    rk_IS_SOCKET_ERROR(count) &&
            (rk_SOCK_ERRNO == WSANOTINITIALISED ||
             rk_SOCK_ERRNO == WSAENOTSOCK)) {
	    use_read = 1;

	    count = _read (sock, cbuf, rem);
	}
#else
	count = recv (sock, cbuf, rem, 0);
#endif
	if (count < 0) {

	    /* With WinSock, the error EINTR (WSAEINTR), is used to
	       indicate that a blocking call was cancelled using
	       WSACancelBlockingCall(). */

#ifndef HAVE_WINSOCK
	    if (rk_SOCK_ERRNO == EINTR)
		continue;
#endif
	    return count;
	} else if (count == 0) {
	    return count;
	}
	cbuf += count;
	rem -= count;
    }
    return nbytes;
}

#endif
