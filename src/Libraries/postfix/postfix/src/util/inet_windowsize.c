/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
/* System libraries. */

#include <sys_defs.h>
#include <sys/socket.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* Application storage. */

 /*
  * Tunable to work around broken routers.
  */
int     inet_windowsize = 0;

/* set_inet_windowsize - set TCP send/receive window size */

void    set_inet_windowsize(int sock, int windowsize)
{

    /*
     * Sanity check.
     */
    if (windowsize <= 0)
	msg_panic("inet_windowsize: bad window size %d", windowsize);

    /*
     * Generic implementation: set the send and receive buffer size before
     * listen() or connect().
     */
    if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (void *) &windowsize,
		   sizeof(windowsize)) < 0)
	msg_warn("setsockopt SO_SNDBUF %d: %m", windowsize);
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (void *) &windowsize,
		   sizeof(windowsize)) < 0)
	msg_warn("setsockopt SO_RCVBUF %d: %m", windowsize);
}
