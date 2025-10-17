/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "sane_connect.h"

/* sane_connect - sanitize connect() results */

int     sane_connect(int sock, struct sockaddr *sa, SOCKADDR_SIZE len)
{

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
    if (sa->sa_family == AF_INET) {
	int     on = 1;

	(void) setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE,
			  (void *) &on, sizeof(on));
    }
    return (connect(sock, sa, len));
}
