/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#include <netinet/in.h>
#include <netdb.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* Application-specific. */

#include <biff_notify.h>

/* biff_notify - notify recipient via the biff "protocol" */

void    biff_notify(const char *text, ssize_t len)
{
    static struct sockaddr_in sin;
    static int sock = -1;
    struct hostent *hp;
    struct servent *sp;

    /*
     * Initialize a socket address structure, or re-use an existing one.
     */
    if (sin.sin_family == 0) {
	if ((sp = getservbyname("biff", "udp")) == 0) {
	    msg_warn("service not found: biff/udp");
	    return;
	}
	if ((hp = gethostbyname("localhost")) == 0) {
	    msg_warn("host not found: localhost");
	    return;
	}
	if ((int) hp->h_length > (int) sizeof(sin.sin_addr)) {
	    msg_warn("bad address size %d for localhost", hp->h_length);
	    return;
	}
	sin.sin_family = hp->h_addrtype;
	sin.sin_port = sp->s_port;
	memcpy((void *) &sin.sin_addr, hp->h_addr_list[0], hp->h_length);
    }

    /*
     * Open a socket, or re-use an existing one.
     */
    if (sock < 0) {
	if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
	    msg_warn("socket: %m");
	    return;
	}
	close_on_exec(sock, CLOSE_ON_EXEC);
    }

    /*
     * Biff!
     */
    if (sendto(sock, text, len, 0, (struct sockaddr *) &sin, sizeof(sin)) != len)
	msg_warn("biff_notify: %m");
}
