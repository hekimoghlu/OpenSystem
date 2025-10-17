/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>

/* Application-specific. */

#include "msg.h"
#include "stringops.h"
#include "find_inet.h"

#ifndef INADDR_NONE
#define INADDR_NONE 0xffffffff
#endif

/* find_inet_addr - translate numerical or symbolic host name */

unsigned find_inet_addr(const char *host)
{
    struct in_addr addr;
    struct hostent *hp;

    addr.s_addr = inet_addr(host);
    if ((addr.s_addr == INADDR_NONE) || (addr.s_addr == 0)) {
	if ((hp = gethostbyname(host)) == 0)
	    msg_fatal("host not found: %s", host);
	if (hp->h_addrtype != AF_INET)
	    msg_fatal("unexpected address family: %d", hp->h_addrtype);
	if (hp->h_length != sizeof(addr))
	    msg_fatal("unexpected address length %d", hp->h_length);
	memcpy((void *) &addr, hp->h_addr, hp->h_length);
    }
    return (addr.s_addr);
}

/* find_inet_port - translate numerical or symbolic service name */

int     find_inet_port(const char *service, const char *protocol)
{
    struct servent *sp;
    int     port;

    if (alldig(service) && (port = atoi(service)) != 0) {
	if (port < 0 || port > 65535)
	    msg_fatal("bad port number: %s", service);
	return (htons(port));
    } else {
	if ((sp = getservbyname(service, protocol)) == 0)
	    msg_fatal("unknown service: %s/%s", service, protocol);
	return (sp->s_port);
    }
}
