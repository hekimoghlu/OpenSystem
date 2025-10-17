/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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

/* getaddrinfo via string specifying host and port */

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
roken_getaddrinfo_hostspec2(const char *hostspec,
			    int socktype,
			    int port,
			    struct addrinfo **ai)
{
    const char *p;
    char portstr[NI_MAXSERV];
    char host[MAXHOSTNAMELEN];
    struct addrinfo hints;
    int hostspec_len;

    struct hst {
	const char *prefix;
	int socktype;
	int protocol;
	int port;
    } *hstp, hst[] = {
	{ "http://", SOCK_STREAM, IPPROTO_TCP, 80 },
	{ "http/", SOCK_STREAM, IPPROTO_TCP, 80 },
	{ "tcp/", SOCK_STREAM, IPPROTO_TCP, 0 },
	{ "udp/", SOCK_DGRAM, IPPROTO_UDP, 0 },
	{ NULL, 0, 0, 0 }
    };

    memset(&hints, 0, sizeof(hints));

    hints.ai_socktype = socktype;

    for(hstp = hst; hstp->prefix; hstp++) {
	if(strncmp(hostspec, hstp->prefix, strlen(hstp->prefix)) == 0) {
	    hints.ai_socktype = hstp->socktype;
	    hints.ai_protocol = hstp->protocol;
	    if(port == 0)
		port = hstp->port;
	    hostspec += strlen(hstp->prefix);
	    break;
	}
    }

    p = strchr (hostspec, ':');
    if (p != NULL) {
	char *end;

	port = strtol (p + 1, &end, 0);
	hostspec_len = p - hostspec;
    } else {
	hostspec_len = strlen(hostspec);
    }
    snprintf (portstr, sizeof(portstr), "%u", port);

    snprintf (host, sizeof(host), "%.*s", hostspec_len, hostspec);
    return getaddrinfo (host, portstr, &hints, ai);
}

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
roken_getaddrinfo_hostspec(const char *hostspec,
			   int port,
			   struct addrinfo **ai)
{
    return roken_getaddrinfo_hostspec2(hostspec, 0, port, ai);
}
