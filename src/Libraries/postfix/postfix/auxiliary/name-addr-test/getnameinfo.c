/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

int     main(int argc, char **argv)
{
    char    hostbuf[NI_MAXHOST];	/* XXX */
    struct addrinfo hints;
    struct addrinfo *res0;
    struct addrinfo *res;
    const char *host;
    const char *addr;
    int     err;

#define NO_SERVICE ((char *) 0)

    if (argc != 2) {
	fprintf(stderr, "usage: %s ipaddres\n", argv[0]);
	exit(1);
    }

    /*
     * Convert address to internal form.
     */
    host = argv[1];
    memset((char *) &hints, 0, sizeof(hints));
    hints.ai_family = (strchr(host, ':') ? AF_INET6 : AF_INET);
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags |= AI_NUMERICHOST;
    if ((err = getaddrinfo(host, NO_SERVICE, &hints, &res0)) != 0) {
	fprintf(stderr, "getaddrinfo %s: %s\n", host, gai_strerror(err));
	exit(1);
    }

    /*
     * Convert host address to name.
     */
    for (res = res0; res != 0; res = res->ai_next) {
	err = getnameinfo(res->ai_addr, res->ai_addrlen,
			  hostbuf, sizeof(hostbuf),
			  NO_SERVICE, 0, NI_NAMEREQD);
	if (err) {
	    fprintf(stderr, "getnameinfo %s: %s\n", host, gai_strerror(err));
	    exit(1);
	}
	printf("Hostname:\t%s\n", hostbuf);
	addr = (res->ai_family == AF_INET ?
		(char *) &((struct sockaddr_in *) res->ai_addr)->sin_addr :
		(char *) &((struct sockaddr_in6 *) res->ai_addr)->sin6_addr);
	if (inet_ntop(res->ai_family, addr, hostbuf, sizeof(hostbuf)) == 0) {
	    perror("inet_ntop:");
	    exit(1);
	}
	printf("Address:\t%s\n", hostbuf);
    }
    freeaddrinfo(res0);
    exit(0);
}
