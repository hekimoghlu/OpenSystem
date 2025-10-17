/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
    const char *addr;
    int     err;

#define NO_SERVICE ((char *) 0)

    if (argc != 2) {
	fprintf(stderr, "usage: %s hostname\n", argv[0]);
	exit(1);
    }
    memset((char *) &hints, 0, sizeof(hints));
    hints.ai_family = PF_UNSPEC;
    hints.ai_flags = AI_CANONNAME;
    hints.ai_socktype = SOCK_STREAM;
    if ((err = getaddrinfo(argv[1], NO_SERVICE, &hints, &res0)) != 0) {
	fprintf(stderr, "host %s not found: %s\n", argv[1], gai_strerror(err));
	exit(1);
    }
    printf("Hostname:\t%s\n", res0->ai_canonname);
    printf("Addresses:\t");
    for (res = res0; res != 0; res = res->ai_next) {
	addr = (res->ai_family == AF_INET ?
		(char *) &((struct sockaddr_in *) res->ai_addr)->sin_addr :
		(char *) &((struct sockaddr_in6 *) res->ai_addr)->sin6_addr);
	if (inet_ntop(res->ai_family, addr, hostbuf, sizeof(hostbuf)) == 0) {
	    perror("inet_ntop:");
	    exit(1);
	}
	printf("%s ", hostbuf);
    }
    printf("\n");
    freeaddrinfo(res0);
    exit(0);
}
