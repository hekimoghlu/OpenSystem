/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
/* $Id: sample-gai.c,v 1.4 2009/09/02 23:48:02 tbox Exp $ */

#include <config.h>

#include <isc/net.h>
#include <isc/print.h>

#include <irs/netdb.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void
do_gai(int family, char *hostname) {
	struct addrinfo hints, *res, *res0;
	int error;
	char namebuf[1024], addrbuf[1024], servbuf[1024];

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = family;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_CANONNAME;
	error = getaddrinfo(hostname, "http", &hints, &res0);
	if (error) {
		fprintf(stderr, "getaddrinfo failed for %s,family=%d: %s\n",
			hostname, family, gai_strerror(error));
		return;
	}

	for (res = res0; res; res = res->ai_next) {
		error = getnameinfo(res->ai_addr,
				    (socklen_t)res->ai_addrlen,
				    addrbuf, sizeof(addrbuf),
				    NULL, 0, NI_NUMERICHOST);
		if (error == 0)
			error = getnameinfo(res->ai_addr,
					    (socklen_t)res->ai_addrlen,
					    namebuf, sizeof(namebuf),
					    servbuf, sizeof(servbuf), 0);
		if (error != 0) {
			fprintf(stderr, "getnameinfo failed: %s\n",
				gai_strerror(error));
		} else {
			printf("%s(%s/%s)=%s:%s\n", hostname,
			       res->ai_canonname, addrbuf, namebuf, servbuf);
		}
	}

	freeaddrinfo(res0);
}

int
main(int argc, char *argv[]) {
	if (argc < 2)
		exit(1);

	do_gai(AF_INET, argv[1]);
	do_gai(AF_INET6, argv[1]);
	do_gai(AF_UNSPEC, argv[1]);

	return (0);
}
