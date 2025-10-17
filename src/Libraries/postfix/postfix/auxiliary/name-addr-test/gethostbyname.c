/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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

main(argc, argv)
int     argc;
char  **argv;
{
    struct hostent *hp;

    if (argc != 2) {
	fprintf(stderr, "usage: %s hostname\n", argv[0]);
	exit(1);
    }
    if (hp = gethostbyname(argv[1])) {
	printf("Hostname:\t%s\n", hp->h_name);
	printf("Aliases:\t");
	while (hp->h_aliases[0])
	    printf("%s ", *hp->h_aliases++);
	printf("\n");
	printf("Addresses:\t");
	while (hp->h_addr_list[0])
	    printf("%s ", inet_ntoa(*(struct in_addr *) * hp->h_addr_list++));
	printf("\n");
	exit(0);
    } else {
	fprintf(stderr, "host %s not found\n", argv[1]);
	exit(1);
    }
}
