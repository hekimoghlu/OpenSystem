/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/net/addr2ascii.c,v 1.2 2002/03/22 21:52:28 obrien Exp $");

#include <sys/types.h>
#include <sys/socket.h>

#include <errno.h>
#include <string.h>

#include <net/if_dl.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <stdlib.h>

/*-
 * Convert a network address from binary to printable numeric format.
 * This API is copied from INRIA's IPv6 implementation, but it is a
 * bit bogus in two ways:
 *
 *	1) There is no value in passing both an address family and
 *	   an address length; either one should imply the other,
 *	   or we should be passing sockaddrs instead.
 *	2) There should by contrast be /added/ a length for the buffer
 *	   that we pass in, so that programmers are spared the need to
 *	   manually calculate (read: ``guess'') the maximum length.
 *
 * Flash: the API is also the same in the NRL implementation, and seems to
 * be some sort of standard, so we appear to be stuck with both the bad
 * naming and the poor choice of arguments.
 */
char *
addr2ascii(int af, const void *addrp, int len, char *buf)
{
	if (buf == NULL) {
		static char *staticbuf = NULL;
		
		if (staticbuf == NULL) {
			staticbuf = malloc(64); // 64 for AF_LINK > 16 for AF_INET
			if (staticbuf == NULL) {
				return NULL;
			}
		}
		
		buf = staticbuf;
	}

	switch(af) {
	case AF_INET:
		if (len != sizeof(struct in_addr)) {
			errno = ENAMETOOLONG;
			return 0;
		}
		strcpy(buf, inet_ntoa(*(const struct in_addr *)addrp));
		break;

	case AF_LINK:
		if (len != sizeof(struct sockaddr_dl)) {
			errno = ENAMETOOLONG;
			return 0;
		}
		strcpy(buf, link_ntoa((const struct sockaddr_dl *)addrp));
		break;
			
	default:
		errno = EPROTONOSUPPORT;
		return 0;
	}
	return buf;
}
