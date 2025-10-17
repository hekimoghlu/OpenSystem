/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)gethostnamadr.c	8.1 (Berkeley) 6/4/93";
static char rcsid[] = "$FreeBSD: src/lib/libc/net/map_v4v6.c,v 1.5.2.1 2001/03/05 10:47:08 obrien Exp $";
#endif /* LIBC_SCCS and not lint */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <arpa/nameser_compat.h>
#include <nameser.h>

#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <ctype.h>
#include <syslog.h>

typedef union {
	int32_t al;
	char ac;
} align;

void
_map_v4v6_address(src, dst)
	const char *src;
	char *dst;
{
	u_char *p = (u_char *)dst;
	char tmp[INADDRSZ];
	int i;

	/* Stash a temporary copy so our caller can update in place. */
	bcopy(src, tmp, INADDRSZ);
	/* Mark this ipv6 addr as a mapped ipv4. */
	for (i = 0; i < 10; i++)
		*p++ = 0x00;
	*p++ = 0xff;
	*p++ = 0xff;
	/* Retrieve the saved copy and we're done. */
	bcopy(tmp, (void*)p, INADDRSZ);
}

void
_map_v4v6_hostent(hp, bpp, lenp)
	struct hostent *hp;
	char **bpp;
	int *lenp;
{
	char **ap;

	if (hp->h_addrtype != AF_INET || hp->h_length != INADDRSZ)
		return;
	hp->h_addrtype = AF_INET6;
	hp->h_length = IN6ADDRSZ;
	for (ap = hp->h_addr_list; *ap; ap++) {
		int i = sizeof(align) - ((u_long)*bpp % sizeof(align));

		if (*lenp < (i + IN6ADDRSZ)) {
			/* Out of memory.  Truncate address list here.  XXX */
			*ap = NULL;
			return;
		}
		*bpp += i;
		*lenp -= i;
		_map_v4v6_address(*ap, *bpp);
		*ap = *bpp;
		*bpp += IN6ADDRSZ;
		*lenp -= IN6ADDRSZ;
	}
}
