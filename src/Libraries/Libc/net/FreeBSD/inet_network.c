/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcomma"

#if defined(LIBC_SCCS) && !defined(lint)
static const char sccsid[] = "@(#)inet_network.c	8.1 (Berkeley) 6/4/93";

/* the algorithms only can deal with ASCII, so we optimize for it */
#define USE_ASCII

#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/inet/inet_network.c,v 1.5 2008/01/14 22:55:20 cperciva Exp $");

#include "port_before.h"

#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ctype.h>

#include "port_after.h"

/*%
 * Internet network address interpretation routine.
 * The library routines call this routine to interpret
 * network numbers.
 */
in_addr_t
inet_network(const char *cp)
{
	in_addr_t val, base, n;
	char c;
	in_addr_t parts[4], *pp = parts;
	int i, digit;

again:
	val = 0; base = 10; digit = 0;
	if (*cp == '0')
		digit = 1, base = 8, cp++;
	if (*cp == 'x' || *cp == 'X')
		base = 16, cp++;
	while ((c = *cp) != 0) {
		if (isdigit((unsigned char)c)) {
			if (base == 8U && (c == '8' || c == '9'))
				return (INADDR_NONE);
			val = (val * base) + (c - '0');
			cp++;
			digit = 1;
			continue;
		}
		if (base == 16U && isxdigit((unsigned char)c)) {
			val = (val << 4) +
			      (c + 10 - (islower((unsigned char)c) ? 'a' : 'A'));
			cp++;
			digit = 1;
			continue;
		}
		break;
	}
	if (!digit)
		return (INADDR_NONE);
	if (pp >= parts + 4 || val > 0xffU)
		return (INADDR_NONE);
	if (*cp == '.') {
		*pp++ = val, cp++;
		goto again;
	}
	if (*cp && !isspace(*cp&0xff))
		return (INADDR_NONE);
	*pp++ = val;
	n = pp - parts;
	if (n > 4U)
		return (INADDR_NONE);
	for (val = 0, i = 0; i < n; i++) {
		val <<= 8;
		val |= parts[i] & 0xff;
	}
	return (val);
}

/*
 * Weak aliases for applications that use certain private entry points,
 * and fail to include <arpa/inet.h>.
 */
#undef inet_network
__weak_reference(__inet_network, inet_network);

#pragma clang diagnostic pop
/*! \file */
