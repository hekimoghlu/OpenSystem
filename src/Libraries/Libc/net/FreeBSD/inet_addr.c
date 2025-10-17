/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
/*
 * Portions Copyright (c) 1993 by Digital Equipment Corporation.
 * 
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies, and that
 * the name of Digital Equipment Corporation not be used in advertising or
 * publicity pertaining to distribution of the document or software without
 * specific, written prior permission.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS" AND DIGITAL EQUIPMENT CORP. DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS.   IN NO EVENT SHALL DIGITAL EQUIPMENT
 * CORPORATION BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 */

/*
 * Copyright (c) 2004 by Internet Systems Consortium, Inc. ("ISC")
 * Portions Copyright (c) 1996-1999 by Internet Software Consortium.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND ISC DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS.  IN NO EVENT SHALL ISC BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
 * OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcomma"

#if defined(LIBC_SCCS) && !defined(lint)
static const char sccsid[] = "@(#)inet_addr.c	8.1 (Berkeley) 6/17/93";
static const char rcsid[] = "$Id: inet_addr.c,v 1.4.18.1 2005/04/27 05:00:52 sra Exp $";
#endif /* LIBC_SCCS and not lint */

/* the algorithms only can deal with ASCII, so we optimize for it */
#define USE_ASCII

#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/inet/inet_addr.c,v 1.4 2007/06/03 17:20:26 ume Exp $");

#include "port_before.h"

#include <sys/types.h>
#include <sys/param.h>

#include <netinet/in.h>
#include <arpa/inet.h>

#include <ctype.h>

#include "port_after.h"

/*%
 * Ascii internet address interpretation routine.
 * The value returned is in network order.
 */
in_addr_t		/* XXX should be struct in_addr :( */
inet_addr(const char *cp) {
	struct in_addr val;

	if (inet_aton(cp, &val))
		return (val.s_addr);
	return (INADDR_NONE);
}

/*%
 * Check whether "cp" is a valid ascii representation
 * of an Internet address and convert to a binary address.
 * Returns 1 if the address is valid, 0 if not.
 * This replaces inet_addr, the return value from which
 * cannot distinguish between failure and a local broadcast address.
 * strict == 1 disallows trailing characters.
 */
int
_inet_aton_check(const char *cp, struct in_addr *addr, int strict)
{
	u_long val;
	int base, n;
	char c;
	u_int8_t parts[4];
	u_int8_t *pp = parts;
	int digit;

	c = *cp;
	for (;;) {
		/*
		 * Collect number up to ``.''.
		 * Values are specified as for C:
		 * 0x=hex, 0=octal, isdigit=decimal.
		 */
		if (!isdigit((unsigned char)c))
			return (0);
		val = 0; base = 10; digit = 0;
		if (c == '0') {
			c = *++cp;
			if (c == 'x' || c == 'X')
				base = 16, c = *++cp;
			else {
				base = 8;
				digit = 1 ;
			}
		}
		for (;;) {
			if (isascii(c) && isdigit((unsigned char)c)) {
				if (base == 8 && (c == '8' || c == '9'))
					return (0);
				val = (val * base) + (c - '0');
				c = *++cp;
				digit = 1;
			} else if (base == 16 && isascii(c) && 
				   isxdigit((unsigned char)c)) {
				val = (val << 4) |
					(c + 10 - (islower((unsigned char)c) ? 'a' : 'A'));
				c = *++cp;
				digit = 1;
			} else
				break;
		}
		if (c == '.') {
			/*
			 * Internet format:
			 *	a.b.c.d
			 *	a.b.c	(with c treated as 16 bits)
			 *	a.b	(with b treated as 24 bits)
			 */
			if (pp >= parts + 3 || val > 0xffU)
				return (0);
			*pp++ = val;
			c = *++cp;
		} else
			break;
	}
	/*
	 * Check for trailing characters.
	 */
	if (c != '\0') {
		if (strict) return (0);
		if (!isascii(c) || !isspace(c)) return (0);
	}
	/*
	 * Did we get a valid digit?
	 */
	if (!digit)
		return (0);
	/*
	 * Concoct the address according to
	 * the number of parts specified.
	 */
	n = pp - parts + 1;
	switch (n) {
	case 1:				/*%< a -- 32 bits */
		break;

	case 2:				/*%< a.b -- 8.24 bits */
		if (val > 0xffffffU)
			return (0);
		val |= parts[0] << 24;
		break;

	case 3:				/*%< a.b.c -- 8.8.16 bits */
		if (val > 0xffffU)
			return (0);
		val |= (parts[0] << 24) | (parts[1] << 16);
		break;

	case 4:				/*%< a.b.c.d -- 8.8.8.8 bits */
		if (val > 0xffU)
			return (0);
		val |= (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8);
		break;
	}
	if (addr != NULL)
		addr->s_addr = htonl(val);
	return (1);
}

int
inet_aton(const char *cp, struct in_addr *addr)
{
	return _inet_aton_check(cp, addr, 0);
}

/*
 * Weak aliases for applications that use certain private entry points,
 * and fail to include <arpa/inet.h>.
 */
#undef inet_addr
__weak_reference(__inet_addr, inet_addr);
#undef inet_aton
__weak_reference(__inet_aton, inet_aton);

#pragma clang diagnostic pop
/*! \file */
