/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"
#include <stddef.h>
#include <string.h>

#include "netdissect-ctype.h"

#include "strtoaddr.h"

#ifndef NS_INADDRSZ
#define NS_INADDRSZ	4	/* IPv4 T_A */
#endif

#ifndef NS_IN6ADDRSZ
#define NS_IN6ADDRSZ	16	/* IPv6 T_AAAA */
#endif

#ifndef NS_INT16SZ
#define NS_INT16SZ	2	/* #/bytes of data in a uint16_t */
#endif

/*%
 * WARNING: Don't even consider trying to compile this on a system where
 * sizeof(int) < 4.  sizeof(int) > 4 is fine; all the world's not a VAX.
 */

/* int
 * strtoaddr(src, dst)
 *	convert presentation level IPv4 address to network order binary form.
 * return:
 *	1 if `src' is a valid input, else 0.
 * notice:
 *	does not touch `dst' unless it's returning 1.
 * author:
 *	Paul Vixie, 1996.
 */
int
strtoaddr(const char *src, void *dst)
{
	uint32_t val;
	u_int digit;
	ptrdiff_t n;
	unsigned char c;
	u_int parts[4];
	u_int *pp = parts;

	c = *src;
	for (;;) {
		/*
		 * Collect number up to ``.''.
		 * Values are specified as for C:
		 * 0x=hex, 0=octal, isdigit=decimal.
		 */
		if (!ND_ASCII_ISDIGIT(c))
			return (0);
		val = 0;
		if (c == '0') {
			c = *++src;
			if (c == 'x' || c == 'X')
				return (0);
			else if (ND_ASCII_ISDIGIT(c) && c != '9')
				return (0);
		}
		for (;;) {
			if (ND_ASCII_ISDIGIT(c)) {
				digit = c - '0';
				val = (val * 10) + digit;
				c = *++src;
			} else
				break;
		}
		if (c == '.') {
			/*
			 * Internet format:
			 *	a.b.c.d
			 *	a.b.c	(with c treated as 16 bits)
			 *	a.b	(with b treated as 24 bits)
			 *	a	(with a treated as 32 bits)
			 */
			if (pp >= parts + 3)
				return (0);
			*pp++ = val;
			c = *++src;
		} else
			break;
	}
	/*
	 * Check for trailing characters.
	 */
	if (c != '\0' && c != ' ' && c != '\t')
		return (0);
	/*
	 * Find the number of parts specified.
	 * It must be 4; we only support dotted quads, we don't
	 * support shorthand.
	 */
	n = pp - parts + 1;
	if (n != 4)
		return (0);
	/*
	 * parts[0-2] were set to the first 3 parts of the address;
	 * val was set to the 4th part.
	 *
	 * Check if any part is bigger than 255.
	 */
	if ((parts[0] | parts[1] | parts[2] | val) > 0xff)
		return (0);
	/*
	 * Add the other three parts to val.
	 */
	val |= (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8);
	if (dst) {
		val = htonl(val);
		memcpy(dst, &val, NS_INADDRSZ);
	}
	return (1);
}

/* int
 * strtoaddr6(src, dst)
 *	convert presentation level IPv6 address to network order binary form.
 * return:
 *	1 if `src' is a valid [RFC1884 2.2] address, else 0.
 * notice:
 *	(1) does not touch `dst' unless it's returning 1.
 *	(2) :: in a full address is silently ignored.
 * credit:
 *	inspired by Mark Andrews.
 * author:
 *	Paul Vixie, 1996.
 */
int
strtoaddr6(const char *src, void *dst)
{
	static const char xdigits_l[] = "0123456789abcdef",
			  xdigits_u[] = "0123456789ABCDEF";
	u_char tmp[NS_IN6ADDRSZ], *tp, *endp, *colonp;
	const char *xdigits, *curtok;
	int ch, seen_xdigits;
	u_int val;

	memset((tp = tmp), '\0', NS_IN6ADDRSZ);
	endp = tp + NS_IN6ADDRSZ;
	colonp = NULL;
	/* Leading :: requires some special handling. */
	if (*src == ':')
		if (*++src != ':')
			return (0);
	curtok = src;
	seen_xdigits = 0;
	val = 0;
	while ((ch = *src++) != '\0') {
		const char *pch;

		if ((pch = strchr((xdigits = xdigits_l), ch)) == NULL)
			pch = strchr((xdigits = xdigits_u), ch);
		if (pch != NULL) {
			val <<= 4;
			val |= (int)(pch - xdigits);
			if (++seen_xdigits > 4)
				return (0);
			continue;
		}
		if (ch == ':') {
			curtok = src;
			if (!seen_xdigits) {
				if (colonp)
					return (0);
				colonp = tp;
				continue;
			} else if (*src == '\0')
				return (0);
			if (tp + NS_INT16SZ > endp)
				return (0);
			*tp++ = (u_char) (val >> 8) & 0xff;
			*tp++ = (u_char) val & 0xff;
			seen_xdigits = 0;
			val = 0;
			continue;
		}
		if (ch == '.' && ((tp + NS_INADDRSZ) <= endp) &&
		    strtoaddr(curtok, tp) > 0) {
			tp += NS_INADDRSZ;
			seen_xdigits = 0;
			break;	/*%< '\\0' was seen by strtoaddr(). */
		}
		return (0);
	}
	if (seen_xdigits) {
		if (tp + NS_INT16SZ > endp)
			return (0);
		*tp++ = (u_char) (val >> 8) & 0xff;
		*tp++ = (u_char) val & 0xff;
	}
	if (colonp != NULL) {
		/*
		 * Since some memmove()'s erroneously fail to handle
		 * overlapping regions, we'll do the shift by hand.
		 */
		const ptrdiff_t n = tp - colonp;
		int i;

		if (tp == endp)
			return (0);
		for (i = 1; i <= n; i++) {
			endp[- i] = colonp[n - i];
			colonp[n - i] = 0;
		}
		tp = endp;
	}
	if (tp != endp)
		return (0);
	memcpy(dst, tmp, NS_IN6ADDRSZ);
	return (1);
}
