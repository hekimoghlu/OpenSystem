/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
#include "apr_private.h"
#include "apr_arch_networkio.h"

#if APR_HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#if APR_HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#if APR_HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#if APR_HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#include <string.h>
#if APR_HAVE_ERRNO_H
#include <errno.h>
#endif

#ifndef IN6ADDRSZ
#define IN6ADDRSZ   16
#endif

#ifndef INT16SZ
#define INT16SZ sizeof(apr_int16_t)
#endif

#ifndef INADDRSZ
#define INADDRSZ    4
#endif

#ifndef __P
#define __P(x) x
#endif

#if !defined(EAFNOSUPPORT) && defined(WSAEAFNOSUPPORT)
#define EAFNOSUPPORT WSAEAFNOSUPPORT
#endif

/*
 * WARNING: Don't even consider trying to compile this on a system where
 * sizeof(int) < 4.  sizeof(int) > 4 is fine; all the world's not a VAX.
 */

static int	inet_pton4 __P((const char *src, unsigned char *dst));
#if APR_HAVE_IPV6
static int	inet_pton6 __P((const char *src, unsigned char *dst));
#endif

/* int
 * inet_pton(af, src, dst)
 *	convert from presentation format (which usually means ASCII printable)
 *	to network format (which is usually some kind of binary format).
 * return:
 *	1 if the address was valid for the specified address family
 *	0 if the address wasn't valid (`dst' is untouched in this case)
 *	-1 if some other error occurred (`dst' is untouched in this case, too)
 * author:
 *	Paul Vixie, 1996.
 */
int
apr_inet_pton(int af, const char *src, void *dst)
{
	switch (af) {
	case AF_INET:
		return (inet_pton4(src, dst));
#if APR_HAVE_IPV6
	case AF_INET6:
		return (inet_pton6(src, dst));
#endif
	default:
		errno = EAFNOSUPPORT;
		return (-1);
	}
	/* NOTREACHED */
}

/* int
 * inet_pton4(src, dst)
 *	like inet_aton() but without all the hexadecimal and shorthand.
 * return:
 *	1 if `src' is a valid dotted quad, else 0.
 * notice:
 *	does not touch `dst' unless it's returning 1.
 * author:
 *	Paul Vixie, 1996.
 */
static int
inet_pton4(const char *src, unsigned char *dst)
{
    static const char digits[] = "0123456789";
    int saw_digit, octets, ch;
    unsigned char tmp[INADDRSZ], *tp;

    saw_digit = 0;
    octets = 0;
    *(tp = tmp) = 0;
    while ((ch = *src++) != '\0') {
	const char *pch;

	if ((pch = strchr(digits, ch)) != NULL) {
	    unsigned int new = *tp * 10 + (unsigned int)(pch - digits);

	    if (new > 255)
		return (0);
	    *tp = new;
	    if (! saw_digit) {
		if (++octets > 4)
		    return (0);
		saw_digit = 1;
	    }
	} else if (ch == '.' && saw_digit) {
	    if (octets == 4)
		return (0);
	    *++tp = 0;
	    saw_digit = 0;
	} else
		return (0);
    }
    if (octets < 4)
	return (0);

    memcpy(dst, tmp, INADDRSZ);
    return (1);
}

#if APR_HAVE_IPV6
/* int
 * inet_pton6(src, dst)
 *	convert presentation level address to network order binary form.
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
static int
inet_pton6(const char *src, unsigned char *dst)
{
	static const char xdigits_l[] = "0123456789abcdef",
			  xdigits_u[] = "0123456789ABCDEF";
	unsigned char tmp[IN6ADDRSZ], *tp, *endp, *colonp;
	const char *xdigits, *curtok;
	int ch, saw_xdigit;
	unsigned int val;

	memset((tp = tmp), '\0', IN6ADDRSZ);
	endp = tp + IN6ADDRSZ;
	colonp = NULL;
	/* Leading :: requires some special handling. */
	if (*src == ':')
		if (*++src != ':')
			return (0);
	curtok = src;
	saw_xdigit = 0;
	val = 0;
	while ((ch = *src++) != '\0') {
		const char *pch;

		if ((pch = strchr((xdigits = xdigits_l), ch)) == NULL)
			pch = strchr((xdigits = xdigits_u), ch);
		if (pch != NULL) {
			val <<= 4;
			val |= (pch - xdigits);
			if (val > 0xffff)
				return (0);
			saw_xdigit = 1;
			continue;
		}
		if (ch == ':') {
			curtok = src;
			if (!saw_xdigit) {
				if (colonp)
					return (0);
				colonp = tp;
				continue;
			}
			if (tp + INT16SZ > endp)
				return (0);
			*tp++ = (unsigned char) (val >> 8) & 0xff;
			*tp++ = (unsigned char) val & 0xff;
			saw_xdigit = 0;
			val = 0;
			continue;
		}
		if (ch == '.' && ((tp + INADDRSZ) <= endp) &&
		    inet_pton4(curtok, tp) > 0) {
			tp += INADDRSZ;
			saw_xdigit = 0;
			break;	/* '\0' was seen by inet_pton4(). */
		}
		return (0);
	}
	if (saw_xdigit) {
		if (tp + INT16SZ > endp)
			return (0);
		*tp++ = (unsigned char) (val >> 8) & 0xff;
		*tp++ = (unsigned char) val & 0xff;
	}
	if (colonp != NULL) {
		/*
		 * Since some memmove()'s erroneously fail to handle
		 * overlapping regions, we'll do the shift by hand.
		 */
		const apr_ssize_t n = tp - colonp;
		apr_ssize_t i;

		for (i = 1; i <= n; i++) {
			endp[- i] = colonp[n - i];
			colonp[n - i] = 0;
		}
		tp = endp;
	}
	if (tp != endp)
		return (0);
	memcpy(dst, tmp, IN6ADDRSZ);
	return (1);
}
#endif
