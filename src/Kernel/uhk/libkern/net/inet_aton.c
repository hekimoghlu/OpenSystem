/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
#include <sys/param.h>
#include <sys/systm.h>

#include <netinet/in.h>

/* XXX ctype.h is missing, see libkern/stdio/scanf.c */
#if 1
static inline int
isspace(char c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\12';
}
#endif

int
inet_aton(const char *cp, struct in_addr *addr)
{
	u_long parts[4];
	in_addr_t val = 0;
	const char *c;
	char *endptr;
	int gotend, n;

	c = (const char *)cp;
	n = 0;

	/*
	 * Run through the string, grabbing numbers until
	 * the end of the string, or some error
	 */
	gotend = 0;
	while (!gotend) {
		unsigned long l;

		l = strtoul(c, &endptr, 0);

		if (l == ULONG_MAX || (l == 0 && endptr == c)) {
			return 0;
		}

		val = (in_addr_t)l;

		/*
		 * If the whole string is invalid, endptr will equal
		 * c.. this way we can make sure someone hasn't
		 * gone '.12' or something which would get past
		 * the next check.
		 */
		if (endptr == c) {
			return 0;
		}
		parts[n] = val;
		c = endptr;

		/* Check the next character past the previous number's end */
		switch (*c) {
		case '.':

			/* Make sure we only do 3 dots .. */
			if (n == 3) {   /* Whoops. Quit. */
				return 0;
			}
			n++;
			c++;
			break;

		case '\0':
			gotend = 1;
			break;

		default:
			if (isspace((unsigned char)*c)) {
				gotend = 1;
				break;
			} else {
				/* Invalid character, then fail. */
				return 0;
			}
		}
	}

	/* Concoct the address according to the number of parts specified. */
	switch (n) {
	case 0:                         /* a -- 32 bits */

		/*
		 * Nothing is necessary here.  Overflow checking was
		 * already done in strtoul().
		 */
		break;
	case 1:                         /* a.b -- 8.24 bits */
		if (val > 0xffffff || parts[0] > 0xff) {
			return 0;
		}
		val |= parts[0] << 24;
		break;

	case 2:                         /* a.b.c -- 8.8.16 bits */
		if (val > 0xffff || parts[0] > 0xff || parts[1] > 0xff) {
			return 0;
		}
		val |= (parts[0] << 24) | (parts[1] << 16);
		break;

	case 3:                         /* a.b.c.d -- 8.8.8.8 bits */
		if (val > 0xff || parts[0] > 0xff || parts[1] > 0xff ||
		    parts[2] > 0xff) {
			return 0;
		}
		val |= (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8);
		break;
	}

	if (addr != NULL) {
		addr->s_addr = htonl(val);
	}
	return 1;
}

