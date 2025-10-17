/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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

#include <pcap-types.h>

#include <memory.h>
#include <stdio.h>
#include <string.h>

#include "pcap-int.h"

#include <pcap/namedb.h>

#ifdef HAVE_OS_PROTO_H
#include "os-proto.h"
#endif

static inline int skip_space(FILE *);
static inline int skip_line(FILE *);

/* Hex digit to integer. */
static inline u_char
xdtoi(u_char c)
{
	if (c >= '0' && c <= '9')
		return (u_char)(c - '0');
	else if (c >= 'a' && c <= 'f')
		return (u_char)(c - 'a' + 10);
	else
		return (u_char)(c - 'A' + 10);
}

/*
 * Skip linear white space (space and tab) and any CRs before LF.
 * Stop when we hit a non-white-space character or an end-of-line LF.
 */
static inline int
skip_space(FILE *f)
{
	int c;

	do {
		c = getc(f);
	} while (c == ' ' || c == '\t' || c == '\r');

	return c;
}

static inline int
skip_line(FILE *f)
{
	int c;

	do
		c = getc(f);
	while (c != '\n' && c != EOF);

	return c;
}

struct pcap_etherent *
pcap_next_etherent(FILE *fp)
{
	register int c, i;
	u_char d;
	char *bp;
	size_t namesize;
	static struct pcap_etherent e;

	memset((char *)&e, 0, sizeof(e));
	for (;;) {
		/* Find addr */
		c = skip_space(fp);
		if (c == EOF)
			return (NULL);
		if (c == '\n')
			continue;

		/* If this is a comment, or first thing on line
		   cannot be Ethernet address, skip the line. */
		if (!PCAP_ISXDIGIT(c)) {
			c = skip_line(fp);
			if (c == EOF)
				return (NULL);
			continue;
		}

		/* must be the start of an address */
		for (i = 0; i < 6; i += 1) {
			d = xdtoi((u_char)c);
			c = getc(fp);
			if (c == EOF)
				return (NULL);
			if (PCAP_ISXDIGIT(c)) {
				d <<= 4;
				d |= xdtoi((u_char)c);
				c = getc(fp);
				if (c == EOF)
					return (NULL);
			}
			e.addr[i] = d;
			if (c != ':')
				break;
			c = getc(fp);
			if (c == EOF)
				return (NULL);
		}

		/* Must be whitespace */
		if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
			c = skip_line(fp);
			if (c == EOF)
				return (NULL);
			continue;
		}
		c = skip_space(fp);
		if (c == EOF)
			return (NULL);

		/* hit end of line... */
		if (c == '\n')
			continue;

		if (c == '#') {
			c = skip_line(fp);
			if (c == EOF)
				return (NULL);
			continue;
		}

		/* pick up name */
		bp = e.name;
		/* Use 'namesize' to prevent buffer overflow. */
		namesize = sizeof(e.name) - 1;
		do {
			*bp++ = (u_char)c;
			c = getc(fp);
			if (c == EOF)
				return (NULL);
		} while (c != ' ' && c != '\t' && c != '\r' && c != '\n'
		    && --namesize != 0);
		*bp = '\0';

		/* Eat trailing junk */
		if (c != '\n')
			(void)skip_line(fp);

		return &e;
	}
}
