/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
__FBSDID("$FreeBSD: src/lib/libc/string/strspn.c,v 1.5 2005/04/02 18:52:44 das Exp $");

#include <sys/types.h>
#include <limits.h>
#include <string.h>

#define	IDX(c)	((u_char)(c) / LONG_BIT)
#define	BIT(c)	((u_long)1 << ((u_char)(c) % LONG_BIT))

size_t
strspn(const char *s, const char *charset)
{
	/*
	 * NB: idx and bit are temporaries whose use causes gcc 3.4.2 to
	 * generate better code.  Without them, gcc gets a little confused.
	 */
	const char *s1;
	u_long bit;
	u_long tbl[(UCHAR_MAX + 1) / LONG_BIT];
	int idx;

	if(*s == '\0')
		return (0);

#if LONG_BIT == 64	/* always better to unroll on 64-bit architectures */
	tbl[3] = tbl[2] = tbl[1] = tbl[0] = 0;
#else
	for (idx = 0; idx < sizeof(tbl) / sizeof(tbl[0]); idx++)
		tbl[idx] = 0;
#endif
	for (; *charset != '\0'; charset++) {
		idx = IDX(*charset);
		bit = BIT(*charset);
		tbl[idx] |= bit;
	}

	for(s1 = s; ; s1++) {
		idx = IDX(*s1);
		bit = BIT(*s1);
		if ((tbl[idx] & bit) == 0)
			break;
	}
	return (s1 - s);
}
