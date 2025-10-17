/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
static char sccsid[] = "@(#)swab.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/string/swab.c,v 1.7 2007/01/09 00:28:12 imp Exp $");

#include <unistd.h>

void
swab(const void * __restrict from, void * __restrict to, ssize_t len)
{
	unsigned long temp;
	int n;
	char *fp, *tp;

	if (len <= 0)
		return;
	n = len >> 1;
	fp = (char *)from;
	tp = (char *)to;
#define	STEP	temp = *fp++,*tp++ = *fp++,*tp++ = temp
	/* round to multiple of 8 */
	for (; n & 0x7; --n)
		STEP;
	for (n >>= 3; n > 0; --n) {
		STEP; STEP; STEP; STEP;
		STEP; STEP; STEP; STEP;
	}
}
