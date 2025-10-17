/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
static char sccsid[] = "@(#)bsearch.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/bsearch.c,v 1.4 2007/01/09 00:28:09 imp Exp $");

#include <stddef.h>
#include <stdlib.h>

/*
 * Perform a binary search.
 *
 * The code below is a bit sneaky.  After a comparison fails, we
 * divide the work in half by moving either left or right. If lim
 * is odd, moving left simply involves halving lim: e.g., when lim
 * is 5 we look at item 2, so we change lim to 2 so that we will
 * look at items 0 & 1.  If lim is even, the same applies.  If lim
 * is odd, moving right again involes halving lim, this time moving
 * the base up one item past p: e.g., when lim is 5 we change base
 * to item 3 and make lim 2 so that we will look at items 3 and 4.
 * If lim is even, however, we have to shrink it by one before
 * halving: e.g., when lim is 4, we still looked at item 2, so we
 * have to make lim 3, then halve, obtaining 1, so that we will only
 * look at item 3.
 */
void *
bsearch(const void *key, const void *base0, size_t nmemb, size_t size,
    int (*compar)(const void *, const void *))
{
	const char *base = base0;
	size_t lim;
	int cmp;
	const void *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = (*compar)(key, p);
		if (cmp == 0)
			return ((void *)p);
		if (cmp > 0) {	/* key > p: move right */
			base = (char *)p + size;
			lim--;
		}		/* else move left */
	}
	return (NULL);
}

#ifdef __BLOCKS__
void *
bsearch_b(const void *key, const void *base0, size_t nmemb, size_t size,
    int (^compar)(const void *, const void *))
{
	const char *base = base0;
	size_t lim;
	int cmp;
	const void *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = compar(key, p);
		if (cmp == 0)
			return ((void *)p);
		if (cmp > 0) {	/* key > p: move right */
			base = (char *)p + size;
			lim--;
		}		/* else move left */
	}
	return (NULL);
}
#endif /* __BLOCKS__ */
