/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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
#include <sys/types.h>
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/lsearch.c,v 1.1 2002/10/16 14:29:22 robert Exp $");

#define	_SEARCH_PRIVATE
#include <search.h>
#include <stdint.h>	/* for uint8_t */
#include <stdlib.h>	/* for NULL */
#include <string.h>	/* for memcpy() prototype */

static void *lwork(const void *, const void *, size_t *, size_t,
    int (*)(const void *, const void *), int);

void *lsearch(const void *key, void *base, size_t *nelp, size_t width,
    int (*compar)(const void *, const void *))
{

	return (lwork(key, base, nelp, width, compar, 1));
}

void *lfind(const void *key, const void *base, size_t *nelp, size_t width,
    int (*compar)(const void *, const void *))
{

	return (lwork(key, base, nelp, width, compar, 0));
}

static void *
lwork(const void *key, const void *base, size_t *nelp, size_t width,
    int (*compar)(const void *, const void *), int addelem)
{
	uint8_t *ep, *endp;

	/*
	 * Cast to an integer value first to avoid the warning for removing
	 * 'const' via a cast.
	 */
	ep = (uint8_t *)(uintptr_t)base;
	for (endp = (uint8_t *)(ep + width * *nelp); ep < endp; ep += width) {
		if (compar(key, ep) == 0)
			return (ep);
	}

	/* lfind() shall return when the key was not found. */
	if (!addelem)
		return (NULL);

	/*
	 * lsearch() adds the key to the end of the table and increments
	 * the number of elements.
	 */
	memcpy(endp, key, width);
	++*nelp;

	return (endp);
}
