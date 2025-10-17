/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#pragma clang diagnostic ignored "-Wstrict-prototypes"

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)heapsort.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/stdlib/heapsort.c,v 1.6 2008/01/13 02:11:10 das Exp $");

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>

/*
 * Swap two areas of size number of bytes.  Although qsort(3) permits random
 * blocks of memory to be sorted, sorting pointers is almost certainly the
 * common case (and, were it not, could easily be made so).  Regardless, it
 * isn't worth optimizing; the SWAP's get sped up by the cache, and pointer
 * arithmetic gets lost in the time required for comparison function calls.
 */
#define	SWAP(a, b, count, size, tmp) { \
	count = size; \
	do { \
		tmp = *a; \
		*a++ = *b; \
		*b++ = tmp; \
	} while (--count); \
}

/* Copy one block of size size to another. */
#define COPY(a, b, count, size, tmp1, tmp2) { \
	count = size; \
	tmp1 = a; \
	tmp2 = b; \
	do { \
		*tmp1++ = *tmp2++; \
	} while (--count); \
}

/*
 * Build the list into a heap, where a heap is defined such that for
 * the records K1 ... KN, Kj/2 >= Kj for 1 <= j/2 <= j <= N.
 *
 * There two cases.  If j == nmemb, select largest of Ki and Kj.  If
 * j < nmemb, select largest of Ki, Kj and Kj+1.
 */
#define CREATE(initval, nmemb, par_i, child_i, par, child, size, count, tmp) { \
	for (par_i = initval; (child_i = par_i * 2) <= nmemb; \
	    par_i = child_i) { \
		child = base + child_i * size; \
		if (child_i < nmemb && compar(child, child + size) < 0) { \
			child += size; \
			++child_i; \
		} \
		par = base + par_i * size; \
		if (compar(child, par) <= 0) \
			break; \
		SWAP(par, child, count, size, tmp); \
	} \
}

/*
 * Select the top of the heap and 'heapify'.  Since by far the most expensive
 * action is the call to the compar function, a considerable optimization
 * in the average case can be achieved due to the fact that k, the displaced
 * elememt, is ususally quite small, so it would be preferable to first
 * heapify, always maintaining the invariant that the larger child is copied
 * over its parent's record.
 *
 * Then, starting from the *bottom* of the heap, finding k's correct place,
 * again maintianing the invariant.  As a result of the invariant no element
 * is 'lost' when k is assigned its correct place in the heap.
 *
 * The time savings from this optimization are on the order of 15-20% for the
 * average case. See Knuth, Vol. 3, page 158, problem 18.
 *
 * XXX Don't break the #define SELECT line, below.  Reiser cpp gets upset.
 */
#define SELECT(par_i, child_i, nmemb, par, child, size, k, count, tmp1, tmp2) { \
	for (par_i = 1; (child_i = par_i * 2) <= nmemb; par_i = child_i) { \
		child = base + child_i * size; \
		if (child_i < nmemb && compar(child, child + size) < 0) { \
			child += size; \
			++child_i; \
		} \
		par = base + par_i * size; \
		COPY(par, child, count, size, tmp1, tmp2); \
	} \
	for (;;) { \
		child_i = par_i; \
		par_i = child_i / 2; \
		child = base + child_i * size; \
		par = base + par_i * size; \
		if (child_i == 1 || compar(k, par) < 0) { \
			COPY(child, k, count, size, tmp1, tmp2); \
			break; \
		} \
		COPY(child, par, count, size, tmp1, tmp2); \
	} \
}

/*
 * Heapsort -- Knuth, Vol. 3, page 145.  Runs in O (N lg N), both average
 * and worst.  While heapsort is faster than the worst case of quicksort,
 * the BSD quicksort does median selection so that the chance of finding
 * a data set that will trigger the worst case is nonexistent.  Heapsort's
 * only advantage over quicksort is that it requires little additional memory.
 */
int
heapsort(vbase, nmemb, size, compar)
	void *vbase;
	size_t nmemb, size;
	int (*compar)(const void *, const void *);
{
	size_t cnt, i, j, l;
	char tmp, *tmp1, *tmp2;
	char *base, *k, *p, *t;

	if (nmemb <= 1)
		return (0);

	if (!size) {
		errno = EINVAL;
		return (-1);
	}

	if ((k = malloc(size)) == NULL)
		return (-1);

	/*
	 * Items are numbered from 1 to nmemb, so offset from size bytes
	 * below the starting address.
	 */
	base = (char *)vbase - size;

	for (l = nmemb / 2 + 1; --l;)
		CREATE(l, nmemb, i, j, t, p, size, cnt, tmp);

	/*
	 * For each element of the heap, save the largest element into its
	 * final slot, save the displaced element (k), then recreate the
	 * heap.
	 */
	while (nmemb > 1) {
		COPY(k, base + nmemb * size, cnt, size, tmp1, tmp2);
		COPY(base + nmemb * size, base + size, cnt, size, tmp1, tmp2);
		--nmemb;
		SELECT(i, j, nmemb, t, p, size, k, cnt, tmp1, tmp2);
	}
	free(k);
	return (0);
}
#pragma clang diagnostic pop
