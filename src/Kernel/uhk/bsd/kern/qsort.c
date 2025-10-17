/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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
 * Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved
 *
 * Copyright (c) 1992, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)qsort.c	8.1 (Berkeley) 6/4/93
 */


#include <sys/types.h>
//#include <stdlib.h>
#include <sys/kpi_private.h>

__private_extern__
void
qsort(void *a, size_t n, size_t es, int (*cmp)(const void *, const void *));

static inline char      *med3(char *, char *, char *, int (*)(const void *, const void *));
static inline void       swapfunc(char *, char *, long, int);

#define min(a, b)       ((a) < (b) ? (a) : (b))

/*
 * Qsort routine from Bentley & McIlroy's "Engineering a Sort Function".
 */
#define swapcode(TYPE, parmi, parmj, n)                 \
	long i = (n) / sizeof (TYPE);                   \
	TYPE *pi = (TYPE *) (parmi);                    \
	TYPE *pj = (TYPE *) (parmj);                    \
	do {                                            \
	        TYPE	t = *pi;                        \
	        *pi++ = *pj;                            \
	        *pj++ = t;                              \
	} while (--i > 0);

#define SWAPINIT(a, es) swaptype = ((char *)a - (char *)0) % sizeof(long) || \
	es % sizeof(long) ? 2 : es == sizeof(long)? 0 : 1;

static inline void
swapfunc(char *a, char *b, long n, int swaptype)
{
	if (swaptype <= 1) {
		swapcode(long, a, b, n);
	} else {
		swapcode(char, a, b, n);
	}
}

#define swap(a, b)                                      \
	if (swaptype == 0) {                            \
	        long t = *(long *)(a);                  \
	        *(long *)(a) = *(long *)(b);            \
	        *(long *)(b) = t;                       \
	} else                                          \
	        swapfunc(a, b, es, swaptype)

#define vecswap(a, b, n)        if ((n) > 0) swapfunc(a, b, n, swaptype)

static inline char *
med3(char *a, char *b, char *c, int (*cmp)(const void *, const void *))
{
	return cmp(a, b) < 0 ?
	       (cmp(b, c) < 0 ? b : (cmp(a, c) < 0 ? c : a))
	       :(cmp(b, c) > 0 ? b : (cmp(a, c) < 0 ? a : c));
}

__private_extern__
void
qsort(void *a, size_t n, size_t es, int (*cmp)(const void *, const void *))
{
	char *pa, *pb, *pc, *pd, *pl, *pm, *pn;
	int swaptype, swap_cnt;
	long d, r;

loop:   SWAPINIT(a, es);
	swap_cnt = 0;
	if (n < 7) {
		for (pm = (char *)a + es; pm < (char *) a + n * es; pm += es) {
			for (pl = pm; pl > (char *) a && cmp(pl - es, pl) > 0;
			    pl -= es) {
				swap(pl, (char *)(pl - es));
			}
		}
		return;
	}
	pm = (char *)a + (n / 2) * es;
	if (n > 7) {
		pl = a;
		pn = (char *)a + (n - 1) * es;
		if (n > 40) {
			d = (n / 8) * es;
			pl = med3(pl, pl + d, pl + 2 * d, cmp);
			pm = med3(pm - d, pm, pm + d, cmp);
			pn = med3(pn - 2 * d, pn - d, pn, cmp);
		}
		pm = med3(pl, pm, pn, cmp);
	}
	swap(a, pm);
	pa = pb = (char *)a + es;

	pc = pd = (char *)a + (n - 1) * es;
	for (;;) {
		while (pb <= pc && (r = cmp(pb, a)) <= 0) {
			if (r == 0) {
				swap_cnt = 1;
				swap(pa, pb);
				pa += es;
			}
			pb += es;
		}
		while (pb <= pc && (r = cmp(pc, a)) >= 0) {
			if (r == 0) {
				swap_cnt = 1;
				swap(pc, pd);
				pd -= es;
			}
			pc -= es;
		}
		if (pb > pc) {
			break;
		}
		swap(pb, pc);
		swap_cnt = 1;
		pb += es;
		pc -= es;
	}
	if (swap_cnt == 0) {  /* Switch to insertion sort */
		for (pm = (char *)a + es; pm < (char *) a + n * es; pm += es) {
			for (pl = pm; pl > (char *) a && cmp(pl - es, pl) > 0;
			    pl -= es) {
				swap(pl, pl - es);
			}
		}
		return;
	}

	pn = (char *)a + n * es;
	r = min(pa - (char *)a, pb - pa);
	vecswap(a, pb - r, r);
	r = min(pd - pc, pn - pd - es);
	vecswap(pb, pn - r, r);
	if ((size_t)(r = pb - pa) > es) {
		qsort(a, r / es, es, cmp);
	}
	if ((size_t)(r = pd - pc) > es) {
		/* Iterate rather than recurse to save stack space */
		a = pn - r;
		n = r / es;
		goto loop;
	}
/*		qsort(pn - r, r / es, es, cmp);*/
}

/* private KPI */
void
kx_qsort(void *array, size_t nm, size_t member_size, int (*cmpf)(const void *, const void *))
{
	qsort(array, nm, member_size, cmpf);
}
