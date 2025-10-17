/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
#ifndef CRAY
#define STACKMIN 512
#define MINBLK (2*sizeof(struct mem) + 16)
#define F _malloc_free_
#define SBGULP 8192
#include "string.h"	/* for memcpy */

#ifdef KR_headers
#define Char char
#define Unsigned unsigned
#define Int /*int*/
#else
#define Char void
#define Unsigned size_t
#define Int int
#endif

typedef struct mem {
	struct mem *next;
	Unsigned len;
	} mem;

mem *F;

 Char *
#ifdef KR_headers
malloc(size)
	register Unsigned size;
#else
malloc(register Unsigned size)
#endif
{
	register mem *p, *q, *r, *s;
	unsigned register k, m;
	extern Char *sbrk(Int);
	char *top, *top1;

	size = (size+7) & ~7;
	r = (mem *) &F;
	for (p = F, q = 0; p; r = p, p = p->next) {
		if ((k = p->len) >= size && (!q || m > k)) {
			m = k;
			q = p;
			s = r;
			}
		}
	if (q) {
		if (q->len - size >= MINBLK) { /* split block */
			p = (mem *) (((char *) (q+1)) + size);
			p->next = q->next;
			p->len = q->len - size - sizeof(mem);
			s->next = p;
			q->len = size;
			}
		else
			s->next = q->next;
		}
	else {
		top = (Char *)(((long)sbrk(0) + 7) & ~7);
		if (F && (char *)(F+1) + F->len == top) {
			q = F;
			F = F->next;
			}
		else
			q = (mem *) top;
		top1 = (char *)(q+1) + size;
		if (sbrk((int)(top1-top+SBGULP)) == (Char *) -1)
			return 0;
		r = (mem *)top1;
		r->len = SBGULP - sizeof(mem);
		r->next = F;
		F = r;
		q->len = size;
		}
	return (Char *) (q+1);
	}

 void
#ifdef KR_headers
free(f)
	Char *f;
#else
free(Char *f)
#endif
{
	mem *p, *q, *r;
	char *pn, *qn;

	if (!f) return;
	q = (mem *) ((char *)f - sizeof(mem));
	qn = (char *)f + q->len;
	for (p = F, r = (mem *) &F; ; r = p, p = p->next) {
		if (qn == (Char *) p) {
			q->len += p->len + sizeof(mem);
			p = p->next;
			}
		pn = p ? ((char *) (p+1)) + p->len : 0;
		if (pn == (Char *) q) {
			p->len += sizeof(mem) + q->len;
			q->len = 0;
			q->next = p;
			r->next = p;
			break;
			}
		if (pn < (char *) q) {
			r->next = q;
			q->next = p;
			break;
			}
		}
	}

 Char *
#ifdef KR_headers
realloc(f, size)
	Char *f;
	Unsigned size;
#else
realloc(Char *f, Unsigned size)
#endif
{
	mem *p;
	Char *q, *f1;
	Unsigned s1;

	if (!f) return malloc(size);
	p = (mem *) ((char *)f - sizeof(mem));
	s1 = p->len;
	free(f);
	if (s1 > size)
		s1 = size + 7 & ~7;
	if (!p->len) {
		f1 = (Char *)(p->next + 1);
		memcpy(f1, f, s1);
		f = f1;
		}
	q = malloc(size);
	if (q && q != f)
		memcpy(q, f, s1);
	return q;
	}

/* The following (calloc) should really be in a separate file, */
/* but defining it here sometimes avoids confusion on systems */
/* that do not provide calloc in its own file. */

 Char *
#ifdef KR_headers
calloc(n, m) Unsigned m, n;
#else
calloc(Unsigned n, Unsigned m)
#endif
{
	Char *rv;
	rv = malloc(n *= m);
	if (n && rv)
		memset(rv, 0, n);
	return rv;
	}
#endif
