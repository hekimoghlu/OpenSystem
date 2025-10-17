/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
 * Hash table functions
 */

#include "gen_locl.h"

RCSID("$Id$");

static Hashentry *_search(Hashtab * htab,	/* The hash table */
			  void *ptr);	/* And key */

Hashtab *
hashtabnew(int sz,
	   int (*cmp) (void *, void *),
	   unsigned (*hash) (void *))
{
    Hashtab *htab;
    int i;

    assert(sz > 0);

    htab = (Hashtab *) malloc(sizeof(Hashtab) + (sz - 1) * sizeof(Hashentry *));
    if (htab == NULL)
	return NULL;

    for (i = 0; i < sz; ++i)
	htab->tab[i] = NULL;

    htab->cmp = cmp;
    htab->hash = hash;
    htab->sz = sz;
    return htab;
}

/* Intern search function */

static Hashentry *
_search(Hashtab * htab, void *ptr)
{
    Hashentry *hptr;

    assert(htab && ptr);

    for (hptr = htab->tab[(*htab->hash) (ptr) % htab->sz];
	 hptr;
	 hptr = hptr->next)
	if ((*htab->cmp) (ptr, hptr->ptr) == 0)
	    break;
    return hptr;
}

/* Search for element in hash table */

void *
hashtabsearch(Hashtab * htab, void *ptr)
{
    Hashentry *tmp;

    tmp = _search(htab, ptr);
    return tmp ? tmp->ptr : tmp;
}

/* add element to hash table */
/* if already there, set new value */
/* !NULL if succesful */

void *
hashtabadd(Hashtab * htab, void *ptr)
{
    Hashentry *h = _search(htab, ptr);
    Hashentry **tabptr;

    assert(htab && ptr);

    if (h)
	free((void *) h->ptr);
    else {
	h = (Hashentry *) malloc(sizeof(Hashentry));
	if (h == NULL) {
	    return NULL;
	}
	tabptr = &htab->tab[(*htab->hash) (ptr) % htab->sz];
	h->next = *tabptr;
	*tabptr = h;
	h->prev = tabptr;
	if (h->next)
	    h->next->prev = &h->next;
    }
    h->ptr = ptr;
    return h;
}

/* delete element with key key. Iff freep, free Hashentry->ptr */

int
_hashtabdel(Hashtab * htab, void *ptr, int freep)
{
    Hashentry *h;

    assert(htab && ptr);

    h = _search(htab, ptr);
    if (h) {
	if (freep)
	    free(h->ptr);
	if ((*(h->prev) = h->next))
	    h->next->prev = h->prev;
	free(h);
	return 0;
    } else
	return -1;
}

/* Do something for each element */

void
hashtabforeach(Hashtab * htab, int (*func) (void *ptr, void *arg),
	       void *arg)
{
    Hashentry **h, *g;

    assert(htab);

    for (h = htab->tab; h < &htab->tab[htab->sz]; ++h)
	for (g = *h; g; g = g->next)
	    if ((*func) (g->ptr, arg))
		return;
}

/* standard hash-functions for strings */

unsigned
hashadd(const char *s)
{				/* Standard hash function */
    unsigned i;

    assert(s);

    for (i = 0; *s; ++s)
	i += *s;
    return i;
}

unsigned
hashcaseadd(const char *s)
{				/* Standard hash function */
    unsigned i;

    assert(s);

    for (i = 0; *s; ++s)
	i += toupper((unsigned char)*s);
    return i;
}

#define TWELVE (sizeof(unsigned))
#define SEVENTYFIVE (6*sizeof(unsigned))
#define HIGH_BITS (~((unsigned)(~0) >> TWELVE))

unsigned
hashjpw(const char *ss)
{				/* another hash function */
    unsigned h = 0;
    unsigned g;
    const unsigned char *s = (const unsigned char *)ss;

    for (; *s; ++s) {
	h = (h << TWELVE) + *s;
	if ((g = h & HIGH_BITS))
	    h = (h ^ (g >> SEVENTYFIVE)) & ~HIGH_BITS;
    }
    return h;
}
