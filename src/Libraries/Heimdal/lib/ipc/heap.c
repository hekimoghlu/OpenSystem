/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
 * Heap
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
RCSID("$Id$");
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "heap.h"

struct heap {
    heap_cmp_fn cmp;
    unsigned max_sz;
    unsigned sz;
    heap_element *data;
};

/*
 * Allocate a new heap of size `sz' with compare function `cmp'.
 */

Heap *
heap_new (unsigned sz, heap_cmp_fn cmp)
{
    Heap *ret;
    unsigned i;

    assert(sz != 0);

    ret = malloc (sizeof(*ret));
    if (ret == NULL)
	return ret;

    ret->cmp    = cmp;
    ret->max_sz = sz;
    ret->sz     = 0;
    ret->data   = malloc (sz * sizeof(*ret->data));
    if (ret->data == NULL) {
	free (ret);
	return NULL;
    }
    for (i = 0; i < sz; ++i) {
	ret->data[i].data = NULL;
	ret->data[i].ptr  = NULL;
    }
    return ret;
}

static inline unsigned
parent (unsigned n)
{
    return (n + 1) / 2 - 1;
}

static inline unsigned
left_child (unsigned n)
{
    return 2 * n + 1;
}

static inline unsigned
right_child (unsigned n)
{
    return 2 * n + 2;
}

static heap_ptr dummy;

/*
 *
 */

static void
assign (Heap *h, unsigned n, heap_element el)
{
    h->data[n] = el;
    *(h->data[n].ptr) = n;
}

/*
 *
 */

static void
upheap (Heap *h, unsigned n)
{
    heap_element v = h->data[n];

    while (n > 0 && (*h->cmp)(h->data[parent(n)].data, v.data) > 0) {
	assign (h, n, h->data[parent(n)]);
	n = parent(n);
    }
    assign (h, n, v);
}

/*
 *
 */

static void
downheap (Heap *h, unsigned n)
{
    heap_element v = h->data[n];

    while (n < h->sz / 2) {
	int cmp1, cmp2;
	unsigned new_n;

	assert (left_child(n) < h->sz);

	new_n = left_child(n);

	cmp1 = (*h->cmp)(v.data, h->data[new_n].data);

	if (right_child(n) < h->sz) {
	    cmp2 = (*h->cmp)(v.data, h->data[right_child(n)].data);
	    if (cmp2 > cmp1) {
		cmp1  = cmp2;
		new_n = right_child(n);
	    }
	}

	if (cmp1 > 0) {
	    assign (h, n, h->data[new_n]);
	    n = new_n;
	} else {
	    break;
	}
    }
    assign (h, n, v);
}

/*
 * Insert a new element `data' into `h'.
 * Return 0 if succesful or else -1.
 */

int
heap_insert (Heap *h, const void *data, heap_ptr *ptr)
{
    assert (data != NULL);

    if (h->sz == h->max_sz) {
	unsigned new_sz = h->max_sz * 2;
	heap_element *tmp;

	tmp = realloc (h->data, new_sz * sizeof(*h->data));
	if (tmp == NULL)
	    return -1;
	h->max_sz = new_sz;
	h->data   = tmp;
    }
    if (ptr == NULL)
	ptr = &dummy;

    h->data[h->sz].data = data;
    h->data[h->sz].ptr  = ptr;
    upheap (h, h->sz);
    ++h->sz;
    return 0;
}

/*
 * Return the head of the heap `h' (or NULL if it's empty).
 */

const void *
heap_head (Heap *h)
{
    if (h->sz == 0)
	return NULL;
    else
	return h->data[0].data;
}

/*
 * Remove element `n' from the heap `h'
 */

static void
remove_this (Heap *h, unsigned n)
{
    assert (n < h->sz);

    --h->sz;
    h->data[n] = h->data[h->sz];
    h->data[h->sz].data = NULL;
    h->data[h->sz].ptr  = NULL;
    if (n != h->sz) {
	downheap (h, n);
	upheap (h, n);
    }
}

/*
 * Remove the head from the heap `h'.
 */

void
heap_remove_head (Heap *h)
{
    remove_this (h, 0);
}

/*
 * Remove this very element from the heap `h'.
 * Return 0 if succesful and -1 if it couldn't be found.
 */

int
heap_remove (Heap *h, heap_ptr ptr)
{
    if (h->sz == 0)
	return -1;

    assert (h->data[ptr].ptr != &dummy);

    remove_this (h, ptr);
    return 0;
}

/*
 * Delete the heap `h'
 */

void
heap_delete (Heap *h)
{
    free (h->data);
    free (h);
}

/*
 *
 */

static int
do_verify (Heap *h, unsigned n)
{
    if (left_child(n) < h->sz) {
	if((*h->cmp)(h->data[n].data, h->data[left_child(n)].data) > 0)
	    return 0;
	if (!do_verify (h, left_child(n)))
	    return 0;
    }
    if (right_child(n) < h->sz) {
	if((*h->cmp)(h->data[n].data, h->data[right_child(n)].data) > 0)
	    return 0;
	if (!do_verify (h, right_child(n)))
	    return 0;
    }
    return 1;
}

/*
 * Verify that `h' is really a heap.
 */

int
heap_verify (Heap *h)
{
    return do_verify (h, 0);
}
