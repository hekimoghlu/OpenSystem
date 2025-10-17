/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "array.h"
#include "memory.h"

struct array {
	int a_count;
	int a_size;
	bool a_sorted;
	void *a_array[];
};

int
array_count(const array_t *a)
{
	return a ? a->a_count : 0;
}

void *
array_get(const array_t *a, int n)
{
	int count = array_count(a);
	if (n < 0) {
		n += count;
	}
	if (n < 0 || n >= count) {
		return NULL;
	}
	return a->a_array[n];
}

static int
array_alloc_size(int n)
{
	if (n < 16) {
		return 16;
	}
	int inc = 1 << (fls(n) - 2);
	int size = inc;
	while (size < n) size += inc;
	return size;
}

static array_t *
array_grow(array_t **array, int n)
{
	array_t *a = *array;
	array_t *b;
	int count = array_count(a);
	int size;

	if (!n) {
		return a;
	}

	if (a && count + n <= a->a_size) {
		a->a_sorted = false;
		return a;
	}

	size = array_alloc_size(count + n);
	b = xmalloc(sizeof(array_t) + size * sizeof(void *));
	b->a_count = count;
	b->a_size = size;
	b->a_sorted = false;
	if (count)
		memcpy(b->a_array, a->a_array, count * sizeof(void *));
	free(a);
	return *array = b;
}

void
array_add(array_t **array, void *v)
{
	array_t *a = array_grow(array, 1);
	a->a_array[a->a_count++] = v;
}

void
array_concat(array_t **array, array_t **other)
{
	array_t *a, *b = *other;
	int n;

	*other = NULL;
	n = array_count(b);
	a = array_grow(array, n);
	if (n) {
		memcpy(a->a_array + a->a_count, b->a_array, n * sizeof(void *));
		a->a_count += n;
	}
	free(b);
}

void
array_clear(array_t *a, void (*freeval)(void *, void *), void *priv)
{
	if (a) {
		if (freeval) {
			for (int i = 0; i < a->a_count; i++) {
				freeval(a->a_array[i], priv);
			}
		}
		a->a_count = 0;
	}
}

void
array_free(array_t **array, void (*freeval)(void *, void *), void *priv)
{
	array_clear(*array, freeval, priv);
	free(*array);
	*array = NULL;
}

int
array_iter(const array_t *a, int (*cb)(void *, void *), void *priv)
{
	int cumrc = 0;
	int count = array_count(a);
	int cbrc;

	for (int i = 0; i < count; i++) {
		if ((cbrc = cb(a->a_array[i], priv)) < 0)
			return (cbrc);
		cumrc += cbrc;
	}

	return (cumrc);
}

int
array_filter(array_t *a, int (*cb)(void *, void *), void *priv)
{
	int cbrc, r, w;
	int count = array_count(a);

	for (r = 0, w = 0; r < count; r++) {
		cbrc = cb(a->a_array[r], priv);
		if (cbrc == ARRAY_ABORT)
			break;
		if (cbrc == ARRAY_KEEP)
			a->a_array[w++] = a->a_array[r];
	}
	if (r != w && r < count)
		memmove(a->a_array + w, a->a_array + r, (count - r) * sizeof(void *));
	w += count - r;
	if (w != count)
		a->a_count = w;
	return count - w;
}

static int
array_cmp(void *priv, const void *e1, const void *e2)
{
	int (*fun)(void *, void *) = priv;
	return fun(*(void **)e1, *(void **)e2);
}

void
array_sort(array_t *a, int (*cmp)(void *, void *))
{
	int count = array_count(a);
	if (count && !a->a_sorted) {
		a->a_sorted = true;
		qsort_r(a->a_array, count, sizeof(void *), cmp, array_cmp);
	}
}
