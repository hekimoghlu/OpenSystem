/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
#include "baselocl.h"

struct hashentry {
    struct hashentry **prev;
    struct hashentry *next;
    heim_object_t key;
    heim_object_t value;
};

struct heim_dict_data {
    size_t size;
    struct hashentry **tab;
};

static void
dict_dealloc(void *ptr)
{
    heim_dict_t dict = ptr;
    struct hashentry **h, *g, *i;

    for (h = dict->tab; h < &dict->tab[dict->size]; ++h) {
	for (g = h[0]; g; g = i) {
	    i = g->next;
	    heim_release(g->key);
	    heim_release(g->value);
	    free(g);
	}
    }
    free(dict->tab);
}

struct heim_type_data dict_object = {
    HEIM_TID_DICT,
    "dict-object",
    NULL,
    dict_dealloc,
    NULL,
    NULL,
    NULL
};

static size_t
isprime(size_t p)
{
    size_t q, i;

    for(i = 2 ; i < p; i++) {
	q = p / i;

	if (i * q == p)
	    return 0;
	if (i * i > p)
	    return 1;
    }
    return 1;
}

static size_t
findprime(size_t p)
{
    if (p % 2 == 0)
	p++;

    while (isprime(p) == 0)
	p += 2;

    return p;
}

/**
 * Allocate an array
 *
 * @return A new allocated array, free with heim_release()
 */

heim_dict_t
heim_dict_create(size_t size)
{
    heim_dict_t dict;

    dict = _heim_alloc_object(&dict_object, sizeof(*dict));

    dict->size = findprime(size);
    if (dict->size == 0) {
	heim_release(dict);
	return NULL;
    }

    dict->tab = calloc(dict->size, sizeof(dict->tab[0]));
    if (dict->tab == NULL) {
	dict->size = 0;
	heim_release(dict);
	return NULL;
    }

    return dict;
}

/**
 * Get type id of an dict
 *
 * @return the type id
 */

heim_tid_t
heim_dict_get_type_id(void)
{
    return HEIM_TID_DICT;
}

/* Intern search function */

static struct hashentry *
_search(heim_dict_t dict, heim_object_t ptr)
{
    unsigned long v = heim_get_hash(ptr);
    struct hashentry *p;

    for (p = dict->tab[v % dict->size]; p != NULL; p = p->next)
	if (heim_cmp(ptr, p->key) == 0)
	    return p;

    return NULL;
}

/**
 * Search for element in hash table
 *
 * @value dict the dict to search in
 * @value key the key to search for
 *
 * @return a retained copy of the value for key or NULL if not found
 */

heim_object_t
heim_dict_copy_value(heim_dict_t dict, heim_object_t key)
{
    struct hashentry *p;
    p = _search(dict, key);
    if (p == NULL)
	return NULL;

    return heim_retain(p->value);
}

/**
 * Add key and value to dict
 *
 * @value dict the dict to add too
 * @value key the key to add
 * @value value the value to add
 *
 * @return 0 if added, errno if not
 */

int
heim_dict_set_value(heim_dict_t dict, heim_object_t key, heim_object_t value)
{
    struct hashentry **tabptr, *h;

    h = _search(dict, key);
    if (h) {
	heim_release(h->value);
	h->value = heim_retain(value);
    } else {
	unsigned long v;

	h = malloc(sizeof(*h));
	if (h == NULL)
	    return ENOMEM;

	h->key = heim_retain(key);
	h->value = heim_retain(value);

	v = heim_get_hash(key);

	tabptr = &dict->tab[v % dict->size];
	h->next = *tabptr;
	*tabptr = h;
	h->prev = tabptr;
	if (h->next)
	    h->next->prev = &h->next;
    }

    return 0;
}

/**
 * Delete element with key key
 *
 * @value dict the dict to delete from
 * @value key the key to delete
 */

void
heim_dict_delete_key(heim_dict_t dict, heim_object_t key)
{
    struct hashentry *h = _search(dict, key);

    if (h == NULL)
	return;

    heim_release(h->key);
    heim_release(h->value);

    if ((*(h->prev) = h->next) != NULL)
	h->next->prev = h->prev;

    free(h);
}

/**
 * Do something for each element
 *
 * @value dict the dict to interate over
 * @value func the function to search for
 * @value arg argument to func
 */

void
heim_dict_iterate_f(heim_dict_t dict, void *arg, heim_dict_iterator_f_t func)
{
    struct hashentry **h, *g;

    for (h = dict->tab; h < &dict->tab[dict->size]; ++h)
	for (g = *h; g; g = g->next)
	    func(g->key, g->value, arg);
}

#ifdef __BLOCKS__
/**
 * Do something for each element
 *
 * @value dict the dict to interate over
 * @value func the function to search for
 */

void
heim_dict_iterate(heim_dict_t dict, void (^func)(heim_object_t, heim_object_t))
{
    struct hashentry **h, *g;

    for (h = dict->tab; h < &dict->tab[dict->size]; ++h)
	for (g = *h; g; g = g->next)
	    func(g->key, g->value);
}
#endif
