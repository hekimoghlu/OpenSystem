/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
#include "libinfo_common.h"

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <libkern/OSAtomic.h>
#include "si_data.h"
#include "si_module.h"

LIBINFO_EXPORT
si_list_t *
si_list_add(si_list_t *l, si_item_t *e)
{
	size_t size;

	if (e == NULL) return l;

	if (l == NULL)
	{
		l = (si_list_t *)calloc(1, sizeof(si_list_t));
		if (l != NULL) l->refcount = 1;
	}

	if (l != NULL)
	{
		size = (l->count + 1) * sizeof(si_item_t *);

		l->entry = (si_item_t **)reallocf(l->entry, size);
		if (l->entry != NULL) l->entry[l->count++] = si_item_retain(e);
	}

	if ((l == NULL) || (l->entry == NULL))
	{
		free(l);
		l = NULL;
		errno = ENOMEM;
	}

	return l;
}

LIBINFO_EXPORT
si_list_t *
si_list_concat(si_list_t *l, si_list_t *x)
{
	si_item_t *item;
	size_t newcount;
	size_t size;
	int i;

	if ((x == NULL) || (x->count == 0)) return l;

	if (l == NULL)
	{
		l = (si_list_t *)calloc(1, sizeof(si_list_t));
		l->refcount = 1;
	}

	if (l != NULL)
	{
		newcount = (size_t)l->count + (size_t)x->count;
		size = newcount * sizeof(si_item_t *);

		l->entry = (si_item_t **)reallocf(l->entry, size);
		if (l->entry)
		{
			for (i = 0; i < x->count; ++i)
			{
				item = x->entry[i];
				si_item_retain(item);
				l->entry[l->count + i] = item;
			}

			l->count += x->count;
		}
		else
		{
			l->count = 0;
			free(l);
			l = NULL;
		}
	}

	if (l == NULL) errno = ENOMEM;

	return l;
}

LIBINFO_EXPORT
si_item_t *
si_list_next(si_list_t *list)
{
	if (list == NULL) return NULL;
	if (list->curr >= list->count) return NULL;

	return list->entry[list->curr++];
}

LIBINFO_EXPORT
void
si_list_reset(si_list_t *list)
{
	if (list != NULL) list->curr = 0;
}

LIBINFO_EXPORT
si_list_t *
si_list_retain(si_list_t *list)
{
	int32_t rc;

	if (list == NULL) return NULL;

	rc = OSAtomicIncrement32Barrier(&list->refcount);
	assert(rc >= 1);

	return list;
}

LIBINFO_EXPORT
void
si_list_release(si_list_t *list)
{
	int32_t rc, i;

	if (list == NULL) return;

	rc = OSAtomicDecrement32Barrier(&list->refcount);
	assert(rc >= 0);

	if (rc == 0)
	{
		for (i = 0; i < list->count; i++)
		{
			si_item_release(list->entry[i]);
		}

		free(list->entry);
		free(list);
	}
}

LIBINFO_EXPORT
si_item_t *
si_item_retain(si_item_t *item)
{
	int32_t rc;

	if (item == NULL) return NULL;

	rc = OSAtomicIncrement32Barrier(&item->refcount);
	assert(rc >= 1);

	return item;
}

LIBINFO_EXPORT
void
si_item_release(si_item_t *item)
{
	int32_t rc;

	if (item == NULL) return;

	rc = OSAtomicDecrement32Barrier(&item->refcount);
	assert(rc >= 0);

	if (rc == 0) free(item);
}
