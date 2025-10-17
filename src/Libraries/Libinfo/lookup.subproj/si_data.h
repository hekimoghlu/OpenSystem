/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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
#ifndef __SI_DATA_H__
#define __SI_DATA_H__

#include <stdint.h>

typedef struct
{
	void *src;
	uint32_t type;
	int32_t refcount;
	uint64_t validation_a;
	uint64_t validation_b;
} si_item_t;

typedef struct
{
	int32_t refcount;
	uint32_t count;
	uint32_t curr;
	si_item_t **entry;
} si_list_t;

/* list construction - these do not retain items */
si_list_t *si_list_add(si_list_t *l, si_item_t *e);
si_list_t *si_list_concat(si_list_t *l, si_list_t *x);

si_list_t *si_list_retain(si_list_t *l);
void si_list_release(si_list_t *l);

si_item_t *si_list_next(si_list_t *list);
void si_list_reset(si_list_t *list);

si_item_t *si_item_retain(si_item_t *item);
void si_item_release(si_item_t *item);

#endif /* ! __SI_DATA_H__ */
