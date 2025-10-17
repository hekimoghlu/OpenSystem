/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#include <stddef.h>

#define POOL_CLEAR	(1<<0)		/* zero fill allocations	*/
#define POOL_QALIGN	(1<<1)		/* align data to quanta		*/
#define POOL_INTERN	(1<<2)		/* Allocate extent structures	*/
#define POOL_APPEND	(1<<3)		/*   or appended to extent data	*/

typedef void *alloc_pool_t;

alloc_pool_t pool_create(size_t size, size_t quantum, void (*bomb)(char *), int flags);
void pool_destroy(alloc_pool_t pool);
void *pool_alloc(alloc_pool_t pool, size_t size, char *bomb);
void pool_free(alloc_pool_t pool, size_t size, void *addr);

#define pool_talloc(pool, type, count, bomb) \
	((type *)pool_alloc(pool, sizeof(type) * count, bomb))

#define pool_tfree(pool, type, count, addr) \
	(pool_free(pool, sizeof(type) * count, addr))

