/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
/* $Id$ */

/*! \file */

#include <config.h>

#include <string.h>

#include <isc/mem.h>
#include <isc/random.h>
#include <isc/pool.h>
#include <isc/util.h>

/***
 *** Types.
 ***/

struct isc_pool {
	isc_mem_t *			mctx;
	unsigned int			count;
	isc_pooldeallocator_t		free;
	isc_poolinitializer_t		init;
	void *				initarg;
	void **				pool;
};

/***
 *** Functions.
 ***/

static isc_result_t
alloc_pool(isc_mem_t *mctx, unsigned int count, isc_pool_t **poolp) {
	isc_pool_t *pool;

	pool = isc_mem_get(mctx, sizeof(*pool));
	if (pool == NULL)
		return (ISC_R_NOMEMORY);
	pool->count = count;
	pool->free = NULL;
	pool->init = NULL;
	pool->initarg = NULL;
	pool->mctx = NULL;
	isc_mem_attach(mctx, &pool->mctx);
	pool->pool = isc_mem_get(mctx, count * sizeof(void *));
	if (pool->pool == NULL) {
		isc_mem_put(mctx, pool, sizeof(*pool));
		return (ISC_R_NOMEMORY);
	}
	memset(pool->pool, 0, count * sizeof(void *));

	*poolp = pool;
	return (ISC_R_SUCCESS);
}

isc_result_t
isc_pool_create(isc_mem_t *mctx, unsigned int count,
		   isc_pooldeallocator_t release,
		   isc_poolinitializer_t init, void *initarg,
		   isc_pool_t **poolp)
{
	isc_pool_t *pool = NULL;
	isc_result_t result;
	unsigned int i;

	INSIST(count > 0);

	/* Allocate the pool structure */
	result = alloc_pool(mctx, count, &pool);
	if (result != ISC_R_SUCCESS)
		return (result);

	pool->free = release;
	pool->init = init;
	pool->initarg = initarg;

	/* Populate the pool */
	for (i = 0; i < count; i++) {
		result = init(&pool->pool[i], initarg);
		if (result != ISC_R_SUCCESS) {
			isc_pool_destroy(&pool);
			return (result);
		}
	}

	*poolp = pool;
	return (ISC_R_SUCCESS);
}

void *
isc_pool_get(isc_pool_t *pool) {
	isc_uint32_t i;
	isc_random_get(&i);
	return (pool->pool[i % pool->count]);
}

int
isc_pool_count(isc_pool_t *pool) {
	REQUIRE(pool != NULL);
	return (pool->count);
}

isc_result_t
isc_pool_expand(isc_pool_t **sourcep, unsigned int count,
		   isc_pool_t **targetp)
{
	isc_result_t result;
	isc_pool_t *pool;

	REQUIRE(sourcep != NULL && *sourcep != NULL);
	REQUIRE(targetp != NULL && *targetp == NULL);

	pool = *sourcep;
	if (count > pool->count) {
		isc_pool_t *newpool = NULL;
		unsigned int i;

		/* Allocate a new pool structure */
		result = alloc_pool(pool->mctx, count, &newpool);
		if (result != ISC_R_SUCCESS)
			return (result);

		newpool->free = pool->free;
		newpool->init = pool->init;
		newpool->initarg = pool->initarg;

		/* Copy over the objects from the old pool */
		for (i = 0; i < pool->count; i++) {
			newpool->pool[i] = pool->pool[i];
			pool->pool[i] = NULL;
		}

		/* Populate the new entries */
		for (i = pool->count; i < count; i++) {
			result = pool->init(&newpool->pool[i], pool->initarg);
			if (result != ISC_R_SUCCESS) {
				isc_pool_destroy(&pool);
				return (result);
			}
		}

		isc_pool_destroy(&pool);
		pool = newpool;
	}

	*sourcep = NULL;
	*targetp = pool;
	return (ISC_R_SUCCESS);
}

void
isc_pool_destroy(isc_pool_t **poolp) {
	unsigned int i;
	isc_pool_t *pool = *poolp;
	for (i = 0; i < pool->count; i++) {
		if (pool->free != NULL && pool->pool[i] != NULL)
			pool->free(&pool->pool[i]);
	}
	isc_mem_put(pool->mctx, pool->pool, pool->count * sizeof(void *));
	isc_mem_putanddetach(&pool->mctx, pool, sizeof(*pool));
	*poolp = NULL;
}
