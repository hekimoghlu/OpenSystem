/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
/*! \file */

#include <config.h>

#include <stddef.h>

#include <isc/counter.h>
#include <isc/magic.h>
#include <isc/mem.h>
#include <isc/util.h>

#define COUNTER_MAGIC			ISC_MAGIC('C', 'n', 't', 'r')
#define VALID_COUNTER(r)		ISC_MAGIC_VALID(r, COUNTER_MAGIC)

struct isc_counter {
	unsigned int	magic;
	isc_mem_t	*mctx;
	isc_mutex_t	lock;
	unsigned int	references;
	unsigned int	limit;
	unsigned int	used;
};

isc_result_t
isc_counter_create(isc_mem_t *mctx, int limit, isc_counter_t **counterp) {
	isc_result_t result;
	isc_counter_t *counter;

	REQUIRE(counterp != NULL && *counterp == NULL);

	counter = isc_mem_get(mctx, sizeof(*counter));
	if (counter == NULL)
		return (ISC_R_NOMEMORY);

	result = isc_mutex_init(&counter->lock);
	if (result != ISC_R_SUCCESS) {
		isc_mem_put(mctx, counter, sizeof(*counter));
		return (result);
	}

	counter->mctx = NULL;
	isc_mem_attach(mctx, &counter->mctx);

	counter->references = 1;
	counter->limit = limit;
	counter->used = 0;

	counter->magic = COUNTER_MAGIC;
	*counterp = counter;
	return (ISC_R_SUCCESS);
}

isc_result_t
isc_counter_increment(isc_counter_t *counter) {
	isc_result_t result = ISC_R_SUCCESS;

	LOCK(&counter->lock);
	counter->used++;
	if (counter->limit != 0 && counter->used >= counter->limit)
		result = ISC_R_QUOTA;
	UNLOCK(&counter->lock);

	return (result);
}

unsigned int
isc_counter_used(isc_counter_t *counter) {
	REQUIRE(VALID_COUNTER(counter));

	return (counter->used);
}

void
isc_counter_setlimit(isc_counter_t *counter, int limit) {
	REQUIRE(VALID_COUNTER(counter));

	LOCK(&counter->lock);
	counter->limit = limit;
	UNLOCK(&counter->lock);
}

void
isc_counter_attach(isc_counter_t *source, isc_counter_t **targetp) {
	REQUIRE(VALID_COUNTER(source));
	REQUIRE(targetp != NULL && *targetp == NULL);

	LOCK(&source->lock);
	source->references++;
	INSIST(source->references > 0);
	UNLOCK(&source->lock);

	*targetp = source;
}

static void
destroy(isc_counter_t *counter) {
	counter->magic = 0;
	isc_mutex_destroy(&counter->lock);
	isc_mem_putanddetach(&counter->mctx, counter, sizeof(*counter));
}

void
isc_counter_detach(isc_counter_t **counterp) {
	isc_counter_t *counter;
	isc_boolean_t want_destroy = ISC_FALSE;

	REQUIRE(counterp != NULL && *counterp != NULL);
	counter = *counterp;
	REQUIRE(VALID_COUNTER(counter));

	*counterp = NULL;

	LOCK(&counter->lock);
	INSIST(counter->references > 0);
	counter->references--;
	if (counter->references == 0)
		want_destroy = ISC_TRUE;
	UNLOCK(&counter->lock);

	if (want_destroy)
		destroy(counter);
}
