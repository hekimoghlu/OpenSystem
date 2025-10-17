/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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

//
//  magazine_rack.c
//  libmalloc
//
//  Created by Matt Wright on 8/29/16.
//
//

#include <darwintest.h>
#include "magazine_testing.h"

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(basic_magazine_init, "allocate magazine counts")
{
	struct rack_s rack;

	for (int i=1; i < 64; i++) {
		memset(&rack, 'a', sizeof(rack));
		rack_init(&rack, RACK_TYPE_NONE, i, 0);
		T_ASSERT_NOTNULL(rack.magazines, "%d magazine initialisation", i);
	}
}

T_DECL(basic_magazine_deinit, "allocate deallocate magazines")
{
	struct rack_s rack;
	memset(&rack, 'a', sizeof(rack));

	rack_init(&rack, RACK_TYPE_NONE, 1, 0);
	T_ASSERT_NOTNULL(rack.magazines, "magazine init");

	rack_destroy(&rack);
	T_ASSERT_NULL(rack.magazines, "magazine deinit");
}

void *
pressure_thread(void *arg)
{
	T_LOG("pressure thread started\n");
	while (1) {
		malloc_zone_pressure_relief(0, 0);
	}
}

void *
thread(void *arg)
{
	uintptr_t sz = (uintptr_t)arg;
	T_LOG("thread started (allocation size: %lu bytes)\n", sz);
	void *temp = malloc(sz);

	uint64_t c = 100;
	while (c-- > 0) {
		uint32_t num = arc4random_uniform(100000);
		void **allocs = malloc(sizeof(void *) * num);

		for (int i=0; i<num; i++) {
			allocs[i] = malloc(sz);
		}
		for (int i=0; i<num; i++) {
			free(allocs[num - 1 - i]);
		}
		free((void *)allocs);
	}
	free(temp);
	return NULL;
}

T_DECL(rack_tiny_region_remove, "exercise region deallocation race (rdar://66713029)")
{
	pthread_t p1;
	pthread_create(&p1, NULL, pressure_thread, NULL);

	const int threads = 8;
	pthread_t p[threads];

	for (int i=0; i<threads; i++) {
		pthread_create(&p[i], NULL, thread, (void *)128);
	}
	for (int i=0; i<threads; i++) {
		pthread_join(p[i], NULL);
	}
	T_PASS("finished without crashing");
}

T_DECL(rack_small_region_remove, "exercise region deallocation race (rdar://66713029)")
{
	pthread_t p1;
	pthread_create(&p1, NULL, pressure_thread, NULL);

	const int threads = 8;
	pthread_t p[threads];

	for (int i=0; i<threads; i++) {
		pthread_create(&p[i], NULL, thread, (void *)1024);
	}
	for (int i=0; i<threads; i++) {
		pthread_join(p[i], NULL);
	}
	T_PASS("finished without crashing");
}

T_DECL(rack_medium_region_remove, "exercise region deallocation race (rdar://66713029)",
	   T_META_ENVVAR("MallocMediumZone=1"),
	   T_META_ENVVAR("MallocMediumActivationThreshold=1"),
	   T_META_ENABLED(CONFIG_MEDIUM_ALLOCATOR))
{
	pthread_t p1;
	pthread_create(&p1, NULL, pressure_thread, NULL);

	const int threads = 8;
	pthread_t p[threads];

	for (int i=0; i<threads; i++) {
		pthread_create(&p[i], NULL, thread, (void *)65536);
	}
	for (int i=0; i<threads; i++) {
		pthread_join(p[i], NULL);
	}
	T_PASS("finished without crashing");
}
