/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
//  malloc_msl.c
//  libmalloc
//
//  Test for enabling malloc stack logging while concurrently adding other zones
//

#include <MallocStackLogging/MallocStackLogging.h>
#include <darwintest.h>
#include <malloc/malloc.h>

#if TARGET_OS_WATCH
#define N_ZONE_CREATION_THREADS 4
#else // TARGET_OS_WATCH
#define N_ZONE_CREATION_THREADS 8
#endif // TARGET_OS_WATCH

static void *
msl_thread(void *arg)
{
	T_LOG("enable MSL");
	bool enable = msl_turn_on_stack_logging(msl_mode_lite);
	T_ASSERT_TRUE(enable, "msl_turn_on_stack_logging returned false");
	usleep(500);
	msl_turn_off_stack_logging();
	usleep(500);

	return NULL;
}

static void *
zone_thread(void *arg)
{
	vm_size_t start_size = (vm_size_t)arg;
	while (1) {
		malloc_zone_t *zone = malloc_create_zone(start_size, 0);
		malloc_destroy_zone(zone);
	}
	return NULL;
}

T_DECL(malloc_enable_msl_lite, "enable the malloc stack logging lite zone while constantly registering zones", T_META_TAG_VM_NOT_PREFERRED)
{
	pthread_t zone_threads[N_ZONE_CREATION_THREADS];
	for (int i = 0; i < N_ZONE_CREATION_THREADS; i++) {
		vm_size_t zone_start_size = 1000;
		pthread_create(&zone_threads[i], NULL, zone_thread, (void *)zone_start_size);
	}

	usleep(50);

	pthread_t msl;
	pthread_create(&msl, NULL, msl_thread, NULL);
	pthread_join(msl, NULL);
	T_PASS("finished without crashing");
}
