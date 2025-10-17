/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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
//  malloc_heap_check_test.c
//  libsystem_malloc
//
//  Created by Kim Topley on 4/26/18.
//
#include <stdlib.h>
#include <darwintest.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

static void
run_heap_test()
{
	const int MAX_POINTERS = 1024;
	void **pointers = (void **)calloc(MAX_POINTERS, sizeof(void *));
	for (int iteration = 0; iteration < 100000; iteration++) {
		int index = random() % MAX_POINTERS;
		size_t size = 1 << (random() % 20);
		if (pointers[index]) {
			if (random() % 4 == 0) {
				pointers[index] = realloc(pointers[index], size);
				continue;
			}
			free(pointers[index]);
			pointers[index] = NULL;
		}
		void * pointer = malloc(size);
		T_QUIET; T_ASSERT_NOTNULL(pointer, "Allocation failed for %llu\n",
				(uint64_t)size);
		pointers[index] = pointer;
	}
	for (int i = 0; i < MAX_POINTERS; i++) {
		free(pointers[i]);
	}
	free(pointers);
}

T_DECL(malloc_heap_check_no_nano, "malloc heap checking (no Nano)",
	   T_META_ENVVAR("MallocCheckHeapStart=1"),
	   T_META_ENVVAR("MallocCheckHeapEach=1"),
	   T_META_ENVVAR("MallocNanoZone=0"))
{
	run_heap_test();

	// If we get here without crashing, we pass.
	T_PASS("Heap check succeeded");
}
T_DECL(malloc_heap_check_nano, "malloc heap checking (with Nano)",
	   T_META_ENVVAR("MallocCheckHeapStart=1"),
	   T_META_ENVVAR("MallocCheckHeapEach=1"),
	   T_META_ENVVAR("MallocNanoZone=1"))
{
	run_heap_test();

	// If we get here without crashing, we pass.
	T_PASS("Heap check succeeded");
}

