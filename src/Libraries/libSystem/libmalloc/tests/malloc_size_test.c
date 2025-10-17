/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
//  malloc_size_test.c
//  libmalloc
//
//  Tests for malloc_size() on both good and bad pointers.
//

#include <darwintest.h>
#include <stdlib.h>
#include <malloc/malloc.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

static void
test_malloc_size_valid(size_t min, size_t max, size_t incr)
{
	for (size_t sz = min; sz <= max; sz += incr) {
		void *ptr = malloc(sz);
		T_ASSERT_NOTNULL(ptr, "Allocate size %llu\n", (uint64_t)sz);
		T_ASSERT_GE(malloc_size(ptr), malloc_good_size(sz), "Check size value");
		free(ptr);
	}
}

static void
test_malloc_size_invalid(size_t min, size_t max, size_t incr)
{
	for (size_t sz = min; sz <= max; sz += incr) {
		void *ptr = malloc(sz);
		T_ASSERT_NOTNULL(ptr, "Allocate size %llu\n", (uint64_t)sz);
		T_ASSERT_EQ(malloc_size(ptr + 1), 0UL, "Check offset by 1 size value");
		T_ASSERT_EQ(malloc_size(ptr + sz/2), 0UL, "Check offset by half size value");
		free(ptr);
	}
}

T_DECL(malloc_size_valid, "Test malloc_size() on valid pointers, non-Nano",
	   T_META_ENVVAR("MallocNanoZone=0"))
{
	// Test various sizes, roughly targetting each allocator range.
	test_malloc_size_valid(2, 256, 16);
	test_malloc_size_valid(512, 8192, 256);
	test_malloc_size_valid(8192, 65536, 1024);
}

T_DECL(malloc_size_valid_nanov1, "Test malloc_size() on valid pointers for Nanov1",
	   T_META_ENVVAR("MallocNanoZone=V1"))
{
	test_malloc_size_valid(2, 256, 16);
}

T_DECL(malloc_size_valid_nanov2, "Test malloc_size() on valid pointers for Nanov2",
	   T_META_ENVVAR("MallocNanoZone=V2"))
{
	test_malloc_size_valid(2, 256, 16);
}

T_DECL(malloc_size_invalid, "Test malloc_size() on invalid pointers, non-Nano",
	   T_META_ENVVAR("MallocNanoZone=0"))
{
	// Test various sizes, roughly targetting each allocator range.
	test_malloc_size_invalid(2, 256, 16);
	test_malloc_size_invalid(512, 8192, 256);
	test_malloc_size_invalid(8192, 32768, 1024);
}

T_DECL(malloc_size_invalid_nanov1, "Test malloc_size() on valid pointers for Nanov1",
	   T_META_ENVVAR("MallocNanoZone=V1"))
{
	test_malloc_size_invalid(2, 256, 16);
}

T_DECL(malloc_size_invalid_nanov2, "Test malloc_size() on valid pointers for Nanov2",
	   T_META_ENVVAR("MallocNanoZone=V2"))
{
	test_malloc_size_invalid(2, 256, 16);
}
