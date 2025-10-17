/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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
//  magazine_malloc.c
//  libmalloc
//
//  Created by Kim Topley on 11/8/17.
//
#include <darwintest.h>
#include <stdlib.h>
#include <malloc/malloc.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(malloc_zone_batch, "malloc_zone_batch_malloc and malloc_zone_batch_free")
{
	const unsigned count = 10;
	void **results;
	unsigned number;

	// Use malloc_zone_batch_malloc() with a size that maps to the tiny
	// allocator. This should succeed.
	results = calloc(count, sizeof(void *));
	number = malloc_zone_batch_malloc(malloc_default_zone(), 32, results, count);
	T_ASSERT_EQ(number, count, "allocated from tiny zone");
	for (int i = 0; i < count; i++) {
		T_QUIET; T_ASSERT_NOTNULL(results[i], "pointer %d is NULL", i);
	}
	malloc_zone_batch_free(malloc_default_zone(), results, count);
	free(results);

	// Use malloc_zone_batch_malloc() with a size that maps to the small
	// allocator. This should fail.
	results = calloc(count, sizeof(void *));
	number = malloc_zone_batch_malloc(malloc_default_zone(), 2048, results, count);
	T_ASSERT_EQ(0, number, "could not allocat from small zone");
	for (int i = 0; i < count; i++) {
		T_QUIET; T_ASSERT_NULL(results[i], "pointer %d is not NULL", i);
	}
	malloc_zone_batch_free(malloc_default_zone(), results, count);
	free(results);
}
