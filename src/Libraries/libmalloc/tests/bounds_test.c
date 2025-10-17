/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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

#include <darwintest.h>
#include <malloc/malloc.h>
#include <malloc_private.h>
#include <stdlib.h>
#include <../src/internal.h>

T_DECL(bounds_sanity, "Pointer Bounds Sanity Check",
		T_META_TAG_VM_NOT_PREFERRED)
{
	size_t size = rand() % 1024;
	printf("Allocating %zu bytes...", size);
	void *ptr = malloc(size);
	T_EXPECT_NOTNULL(ptr, "allocation succeeded");
	T_EXPECT_LE(size, malloc_size(ptr), "requested size smaller or equal to \
		actual size");
	size = rand() % 1024;
	printf("Reallocating %zu bytes...", size);
	ptr = realloc(ptr, size);
	T_EXPECT_NOTNULL(ptr, "reallocation succeeded");
	T_EXPECT_LE(size, malloc_size(ptr), "requested size smaller or equal to \
		actual size");
	free(ptr);
	size = rand() % 1024;
	printf("Zero allocating %zu bytes...", size);
	ptr = calloc(1, size);
	T_EXPECT_NOTNULL(ptr, "zero allocation succeeded");
	T_EXPECT_LE(size, malloc_size(ptr), "requested size smaller or equal to \
		actual size");
	free(ptr);
}
