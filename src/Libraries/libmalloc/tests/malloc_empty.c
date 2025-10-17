/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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

#include <stdlib.h>
#include <darwintest.h>
#include <malloc_private.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(empty_malloc_valid, "Zero size allocation returns valid pointer")
{
	void *ptr;

	ptr = malloc(0);
	T_ASSERT_NOTNULL(ptr, "Empty malloc returns pointer");
	free(ptr);

	ptr = calloc(1, 0);
	T_ASSERT_NOTNULL(ptr, "Empty calloc returns pointer");
	free(ptr);

	ptr = realloc(NULL, 0);
	T_ASSERT_NOTNULL(ptr, "Empty realloc returns pointer");
	free(ptr);

	ptr = aligned_alloc(sizeof(void *), 0);
	T_ASSERT_NOTNULL(ptr, "Empty aligned_alloc returns pointer");
	free(ptr);

	ptr = reallocf(NULL, 0);
	T_ASSERT_NOTNULL(ptr, "Empty reallocf returns pointer");
	free(ptr);

	ptr = valloc(0);
	T_ASSERT_NOTNULL(ptr, "Empty valloc returns pointer");
	free(ptr);

	int ret = posix_memalign(&ptr, sizeof(void *), 0);
	T_ASSERT_EQ(ret, 0, "posix_memalign returns success");
	T_ASSERT_NOTNULL(ptr, "Empty posix_memalign returns pointer");
	free(ptr);
}
