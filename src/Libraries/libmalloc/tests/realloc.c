/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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
#include <stdlib.h>
#include <stdint.h>

#include <malloc/malloc.h>

T_DECL(realloc_failure, "realloc failure", T_META_TAG_XZONE,
		T_META_TAG("no_debug"), T_META_TAG_VM_NOT_PREFERRED)
{
	void *a = malloc(16);
	T_ASSERT_NOTNULL(a, "malloc(16)");
	void *b = realloc(a, SIZE_MAX - (1 << 17));
	errno_t error = errno;
	size_t a_sz = malloc_size(a);

	T_ASSERT_NULL(b, "realloc should fail");
	T_ASSERT_EQ(error, ENOMEM, "failure should have been ENOMEM");

	T_ASSERT_GT(a_sz, 0ul, "The original pointer should not have been freed");

	free(a);
}

T_DECL(reallocf_failure, "reallocf failure", T_META_TAG_XZONE,
		T_META_TAG("no_debug"), T_META_TAG_VM_NOT_PREFERRED)
{
	// rdar://134443969: Avoid the tiny zone because it may madvise
	void *a = malloc(65536);
	T_ASSERT_NOTNULL(a, "malloc(65536)");
	void *b = reallocf(a, SIZE_MAX - (1 << 17));
	errno_t error = errno;
	size_t a_sz = malloc_size(a);

	T_ASSERT_NULL(b, "reallocf should fail");
	T_ASSERT_EQ(error, ENOMEM, "failure should have been ENOMEM");

	T_ASSERT_EQ(a_sz, 0ul, "The original pointer should have been freed");
}
