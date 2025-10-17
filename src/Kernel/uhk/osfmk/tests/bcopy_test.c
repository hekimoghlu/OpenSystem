/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#if DEVELOPMENT || DEBUG

#include <kern/bits.h>

static int
sysctl_test_memmove(__unused int64_t in, __unused int64_t *out)
{
	// Ensure our platform-specific memmove implements correct semantics
	extern void *__xnu_memmove(
		void *dst __sized_by(n),
		const void *src __sized_by(n),
		size_t n) __asm("_memmove");

	// Given two buffers
	int dest = 0;
	int source = 42;
	// When I call our platform-specific memmove implementation
	void* memmove_ret = __xnu_memmove(&dest, &source, sizeof(int));
	// Then the value of `src` has been copied to `dst`
	if (dest != 42) {
		return KERN_FAILURE;
	}
	// And `src` is unmodified
	if (source != 42) {
		return KERN_FAILURE;
	}
	// And the return value is the `dest` pointer we passed in
	if (memmove_ret != &dest) {
		return KERN_FAILURE;
	}
	return KERN_SUCCESS;
}

SYSCTL_TEST_REGISTER(test_memmove, sysctl_test_memmove);

#endif /* DEBUG || DEVELOPMENT */
