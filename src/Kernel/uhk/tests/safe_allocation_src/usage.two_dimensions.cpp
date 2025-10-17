/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
// Make sure `safe_allocation` can be used to create a two-dimensional array.
//
// Note that we don't really recommend using that representation for two
// dimensional arrays because other representations are better, but it
// should at least work.
//

#include <libkern/c++/safe_allocation.h>
#include <darwintest.h>
#include "test_utils.h"

struct T {
	int i;
};

template <typename T>
static void
tests()
{
	test_safe_allocation<test_safe_allocation<int> > array(10, libkern::allocate_memory);

	for (int i = 0; i < 10; i++) {
		array[i] = test_safe_allocation<int>(10, libkern::allocate_memory);
		for (int j = 0; j < 10; ++j) {
			array[i][j] = i + j;
		}
	}

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; ++j) {
			CHECK(array[i][j] == i + j);
		}
	}
}

T_DECL(usage_two_dimensions, "safe_allocation.usage.two_dimensions", T_META_TAG_VM_PREFERRED) {
	tests<T>();
}
