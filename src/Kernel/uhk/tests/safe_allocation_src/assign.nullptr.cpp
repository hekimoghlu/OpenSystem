/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
// Tests for
//  safe_allocation& operator=(std::nullptr_t);
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
	// Assign to a non-null allocation
	{
		tracked_safe_allocation<T> array(10, libkern::allocate_memory);
		tracking_allocator::reset();

		tracked_safe_allocation<T>& ref = (array = nullptr);
		CHECK(&ref == &array);
		CHECK(array.data() == nullptr);
		CHECK(array.size() == 0);
		CHECK(tracking_allocator::did_deallocate);
	}

	// Assign to a null allocation
	{
		tracked_safe_allocation<T> array = nullptr;
		tracking_allocator::reset();

		tracked_safe_allocation<T>& ref = (array = nullptr);
		CHECK(&ref == &array);
		CHECK(array.data() == nullptr);
		CHECK(array.size() == 0);
		CHECK(!tracking_allocator::did_deallocate);
	}
}

T_DECL(assign_nullptr, "safe_allocation.assign.nullptr", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
