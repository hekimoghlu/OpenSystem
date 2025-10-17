/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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
//  void swap(safe_allocation& a, safe_allocation& b);
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
	// Swap non-null with non-null
	{
		tracked_safe_allocation<T> a(10, libkern::allocate_memory);
		tracked_safe_allocation<T> b(20, libkern::allocate_memory);
		T* a_mem = a.data();
		T* b_mem = b.data();
		tracking_allocator::reset();

		swap(a, b); // ADL call

		CHECK(!tracking_allocator::did_allocate);
		CHECK(!tracking_allocator::did_deallocate);
		CHECK(a.data() == b_mem);
		CHECK(b.data() == a_mem);
		CHECK(a.size() == 20);
		CHECK(b.size() == 10);
	}

	// Swap non-null with null
	{
		tracked_safe_allocation<T> a(10, libkern::allocate_memory);
		tracked_safe_allocation<T> b = nullptr;
		T* a_mem = a.data();
		tracking_allocator::reset();

		swap(a, b); // ADL call

		CHECK(!tracking_allocator::did_allocate);
		CHECK(!tracking_allocator::did_deallocate);
		CHECK(a.data() == nullptr);
		CHECK(b.data() == a_mem);
		CHECK(a.size() == 0);
		CHECK(b.size() == 10);
	}

	// Swap null with non-null
	{
		tracked_safe_allocation<T> a = nullptr;
		tracked_safe_allocation<T> b(20, libkern::allocate_memory);
		T* b_mem = b.data();
		tracking_allocator::reset();

		swap(a, b); // ADL call

		CHECK(!tracking_allocator::did_allocate);
		CHECK(!tracking_allocator::did_deallocate);
		CHECK(a.data() == b_mem);
		CHECK(b.data() == nullptr);
		CHECK(a.size() == 20);
		CHECK(b.size() == 0);
	}

	// Swap null with null
	{
		tracked_safe_allocation<T> a = nullptr;
		tracked_safe_allocation<T> b = nullptr;
		tracking_allocator::reset();

		swap(a, b); // ADL call

		CHECK(!tracking_allocator::did_allocate);
		CHECK(!tracking_allocator::did_deallocate);
		CHECK(a.data() == nullptr);
		CHECK(b.data() == nullptr);
		CHECK(a.size() == 0);
		CHECK(b.size() == 0);
	}

	// Swap with self
	{
		tracked_safe_allocation<T> a(10, libkern::allocate_memory);
		T* a_mem = a.data();
		tracking_allocator::reset();

		swap(a, a); // ADL call

		CHECK(!tracking_allocator::did_allocate);
		CHECK(!tracking_allocator::did_deallocate);
		CHECK(a.data() == a_mem);
		CHECK(a.size() == 10);
	}
}

T_DECL(swap, "safe_allocation.swap", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
