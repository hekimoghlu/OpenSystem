/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
//      safe_allocation& operator=(safe_allocation&& other);
//

#include <libkern/c++/safe_allocation.h>
#include <darwintest.h>
#include "test_utils.h"
#include <utility>

struct T {
	int i;
};

template <typename T>
static void
tests()
{
	// Move-assign non-null to non-null
	{
		{
			tracked_safe_allocation<T> from(10, libkern::allocate_memory);
			T* memory = from.data();
			{
				tracked_safe_allocation<T> to(20, libkern::allocate_memory);
				tracking_allocator::reset();

				tracked_safe_allocation<T>& ref = (to = std::move(from));
				CHECK(&ref == &to);
				CHECK(to.data() == memory);
				CHECK(to.size() == 10);
				CHECK(from.data() == nullptr);
				CHECK(from.size() == 0);

				CHECK(!tracking_allocator::did_allocate);
				CHECK(tracking_allocator::deallocated_size == 20 * sizeof(T));
				tracking_allocator::reset();
			}
			CHECK(tracking_allocator::deallocated_size == 10 * sizeof(T));
			tracking_allocator::reset();
		}
		CHECK(!tracking_allocator::did_deallocate);
	}

	// Move-assign null to non-null
	{
		{
			tracked_safe_allocation<T> from = nullptr;
			{
				tracked_safe_allocation<T> to(20, libkern::allocate_memory);
				tracking_allocator::reset();

				tracked_safe_allocation<T>& ref = (to = std::move(from));
				CHECK(&ref == &to);
				CHECK(to.data() == nullptr);
				CHECK(to.size() == 0);
				CHECK(from.data() == nullptr);
				CHECK(from.size() == 0);

				CHECK(!tracking_allocator::did_allocate);
				CHECK(tracking_allocator::deallocated_size == 20 * sizeof(T));
				tracking_allocator::reset();
			}
			CHECK(!tracking_allocator::did_deallocate);
			tracking_allocator::reset();
		}
		CHECK(!tracking_allocator::did_deallocate);
	}

	// Move-assign non-null to null
	{
		{
			tracked_safe_allocation<T> from(10, libkern::allocate_memory);
			T* memory = from.data();
			{
				tracked_safe_allocation<T> to = nullptr;
				tracking_allocator::reset();

				tracked_safe_allocation<T>& ref = (to = std::move(from));
				CHECK(&ref == &to);
				CHECK(to.data() == memory);
				CHECK(to.size() == 10);
				CHECK(from.data() == nullptr);
				CHECK(from.size() == 0);

				CHECK(!tracking_allocator::did_allocate);
				CHECK(!tracking_allocator::did_deallocate);
				tracking_allocator::reset();
			}
			CHECK(tracking_allocator::deallocated_size == 10 * sizeof(T));
			tracking_allocator::reset();
		}
		CHECK(!tracking_allocator::did_deallocate);
	}

	// Move-assign null to null
	{
		{
			tracked_safe_allocation<T> from = nullptr;
			{
				tracked_safe_allocation<T> to = nullptr;
				tracking_allocator::reset();

				tracked_safe_allocation<T>& ref = (to = std::move(from));
				CHECK(&ref == &to);
				CHECK(to.data() == nullptr);
				CHECK(to.size() == 0);
				CHECK(from.data() == nullptr);
				CHECK(from.size() == 0);

				CHECK(!tracking_allocator::did_allocate);
				CHECK(!tracking_allocator::did_deallocate);
				tracking_allocator::reset();
			}
			CHECK(!tracking_allocator::did_deallocate);
			tracking_allocator::reset();
		}
		CHECK(!tracking_allocator::did_deallocate);
	}

	// Move-assign to self
	{
		{
			tracked_safe_allocation<T> obj(10, libkern::allocate_memory);
			T* memory = obj.data();

			tracking_allocator::reset();
			tracked_safe_allocation<T>& ref = (obj = std::move(obj));
			CHECK(&ref == &obj);
			CHECK(obj.data() == memory);
			CHECK(obj.size() == 10);
			CHECK(!tracking_allocator::did_allocate);
			CHECK(!tracking_allocator::did_deallocate);
			tracking_allocator::reset();
		}
		CHECK(tracking_allocator::deallocated_size == 10 * sizeof(T));
	}
}

T_DECL(assign_move, "safe_allocation.assign.move", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
