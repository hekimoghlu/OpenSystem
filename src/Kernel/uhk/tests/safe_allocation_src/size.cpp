/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
//      size_t size() const;
//

#include <libkern/c++/safe_allocation.h>
#include <darwintest.h>
#include "test_utils.h"
#include <cstddef>
#include <type_traits>
#include <utility>

struct T {
	int i;
};

template <typename T>
static void
tests()
{
	{
		test_safe_allocation<T> const array(10, libkern::allocate_memory);
		CHECK(array.size() == 10);
	}
	{
		T* memory = reinterpret_cast<T*>(malloc_allocator::allocate(10 * sizeof(T)));
		test_safe_allocation<T> const array(memory, 10, libkern::adopt_memory);
		CHECK(array.size() == 10);
	}
	{
		test_safe_allocation<T> const array(nullptr, 0, libkern::adopt_memory);
		CHECK(array.size() == 0);
	}
	{
		test_safe_allocation<T> const array;
		CHECK(array.size() == 0);
	}

	{
		using Size = decltype(std::declval<test_safe_allocation<T> const&>().size());
		static_assert(std::is_same_v<Size, std::size_t>);
	}
}

T_DECL(size, "safe_allocation.size", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
