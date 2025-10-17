/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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
//  template <typename T, typename Policy>
//  bool operator==(std::nullptr_t, bounded_ptr<T, Policy> const& p);
//
//  template <typename T, typename Policy>
//  bool operator!=(std::nullptr_t, bounded_ptr<T, Policy> const& p);
//
//  template <typename T, typename Policy>
//  bool operator==(bounded_ptr<T, Policy> const& p, std::nullptr_t);
//
//  template <typename T, typename Policy>
//  bool operator!=(bounded_ptr<T, Policy> const& p, std::nullptr_t);
//

#include <libkern/c++/bounded_ptr.h>
#include <darwintest.h>
#include <darwintest_utils.h>
#include "test_utils.h"

#define _assert(...) T_ASSERT_TRUE((__VA_ARGS__), # __VA_ARGS__)

struct T { };

struct non_default_policy {
	static constexpr void
	trap(char const*)
	{
	}
};

template <typename T, typename QualT>
static void
tests()
{
	T t;

	{
		test_bounded_ptr<QualT> const ptr(&t, &t, &t + 1);
		_assert(!(ptr == nullptr));
		_assert(!(nullptr == ptr));
		_assert(ptr != nullptr);
		_assert(nullptr != ptr);
	}
	{
		test_bounded_ptr<QualT> const ptr = nullptr;
		_assert(ptr == nullptr);
		_assert(nullptr == ptr);
		_assert(!(ptr != nullptr));
		_assert(!(nullptr != ptr));
	}

	// Test with a custom policy
	{
		libkern::bounded_ptr<QualT, non_default_policy> const ptr = nullptr;
		_assert(ptr == nullptr);
		_assert(nullptr == ptr);
		_assert(!(ptr != nullptr));
		_assert(!(nullptr != ptr));
	}
}

T_DECL(compare_equal_nullptr, "bounded_ptr.compare.equal.nullptr", T_META_TAG_VM_PREFERRED) {
	tests<T, T>();
	tests<T, T const>();
	tests<T, T volatile>();
	tests<T, T const volatile>();
}
