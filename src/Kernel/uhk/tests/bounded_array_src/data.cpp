/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
//  T* data();
//  T const* data() const;
//

#include <libkern/c++/bounded_array.h>
#include "test_policy.h"
#include <darwintest.h>
#include <type_traits>

struct T { int i; };
inline bool
operator==(T const& a, T const& b)
{
	return a.i == b.i;
}

template <typename T>
static void
tests()
{
	{
		test_bounded_array<T, 5> array = {T{0}, T{1}, T{2}, T{3}, T{4}};
		T* data = array.data();
		CHECK(data != nullptr);
		CHECK(data[0] == T{0});
		CHECK(data[1] == T{1});
		CHECK(data[2] == T{2});
		CHECK(data[3] == T{3});
		CHECK(data[4] == T{4});
	}
	{
		test_bounded_array<T, 5> const array = {T{0}, T{1}, T{2}, T{3}, T{4}};
		T const* data = array.data();
		CHECK(data != nullptr);
		CHECK(data[0] == T{0});
		CHECK(data[1] == T{1});
		CHECK(data[2] == T{2});
		CHECK(data[3] == T{3});
		CHECK(data[4] == T{4});
	}

	{
		test_bounded_array<T, 0> array = {};
		T* data = array.data();
		CHECK(data != nullptr);
	}
	{
		test_bounded_array<T, 0> const array = {};
		T const* data = array.data();
		CHECK(data != nullptr);
	}
}

T_DECL(data, "bounded_array.data", T_META_TAG_VM_PREFERRED) {
	tests<T>();
}
