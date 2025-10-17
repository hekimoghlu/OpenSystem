/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
//  template <size_t N>
//  bounded_array_ref(T (&array)[N]);
//

#include <libkern/c++/bounded_array_ref.h>
#include "test_policy.h"
#include <darwintest.h>
#include <darwintest_utils.h>

struct T { int i; };
inline bool
operator==(T const& a, T const& b)
{
	return a.i == b.i;
};

template <typename T>
static void
tests()
{
	{
		T array[5] = {T{0}, T{1}, T{2}, T{3}, T{4}};
		test_bounded_array_ref<T> view(array);
		CHECK(view.data() == &array[0]);
		CHECK(view.size() == 5);
		CHECK(view[0] == T{0});
		CHECK(view[1] == T{1});
		CHECK(view[2] == T{2});
		CHECK(view[3] == T{3});
		CHECK(view[4] == T{4});
	}

	{
		T array[1] = {T{11}};
		test_bounded_array_ref<T> view(array);
		CHECK(view.data() == &array[0]);
		CHECK(view.size() == 1);
		CHECK(view[0] == T{11});
	}

	// Also test implicit construction
	{
		T array[1] = {T{11}};
		test_bounded_array_ref<T> view = array;
		CHECK(view.data() == &array[0]);
		CHECK(view.size() == 1);
	}
	{
		T array[1] = {T{11}};
		auto check = [&array](test_bounded_array_ref<T> view) {
			    CHECK(view.data() == &array[0]);
			    CHECK(view.size() == 1);
		    };
		check(array);
	}
}

T_DECL(ctor_C_array, "bounded_array_ref.ctor.C_array", T_META_TAG_VM_PREFERRED) {
	tests<T>();
}
