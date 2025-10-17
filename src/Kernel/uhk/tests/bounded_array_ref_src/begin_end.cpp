/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
//  iterator begin() const;
//  iterator end() const;
//

#include <libkern/c++/bounded_array_ref.h>
#include "test_policy.h"
#include <darwintest.h>
#include <type_traits>

struct T { int i; };

template <typename T>
static void
tests()
{
	using AR = test_bounded_array_ref<T>;

	// Check begin()/end() for a non-null array ref
	{
		T array[5] = {T{0}, T{1}, T{2}, T{3}, T{4}};
		AR const view(array);
		test_bounded_ptr<T> begin = view.begin();
		test_bounded_ptr<T> end = view.end();
		CHECK(begin.discard_bounds() == &array[0]);
		CHECK(end.unsafe_discard_bounds() == &array[5]);
	}

	// Check begin()/end() for a null array ref
	{
		AR const view;
		test_bounded_ptr<T> begin = view.begin();
		test_bounded_ptr<T> end = view.end();
		CHECK(begin.unsafe_discard_bounds() == nullptr);
		CHECK(end.unsafe_discard_bounds() == nullptr);
	}

	// Check associated types
	{
		static_assert(std::is_same_v<typename AR::iterator, test_bounded_ptr<T> >);
	}
}

T_DECL(begin_end, "bounded_array_ref.begin_end", T_META_TAG_VM_PREFERRED) {
	tests<T>();
}
