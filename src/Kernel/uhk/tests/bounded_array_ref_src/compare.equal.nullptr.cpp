/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
//  template <typename T, typename P>
//  bool operator==(bounded_array_ref<T, P> const& x, std::nullptr_t);
//
//  template <typename T, typename P>
//  bool operator!=(bounded_array_ref<T, P> const& x, std::nullptr_t);
//
//  template <typename T, typename P>
//  bool operator==(std::nullptr_t, bounded_array_ref<T, P> const& x);
//
//  template <typename T, typename P>
//  bool operator!=(std::nullptr_t, bounded_array_ref<T, P> const& x);
//

#include <libkern/c++/bounded_array_ref.h>
#include "test_policy.h"
#include <darwintest.h>
#include <darwintest_utils.h>

struct T { int i; };

template <typename T>
static void
tests()
{
	{
		T array[5] = {T{0}, T{1}, T{2}, T{3}, T{4}};
		test_bounded_array_ref<T> view(array);
		CHECK(!(view == nullptr));
		CHECK(!(nullptr == view));
		CHECK(view != nullptr);
		CHECK(nullptr != view);
	}
	{
		test_bounded_array_ref<T> view;
		CHECK(view == nullptr);
		CHECK(nullptr == view);
		CHECK(!(view != nullptr));
		CHECK(!(nullptr != view));
	}
}

T_DECL(compare_equal_nullptr, "bounded_array_ref.compare.equal.nullptr", T_META_TAG_VM_PREFERRED) {
	tests<T>();
}
