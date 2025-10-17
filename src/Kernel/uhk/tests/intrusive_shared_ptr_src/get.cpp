/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
//  pointer get() const noexcept;
//

#include <libkern/c++/intrusive_shared_ptr.h>
#include "test_policy.h"
#include <darwintest.h>
#include <utility>

struct T {
	int i;
};

template <typename T>
static constexpr auto
can_call_get_on_temporary(int)->decltype(std::declval<test_shared_ptr<T> >().get(), bool ())
{
	return true;
}

template <typename T>
static constexpr auto
can_call_get_on_temporary(...)->bool
{
	return false;
}

template <typename T>
static void
tests()
{
	{
		T obj{3};
		tracking_policy::reset();
		tracked_shared_ptr<T> const ptr(&obj, libkern::retain);
		T* raw = ptr.get();
		CHECK(raw == &obj);
		CHECK(ptr.get() == raw); // ptr didn't change
		CHECK(tracking_policy::retains == 1);
		CHECK(tracking_policy::releases == 0);
	}

	static_assert(!can_call_get_on_temporary<T>(int{}), "");
}

T_DECL(get, "intrusive_shared_ptr.get", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
