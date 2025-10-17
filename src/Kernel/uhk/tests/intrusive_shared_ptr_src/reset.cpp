/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
//  void reset() noexcept;
//

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest.h>
#include "test_policy.h"

struct T {
	int i;
};

template <typename T>
static void
tests()
{
	T obj{3};

	// reset() on a non-null shared pointer
	{
		tracked_shared_ptr<T> ptr(&obj, libkern::retain);
		tracking_policy::reset();
		ptr.reset();
		CHECK(tracking_policy::releases == 1);
		CHECK(tracking_policy::retains == 0);
		CHECK(ptr.get() == nullptr);
	}

	// reset() on a null shared pointer
	{
		tracked_shared_ptr<T> ptr = nullptr;
		tracking_policy::reset();
		ptr.reset();
		CHECK(tracking_policy::releases == 0);
		CHECK(tracking_policy::retains == 0);
		CHECK(ptr.get() == nullptr);
	}

	// reset() as a self-reference
	{
		tracked_shared_ptr<T> ptr(&obj, libkern::retain);
		tracked_shared_ptr<T> ptr2(&obj, libkern::retain);
		CHECK(!ptr.reset());

		CHECK(&ptr.reset() == &ptr);

		// check short-circuiting
		bool ok =  (ptr.reset() && !ptr2.reset());
		CHECK(ptr2.get() != nullptr);
	}
}

T_DECL(reset, "intrusive_shared_ptr.reset", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
